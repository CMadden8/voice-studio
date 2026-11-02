"""Data safety checks for persistent voice revisions and explicit exports."""
from pathlib import Path
import json

import pytest

import library


def test_seeded_presets_are_stable_and_returned_data_is_detached(tmp_path):
    voices = library.VoiceLibrary(tmp_path / "Voices")
    initial = voices.list_voices()
    assert {voice["name"] for voice in initial} == {preset["name"] for preset in library.PRESETS}
    assert len(initial) == 18
    assert all(voice["builtin"] and voice["version"] == 1 for voice in initial)
    assert [voice["name"] for voice in initial] == sorted([voice["name"] for voice in initial], key=str.casefold)
    initial[0]["settings"]["pitch"] = 8
    assert voices.get(initial[0]["id"])["settings"]["pitch"] != 8
    reloaded = library.VoiceLibrary(tmp_path / "Voices")
    assert [voice["id"] for voice in voices.list_voices()] == [voice["id"] for voice in reloaded.list_voices()]
    assert all(voice["description"] for voice in initial)
    giant = next(voice for voice in initial if voice["name"] == "Giant")
    assert giant["settings"]["pitch"] < -5 and giant["settings"]["resonance"] < .8
    assert .4 <= giant["settings"]["cleanup"] <= .55


def test_character_revisions_duplicate_and_archive_survive_reopen(tmp_path):
    voices = library.VoiceLibrary(tmp_path / "Voices")
    original = voices.create("Gatekeeper", {"pitch": -2, "resonance": .92})
    revision = voices.update(original["id"], {"pitch": -3, "tone": -.1}, name="Gatekeeper elder")
    assert revision["version"] == 2
    assert voices.get(original["id"], 1) == original
    assert voices.get(original["id"]) == revision
    assert [v["version"] for v in voices.history(original["id"])] == [2, 1]
    duplicate = voices.duplicate(original["id"], "Gatekeeper variant")
    assert duplicate["id"] != original["id"] and duplicate["version"] == 1
    assert duplicate["settings"] == revision["settings"]
    voices.archive(original["id"])
    reloaded = library.VoiceLibrary(tmp_path / "Voices")
    assert original["id"] not in {v["id"] for v in reloaded.list_voices()}
    assert reloaded.get(original["id"], 1) == original
    with pytest.raises(ValueError, match="archived"):
        voices.update(original["id"], {})
    # Archiving a seeded voice must not make it reappear during seeding.
    builtin = next(v for v in voices.list_voices() if v["builtin"])
    voices.archive(builtin["id"])
    assert builtin["id"] not in {v["id"] for v in library.VoiceLibrary(tmp_path / "Voices").list_voices()}


@pytest.mark.parametrize("settings", [{"pitch": float("nan")}, {"pitch": float("inf")},
                                        {"pitch": 9}, {"speed": 0}, {"resonance": .5},
                                        {"room": True}, {"unknown": 1}, {"tone": "loud"},
                                        {"cleanup": -.1}, {"cleanup": 1.1}])
def test_invalid_settings_never_touch_existing_revision(tmp_path, settings):
    voices = library.VoiceLibrary(tmp_path / "Voices")
    voice = voices.create("My character", {})
    before = voices._path(voice["id"]).read_bytes()
    with pytest.raises(ValueError):
        voices.update(voice["id"], settings)
    assert voices._path(voice["id"]).read_bytes() == before
    assert voices.get(voice["id"]) == voice


def test_failed_atomic_save_keeps_all_previous_revisions(monkeypatch, tmp_path):
    voices = library.VoiceLibrary(tmp_path / "Voices")
    voice = voices.create("My character", {})
    revision = voices.update(voice["id"], {"pitch": 2})
    before = voices._path(voice["id"]).read_bytes()

    def fail_replace(_self, _target):
        raise OSError("simulated write failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        voices.update(voice["id"], {"pitch": 3})
    assert voices._path(voice["id"]).read_bytes() == before
    assert voices.get(voice["id"]) == revision
    assert len(voices.history(voice["id"])) == 2
    assert not list(voices.folder.glob("*.tmp"))


def test_validation_of_names_ids_and_revision_numbers(tmp_path):
    voices = library.VoiceLibrary(tmp_path / "Voices")
    for name in ("", "  ", "bad\x00name", "a" * 81):
        with pytest.raises(ValueError):
            voices.create(name, {})
    with pytest.raises(ValueError, match="already exists"):
        voices.create("original", {})
    with pytest.raises(ValueError, match="identifier"):
        voices.get("../../escape")
    voice = voices.create("Name with: character?", {})
    assert voices._path(voice["id"]).parent == voices.folder
    for version in (0, -1, 2, True, "1"):
        with pytest.raises(KeyError):
            voices.get(voice["id"], version)


@pytest.mark.parametrize(("source", "expected"), [
    ("CON", "_CON"), ("con.wav", "_con.wav"), ("LPT1", "_LPT1"),
    (" ../Gatekeeper: hello? ", "-Gatekeeper- hello-"), ("...", "Untitled"),
    ("Áine's voice", "Áine's voice"), ("name. ", "name"),
])
def test_windows_safe_names(source, expected):
    assert library.safe_name(source) == expected


def test_unique_copy_preserves_original_and_numbered_previous_exports(tmp_path):
    source = tmp_path / "Recording.wav"
    source.write_bytes(b"first take\x00\xff")
    destination = tmp_path / "Exports"
    first = library.unique_copy(source, destination, "Gatekeeper.wav")
    assert first.read_bytes() == source.read_bytes()
    assert library.unique_copy(source, destination, "Gatekeeper.wav") == first
    source.write_bytes(b"second take\x00\xff")
    second = library.unique_copy(source, destination, "Gatekeeper.wav")
    assert second.name == "Gatekeeper (2).wav"
    assert first.read_bytes() == b"first take\x00\xff"
    assert second.read_bytes() == source.read_bytes() == b"second take\x00\xff"
    assert library.unique_copy(source, destination, "Gatekeeper.wav") == second
    assert library.unique_copy(source, source.parent) == source


def test_interrupted_copy_removes_own_partial_file_only(monkeypatch, tmp_path):
    source = tmp_path / "Recording.wav"
    source.write_bytes(b"keep this recording")
    exports = tmp_path / "Exports"
    exports.mkdir()
    previous = exports / source.name
    previous.write_bytes(b"previously approved take")

    def fail_copy(_source, target, **_kwargs):
        target.write(b"incomplete")
        raise OSError("simulated copy failure")

    monkeypatch.setattr(library.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="simulated"):
        library.unique_copy(source, exports)
    assert source.read_bytes() == b"keep this recording"
    assert previous.read_bytes() == b"previously approved take"
    assert not (exports / "Recording (2).wav").exists()


def test_legacy_voice_reads_expand_defaults_without_rewriting_history(tmp_path):
    import uuid
    folder = tmp_path / "Voices"
    folder.mkdir()
    voice_id = uuid.uuid4().hex
    legacy_voice = {"id": voice_id, "name": "My existing guard", "version": 1,
                    "settings": {"pitch": -2., "resonance": .9, "tone": -.1,
                                 "texture": 0., "room": 0., "speed": 1.}, "builtin": False}
    path = folder / f"{voice_id}.json"
    path.write_text(json.dumps({"schema": 1, "id": voice_id, "archived": False,
                                "versions": [legacy_voice]}), encoding="utf-8")
    before = path.read_bytes()
    voices = library.VoiceLibrary(folder)
    normal = voices.get(voice_id)
    assert normal["settings"]["cleanup"] == 0.
    assert normal["description"] == ""
    assert voices.history(voice_id) == [normal]
    assert next(v for v in voices.list_voices() if v["id"] == voice_id) == normal
    assert path.read_bytes() == before, "Merely opening a library must not edit old voice files"
    normal["settings"]["cleanup"] = .5
    revised = voices.update(voice_id, normal["settings"], description="A guarded, clean character.")
    assert revised["version"] == 2 and revised["settings"]["cleanup"] == .5
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["versions"][0] == legacy_voice, "New revisions must retain the original stored revision unchanged"
    assert voices.get(voice_id, 1)["settings"]["cleanup"] == 0.
    duplicate = voices.duplicate(voice_id, "Another clean guard")
    assert duplicate["description"] == revised["description"]


def test_catalogue_upgrade_keeps_edits_and_archived_builtins_byte_exact(tmp_path):
    voices = library.VoiceLibrary(tmp_path / "Voices")
    guard = next(v for v in voices.list_voices() if v["name"] == "Deep guard")
    revised = voices.update(guard["id"], {"pitch": 2., "cleanup": .2}, name="My guard")
    voices.archive(guard["id"])
    path = voices._path(guard["id"])
    before = path.read_bytes()
    reloaded = library.VoiceLibrary(voices.folder)
    assert path.read_bytes() == before
    assert reloaded.get(guard["id"]) == revised
    assert guard["id"] not in {v["id"] for v in reloaded.list_voices()}


@pytest.mark.parametrize("archived", [False, True])
def test_new_catalogue_does_not_shadow_a_user_character_with_same_name(tmp_path, archived):
    import uuid
    folder = tmp_path / "Voices"
    folder.mkdir()
    voice_id = uuid.uuid4().hex
    voice = {"id": voice_id, "name": "Giant", "version": 1, "settings": {"pitch": -1.}, "builtin": False}
    path = folder / f"{voice_id}.json"
    path.write_text(json.dumps({"schema": 1, "id": voice_id, "archived": archived, "versions": [voice]}), encoding="utf-8")
    before = path.read_bytes()
    voices = library.VoiceLibrary(folder)
    assert path.read_bytes() == before
    giants = [v for v in voices.list_voices() if v["name"] == "Giant"]
    assert [v["id"] for v in giants] == ([] if archived else [voice_id])


def test_descriptions_are_validated_and_versioned(tmp_path):
    voices = library.VoiceLibrary(tmp_path / "Voices")
    first = voices.create("A described voice", {}, description="First direction.")
    second = voices.update(first["id"], {"cleanup": .4}, description="Second direction.")
    assert first["description"] == "First direction."
    assert voices.get(first["id"], 1)["description"] == "First direction."
    assert second["description"] == "Second direction."
    before = voices._path(first["id"]).read_bytes()
    for description in (None, 42, "bad\x00description", "a" * 361):
        with pytest.raises(ValueError):
            voices.create("Invalid description", {}, description=description)
    with pytest.raises(ValueError):
        voices.update(first["id"], {}, description="a" * 361)
    assert voices._path(first["id"]).read_bytes() == before
