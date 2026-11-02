"""End-to-end recorder/preset/export checks with real non-AI audio workers.

The conventional workflow uses real audio workers. Optional Resolve integration
checks control the worker boundary, so they never open or alter a Resolve project.
All files and preset changes live in pytest's temporary directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


def wait_until(qt, condition, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt.processEvents()
        if condition():
            return
        time.sleep(.015)
    raise AssertionError("The voice app did not finish its real audio worker in time")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def qt():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def speech():
    rate = 48000
    t = np.arange(round(2.8 * rate)) / rate
    phase = 2 * np.pi * (155 * t + 3.5 * np.sin(2 * np.pi * 1.1 * t))
    voiced = sum(np.sin(phase * harmonic) / harmonic for harmonic in range(1, 9))
    envelope = (.65 + .3 * np.sin(2 * np.pi * 2.7 * t))
    envelope *= (t > .12) & (t < 2.55)
    # Include normal unvoiced material and pauses between voiced phrases.
    envelope[(t > 1.05) & (t < 1.28)] = 0
    audio = (voiced * envelope * .12).astype(np.float32)
    rng = np.random.default_rng(17)
    unvoiced = (t > 1.10) & (t < 1.22)
    audio[unvoiced] = rng.normal(0, .035, int(unvoiced.sum()))
    return audio, rate


@pytest.fixture
def window(qt, tmp_path, monkeypatch):
    import app as ui
    app_window = ui.VoiceStudio(root=tmp_path / "voice-project", restore=False)
    app_window.show()
    qt.processEvents()
    played = []
    monkeypatch.setattr(app_window, "_play", lambda audio, rate, kind: played.append((np.asarray(audio).copy(), rate, kind)))
    yield app_window, played
    app_window.close()
    qt.processEvents()


def nonneutral_voice(window):
    choices = window.library.list_voices()
    preferred = next((voice for voice in choices if voice["name"].lower() == "deep guard"), None)
    return preferred or next(voice for voice in choices if voice["name"].lower() != "original")


def settle(qt, window, previous=None):
    wait_until(qt, lambda: window.job is None and window.rendered is not None
               and window.rendered_path is not None
               and (previous is None or window.rendered_path != previous))


def test_record_stop_and_original_comparison_use_real_voice_worker(window, qt, speech, monkeypatch):
    app_window, played = window
    audio, rate = speech
    voice = nonneutral_voice(app_window)
    app_window._select_voice(voice["id"])

    def start(_device=None):
        app_window.recorder.stream = object()
        app_window.recorder.started = time.monotonic()

    def stop():
        app_window.recorder.stream = None
        return audio.copy(), rate

    monkeypatch.setattr(app_window.recorder, "start", start)
    monkeypatch.setattr(app_window.recorder, "stop", stop)
    QTest.mouseClick(app_window.record_button, Qt.MouseButton.LeftButton)
    assert app_window.recorder.stream is not None
    assert not app_window.listen_button.isEnabled()
    QTest.mouseClick(app_window.record_button, Qt.MouseButton.LeftButton)
    settle(qt, app_window)
    assert app_window.source.exists()
    assert app_window.source.is_relative_to(app_window.root)
    captured, captured_rate = sf.read(app_window.source, dtype="float32")
    assert captured_rate == rate
    assert np.allclose(captured, audio, atol=1 / 32768)
    assert np.isfinite(app_window.rendered).all()
    assert app_window.listen_button.isEnabled()
    assert not list(app_window.export_root.rglob("*.wav")), "Raw takes must not fill game export folders"
    QTest.mouseClick(app_window.original_button, Qt.MouseButton.LeftButton)
    assert played and played[-1][2] == "original"
    assert np.allclose(played[-1][0], captured, atol=1 / 32768)


def test_voice_revisions_and_exports_keep_approved_audio_and_exact_original(window, qt, speech, tmp_path):
    app_window, _played = window
    audio, rate = speech
    imported = tmp_path / "Line - imported.wav"
    sf.write(imported, audio, rate, subtype="PCM_16")
    imported_hash = digest(imported)
    app_window.load_recording(imported)
    settle(qt, app_window)
    assert app_window.source != imported and digest(app_window.source) == imported_hash
    previous = app_window.rendered_path
    app_window.pitch.setValue(app_window.pitch.minimum() // 2)
    settle(qt, app_window, previous)
    created = app_window.save_new_voice("Test sentinel guard")
    settle(qt, app_window)
    assert created["version"] == 1
    assert app_window.selected_voice["id"] == created["id"]
    folder = tmp_path / "explicit-game-export"
    paths = [Path(p) for p in app_window.export_audio("Gate warning", folder, include_original=True)]
    assert len([path for path in paths if path.suffix.lower() == ".wav"]) == 2
    assert all(path.is_relative_to(folder) and path.exists() for path in paths)
    originals = [path for path in paths if path.suffix.lower() == ".wav" and digest(path) == imported_hash]
    assert len(originals) == 1, "Original export must preserve the imported WAV byte-for-byte"
    prior_hashes = {path: digest(path) for path in paths}
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in app_window.export_history_path.glob("*.json")]
    assert len(reports) == 1
    first_report = reports[0]
    assert first_report["voice"]["id"] == created["id"]
    assert first_report["voice"]["version"] == 1
    assert first_report["settings"] == created["settings"]
    assert first_report["backend"] and first_report["ai"] is False
    assert first_report["modified_preset"] is False
    assert first_report["source_sha256"] == imported_hash
    assert set(first_report["files"]) == {str(path) for path in paths}
    assert first_report["sha256"] == {str(path): digest(path) for path in paths}
    previous = app_window.rendered_path
    next_pitch = max(app_window.pitch.minimum(), min(app_window.pitch.maximum(), app_window.pitch.value() + 12))
    app_window.pitch.setValue(next_pitch)
    settle(qt, app_window, previous)
    app_window.update_voice()
    settle(qt, app_window)
    assert app_window.selected_voice["version"] == 2
    assert app_window.library.get(created["id"], version=1)["settings"] == created["settings"]
    second_paths = [Path(p) for p in app_window.export_audio("Gate warning", folder, include_original=True)]
    assert all(digest(path) == original_digest for path, original_digest in prior_hashes.items())
    assert any(path not in prior_hashes for path in second_paths), "A changed approved line needs a distinct output"
    assert digest(imported) == imported_hash and digest(app_window.source) == imported_hash
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in app_window.export_history_path.glob("*.json")]
    assert len(reports) == 2
    assert {report["voice"]["version"] for report in reports} == {1, 2}
    assert first_report in reports, "Updating a voice must not rewrite an earlier export's provenance"


def test_unvoiced_import_is_processed_and_restore_does_not_autoplay(window, qt, tmp_path):
    app_window, played = window
    rate = 48000
    t = np.arange(rate * 2) / rate
    rng = np.random.default_rng(29)
    whisper = (rng.normal(0, .025, len(t)) * (.65 + .3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    path = tmp_path / "Whisper without stable pitch.wav"
    sf.write(path, whisper, rate, subtype="PCM_16")
    source_hash = digest(path)
    app_window._select_voice(nonneutral_voice(app_window)["id"])
    app_window.load_recording(path)
    settle(qt, app_window)
    assert np.isfinite(app_window.rendered).all()
    assert np.max(np.abs(app_window.rendered)) > .001
    assert digest(app_window.source) == source_hash
    assert not list(app_window.export_root.rglob("*.wav"))
    count = len(played)
    app_window._restore_session()
    settle(qt, app_window)
    assert len(played) == count, "Reopening a project should not unexpectedly play a recording"
    assert digest(app_window.source) == source_hash


def test_rapid_edits_export_only_the_latest_render_snapshot(window, qt, speech, tmp_path):
    app_window, _played = window
    audio, rate = speech
    path = tmp_path / "Rapidly edited dialogue.wav"
    sf.write(path, audio, rate, subtype="PCM_16")
    app_window.load_recording(path)
    # Change several voices while the first real worker is still running.
    # A worker that finishes late must never make an earlier voice saveable.
    choices = [voice for voice in app_window.library.list_voices() if voice["name"].lower() != "original"]
    for choice in choices[:3]:
        app_window._select_voice(choice["id"])
    expected = app_window.selected_voice.copy()
    folder = tmp_path / "approved-lines"
    with pytest.raises(ValueError, match="finish"):
        app_window.export_audio("Do not save an old voice", folder)
    assert not folder.exists()
    settle(qt, app_window)
    assert app_window.rendered_snapshot["voice"]["id"] == expected["id"]
    assert app_window.rendered_snapshot["settings"] == app_window.settings()
    prior = app_window.rendered_path
    app_window.tone.setValue(-60 if app_window.tone.value() != -60 else 60)
    with pytest.raises(ValueError, match="finish"):
        app_window.export_audio("Do not save a stale draft", folder)
    assert not folder.exists()
    settle(qt, app_window, prior)
    expected_settings = app_window.settings().copy()
    assert app_window.rendered_snapshot["modified_preset"] is True
    app_window.export_audio("Latest edited voice", folder, include_original=False)
    reports = [json.loads(report.read_text(encoding="utf-8"))
               for report in app_window.export_history_path.glob("*.json")]
    assert len(reports) == 1
    assert reports[0]["settings"] == expected_settings
    assert reports[0]["voice"]["id"] == expected["id"]
    assert reports[0]["modified_preset"] is True and reports[0]["ai"] is False


def test_closing_during_capture_retains_recoverable_original(window, qt, speech, monkeypatch):
    app_window, _played = window
    audio, rate = speech

    def start(_device=None):
        app_window.recorder.stream = object()
        app_window.recorder.started = time.monotonic()

    def stop():
        app_window.recorder.stream = None
        return audio.copy(), rate

    monkeypatch.setattr(app_window.recorder, "start", start)
    monkeypatch.setattr(app_window.recorder, "stop", stop)
    QTest.mouseClick(app_window.record_button, Qt.MouseButton.LeftButton)
    app_window.close()
    qt.processEvents()
    recovered = list(app_window.recordings.glob("Recovered-*.wav"))
    assert len(recovered) == 1
    saved, saved_rate = sf.read(recovered[0], dtype="float32")
    assert saved_rate == rate and np.allclose(saved, audio, atol=1 / 32768)
    session = json.loads(app_window.state_path.read_text(encoding="utf-8"))
    assert Path(session["source"]) == recovered[0]
    assert not list(app_window.export_root.rglob("*.wav"))


def test_worker_cache_write_failure_reports_completion_instead_of_hanging(qt, tmp_path, monkeypatch):
    import app as ui
    cache = tmp_path / "unwritable-cache"
    cache.mkdir()
    context = {"revision": 7}
    job = ui.RenderJob({"kind": "render"}, context, cache)
    completed = []
    job.completed.connect(completed.append)

    def fail_write(_path, _value):
        raise OSError("simulated cache permission failure")

    def must_not_spawn(*_args, **_kwargs):
        raise AssertionError("No audio process should start after cache write failure")

    monkeypatch.setattr(ui, "write_json", fail_write)
    monkeypatch.setattr(ui.subprocess, "Popen", must_not_spawn)
    job.run()
    qt.processEvents()
    assert len(completed) == 1
    assert completed[0]["ok"] is False
    assert completed[0]["error"] and completed[0]["context"] == context
    assert not list(cache.iterdir())


def test_precancelled_worker_never_writes_or_spawns(qt, tmp_path, monkeypatch):
    import app as ui
    cache = tmp_path / "cancelled-cache"
    cache.mkdir()
    job = ui.RenderJob({"kind": "render"}, {}, cache)
    completed = []
    job.completed.connect(completed.append)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("A cancelled preview must not write a job or start a process")

    monkeypatch.setattr(ui, "write_json", must_not_run)
    monkeypatch.setattr(ui.subprocess, "Popen", must_not_run)
    job.cancel()
    job.run()
    qt.processEvents()
    assert completed == [] and job.process is None
    assert not list(cache.iterdir())


def test_cancellation_during_spawn_stops_the_new_process(qt, tmp_path, monkeypatch):
    import app as ui
    cache = tmp_path / "spawn-race-cache"
    cache.mkdir()
    job = ui.RenderJob({"kind": "render"}, {}, cache)
    completed = []
    job.completed.connect(completed.append)

    class JustStartedProcess:
        killed = 0
        waited = 0

        def kill(self):
            self.killed += 1

        def communicate(self, **_kwargs):
            self.waited += 1
            return "", ""

    process = JustStartedProcess()

    def spawn_during_cancel(*_args, **_kwargs):
        # Simulate a close arriving after the last pre-spawn check, before
        # Popen returns and the worker can assign its process reference.
        job.cancel()
        return process

    monkeypatch.setattr(ui.subprocess, "Popen", spawn_during_cancel)
    job.run()
    qt.processEvents()
    assert process.killed == 1 and process.waited == 1
    assert completed == [] and not list(cache.iterdir())


def test_giant_and_cleanup_controls_render_and_export_the_effective_recipe(window, qt, speech, tmp_path):
    app_window, played = window
    audio, rate = speech
    noise = np.random.default_rng(64).normal(0, .006, len(audio))
    source = tmp_path / "Quiet dialogue with microphone hiss.wav"
    sf.write(source, audio + noise, rate, subtype="PCM_16")
    source_hash = digest(source)
    giant = next(voice for voice in app_window.library.list_voices() if voice["name"] == "Giant")
    assert giant.get("description"), "Character presets need a useful description"
    assert giant["settings"]["pitch"] < 0 and giant["settings"]["resonance"] < 1
    app_window._select_voice(giant["id"])
    assert app_window.settings() == giant["settings"]
    assert app_window.voice_description.text() == giant["description"]
    app_window.load_recording(source)
    settle(qt, app_window)
    assert app_window.cleaned_path and app_window.cleaned_path.exists()
    assert app_window.cleaned_button.isEnabled()
    cleaned, cleaned_rate = sf.read(app_window.cleaned_path, dtype="float32")
    app_window.play_cleaned()
    assert played[-1][2] == "cleaned" and played[-1][1] == cleaned_rate
    assert np.array_equal(played[-1][0], cleaned)
    folder = tmp_path / "explicit-character-lines"
    approved = app_window.export_audio("Giant warning", folder, include_original=True)
    approved_hashes = {Path(path): digest(path) for path in approved}
    old_render = app_window.rendered_path
    app_window.cleanup.setValue(70)
    assert app_window.settings()["cleanup"] == .70
    with pytest.raises(ValueError, match="finish"):
        app_window.export_audio("Stale cleanup must not export", folder)
    settle(qt, app_window, old_render)
    assert app_window.rendered_snapshot["settings"]["cleanup"] == .70
    cleanup_report = app_window.rendered_snapshot["cleanup"]
    assert cleanup_report["strength"] == .70 and cleanup_report["enabled"] is True
    assert cleanup_report["method"]
    assert {"profile_reliable", "noise_reduction_db", "input_noise_dbfs", "cleaned_noise_dbfs"} <= set(cleanup_report)
    assert app_window.rendered_snapshot["modified_preset"] is True
    assert np.isfinite(app_window.rendered).all()
    assert digest(source) == source_hash and digest(app_window.source) == source_hash
    app_window.export_audio("Giant warning", folder, include_original=True)
    assert all(digest(path) == original_hash for path, original_hash in approved_hashes.items())
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in app_window.export_history_path.glob("*.json")]
    edited = next(report for report in reports if report["settings"]["cleanup"] == .70)
    assert edited["voice"]["id"] == giant["id"]
    assert edited["voice"]["settings"]["cleanup"] == .70
    assert edited["ai"] is False and edited["modified_preset"] is True
    assert edited["cleanup"] == cleanup_report, "Exports must retain actual cleanup diagnostics"
    original_voice = next(voice for voice in app_window.library.list_voices() if voice["name"] == "Original")
    app_window._select_voice(original_voice["id"])
    settle(qt, app_window)
    assert app_window.cleanup.value() == 0
    assert not app_window.cleaned_button.isEnabled()
    unchanged, unchanged_rate = sf.read(app_window.source, dtype="float32")
    assert unchanged_rate == rate and np.array_equal(app_window.rendered, unchanged)


def test_legacy_session_gets_working_cleanup_without_rewriting_saved_voice(window, qt, speech, tmp_path):
    app_window, played = window
    audio, rate = speech
    source = tmp_path / "Existing approved performance.wav"
    sf.write(source, audio, rate, subtype="PCM_16")
    legacy = app_window.library.create("Legacy keeper", {"pitch": -1.5, "tone": -.2, "cleanup": 0.})
    app_window._refresh_voice_list()
    app_window._select_voice(legacy["id"])
    app_window.load_recording(source)
    settle(qt, app_window)
    local_source = app_window.source
    local_hash = digest(local_source)
    voice_file = app_window.library.folder / (legacy["id"] + ".json")
    voice_bytes = voice_file.read_bytes()
    saved = json.loads(app_window.state_path.read_text(encoding="utf-8"))
    saved["settings"].pop("cleanup")
    app_window.state_path.write_text(json.dumps(saved), encoding="utf-8")
    previous = app_window.rendered_path
    prior_play_count = len(played)
    app_window._restore_session()
    settle(qt, app_window, previous)
    assert app_window.source == local_source and digest(local_source) == local_hash
    assert app_window.settings()["cleanup"] == .45
    assert app_window.settings()["pitch"] == -1.5 and app_window.settings()["tone"] == -.2
    assert app_window.selected_voice["settings"]["cleanup"] == 0
    assert app_window.selected_voice["version"] == 1 and voice_file.read_bytes() == voice_bytes
    assert app_window.rendered_snapshot["modified_preset"] is True
    assert len(played) == prior_play_count, "A migrated session must still restore without autoplay"
    assert not list(app_window.export_root.rglob("*.wav"))
    original = next(voice for voice in app_window.library.list_voices() if voice["name"] == "Original")
    app_window._select_voice(original["id"])
    settle(qt, app_window)
    saved = json.loads(app_window.state_path.read_text(encoding="utf-8"))
    saved["settings"].pop("cleanup")
    app_window.state_path.write_text(json.dumps(saved), encoding="utf-8")
    previous = app_window.rendered_path
    app_window._restore_session()
    settle(qt, app_window, previous)
    assert app_window.settings()["cleanup"] == 0, "The Original preset must remain an untouched comparison"
    assert app_window.rendered_snapshot["modified_preset"] is False


@pytest.fixture
def controlled_workers(monkeypatch):
    """Control asynchronous completion without accessing a real Resolve project."""
    import app as ui
    created = []

    class ControlledRenderJob(QObject):
        completed = Signal(dict)
        finished = Signal()

        def __init__(self, specification, context, cache, parent=None):
            super().__init__(parent)
            self.specification = copy.deepcopy(specification)
            self.context = copy.deepcopy(context)
            self.cancelled = False
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

        def wait(self, _timeout):
            return True

        def complete(self, response):
            self.completed.emit({**response, "context": copy.deepcopy(self.context)})
            self.finished.emit()

    monkeypatch.setattr(ui, "RenderJob", ControlledRenderJob)
    return created


def complete_controlled_render(worker, audio, rate, *, provider="resolve"):
    """Supply distinct preview files plus the worker's completed provenance."""
    output = Path(worker.specification["output"])
    cleaned = Path(worker.specification["cleanup_output"])
    sf.write(output, np.asarray(audio) * .45, rate, subtype="FLOAT")
    sf.write(cleaned, np.asarray(audio) * .75, rate, subtype="FLOAT")
    strength = worker.context["settings"]["cleanup"]
    uses_ai = provider == "resolve"
    result = {
        "audio": str(output), "cleaned_audio": str(cleaned),
        "backend": "Conventional voice effects",
        "warning": "", "ai": uses_ai, "cleanup_provider": provider,
        "cleanup": {"enabled": strength > 0, "strength": strength,
                    "method": "Resolve Voice Isolation" if uses_ai else "Spectral noise reduction"},
    }
    if uses_ai:
        result["resolve"] = {"amount": round(strength * 100), "version": "21.0.test",
                             "project": "Voice Studio Cleanup"}
    worker.complete({"ok": True, "result": result})
    return result


def test_optional_resolve_choice_preserves_voice_recipes_and_exports_ai_provenance(
        window, qt, speech, tmp_path, controlled_workers):
    app_window, played = window
    audio, rate = speech
    assert app_window.cleanup_provider.currentData() == "conventional"
    assert app_window.ai_badge.text() == "LOCAL · NO AI"
    giant = next(voice for voice in app_window.library.list_voices() if voice["name"] == "Giant")
    voice_file = app_window.library.folder / (giant["id"] + ".json")
    old_voice_bytes = voice_file.read_bytes()
    app_window._select_voice(giant["id"])
    recipe = app_window.settings()
    app_window.cleanup_provider.setCurrentIndex(app_window.cleanup_provider.findData("resolve"))
    assert app_window.settings() == recipe, "The provider is a take setting, not part of a voice recipe"
    assert app_window.ai_badge.text() == "LOCAL · AI CLEANUP"
    app_window._select_voice(nonneutral_voice(app_window)["id"])
    assert app_window.cleanup_provider.currentData() == "resolve"
    app_window._select_voice(giant["id"])
    app_window.cleanup.setValue(75)
    source = tmp_path / "Private original performance.wav"
    sf.write(source, audio, rate, subtype="PCM_16")
    source_hash = digest(source)
    app_window.load_recording(source)
    worker = controlled_workers[-1]
    assert worker.started
    assert worker.specification["cleanup_provider"] == "resolve"
    assert worker.context["cleanup_provider"] == "resolve"
    assert worker.specification["settings"]["cleanup"] == .75
    assert set(worker.specification["settings"]) == set(recipe)
    with pytest.raises(ValueError, match="finish"):
        app_window.export_audio("Pending AI cleanup", tmp_path / "exports")
    expected = complete_controlled_render(worker, audio, rate)
    qt.processEvents()
    assert app_window.rendered_snapshot["ai"] is True
    assert app_window.rendered_snapshot["cleanup_provider"] == "resolve"
    assert app_window.rendered_snapshot["resolve"] == expected["resolve"]
    assert app_window.cleaned_button.isEnabled() and app_window.listen_button.isEnabled()
    app_window.play_cleaned()
    assert played[-1][2] == "cleaned"
    assert np.array_equal(played[-1][0], sf.read(app_window.cleaned_path, dtype="float32")[0])
    files = app_window.export_audio("Giant AI cleanup", tmp_path / "exports", include_original=True)
    assert len(files) == 2 and any(digest(path) == source_hash for path in files)
    report_path, = app_window.export_history_path.glob("*.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ai"] is True and report["cleanup_provider"] == "resolve"
    assert report["resolve"] == expected["resolve"]
    assert report["cleanup"] == expected["cleanup"]
    assert report["source_sha256"] == source_hash
    assert report["sha256"] == {str(path): digest(path) for path in files}
    assert digest(source) == digest(app_window.source) == source_hash
    assert voice_file.read_bytes() == old_voice_bytes
    assert "cleanup_provider" not in report["voice"]["settings"]
    app_window.cleanup.setValue(0)
    assert app_window.cleanup_provider.currentData() == "resolve"
    assert app_window.ai_badge.text() == "LOCAL · NO AI", "An inactive AI choice must not claim AI processing"


def test_resolve_failure_keeps_original_and_never_substitutes_a_conventional_render(
        window, qt, speech, tmp_path, controlled_workers):
    app_window, played = window
    audio, rate = speech
    app_window.cleanup_provider.setCurrentIndex(app_window.cleanup_provider.findData("resolve"))
    app_window.cleanup.setValue(80)
    source = tmp_path / "Original stays available.wav"
    sf.write(source, audio, rate, subtype="PCM_16")
    source_hash = digest(source)
    app_window.load_recording(source)
    worker = controlled_workers[-1]
    worker.complete({"ok": False, "error": "DaVinci Resolve is not running. Open Voice Studio Cleanup and try again."})
    QTest.qWait(450)
    qt.processEvents()
    assert len(controlled_workers) == 1, "A failed AI cleanup must not silently start a conventional worker"
    assert app_window.job is None and app_window.rendered_snapshot is None
    assert app_window.rendered is None and app_window.rendered_path is None and app_window.cleaned_path is None
    assert "Resolve is not running" in app_window.message.text()
    assert not app_window.listen_button.isEnabled() and not app_window.save_button.isEnabled()
    assert app_window.original_button.isEnabled()
    app_window.play_original()
    assert played[-1][2] == "original"
    assert np.array_equal(played[-1][0], sf.read(source, dtype="float32")[0])
    assert digest(source) == digest(app_window.source) == source_hash
    assert not list(app_window.export_history_path.glob("*.json"))
    with pytest.raises(ValueError, match="finish"):
        app_window.export_audio("No substituted output", tmp_path / "exports")


def test_resolve_provider_and_strength_restore_without_autoplay_or_preset_mutation(
        window, qt, speech, tmp_path, controlled_workers, monkeypatch):
    import app as ui
    app_window, _played = window
    audio, rate = speech
    giant = next(voice for voice in app_window.library.list_voices() if voice["name"] == "Giant")
    voice_file = app_window.library.folder / (giant["id"] + ".json")
    voice_bytes = voice_file.read_bytes()
    app_window._select_voice(giant["id"])
    app_window.cleanup_provider.setCurrentIndex(app_window.cleanup_provider.findData("resolve"))
    app_window.cleanup.setValue(82)
    source = tmp_path / "Restore this take.wav"
    sf.write(source, audio, rate, subtype="PCM_16")
    app_window.load_recording(source)
    complete_controlled_render(controlled_workers[-1], audio, rate)
    session = json.loads(app_window.state_path.read_text(encoding="utf-8"))
    preferences = json.loads(app_window.preferences_path.read_text(encoding="utf-8"))
    assert session["cleanup_provider"] == preferences["cleanup_provider"] == "resolve"
    assert session["settings"]["cleanup"] == .82
    restored = ui.VoiceStudio(root=app_window.root, restore=True)
    played = []
    monkeypatch.setattr(restored, "_play", lambda *args: played.append(args))
    try:
        wait_until(qt, lambda: restored.source == app_window.source and restored.job is not None)
        assert restored.cleanup_provider.currentData() == "resolve"
        assert restored.cleanup.value() == 82
        assert restored.ai_badge.text() == "LOCAL · AI CLEANUP"
        assert restored.source == app_window.source
        worker = controlled_workers[-1]
        assert worker.context["autoplay"] is False
        assert worker.specification["cleanup_provider"] == "resolve"
        complete_controlled_render(worker, audio, rate)
        qt.processEvents()
        assert played == []
        assert restored.rendered_snapshot["ai"] is True
        assert restored.rendered_snapshot["voice"]["id"] == giant["id"]
        assert voice_file.read_bytes() == voice_bytes
        assert digest(restored.source) == digest(source)
    finally:
        restored.close()
        qt.processEvents()


def test_switching_cleanup_provider_rejects_stale_ai_preview_and_provenance(
        window, qt, speech, tmp_path, controlled_workers):
    app_window, _played = window
    audio, rate = speech
    app_window.cleanup_provider.setCurrentIndex(app_window.cleanup_provider.findData("resolve"))
    app_window.cleanup.setValue(60)
    source = tmp_path / "Switch during cleanup.wav"
    sf.write(source, audio, rate, subtype="PCM_16")
    app_window.load_recording(source)
    old_worker = controlled_workers[-1]
    app_window.cleanup_provider.setCurrentIndex(app_window.cleanup_provider.findData("conventional"))
    assert app_window.ai_badge.text() == "LOCAL · NO AI"
    complete_controlled_render(old_worker, audio, rate)
    qt.processEvents()
    assert len(controlled_workers) == 2
    assert app_window.rendered is None and app_window.rendered_snapshot is None
    new_worker = controlled_workers[-1]
    assert new_worker.context["cleanup_provider"] == "conventional"
    assert new_worker.specification["cleanup_provider"] == "conventional"
    complete_controlled_render(new_worker, audio, rate, provider="conventional")
    qt.processEvents()
    assert app_window.rendered_snapshot["ai"] is False
    assert app_window.rendered_snapshot["cleanup_provider"] == "conventional"
    app_window.export_audio("Conventional result", tmp_path / "exports")
    report_path, = app_window.export_history_path.glob("*.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ai"] is False and report["cleanup_provider"] == "conventional"
    assert not report.get("resolve"), "A superseded AI result must not leak its provenance into an export"


def test_export_rejects_a_replaced_original_before_creating_any_output(window, qt, speech, tmp_path):
    app_window, _played = window
    audio, rate = speech
    imported = tmp_path / "Approved original.wav"
    sf.write(imported, audio, rate, subtype="PCM_16")
    original_hash = digest(imported)
    app_window.load_recording(imported)
    settle(qt, app_window)
    assert app_window.rendered_snapshot["source_sha256"] == original_hash
    assert app_window.source.is_relative_to(app_window.recordings)
    assert digest(app_window.source) == original_hash
    # Replacing the private source after preview must not pair stale processed
    # audio with a different performance, even if the path itself is unchanged.
    sf.write(app_window.source, np.asarray(audio) * -.35, rate, subtype="PCM_16")
    assert digest(app_window.source) != original_hash
    assert digest(imported) == original_hash
    folder = tmp_path / "must-stay-empty"
    history_before = {path: path.read_bytes() for path in app_window.export_history_path.glob("*.json")}
    with pytest.raises(ValueError, match="The original recording changed"):
        app_window.export_audio("Stale preview", folder, include_original=True)
    assert not folder.exists() or not any(folder.iterdir())
    assert {path: path.read_bytes() for path in app_window.export_history_path.glob("*.json")} == history_before
