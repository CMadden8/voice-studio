"""Provider boundaries and cache safety without launching or importing Resolve."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import engine
import resolve_bridge


@pytest.fixture
def job(tmp_path):
    source = tmp_path / "original.wav"
    t = np.arange(24_000) / 24_000
    sf.write(source, .2 * np.sin(2 * np.pi * 180 * t), 24_000, subtype="PCM_24")
    return {"kind": "render", "input": str(source), "output": str(tmp_path / "preview.wav"),
            "cleanup_output": str(tmp_path / "cleaned.wav"), "cleanup_provider": "resolve",
            "settings": {"cleanup": .65}}


@pytest.fixture
def fake_resolve(monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "_resolve_installation_signature", lambda: {"test_version": 1})
    def start(**kwargs):
        calls.append(("start", kwargs))
        return {"available": True}
    def isolate(source, output, **kwargs):
        data, rate = sf.read(source, dtype="float32")
        assert rate == engine.SAMPLE_RATE and data.ndim == 1
        assert sf.info(source).format == "WAV"
        calls.append(("isolate", kwargs))
        sf.write(output, data * .5, rate, subtype="PCM_24")
        return {"audio": str(output), "provider": "resolve", "ai": True,
                "amount": kwargs["amount"], "resolve_version": "21.1-test",
                "project": "Voice Studio Cleanup", "job_id": "owned-job",
                "isolation": {"isEnabled": True, "amount": kwargs["amount"]}, "warning": None}
    monkeypatch.setattr(resolve_bridge, "ensure_started", start)
    monkeypatch.setattr(resolve_bridge, "isolate_audio", isolate)
    return calls


def test_ai_cleanup_precedes_conventional_character_and_preserves_source(job, fake_resolve, monkeypatch):
    original_hash = hashlib.sha256(Path(job["input"]).read_bytes()).hexdigest()
    original, _ = engine.read_audio(job["input"])
    real_render = engine.render
    seen = []
    def check_render(audio, rate, settings, **kwargs):
        assert settings["cleanup"] == 0
        np.testing.assert_allclose(audio, original * .5, atol=2e-7)
        seen.append(True)
        return real_render(audio, rate, settings, **kwargs)
    monkeypatch.setattr(engine, "render", check_render)
    result = engine.run_job(job)
    assert seen and len(fake_resolve) == 2
    assert fake_resolve[1][1] == {"amount": 65, "timeout": 180}
    assert result["ai"] is True and result["cleanup_provider"] == "resolve"
    assert result["settings"]["cleanup"] == .65
    assert result["resolve"]["resolve_version"] == "21.1-test"
    assert result["resolve"]["cache_hit"] is False
    assert result["cleanup"]["noise_reduction_db"] is None
    assert result["cleanup"]["enabled"] is True
    np.testing.assert_array_equal(engine.read_audio(result["cleaned_audio"])[0], engine.read_audio(result["audio"])[0])
    assert hashlib.sha256(Path(job["input"]).read_bytes()).hexdigest() == original_hash
    json.dumps(result)  # Immutable export snapshot metadata must be serializable.


def test_preset_change_reuses_ai_cleanup_without_resolve_connection(job, fake_resolve):
    first = engine.run_job(job)
    job["settings"]["tone"] = .3
    second = engine.run_job(job)
    assert len(fake_resolve) == 2
    assert second["resolve"]["cache_hit"] is True
    assert first["resolve"]["isolation"] == second["resolve"]["isolation"]
    assert not np.array_equal(engine.read_audio(first["cleaned_audio"])[0], engine.read_audio(second["audio"])[0])


@pytest.mark.parametrize("change", ["strength", "source", "installation", "cache_audio", "cache_manifest"])
def test_cleanup_cache_invalidates_on_material_change(job, fake_resolve, monkeypatch, change):
    engine.run_job(job)
    cache = Path(job["output"]).parent / ".resolve-cleanup"
    if change == "strength":
        job["settings"]["cleanup"] = .8
    elif change == "source":
        source = Path(job["input"])
        data, rate = sf.read(source)
        sf.write(source, data * .9, rate)
    elif change == "installation":
        monkeypatch.setattr(engine, "_resolve_installation_signature", lambda: {"test_version": 2})
    elif change == "cache_audio":
        next(path for path in cache.glob("*.wav") if not path.stem.endswith("-source")).write_bytes(b"broken")
    else:
        next(cache.glob("*.json")).write_text("broken", encoding="utf-8")
    engine.run_job(job)
    assert len(fake_resolve) == 4


@pytest.mark.parametrize("stage", ["start", "isolate"])
def test_resolve_failure_does_not_fall_back_or_export(job, fake_resolve, monkeypatch, stage):
    def fail(*args, **kwargs):
        raise resolve_bridge.ResolveError("Close your editing project before using Resolve cleanup.")
    monkeypatch.setattr(resolve_bridge, "ensure_started" if stage == "start" else "isolate_audio", fail)
    with pytest.raises(engine.VoiceError, match="Close your editing project"):
        engine.run_job(job)
    assert not Path(job["output"]).exists()
    assert not Path(job["cleanup_output"]).exists()
    assert not list((Path(job["output"]).parent / ".resolve-cleanup").glob("*.json"))


@pytest.mark.parametrize("provider", ["conventional", "resolve"])
def test_cleanup_off_never_starts_resolve_and_remains_exact(job, fake_resolve, provider):
    job["cleanup_provider"] = provider
    job["settings"]["cleanup"] = 0
    result = engine.run_job(job)
    assert fake_resolve == []
    assert result["ai"] is False and result["cleanup_provider"] == "none"
    np.testing.assert_array_equal(engine.read_audio(job["input"])[0], engine.read_audio(job["output"])[0])


def test_conventional_cleanup_remains_non_ai(job, fake_resolve):
    job["cleanup_provider"] = "conventional"
    result = engine.run_job(job)
    assert fake_resolve == []
    assert result["ai"] is False and result["cleanup_provider"] == "conventional"
    assert result["cleanup"]["strength"] == .65


@pytest.mark.parametrize("target", ["output", "cleanup_output"])
def test_original_cannot_be_an_output_in_either_provider(job, fake_resolve, target):
    job[target] = job["input"]
    with pytest.raises(engine.VoiceError, match="original recording safe"):
        engine.run_job(job)
    assert fake_resolve == []


def test_incorrect_ai_output_duration_is_rejected(job, fake_resolve, monkeypatch):
    def wrong(source, output, **kwargs):
        sf.write(output, np.zeros(480), 48_000)
        return {"ai": True}
    monkeypatch.setattr(resolve_bridge, "isolate_audio", wrong)
    with pytest.raises(engine.VoiceError, match="different duration"):
        engine.run_job(job)
    assert not Path(job["output"]).exists()


def test_unknown_provider_is_an_explicit_error(job, fake_resolve):
    job["cleanup_provider"] = "cloud"
    with pytest.raises(engine.VoiceError, match="installed cleanup"):
        engine.run_job(job)
    assert fake_resolve == []


def test_source_mutation_during_resolve_is_rejected(job, fake_resolve, monkeypatch):
    real = resolve_bridge.isolate_audio
    def corrupt(source, output, **kwargs):
        result = real(source, output, **kwargs)
        Path(job["input"]).write_bytes(b"changed elsewhere")
        return result
    monkeypatch.setattr(resolve_bridge, "isolate_audio", corrupt)
    with pytest.raises(engine.VoiceError, match="original recording changed"):
        engine.run_job(job)
    assert not Path(job["output"]).exists()


@pytest.mark.parametrize("provider", ["conventional", "resolve"])
@pytest.mark.parametrize("strength", [0, .5])
def test_every_render_binds_the_exact_source_hash(job, fake_resolve, provider, strength):
    job["cleanup_provider"] = provider
    job["settings"]["cleanup"] = strength
    expected = hashlib.sha256(Path(job["input"]).read_bytes()).hexdigest()
    result = engine.run_job(job)
    assert result["source_sha256"] == expected
    assert result["ai"] is (provider == "resolve" and strength > 0)


@pytest.mark.parametrize("provider", ["conventional", "resolve"])
@pytest.mark.parametrize("strength", [0, .5])
def test_source_replaced_during_decode_never_uses_or_caches_stale_audio(
        job, fake_resolve, monkeypatch, provider, strength):
    job["cleanup_provider"] = provider
    job["settings"]["cleanup"] = strength
    real_read = engine.read_audio
    def replaced(path):
        result = real_read(path)
        if Path(path).resolve() == Path(job["input"]).resolve():
            sf.write(path, result[0] * .75, result[1], subtype="FLOAT")
        return result
    monkeypatch.setattr(engine, "read_audio", replaced)
    with pytest.raises(engine.VoiceError, match="original recording changed"):
        engine.run_job(job)
    assert fake_resolve == []
    assert not Path(job["output"]).exists()
    assert not Path(job["cleanup_output"]).exists()
    assert not (Path(job["output"]).parent / ".resolve-cleanup").exists()


@pytest.mark.parametrize("provider", ["conventional", "resolve"])
@pytest.mark.parametrize("strength", [0, .5])
def test_source_replaced_during_character_render_leaves_no_export_previews(
        job, fake_resolve, monkeypatch, provider, strength):
    job["cleanup_provider"] = provider
    job["settings"]["cleanup"] = strength
    real_render = engine.render
    def replaced(*args, **kwargs):
        result = real_render(*args, **kwargs)
        source = Path(job["input"])
        data, rate = sf.read(source)
        sf.write(source, data * .75, rate, subtype="FLOAT")
        return result
    monkeypatch.setattr(engine, "render", replaced)
    with pytest.raises(engine.VoiceError, match="original recording changed"):
        engine.run_job(job)
    assert not Path(job["output"]).exists()
    assert not Path(job["cleanup_output"]).exists()
