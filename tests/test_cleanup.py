"""Numerical non-AI noise reduction and original-file safety checks."""
import json

import numpy as np
import pytest
from scipy import signal
import soundfile as sf

from engine import SAMPLE_RATE, VoiceError, cleanup_audio, render, run_job


@pytest.fixture
def noisy_speech():
    rate = SAMPLE_RATE
    t = np.arange(rate * 4) / rate
    carrier = .07 * sum(np.sin(2 * np.pi * 145 * k * t) / k for k in range(1, 9))
    envelope = np.zeros(len(t))
    # Speech starts immediately; its first half-second is NOT a noise print.
    for start, end in [(0., .65), (.95, 1.55), (1.8, 2.45), (2.8, 3.45)]:
        mask = (t >= start) & (t < end)
        envelope[mask] = np.sin(np.linspace(0, np.pi, int(mask.sum()))) ** .35
    clean = (carrier * envelope).astype(np.float32)
    noise = np.random.default_rng(31).normal(0, .008, len(t)) + .004 * np.sin(2 * np.pi * 50 * t)
    return clean, (clean + noise).astype(np.float32), envelope


def snr(reference, actual):
    return 10 * np.log10(np.sum(reference.astype(np.float64) ** 2) /
                         max(np.sum((actual.astype(np.float64) - reference) ** 2), 1e-24))


def test_noise_reduction_improves_known_snr_without_erasing_speech(noisy_speech):
    clean, noisy, envelope = noisy_speech
    output, diagnostics = cleanup_audio(noisy, SAMPLE_RATE, .45)
    assert snr(clean, output) - snr(clean, noisy) > 4.
    quiet = envelope == 0
    assert 20 * np.log10(np.std(noisy[quiet]) / np.std(output[quiet])) > 7.
    active = envelope > .3
    assert .9 < np.sqrt(np.mean(output[active] ** 2)) / np.sqrt(np.mean(clean[active] ** 2)) < 1.06
    # Protect the beginning rather than subtracting the opening phrase.
    assert np.std(output[:SAMPLE_RATE // 2]) > np.std(clean[:SAMPLE_RATE // 2]) * .85
    assert diagnostics["profile_reliable"]
    assert diagnostics["noise_reduction_db"] > 7
    assert len(output) == len(noisy) and np.isfinite(output).all()


def test_cleanup_off_is_byte_exact_and_strength_is_controllable(noisy_speech):
    _clean, noisy, envelope = noisy_speech
    unchanged, off = cleanup_audio(noisy, SAMPLE_RATE, 0)
    assert np.array_equal(unchanged, noisy) and not off["enabled"]
    light, _ = cleanup_audio(noisy, SAMPLE_RATE, .25)
    strong, _ = cleanup_audio(noisy, SAMPLE_RATE, .7)
    assert np.std(strong[envelope == 0]) < np.std(light[envelope == 0])


def test_continuous_whisper_is_not_treated_as_room_tone():
    rng = np.random.default_rng(17)
    whisper = signal.sosfilt(signal.butter(2, 500, "highpass", fs=SAMPLE_RATE, output="sos"),
                             rng.normal(0, .004, SAMPLE_RATE * 2)).astype(np.float32)
    changed, diagnostics = cleanup_audio(whisper, SAMPLE_RATE, 1.)
    assert not diagnostics["profile_reliable"]
    assert np.sqrt(np.mean(changed ** 2)) > np.sqrt(np.mean(whisper ** 2)) * .95
    assert np.corrcoef(whisper, changed)[0, 1] > .99


@pytest.mark.parametrize("length", [1, 10, 1000, SAMPLE_RATE])
def test_short_and_silent_audio_remain_finite_and_silent(length):
    output, _ = cleanup_audio(np.zeros(length, dtype=np.float32), SAMPLE_RATE, .8)
    assert len(output) == length and not np.any(output) and np.isfinite(output).all()


def test_actual_cleaned_ab_file_and_metadata_use_the_same_processing(noisy_speech, tmp_path):
    _clean, noisy, _envelope = noisy_speech
    cleaned_file = tmp_path / "cleaned.wav"
    output, rate, metadata = render(noisy, SAMPLE_RATE, {"cleanup": .45}, cleanup_output=cleaned_file)
    disk, disk_rate = sf.read(cleaned_file, dtype="float32")
    assert disk_rate == rate and np.array_equal(disk, output)
    assert metadata["cleaned_audio"] == str(cleaned_file)
    assert metadata["cleanup"]["strength"] == .45
    assert metadata["cleanup"]["noise_reduction_db"] > 7.
    assert metadata["ai"] is False


def test_worker_cleanup_original_and_altered_outputs_are_distinct(noisy_speech, tmp_path):
    _clean, noisy, _envelope = noisy_speech
    original = tmp_path / "original.wav"
    sf.write(original, noisy, SAMPLE_RATE, subtype="PCM_16")
    saved = original.read_bytes()
    cleaned = tmp_path / "cleaned.wav"
    changed = tmp_path / "giant.wav"
    result = run_job({"kind": "render", "input": str(original), "output": str(changed),
                      "cleanup_output": str(cleaned),
                      "settings": {"pitch": -5., "resonance": .8, "cleanup": .45}})
    assert original.read_bytes() == saved
    assert changed.exists() and cleaned.exists()
    assert result["cleaned_audio"] == str(cleaned) and result["cleanup"]["enabled"]
    assert sf.info(changed).duration == sf.info(original).duration
    with pytest.raises(VoiceError, match="original recording safe"):
        run_job({"kind": "render", "input": str(original), "output": str(changed),
                 "cleanup_output": str(original), "settings": {"cleanup": .45}})
    assert original.read_bytes() == saved
