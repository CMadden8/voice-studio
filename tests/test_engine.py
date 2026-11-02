"""Exercise real non-neural native voice processing, without microphone capture."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import parselmouth
import pytest
from scipy import signal
import soundfile as sf

from engine import DEFAULT_SETTINGS, SAMPLE_RATE, VoiceError, clean_settings, read_audio, render


@pytest.fixture(scope="module")
def vowel():
    rate = SAMPLE_RATE
    size = rate * 2
    excitation = np.zeros(size)
    excitation[np.arange(0, size, round(rate / 140))] = 1.
    audio = signal.lfilter([1.], [1., -.75], excitation)
    for frequency, bandwidth in [(650, 75), (1400, 100), (2450, 140)]:
        radius = np.exp(-np.pi * bandwidth / rate)
        coefficients = [1., -2 * radius * np.cos(2 * np.pi * frequency / rate), radius ** 2]
        audio = signal.lfilter([1.], coefficients, audio)
    audio *= .3 / np.max(np.abs(audio))
    fade = round(rate * .04)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    return audio.astype(np.float32)


def f0(audio):
    frequencies = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE).to_pitch_ac(
        pitch_floor=50, pitch_ceiling=500).selected_array["frequency"]
    return float(np.median(frequencies[frequencies > 0]))


def first_formant(audio):
    formants = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE).to_formant_burg(
        time_step=.01, max_number_of_formants=5, maximum_formant=5000)
    values = [formants.get_value_at_time(1, point) for point in np.arange(.3, 1.5, .05)]
    return float(np.nanmedian(values))


def test_default_processing_keeps_the_original_waveform(vowel):
    output, rate, metadata = render(vowel, SAMPLE_RATE)
    assert rate == SAMPLE_RATE and len(output) == len(vowel)
    assert np.array_equal(vowel, output)
    assert metadata["ai"] is False and "pass-through" in metadata["backend"]


def test_pitch_and_resonance_are_independent_with_real_speech_engine(vowel):
    original_pitch = f0(vowel)
    original_formant = first_formant(vowel)
    deeper, rate, metadata = render(vowel, SAMPLE_RATE, {"pitch": -4, "resonance": 1.})
    bigger, _, _ = render(vowel, SAMPLE_RATE, {"pitch": 0, "resonance": .82})
    assert metadata["backend"] == "Praat/Parselmouth 0.4.7"
    assert abs(f0(deeper) / original_pitch - 2 ** (-4 / 12)) < .035
    assert abs(first_formant(deeper) / original_formant - 1.) < .12
    assert abs(f0(bigger) / original_pitch - 1.) < .035
    assert abs(first_formant(bigger) / original_formant - .82) < .08
    assert len(deeper) == len(vowel) == len(bigger)
    assert rate == SAMPLE_RATE and np.isfinite(deeper).all()


def test_speed_changes_duration_without_transposing_the_voice(vowel):
    output, _, _ = render(vowel, SAMPLE_RATE, {"speed": .8})
    assert len(output) == round(len(vowel) / .8)
    assert abs(f0(output) / f0(vowel) - 1.) < .035


@pytest.mark.parametrize("kind", ["whisper", "growl", "silence"])
def test_unvoiced_rough_and_silent_audio_are_accepted(kind):
    rate = SAMPLE_RATE
    rng = np.random.default_rng(8)
    t = np.arange(rate) / rate
    if kind == "whisper":
        audio = signal.sosfilt(signal.butter(2, 500, "highpass", fs=rate, output="sos"),
                              rng.normal(0, .04, rate))
    elif kind == "growl":
        audio = .2 * np.sin(2 * np.pi * 60 * t) + rng.normal(0, .015, rate)
    else:
        audio = np.zeros(rate)
    output, actual_rate, metadata = render(audio.astype(np.float32), rate,
                                          {"pitch": -3, "resonance": .9, "texture": .2})
    assert actual_rate == rate and len(output) == len(audio)
    assert np.isfinite(output).all() and np.max(np.abs(output)) <= .98001
    if kind != "silence":
        assert np.sqrt(np.mean(output ** 2)) > .001
    else:
        assert not np.any(output)
    if kind == "whisper":
        assert metadata["backend"] == "Rubber Band R3 4.0.0"
        assert "unvoiced" in metadata["warning"]


def test_tone_texture_and_room_are_audible_and_nonclipping(vowel):
    changed, _, _ = render(vowel, SAMPLE_RATE, {"tone": .8, "texture": .5, "room": .3})
    assert len(changed) > len(vowel)
    assert not np.allclose(changed[:len(vowel)], vowel, atol=1e-4)
    assert np.max(np.abs(changed)) <= .98001


def test_audio_import_resamples_and_preserves_opposite_stereo(tmp_path):
    rate = 22050
    t = np.arange(rate) / rate
    mono = .2 * np.sin(2 * np.pi * 200 * t)
    path = tmp_path / "stereo.flac"
    sf.write(path, np.column_stack([mono, -mono]), rate)
    imported, actual_rate = read_audio(path)
    assert actual_rate == SAMPLE_RATE and imported.ndim == 1
    assert abs(len(imported) - SAMPLE_RATE) < 2
    assert np.max(np.abs(imported)) > .19
    invalid = tmp_path / "invalid.wav"
    invalid.write_text("not audio")
    with pytest.raises(VoiceError, match="couldn't open"):
        read_audio(invalid)


@pytest.mark.parametrize("extension,format_name", [("wav", "WAV"), ("flac", "FLAC"),
                                                    ("ogg", "OGG"), ("mp3", "MP3")])
def test_advertised_formats_are_really_encoded_and_decoded(vowel, tmp_path, extension, format_name):
    path = tmp_path / ("voice." + extension)
    sf.write(path, vowel, SAMPLE_RATE, format=format_name)
    assert sf.info(path).format == format_name
    decoded, rate = read_audio(path)
    assert rate == SAMPLE_RATE and abs(len(decoded) - len(vowel)) < 1200
    assert np.sqrt(np.mean(decoded ** 2)) > .01


def test_worker_contract_exports_a_finite_wav_and_keeps_input_unchanged(vowel, tmp_path):
    source = tmp_path / "spoken.wav"
    output = tmp_path / "changed.wav"
    sf.write(source, vowel, SAMPLE_RATE, subtype="PCM_16")
    original_bytes = source.read_bytes()
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"kind": "render", "input": str(source), "output": str(output),
                               "settings": {"pitch": -2, "resonance": .92}}))
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "src" / "engine.py"),
                                "--job", str(job)], capture_output=True, text=True, timeout=30,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert completed.returncode == 0, completed.stderr
    payload = json.loads([line[10:] for line in completed.stdout.splitlines() if line.startswith("VS_RESULT:")][-1])
    assert payload["ok"] is True and payload["result"]["audio"] == str(output)
    assert payload["result"]["ai"] is False
    audio, rate = sf.read(output)
    assert rate == SAMPLE_RATE and np.isfinite(audio).all()
    assert np.max(np.abs(audio)) <= .98001
    assert source.read_bytes() == original_bytes


def test_invalid_settings_and_audio_fail_with_readable_errors():
    assert clean_settings({"pitch": 100})["pitch"] == 8.
    with pytest.raises(VoiceError, match="invalid"):
        clean_settings({"pitch": float("nan")})
    with pytest.raises(VoiceError, match="valid audio"):
        render(np.array([], dtype=np.float32), SAMPLE_RATE)
