"""Reference integrity, bounded calibration and real-worker character workflow."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import parselmouth
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from consistency import CharacterProfiles, analyse_neutral, digest, make_calibration, verify_reference
from engine import SAMPLE_RATE, VoiceError, render, run_job
from library import VoiceLibrary
from test_app import qt, window, settle, wait_until


def spoken_sample(pitch=150., level=1., seconds=3.2):
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    frequency = pitch * (1 + .035 * np.sin(2 * np.pi * 1.3 * t))
    phase = 2 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    audio = sum(np.sin(phase * harmonic) / harmonic for harmonic in range(1, 7))
    envelope = .7 + .25 * np.sin(2 * np.pi * 2.2 * t)
    envelope *= np.minimum(1, t / .07) * np.minimum(1, (seconds - t) / .07)
    return (audio * envelope * .1 * level).astype(np.float32)


def pitch(audio):
    measured = parselmouth.Sound(audio, sampling_frequency=SAMPLE_RATE).to_pitch_ac(
        pitch_floor=50, pitch_ceiling=700).selected_array["frequency"]
    return float(np.median(measured[measured > 0]))


def test_neutral_measurement_and_session_offsets_follow_known_drift():
    reference = {"id": "ref", "voice_id": "voice", "voice_version": 1,
                 "source_sha256": "source", "output_sha256": "output",
                 "measurement": analyse_neutral(spoken_sample(), SAMPLE_RATE)}
    shifted = spoken_sample(150 * 2 ** (1.25 / 12), 10 ** (-3 / 20))
    measured = analyse_neutral(shifted, SAMPLE_RATE)
    correction = make_calibration(reference, measured, sample_sha256="sample", microphone="test mic")
    assert correction["pitch"] == pytest.approx(-1.25, abs=.05)
    assert correction["gain_db"] == pytest.approx(3., abs=.15)


@pytest.mark.parametrize("kind", ["silence", "whisper", "clipping", "short", "quiet"])
def test_calibration_rejects_unreliable_samples(kind):
    audio = spoken_sample()
    if kind == "silence":
        audio[:] = 0
    elif kind == "whisper":
        audio = np.random.default_rng(23).normal(0, .03, len(audio))
    elif kind == "clipping":
        audio[::50] = 1
    elif kind == "short":
        audio = audio[:SAMPLE_RATE]
    else:
        audio *= .005
    with pytest.raises(ValueError):
        analyse_neutral(audio, SAMPLE_RATE)


def test_large_drift_requests_a_retake():
    baseline = analyse_neutral(spoken_sample(), SAMPLE_RATE)
    reference = {"measurement": baseline}
    with pytest.raises(ValueError, match="two semitones"):
        make_calibration(reference, {**baseline, "pitch_hz": baseline["pitch_hz"] * 1.25},
                         sample_sha256="sample", microphone="mic")
    with pytest.raises(ValueError, match="6 dB"):
        make_calibration(reference, {**baseline, "level_db": baseline["level_db"] - 8},
                         sample_sha256="sample", microphone="mic")


def test_fixed_calibration_preserves_pitch_and_level_expression():
    first, second = spoken_sample(160, .45), spoken_sample(210, 1.)
    line = np.concatenate([first, second])
    calibration = {"schema": 1, "pitch": -1., "gain_db": 0.}
    output, rate, metadata = render(line, SAMPLE_RATE, calibration=calibration)
    size = len(first)
    assert pitch(output[:size]) / pitch(first) == pytest.approx(2 ** (-1 / 12), abs=.02)
    assert pitch(output[size:]) / pitch(second) == pytest.approx(2 ** (-1 / 12), abs=.02)
    assert pitch(output[size:]) / pitch(output[:size]) > 1.25
    gain_only, _, _ = render(line, SAMPLE_RATE, calibration={**calibration, "pitch": 0., "gain_db": 3.})
    assert np.allclose(gain_only, line * 10 ** (3 / 20), atol=1e-7)
    assert metadata["session_calibration"] == calibration and metadata["ai"] is False
    assert rate == SAMPLE_RATE and np.isfinite(output).all()


def test_calibration_extends_pitch_without_mutating_saved_recipe():
    settings = {"pitch": -8., "resonance": .9}
    original = copy.deepcopy(settings)
    output, _, metadata = render(spoken_sample(), SAMPLE_RATE, settings,
        calibration={"schema": 1, "pitch": -2., "gain_db": 0.})
    assert settings == original and metadata["settings"]["pitch"] == -8.
    assert metadata["effective_pitch"] == -10.
    assert pitch(output) / pitch(spoken_sample()) == pytest.approx(2 ** (-10 / 12), abs=.025)
    with pytest.raises(VoiceError, match="out of range"):
        render(spoken_sample(), SAMPLE_RATE, calibration={"schema": 1, "pitch": 3, "gain_db": 0})


def test_references_are_revision_specific_and_keep_earlier_audio(tmp_path):
    library = VoiceLibrary(tmp_path / "Voices")
    voice = library.create("Sentinel", {})
    profiles = CharacterProfiles(tmp_path)
    source, output = tmp_path / "source.wav", tmp_path / "output.wav"
    sf.write(source, spoken_sample(), SAMPLE_RATE)
    sf.write(output, spoken_sample(120), SAMPLE_RATE)
    snapshot = {"source_sha256": digest(source), "settings": voice["settings"]}
    source_before = digest(source)
    reference = profiles.save_reference(voice, source, output, analyse_neutral(spoken_sample(), SAMPLE_RATE), snapshot)
    replacement = profiles.save_reference(voice, source, output, reference["measurement"], snapshot)
    assert reference["id"] != replacement["id"]
    assert Path(reference["source"]).exists() and Path(reference["output"]).exists()
    assert digest(reference["source"]) == source_before == digest(source)
    assert profiles.reference(voice)["id"] == replacement["id"]
    newer = library.update(voice["id"], {"pitch": -2})
    assert profiles.reference(newer) is None
    assert profiles.reference(library.get(voice["id"], 1))["id"] == replacement["id"]
    assert profiles.get(voice["id"])["locked"] is True
    verify_reference(replacement)
    Path(replacement["output"]).write_bytes(b"changed after approval")
    with pytest.raises(ValueError, match="changed"):
        verify_reference(replacement)
    Path(replacement["output"]).unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_reference(replacement)


def test_lock_controls_and_real_reference_calibration_export_workflow(window, qt, tmp_path):
    import app as ui
    from character_ui import CharacterSetup
    studio, played = window
    assert studio.lock_voice.isChecked() and not studio.pitch.isEnabled()
    QTest.mouseClick(studio.lock_voice, Qt.MouseButton.LeftButton)
    assert studio.pitch.isEnabled()
    QTest.mouseClick(studio.lock_voice, Qt.MouseButton.LeftButton)
    path = tmp_path / "neutral.wav"
    sf.write(path, spoken_sample(), SAMPLE_RATE, subtype="PCM_16")
    studio.load_recording(path)
    settle(qt, studio)
    original_path, original_hash = studio.source, digest(studio.source)
    voice_document = (studio.library.folder / (studio.selected_voice["id"] + ".json")).read_bytes()
    dialog = CharacterSetup(studio, ui.RenderJob)
    dialog.notes.setText("Measured pace; chest resonance; clipped consonants.")
    dialog.save_reference()
    wait_until(qt, lambda: dialog.worker is None)
    reference = studio.profiles.reference(studio.selected_voice)
    assert reference, dialog.message.text()
    assert studio.reference_button.isEnabled() and studio.reference_matches()
    studio.play_reference()
    assert played[-1][2] == "reference"
    assert studio.source == original_path and digest(studio.source) == original_hash
    shifted = tmp_path / "today-neutral.wav"
    sf.write(shifted, spoken_sample(150 * 2 ** (1 / 12), .85), SAMPLE_RATE, subtype="PCM_16")
    dialog.calibrate_from_path(shifted)
    wait_until(qt, lambda: dialog.worker is None)
    settle(qt, studio)
    assert studio.current_calibration(), dialog.message.text()
    assert studio.source == original_path and digest(studio.source) == original_hash
    assert studio.current_calibration()["pitch"] == pytest.approx(-1, abs=.08)
    assert studio.rendered_snapshot["session_calibration"] == studio.current_calibration()
    files = studio.export_audio("Calibrated line", tmp_path / "approved")
    assert len(files) == 2 and any(digest(p) == original_hash for p in files)
    report_path = next(studio.export_history_path.glob("*.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["session_calibration"]["reference_id"] == reference["id"]
    assert report["session_calibration"]["sample_sha256"] == digest(shifted)
    assert report["ai"] is False
    assert (studio.library.folder / (studio.selected_voice["id"] + ".json")).read_bytes() == voice_document
    # Another voice and a changed microphone cannot inherit this calibration.
    old_voice = studio.selected_voice["id"]
    other = next(v for v in studio.library.list_voices() if v["id"] != old_voice)
    studio._select_voice(other["id"])
    assert studio.current_calibration() is None
    studio._select_voice(old_voice)
    assert studio.current_calibration() is not None
    studio._microphone_changed()
    assert studio.current_calibration() is None
    dialog.reject()
    settle(qt, studio)
    # Saved references survive a restart; the session correction deliberately does not.
    reopened = ui.VoiceStudio(root=studio.root, restore=False)
    try:
        reopened._select_voice(old_voice, render=False)
        assert reopened.profiles.reference(reopened.selected_voice)["id"] == reference["id"]
        assert reopened.current_calibration() is None and reopened.lock_voice.isChecked()
    finally:
        reopened.close()


def test_capture_does_not_replace_dialogue_take(window, qt, monkeypatch):
    from character_ui import CalibrationCapture
    studio, _played = window
    source = studio.source
    capture = CalibrationCapture(studio)
    def start(_device):
        capture.recorder.stream = object()
        capture.recorder.started = 0
    def stop():
        capture.recorder.stream = None
        return spoken_sample(), SAMPLE_RATE
    monkeypatch.setattr(capture.recorder, "start", start)
    monkeypatch.setattr(capture.recorder, "stop", stop)
    capture.toggle_recording()
    capture.toggle_recording()
    assert capture.path.is_file() and studio.source == source
    assert capture.recorder.stream is None


def test_dedicated_reference_recording_uses_real_analysis_and_render_without_replacing_take(window, qt, tmp_path):
    import app as ui
    from character_ui import CharacterSetup
    studio, _played = window
    dialogue = tmp_path / "dialogue.wav"
    neutral = tmp_path / "dedicated-neutral.wav"
    sf.write(dialogue, spoken_sample(190), SAMPLE_RATE)
    sf.write(neutral, spoken_sample(150), SAMPLE_RATE)
    studio.load_recording(dialogue)
    settle(qt, studio)
    original, original_hash = studio.source, digest(studio.source)
    dialog = CharacterSetup(studio, ui.RenderJob)
    dialog.reference_from_path(neutral)
    wait_until(qt, lambda: dialog.worker is None)
    reference = studio.profiles.reference(studio.selected_voice)
    assert reference, dialog.message.text()
    assert reference["source_sha256"] == digest(neutral)
    assert reference["snapshot"]["ai"] is False
    assert reference["snapshot"]["backend"]
    assert studio.source == original and digest(studio.source) == original_hash
    verify_reference(reference)
    dialog.reject()
    settle(qt, studio)
