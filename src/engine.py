"""Conventional character voices, with optional local Resolve AI cleanup.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import uuid

import numpy as np
from scipy import ndimage, signal
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 48000
MAX_SECONDS = 300
DEFAULT_SETTINGS = {"pitch": 0., "resonance": 1., "tone": 0., "texture": 0., "room": 0., "speed": 1., "cleanup": 0.}
RANGES = {"pitch": (-8., 8.), "resonance": (.75, 1.35), "tone": (-1., 1.),
          "texture": (0., 1.), "room": (0., .5), "speed": (.75, 1.25), "cleanup": (0., 1.)}
RUBBERBAND = ROOT / "Runtime" / "rubberband" / "rubberband-4.0.0-gpl-executable-windows" / "rubberband-r3.exe"


class VoiceError(Exception):
    """An audio problem that is safe to show in the interface."""


def clean_settings(settings=None):
    result = dict(DEFAULT_SETTINGS)
    for key, (minimum, maximum) in RANGES.items():
        try:
            value = float((settings or {}).get(key, result[key]))
        except (ValueError, TypeError) as error:
            raise VoiceError("One of the voice settings is invalid. Reset the controls and try again.") from error
        if not math.isfinite(value):
            raise VoiceError("One of the voice settings is invalid. Reset the controls and try again.")
        result[key] = min(maximum, max(minimum, value))
    return result


def _resample(audio, original_rate, target_rate):
    if original_rate == target_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)
    ratio = Fraction(int(target_rate), int(original_rate))
    return signal.resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


def read_audio(path: str | Path):
    try:
        info = sf.info(str(path))
        if info.duration > MAX_SECONDS:
            raise VoiceError("Use a recording of five minutes or less. Split longer dialogue into smaller takes.")
        if not 8000 <= info.samplerate <= 192000:
            raise VoiceError("That recording uses an unusual sample rate. Export it as a standard WAV file.")
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, sf.LibsndfileError) as error:
        raise VoiceError("I couldn't open that file. Use WAV, MP3, FLAC or OGG audio.") from error
    if not len(audio):
        raise VoiceError("That recording is empty. Record or choose another take.")
    if not np.isfinite(audio).all():
        raise VoiceError("That recording contains invalid audio samples. Choose another file.")
    # Opposite stereo channels must not cancel a valid voice recording.
    strongest = int(np.argmax(np.mean(audio.astype(np.float64) ** 2, axis=0)))
    return _resample(audio[:, strongest], rate, SAMPLE_RATE), SAMPLE_RATE


def _transform_rubberband(audio, rate, settings):
    pitch_ratio = 2 ** (settings["pitch"] / 12.)
    resonance = settings["resonance"]
    speed = settings["speed"]
    target_length = max(1, round(len(audio) / speed))
    if abs(settings["pitch"]) < 1e-8 and abs(resonance - 1.) < 1e-8 and abs(speed - 1.) < 1e-8:
        return audio.copy(), "Waveform pass-through"
    if float(np.max(np.abs(audio))) < 1e-8:
        return np.zeros(target_length, dtype=np.float32), "Silent waveform pass-through"
    if not RUBBERBAND.is_file():
        raise VoiceError("The local voice engine is missing. Please reinstall Voice Studio's runtime.")
    # Independently move the spectral envelope by rate rescaling, then use ONE
    # R3 operation to correct fundamental pitch and duration. Formant preservation
    # keeps that rescaled envelope while the pitch is corrected. No F0 detector
    # or trained model is needed, so whispers and rough/unvoiced sounds survive.
    ratio = Fraction(1. / resonance).limit_denominator(1000)
    scaled = signal.resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="voice-studio-") as temporary:
        source = Path(temporary) / "source.wav"
        target = Path(temporary) / "changed.wav"
        sf.write(str(source), scaled, rate, subtype="FLOAT")
        command = [str(RUBBERBAND), "--quiet", "--fine", "--formant", "--frequency",
                   f"{pitch_ratio / resonance:.10f}", "--time", f"{resonance / speed:.10f}",
                   str(source), str(target)]
        try:
            completed = subprocess.run(command, capture_output=True, timeout=240,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired as error:
            raise VoiceError("That take took too long to process. Try a shorter recording.") from error
        if completed.returncode or not target.is_file():
            detail = completed.stderr.decode("utf-8", errors="replace")[-2500:]
            raise RuntimeError("Rubber Band could not process the recording: " + detail)
        changed, changed_rate = sf.read(str(target), dtype="float32", always_2d=True)
        if changed_rate != rate:
            raise RuntimeError("The voice engine returned an unexpected sample rate")
        changed = changed[:, 0]
    # Offline R3 compensates its own delay; trim/pad only the final few samples
    # to make dialogue timing predictable at the requested speed.
    if len(changed) < target_length:
        changed = np.pad(changed, (0, target_length - len(changed)))
    return changed[:target_length], "Rubber Band R3 4.0.0"


def _transform_praat(audio, rate, settings):
    """Speech-specific comparison backend, not an automatic hidden fallback."""
    import parselmouth
    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=rate)
    pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=50., pitch_ceiling=700.)
    frequencies = pitch.selected_array["frequency"]
    voiced = frequencies[frequencies > 0]
    if not len(voiced):
        raise VoiceError("Praat could not measure pitch in this comparison clip; the main waveform engine can still process it.")
    changed = parselmouth.praat.call([sound, pitch], "Change gender", settings["resonance"],
                                     float(np.median(voiced)) * 2 ** (settings["pitch"] / 12.),
                                     1., 1. / settings["speed"])
    values = changed.values[0].astype(np.float32)
    target_length = max(1, round(len(audio) / settings["speed"]))
    return np.pad(values, (0, max(0, target_length - len(values))))[:target_length], "Praat/Parselmouth 0.4.7"


def _shelf(audio, rate, gain_db, frequency, high=True):
    if abs(gain_db) < 1e-8:
        return audio
    amplitude = 10 ** (gain_db / 40.)
    omega = 2 * np.pi * frequency / rate
    cosine, sine = np.cos(omega), np.sin(omega)
    beta = np.sqrt(2 * amplitude) * sine
    if high:
        b = amplitude * np.array([(amplitude + 1) + (amplitude - 1) * cosine + beta,
                                  -2 * ((amplitude - 1) + (amplitude + 1) * cosine),
                                  (amplitude + 1) + (amplitude - 1) * cosine - beta])
        a = np.array([(amplitude + 1) - (amplitude - 1) * cosine + beta,
                      2 * ((amplitude - 1) - (amplitude + 1) * cosine),
                      (amplitude + 1) - (amplitude - 1) * cosine - beta])
    else:
        b = amplitude * np.array([(amplitude + 1) - (amplitude - 1) * cosine + beta,
                                  2 * ((amplitude - 1) - (amplitude + 1) * cosine),
                                  (amplitude + 1) - (amplitude - 1) * cosine - beta])
        a = np.array([(amplitude + 1) + (amplitude - 1) * cosine + beta,
                      -2 * ((amplitude - 1) + (amplitude + 1) * cosine),
                      (amplitude + 1) + (amplitude - 1) * cosine - beta])
    return signal.lfilter(b / a[0], a / a[0], audio)


def cleanup_audio(audio, rate, strength=.45):
    """Conservative spectral noise reduction, with no learned model or gain boost.

    Room tone is estimated from low-energy windows across the complete take.
    We do not assume its beginning is silence. If no distinct quiet region is
    available (including continuous whispers), only gentle rumble/DC removal is
    used: treating soft speech as a noise print would erase the performance.
    """
    original = np.asarray(audio, dtype=np.float32)
    if original.ndim != 1 or not len(original) or not np.isfinite(original).all():
        raise VoiceError("Choose a valid recording before applying cleanup.")
    strength = min(1., max(0., float(strength)))
    input_peak = max(float(np.max(np.abs(original))), 1e-12)
    input_rms = max(float(np.sqrt(np.mean(original.astype(np.float64) ** 2))), 1e-12)
    metadata = {"strength": strength, "enabled": strength > 0,
                "profile_reliable": False, "noise_reduction_db": 0.,
                "input_noise_dbfs": None, "cleaned_noise_dbfs": None,
                "method": "Off", "input_clipped": bool(np.mean(np.abs(original) >= .999) > .0001),
                "input_peak_dbfs": round(20 * math.log10(input_peak), 1),
                "input_rms_dbfs": round(20 * math.log10(input_rms), 1)}
    if not strength or not np.any(original):
        if strength:
            metadata["method"] = "Silence pass-through"
        return original.copy(), metadata
    # A low cutoff retains low voices while removing desk/mic rumble. Never
    # normalize upward: a quiet recording must not turn into amplified hiss.
    cutoff = 35. + 25. * strength
    cleaned = np.asarray(original, dtype=np.float64) - float(np.mean(original))
    sos = signal.butter(2, cutoff, "highpass", fs=rate, output="sos")
    if len(cleaned) > 18:
        cleaned = signal.sosfiltfilt(sos, cleaned)
    else:
        cleaned = signal.sosfilt(sos, cleaned)
    metadata["method"] = "Gentle rumble/DC removal"
    metadata["highpass_hz"] = round(cutoff, 1)
    if len(cleaned) < max(64, round(rate * .10)):
        return np.ascontiguousarray(cleaned, dtype=np.float32), metadata
    nfft = min(2048, max(256, 2 ** round(math.log2(rate * .032))))
    hop = nfft // 4
    _, times, spectrum = signal.stft(cleaned, fs=rate, nperseg=nfft,
                                     noverlap=nfft - hop, boundary="zeros", padded=True)
    power = np.abs(spectrum) ** 2
    # Omit padded endpoint frames from noise estimation.
    valid = (times >= nfft / (2 * rate)) & (times <= len(cleaned) / rate - nfft / (2 * rate))
    weights = np.full(len(power), 2.)
    weights[0] = weights[-1] = 1.
    frame_rms = np.sqrt(np.sum(power * weights[:, None], axis=0) * (2. / 3.))
    valid_rms = frame_rms[valid]
    if len(valid_rms) < 5:
        return np.ascontiguousarray(cleaned, dtype=np.float32), metadata
    low, high = np.percentile(valid_rms, [20, 90])
    # At least ~10 dB of room-tone/speech contrast is required. A nearly
    # continuous whisper/noise-like source has no safe automatic noise print.
    if high < 1e-8 or low > high * .30:
        metadata["input_weak"] = bool(high < .006)
        return np.ascontiguousarray(cleaned, dtype=np.float32), metadata
    quiet = valid & (frame_rms <= low)
    if np.count_nonzero(quiet) < 3:
        return np.ascontiguousarray(cleaned, dtype=np.float32), metadata
    metadata["profile_reliable"] = True
    noise = np.percentile(power[:, quiet], 65, axis=1)
    noise = ndimage.uniform_filter1d(noise, size=3, mode="nearest")
    averaged = ndimage.uniform_filter1d(power, size=5, axis=1, mode="nearest")
    remaining = np.maximum(averaged - noise[:, None] * (1. + .5 * strength), 0.)
    gain = np.sqrt(remaining / np.maximum(averaged, 1e-24))
    gain = np.clip(gain, 10 ** (-24. * strength / 20.), 1.)
    gain = ndimage.uniform_filter1d(gain, size=3, axis=0, mode="nearest")
    gain = ndimage.uniform_filter1d(gain, size=3, axis=1, mode="nearest")
    # Retain quick consonants/attacks where the complete frame is clearly
    # louder than room tone. We never introduce a binary silence gate.
    active = frame_rms > max(float(np.median(frame_rms[quiet])) * 2.5, 1e-8)
    attack_gain = np.sqrt(np.maximum(1. - noise[:, None] / np.maximum(power, 1e-24), 0.))
    gain[:, active] = np.maximum(gain[:, active], attack_gain[:, active])
    reduced = spectrum * gain
    _, output = signal.istft(reduced, fs=rate, nperseg=nfft, noverlap=nfft - hop,
                             input_onesided=True, boundary=True)
    output = np.pad(output, (0, max(0, len(original) - len(output))))[:len(original)]
    # Measure the same selected quiet windows after reconstruction. This is
    # an estimate of room-noise attenuation, not a speech-quality score.
    _, _, check = signal.stft(output, fs=rate, nperseg=nfft, noverlap=nfft - hop,
                              boundary="zeros", padded=True)
    after_rms = np.sqrt(np.sum(np.abs(check) ** 2 * weights[:, None], axis=0) * (2. / 3.))
    before = max(float(np.sqrt(np.mean(frame_rms[quiet] ** 2))), 1e-12)
    after = max(float(np.sqrt(np.mean(after_rms[quiet] ** 2))), 1e-12)
    metadata.update({"method": "Conventional spectral noise reduction + rumble removal",
                     "quiet_frame_fraction": round(float(np.mean(quiet)), 3),
                     "input_noise_dbfs": round(20 * math.log10(before), 1),
                     "cleaned_noise_dbfs": round(20 * math.log10(after), 1),
                     "noise_reduction_db": round(max(0., 20 * math.log10(before / after)), 1),
                     "input_weak": bool(high < .006)})
    return np.ascontiguousarray(output, dtype=np.float32), metadata


def render(audio, rate, settings=None, *, backend="auto", cleanup_output=None, calibration=None):
    from consistency import validate_calibration
    try:
        calibration = validate_calibration(calibration)
    except ValueError as error:
        raise VoiceError(str(error)) from error
    started = time.perf_counter()
    settings = clean_settings(settings)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise VoiceError("Choose a valid audio recording before changing the voice.")
    if len(audio) / rate > MAX_SECONDS:
        raise VoiceError("Use a recording of five minutes or less.")
    audio = _resample(audio, int(rate), SAMPLE_RATE)
    rate = SAMPLE_RATE
    audio, cleanup = cleanup_audio(audio, rate, settings["cleanup"])
    if cleanup_output is not None:
        path = Path(cleanup_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".partial.wav")
        try:
            sf.write(str(temporary), audio, rate, subtype="FLOAT")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    effective = dict(settings)
    if calibration:
        # A fixed offset for the entire session, never a per-line pitch target.
        # Cleanup precedes gain so calibration does not amplify the noise estimate.
        audio = audio * 10 ** (calibration["gain_db"] / 20)
        effective["pitch"] += calibration["pitch"]
    warning = ""
    transform_needed = any(abs(effective[key] - DEFAULT_SETTINGS[key]) > 1e-8
                           for key in ("pitch", "resonance", "speed"))
    if not transform_needed:
        changed, name = audio.copy(), "Waveform pass-through"
    elif backend == "auto":
        import parselmouth
        try:
            changed, name = _transform_praat(audio, rate, effective)
        except (VoiceError, parselmouth.PraatError):
            changed, name = _transform_rubberband(audio, rate, effective)
            warning = "Used waveform processing for unvoiced or rough audio."
    elif backend == "rubberband":
        changed, name = _transform_rubberband(audio, rate, effective)
    elif backend == "praat":
        changed, name = _transform_praat(audio, rate, effective)
    else:
        raise VoiceError("That voice engine is not installed.")
    changed = _shelf(changed, rate, settings["tone"] * 9., 2200., high=True)
    changed = _shelf(changed, rate, settings["tone"] * -4., 250., high=False)
    if settings["texture"]:
        amount = settings["texture"]
        peak = max(.02, float(np.max(np.abs(changed))))
        saturated = np.tanh(changed / peak * (1. + 4. * amount)) * peak
        changed = changed * (1. - amount) + saturated * amount
    if settings["room"]:
        room = settings["room"]
        dry = changed.copy()
        tail = round(.23 * rate)
        changed = np.pad(changed, (0, tail))
        for delay, gain in [(.037, .36), (.061, .29), (.097, .23), (.139, .17), (.211, .12)]:
            start = round(delay * rate)
            changed[start:start + len(dry)] += dry * gain * room
    if not np.isfinite(changed).all():
        raise RuntimeError("The audio engine produced invalid samples")
    peak = float(np.max(np.abs(changed)))
    # Retain quiet performances, and only attenuate if an effect risks clipping.
    attenuation = min(1., .98 / max(peak, 1e-9))
    changed = np.ascontiguousarray(changed * attenuation, dtype=np.float32)
    if cleanup.get("input_clipped"):
        warning = (warning + " The recording has clipped peaks. Lower the microphone gain or move farther away when recording; cleanup cannot restore clipped detail.").strip()
    elif cleanup.get("input_weak") or (cleanup["input_rms_dbfs"] < -40 and cleanup["input_peak_dbfs"] < -25):
        warning = (warning + " The input is very quiet. For less hiss, move closer to the microphone or speak at a normal level.").strip()
    metadata = {"backend": name, "warning": warning, "duration": len(changed) / rate,
                "processing_seconds": round(time.perf_counter() - started, 4),
                "settings": settings, "ai": False, "cleanup": cleanup}
    if calibration:
        metadata["session_calibration"] = calibration
        metadata["effective_pitch"] = effective["pitch"]
    if cleanup_output is not None:
        metadata["cleaned_audio"] = str(cleanup_output)
    return changed, rate, metadata


def _write_wav(path, audio, rate=SAMPLE_RATE):
    """Publish complete preview audio atomically, without partial WAV playback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + "." + uuid.uuid4().hex + ".partial.wav")
    try:
        sf.write(str(temporary), audio, rate, subtype="FLOAT")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_installation_signature():
    # A character tweak must not start Resolve again. Invalidate cached cleanup
    # after a Resolve update without loading its scripting library just to check.
    executable = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Blackmagic Design/DaVinci Resolve/Resolve.exe"
    try:
        info = executable.stat()
        return {"path": str(executable), "modified_ns": info.st_mtime_ns, "bytes": info.st_size}
    except OSError:
        return {"path": str(executable), "missing": True}


def _source_sha256(source):
    digest = hashlib.sha256()
    with Path(source).open("rb") as recording:
        for chunk in iter(lambda: recording.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_source_unchanged(source, expected):
    try:
        if _source_sha256(source) == expected:
            return
    except OSError:
        pass
    raise VoiceError("The original recording changed during processing. Choose the recording again.")


def _resolve_cleanup(source, audio, amount, cache, source_hash):
    """Cache AI cleanup separately from reusable, entirely conventional voices."""
    import resolve_bridge
    identity = {"format": 1, "source_sha256": source_hash, "amount": amount,
                "installation": _resolve_installation_signature()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    cache = Path(cache) / ".resolve-cleanup"
    cache.mkdir(parents=True, exist_ok=True)
    wave = cache / (key + ".wav")
    manifest = cache / (key + ".json")
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["identity"] == identity and saved["cleaned_sha256"] == hashlib.sha256(wave.read_bytes()).hexdigest():
            cleaned, rate = read_audio(wave)
            if rate == SAMPLE_RATE and len(cleaned) == len(audio):
                return cleaned, {**saved["resolve"], "cache_hit": True}, source_hash
    except (OSError, ValueError, KeyError, TypeError, VoiceError):
        pass
    try:
        resolve_bridge.ensure_started(timeout=75)
        # The SDK imports audio into its own project. Keep this prepared source
        # alongside the persistent cache, so that project media stays available.
        prepared = cache / (key + "-source.wav")
        if prepared.resolve() == Path(source).resolve():
            raise VoiceError("Choose the original recording outside the private cleanup cache.")
        _write_wav(prepared, audio)
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=str(cache)) as scratch:
            isolated = Path(scratch) / "isolated.wav"
            result = resolve_bridge.isolate_audio(prepared, isolated, amount=amount, timeout=180)
            cleaned, rate = read_audio(isolated)
            if rate != SAMPLE_RATE or len(cleaned) != len(audio):
                raise VoiceError("Resolve returned audio with a different duration. The cleaned take was not used.")
            _assert_source_unchanged(source, source_hash)
            _write_wav(wave, cleaned)
            provenance = {key: value for key, value in result.items() if key != "audio"}
            saved = {"identity": identity, "cleaned_sha256": hashlib.sha256(wave.read_bytes()).hexdigest(),
                     "resolve": provenance}
            temporary = manifest.with_suffix("." + uuid.uuid4().hex + ".partial.json")
            try:
                temporary.write_text(json.dumps(saved, indent=2), encoding="utf-8")
                temporary.replace(manifest)
            finally:
                temporary.unlink(missing_ok=True)
        # Retain at most eight completed cleanup amounts/takes. Only this
        # module's explicit cache files are removed; sources elsewhere persist.
        entries = sorted(cache.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for old in entries[8:]:
            for obsolete in (old, old.with_suffix(".wav"), old.with_name(old.stem + "-source.wav")):
                try:
                    obsolete.unlink(missing_ok=True)
                except OSError:
                    pass
        return cleaned, {**provenance, "cache_hit": False}, source_hash
    except resolve_bridge.ResolveError as error:
        raise VoiceError(str(error)) from error


def run_job(job):
    if job.get("kind") == "analyse_neutral":
        from consistency import analyse_neutral
        source = Path(job["input"]).resolve()
        source_hash = _source_sha256(source)
        audio, rate = read_audio(source)
        try:
            measurement = analyse_neutral(audio, rate)
        except ValueError as error:
            raise VoiceError(str(error)) from error
        _assert_source_unchanged(source, source_hash)
        return {"measurement": measurement, "source_sha256": source_hash}
    if job.get("kind") != "render":
        raise VoiceError("Unknown voice-processing job")
    source = Path(job["input"]).resolve()
    for target in (job["output"], job.get("cleanup_output")):
        if target is not None and Path(target).resolve() == source:
            raise VoiceError("Choose a separate output file to keep the original recording safe.")
    if job.get("cleanup_output") and Path(job["cleanup_output"]).resolve() == Path(job["output"]).resolve():
        raise VoiceError("The cleaned and changed previews need separate output files.")
    provider = job.get("cleanup_provider", "conventional")
    if provider not in ("conventional", "resolve"):
        raise VoiceError("Choose one of the installed cleanup options.")
    try:
        source_hash = _source_sha256(source)
    except OSError as error:
        raise VoiceError("I couldn't open that file. Use WAV, MP3, FLAC or OGG audio.") from error
    audio, rate = read_audio(source)
    _assert_source_unchanged(source, source_hash)
    settings = clean_settings(job.get("settings"))
    if provider == "resolve" and settings["cleanup"] > 0:
        started = time.perf_counter()
        _, diagnostics = cleanup_audio(audio, rate, 0)
        cleaned, provenance, _ = _resolve_cleanup(source, audio, round(settings["cleanup"] * 100), Path(job["output"]).parent, source_hash)
        changed, rate, metadata = render(cleaned, rate, {**settings, "cleanup": 0}, calibration=job.get("calibration"))
        diagnostics.update({"enabled": True, "strength": settings["cleanup"],
                            "method": "DaVinci Resolve Studio AI Voice Isolation", "noise_reduction_db": None})
        metadata.update({"settings": settings, "ai": True, "cleanup_provider": "resolve",
                         "resolve": provenance, "cleanup": diagnostics,
                         "processing_seconds": round(time.perf_counter() - started, 4)})
        if diagnostics["input_clipped"] and "clipped" not in metadata["warning"]:
            metadata["warning"] = (metadata["warning"] + " The recording has clipped peaks. Lower the microphone gain or move farther away when recording; cleanup cannot restore clipped detail.").strip()
        if provenance.get("warning"):
            metadata["warning"] = (metadata["warning"] + " " + str(provenance["warning"])).strip()
    else:
        # Stage the comparison privately: a source replaced during character
        # processing must not leave either public preview looking successful.
        with tempfile.TemporaryDirectory(prefix="voice-studio-preview-") as scratch:
            staged_cleanup = Path(scratch) / "cleaned.wav" if job.get("cleanup_output") is not None else None
            changed, rate, metadata = render(audio, rate, settings, cleanup_output=staged_cleanup, calibration=job.get("calibration"))
            _assert_source_unchanged(source, source_hash)
            if staged_cleanup is not None:
                cleaned, _ = read_audio(staged_cleanup)
        metadata["cleanup_provider"] = "conventional" if settings["cleanup"] else "none"
    _assert_source_unchanged(source, source_hash)
    metadata["source_sha256"] = source_hash
    if job.get("cleanup_output") is not None:
        _write_wav(job["cleanup_output"], cleaned, rate)
        metadata["cleaned_audio"] = str(job["cleanup_output"])
    output = Path(job["output"])
    _write_wav(output, changed, rate)
    return {"audio": str(output), **metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = run_job(json.loads(arguments.job.read_text(encoding="utf-8")))
        print("VS_RESULT:" + json.dumps({"ok": True, "result": result}), flush=True)
        return 0
    except Exception as error:
        traceback.print_exc()
        message = str(error) if isinstance(error, VoiceError) else "I couldn't process that recording. Try again or choose a shorter take."
        print("VS_RESULT:" + json.dumps({"ok": False, "error": message}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
