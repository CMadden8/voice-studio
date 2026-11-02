"""Character references and bounded session corrections. Conventional DSP only.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

from library import unique_copy, _atomic_json

PHRASE = "The road is quiet tonight. I'll be waiting by the old gate."


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as incoming:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_reference(reference):
    try:
        valid = (digest(reference["source"]) == reference["source_sha256"]
                 and digest(reference["output"]) == reference["output_sha256"])
    except OSError as error:
        raise ValueError("A reference recording is missing. Set a new reference from an approved take.") from error
    if not valid:
        raise ValueError("The saved reference changed. Set a new reference from an approved take.")
    return reference


def analyse_neutral(audio, rate):
    """Measure a neutral spoken sample; never infer an identity or an AI score."""
    import parselmouth
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Record a clear neutral spoken phrase first.")
    duration = len(audio) / rate
    if duration < 2 or duration > 30:
        raise ValueError("Use 2–30 seconds of normal speech. Read the reference phrase once.")
    if np.mean(np.abs(audio) >= .98) > .001:
        raise ValueError("This sample has clipped peaks. Lower the microphone gain and record again.")
    frame = max(1, round(rate * .02))
    usable = audio[:len(audio) // frame * frame].reshape(-1, frame)
    levels = np.sqrt(np.mean(usable ** 2, axis=1))
    gate = max(.003, float(np.percentile(levels, 95)) * .1)
    active = levels[levels > gate]
    if not len(active) or len(active) * .02 < 1:
        raise ValueError("Too little clear speech. Move closer to the microphone and read the whole phrase.")
    level_db = 20 * math.log10(float(np.sqrt(np.mean(active ** 2))))
    if level_db < -38:
        raise ValueError("The sample is too quiet to calibrate reliably. Move closer and speak normally.")
    sound = parselmouth.Sound(audio, sampling_frequency=rate)
    pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=50., pitch_ceiling=700.)
    frequencies = pitch.selected_array["frequency"]
    voiced = frequencies[frequencies > 0]
    if len(voiced) < 80 or len(voiced) / max(1, len(frequencies)) < .2:
        raise ValueError("Not enough stable voiced speech to calibrate. Use a normal speaking voice, without whispering or growling.")
    return {"pitch_hz": float(np.median(voiced)), "level_db": level_db,
            "voiced_seconds": round(len(voiced) * .01, 3),
            "duration": duration, "method": "Praat autocorrelation; active-frame RMS"}


def make_calibration(reference, measured, *, sample_sha256, microphone):
    baseline = reference["measurement"]
    pitch = 12 * math.log2(baseline["pitch_hz"] / measured["pitch_hz"])
    gain = baseline["level_db"] - measured["level_db"]
    if abs(pitch) > 2.01:
        raise ValueError("Your pitch is over two semitones from the reference. Listen to the original reference and repeat it in a similar neutral voice.")
    if abs(gain) > 6.01:
        raise ValueError("Your recording level is over 6 dB from the reference. Match your microphone distance or gain and try again.")
    return {"schema": 1, "pitch": round(max(-2., min(2., pitch)), 4),
            "gain_db": round(max(-6., min(6., gain)), 4),
            "reference_id": reference["id"], "reference_source_sha256": reference["source_sha256"],
            "reference_output_sha256": reference["output_sha256"],
            "voice_id": reference["voice_id"], "voice_version": reference["voice_version"],
            "sample_sha256": sample_sha256, "microphone": microphone,
            "measurement": measured, "created": datetime.now().astimezone().isoformat(),
            "method": "Fixed session pitch and gain offsets; no trained models"}


def validate_calibration(value):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("The session calibration is invalid. Calibrate again.")
    for field, limit in (("pitch", 2), ("gain_db", 6)):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or abs(number) > limit:
            raise ValueError("The session calibration is out of range. Calibrate again.")
    return copy.deepcopy(value)


class CharacterProfiles:
    """Profiles are separate from immutable voice recipes and exported audio."""
    def __init__(self, root):
        self.root = Path(root)
        self.folder = self.root / "Character profiles"
        self.folder.mkdir(parents=True, exist_ok=True)

    def _path(self, voice_id):
        if not isinstance(voice_id, str) or len(voice_id) != 32 or any(c not in "0123456789abcdef" for c in voice_id):
            raise ValueError("Invalid character identifier")
        return self.folder / (voice_id + ".json")

    def get(self, voice_id):
        path = self._path(voice_id)
        if not path.exists():
            return {"schema": 1, "locked": True, "notes": "", "references": {}}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != 1 or not isinstance(data.get("references"), dict):
            raise ValueError("This character profile could not be read. Its saved voice and audio are still kept.")
        return data

    def update(self, voice_id, **changes):
        data = self.get(voice_id)
        data.update(changes)
        if not isinstance(data.get("locked"), bool) or not isinstance(data.get("notes"), str) or len(data["notes"]) > 360:
            raise ValueError("Keep acting notes under 360 characters.")
        _atomic_json(self._path(voice_id), data)
        return copy.deepcopy(data)

    def reference(self, voice):
        return self.get(voice["id"])["references"].get(str(voice["version"]))

    def save_reference(self, voice, source, output, measurement, snapshot):
        reference_id = uuid.uuid4().hex
        folder = self.root / "Character references" / voice["id"] / reference_id
        original = unique_copy(source, folder, name="Neutral original" + Path(source).suffix.lower())
        changed = unique_copy(output, folder, name="Neutral character.wav")
        reference = {"id": reference_id, "voice_id": voice["id"], "voice_version": voice["version"],
                     "phrase": PHRASE, "source": str(original), "output": str(changed),
                     "source_sha256": digest(original), "output_sha256": digest(changed),
                     "measurement": measurement, "snapshot": copy.deepcopy(snapshot),
                     "created": datetime.now().astimezone().isoformat()}
        if reference["source_sha256"] != snapshot["source_sha256"] or digest(source) != snapshot["source_sha256"]:
            raise ValueError("The source changed. Open the original again before setting a reference.")
        data = self.get(voice["id"])
        # A replacement gets a new directory; earlier approved reference files stay intact.
        data["references"][str(voice["version"])] = reference
        _atomic_json(self._path(voice["id"]), data)
        _atomic_json(folder / "reference.json", reference)
        return copy.deepcopy(reference)
