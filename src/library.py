"""Versioned, local character-voice presets and non-destructive file copies.

Presets contain conventional DSP parameters, never trained voice models.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import shutil
import unicodedata
import uuid

from preset_catalog import PRESETS
from persistence import replace_with_retry


DEFAULT_SETTINGS = {
    "pitch": 0., "resonance": 1., "tone": 0.,
    "texture": 0., "room": 0., "speed": 1., "cleanup": 0.,
}
SETTING_RANGES = {
    "pitch": (-8., 8.), "resonance": (.75, 1.35), "tone": (-1., 1.),
    "texture": (0., 1.), "room": (0., .5), "speed": (.75, 1.25), "cleanup": (0., 1.),
}
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{kind}{number}" for kind in ("COM", "LPT") for number in range(1, 10)
}


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("Give the voice a name.")
    name = unicodedata.normalize("NFC", name).strip()
    if not name or len(name) > 80 or any(ord(c) < 32 for c in name):
        raise ValueError("Use a voice name between 1 and 80 characters, without control characters.")
    return name


def _validate_description(description: str) -> str:
    if not isinstance(description, str):
        raise ValueError("Use text for the voice description.")
    description = unicodedata.normalize("NFC", description).strip()
    if len(description) > 360 or any(ord(c) < 32 for c in description):
        raise ValueError("Keep the voice description under 360 characters without control characters.")
    return description


def _normalise_voice(voice: dict) -> dict:
    """Expand legacy defaults in a returned copy, never in stored history."""
    result = copy.deepcopy(voice)
    result["settings"] = validate_settings(result["settings"])
    result["description"] = _validate_description(result.get("description", ""))
    return result


def validate_settings(settings: dict) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("Voice settings must be a dictionary.")
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError("Unknown voice setting: " + ", ".join(sorted(str(k) for k in unknown)))
    result = DEFAULT_SETTINGS.copy()
    for key, value in settings.items():
        low, high = SETTING_RANGES[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low:g} and {high:g}.")
        result[key] = float(value)
    return result


def safe_name(text: str) -> str:
    """Return a readable single Windows filename component, never a path."""
    if not isinstance(text, str):
        raise ValueError("A filename must be text.")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    text = text[:110].rstrip(" .") or "Untitled"
    if text.split(".", 1)[0].upper() in _RESERVED_NAMES:
        text = "_" + text
    return text


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class VoiceLibrary:
    """Each voice owns one atomic document containing its complete history."""
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        existing_names = set()
        for existing in self.folder.glob("*.json"):
            try:
                uuid.UUID(existing.stem)
            except ValueError:
                continue
            existing_names.add(self._read(existing.stem)["versions"][-1]["name"].casefold())
        for preset in PRESETS:
            name, changes = preset["name"], preset["settings"]
            voice_id = uuid.uuid5(uuid.NAMESPACE_URL, "VoiceStudio/builtin/" + name).hex
            path = self._path(voice_id)
            if not path.exists() and name.casefold() not in existing_names:
                voice = {"id": voice_id, "name": name, "version": 1,
                         "settings": validate_settings(changes), "builtin": True,
                         "description": _validate_description(preset["description"])}
                _atomic_json(path, {"schema": 1, "id": voice_id, "archived": False, "versions": [voice]})
                existing_names.add(name.casefold())

    def _path(self, voice_id) -> Path:
        try:
            canonical = uuid.UUID(str(voice_id)).hex
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("That voice identifier is invalid.") from error
        return self.folder / f"{canonical}.json"

    def _read(self, voice_id) -> dict:
        path = self._path(voice_id)
        if not path.exists():
            raise KeyError("That voice is no longer in the library.")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema") != 1 or data.get("id") != path.stem:
                raise ValueError("Invalid voice document")
            versions = data["versions"]
            if not isinstance(versions, list) or not versions:
                raise ValueError("Missing voice history")
            for expected_version, voice in enumerate(versions, 1):
                if voice["id"] != path.stem or voice["version"] != expected_version:
                    raise ValueError("Invalid voice history")
                _validate_name(voice["name"])
                validate_settings(voice["settings"])
                _validate_description(voice.get("description", ""))
            return data
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"The saved voice file could not be read: {path.name}") from error

    def _check_unique_name(self, name: str, except_id=None):
        for voice in self.list_voices():
            if voice["id"] != except_id and voice["name"].casefold() == name.casefold():
                raise ValueError("A voice with that name already exists. Choose another name.")

    def list_voices(self) -> list[dict]:
        voices = []
        for path in self.folder.glob("*.json"):
            try:
                uuid.UUID(path.stem)
            except ValueError:
                continue
            data = self._read(path.stem)
            if not data.get("archived", False):
                voices.append(_normalise_voice(data["versions"][-1]))
        return sorted(voices, key=lambda voice: (voice["name"].casefold(), voice["id"]))

    def get(self, voice_id, version=None) -> dict:
        data = self._read(voice_id)
        if version is None:
            return _normalise_voice(data["versions"][-1])
        if isinstance(version, bool) or not isinstance(version, int) or not 1 <= version <= len(data["versions"]):
            raise KeyError("That voice revision is not available.")
        return _normalise_voice(data["versions"][version - 1])

    def history(self, voice_id) -> list[dict]:
        return [_normalise_voice(voice) for voice in self._read(voice_id)["versions"][::-1]]

    def create(self, name: str, settings: dict, description="") -> dict:
        name = _validate_name(name)
        settings = validate_settings(settings)
        self._check_unique_name(name)
        voice_id = uuid.uuid4().hex
        voice = {"id": voice_id, "name": name, "version": 1, "settings": settings, "builtin": False,
                 "description": _validate_description(description)}
        _atomic_json(self._path(voice_id), {"schema": 1, "id": voice_id, "archived": False, "versions": [voice]})
        return copy.deepcopy(voice)

    def update(self, voice_id, settings: dict, name=None, description=None) -> dict:
        data = self._read(voice_id)
        if data.get("archived", False):
            raise ValueError("Duplicate this archived voice to edit it again.")
        previous = _normalise_voice(data["versions"][-1])
        name = _validate_name(previous["name"] if name is None else name)
        settings = validate_settings(settings)
        self._check_unique_name(name, previous["id"])
        description = _validate_description(previous["description"] if description is None else description)
        voice = {**previous, "name": name, "version": previous["version"] + 1, "settings": settings,
                 "description": description}
        data["versions"].append(voice)
        _atomic_json(self._path(voice_id), data)
        return copy.deepcopy(voice)

    def duplicate(self, voice_id, name: str) -> dict:
        source = self.get(voice_id)
        return self.create(name, source["settings"], description=source["description"])

    def archive(self, voice_id):
        data = self._read(voice_id)
        data["archived"] = True
        _atomic_json(self._path(voice_id), data)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as incoming:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_copy(source: Path, destination_folder: Path, name=None) -> Path:
    """Copy intact, reuse identical named copies, and preserve earlier versions."""
    source = Path(source).resolve(strict=True)
    if not source.is_file():
        raise ValueError("Choose an audio file to copy.")
    folder = Path(destination_folder)
    folder.mkdir(parents=True, exist_ok=True)
    requested = Path(safe_name(source.name if name is None else name))
    source_hash = _hash_file(source)
    number = 1
    while True:
        filename = requested.name if number == 1 else f"{requested.stem} ({number}){requested.suffix}"
        destination = folder / filename
        if destination.exists():
            if destination.resolve() == source or (destination.is_file() and _hash_file(destination) == source_hash):
                return destination
            number += 1
            continue
        try:
            target = destination.open("xb")
        except FileExistsError:
            continue
        try:
            with target, source.open("rb") as incoming:
                shutil.copyfileobj(incoming, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if _hash_file(destination) != source_hash:
                raise OSError("The copied recording could not be verified.")
            try:
                shutil.copystat(source, destination)
            except OSError:
                pass
            return destination
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
