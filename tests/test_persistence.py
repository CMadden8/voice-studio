"""Bounded retries never publish a partial JSON document or hide real errors."""
from pathlib import Path
import json

import pytest

import persistence


def test_transient_lock_eventually_publishes_complete_json(tmp_path, monkeypatch):
    target = tmp_path / "preferences.json"
    temporary = tmp_path / "preferences.pending"
    target.write_text('{"volume": 25}', encoding="utf-8")
    temporary.write_text('{"volume": 75, "cleanup_provider": "resolve"}', encoding="utf-8")
    real_replace = Path.replace
    calls = []
    sleeps = []
    def locked_then_available(source, destination):
        calls.append(True)
        assert json.loads(target.read_text(encoding="utf-8")) == {"volume": 25}
        if len(calls) < 4:
            raise PermissionError("temporary Windows sharing lock")
        return real_replace(source, destination)
    monkeypatch.setattr(Path, "replace", locked_then_available)
    monkeypatch.setattr(persistence.time, "sleep", sleeps.append)
    result = persistence.replace_with_retry(temporary, target)
    assert result == target and len(calls) == 4
    assert sleeps == [.02, .04, .06]
    assert json.loads(target.read_text(encoding="utf-8")) == {"volume": 75, "cleanup_provider": "resolve"}
    assert not temporary.exists()


def test_permanent_lock_preserves_existing_file_and_raises(tmp_path, monkeypatch):
    target = tmp_path / "voices.json"
    temporary = tmp_path / "voices.pending"
    target.write_bytes(b'{"version": 1}')
    temporary.write_bytes(b'{"version": 2}')
    calls = []
    sleeps = []
    error = PermissionError("permanent sharing lock")
    def always_locked(source, destination):
        calls.append(True)
        raise error
    monkeypatch.setattr(Path, "replace", always_locked)
    monkeypatch.setattr(persistence.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as failure:
        persistence.replace_with_retry(temporary, target)
    assert failure.value is error
    assert len(calls) == 6
    assert sleeps == [.02, .04, .06, .08, .1]
    assert sum(sleeps) == pytest.approx(.3)
    assert target.read_bytes() == b'{"version": 1}'
    assert temporary.read_bytes() == b'{"version": 2}'


def test_other_errors_are_not_retried(tmp_path, monkeypatch):
    calls = []
    sleeps = []
    error = FileNotFoundError("missing temporary file")
    def missing(source, destination):
        calls.append(True)
        raise error
    monkeypatch.setattr(Path, "replace", missing)
    monkeypatch.setattr(persistence.time, "sleep", sleeps.append)
    with pytest.raises(FileNotFoundError) as failure:
        persistence.replace_with_retry(tmp_path / "missing", tmp_path / "target")
    assert failure.value is error and len(calls) == 1
    assert sleeps == []
