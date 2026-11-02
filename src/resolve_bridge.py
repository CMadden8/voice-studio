"""Local, opt-in DaVinci Resolve Studio Voice Isolation.

probe() only attaches to a running instance. The opt-in ensure_started() helper
can launch Resolve locally and replace its verified-empty startup placeholder.
It never changes scripting preferences or switches away from actual user work.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import math
import os
from pathlib import Path
import csv
import io
import json
import subprocess
import sys
import socket
import tempfile
import time
import uuid

import numpy as np
import soundfile as sf


PROJECT_NAME = "Voice Studio Cleanup"
SAMPLE_RATE = 48_000
SDK_MODULES = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData")) / "Blackmagic Design/DaVinci Resolve/Support/Developer/Scripting/Modules"
RESOLVE_DIR = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Blackmagic Design/DaVinci Resolve"
_dll_directory = None


class ResolveError(RuntimeError):
    """An actionable error, with no silent substitution of another provider."""


def _connect():
    """Load Blackmagic's official helper and attach; scriptapp never launches it."""
    global _dll_directory
    import sys
    modules = Path(os.environ.get("RESOLVE_SCRIPT_API", str(SDK_MODULES.parent))) / "Modules"
    if not (modules / "DaVinciResolveScript.py").is_file():
        raise ResolveError("DaVinci Resolve's scripting SDK was not found. Install DaVinci Resolve Studio, then reopen Voice Studio.")
    if str(modules) not in sys.path:
        sys.path.insert(0, str(modules))
    try:
        if hasattr(os, "add_dll_directory") and _dll_directory is None:
            _dll_directory = os.add_dll_directory(str(RESOLVE_DIR))
        sdk = importlib.import_module("DaVinciResolveScript")
        resolve = sdk.scriptapp("Resolve")
    except Exception as exc:
        raise ResolveError(f"Resolve's local scripting connection could not load: {exc}") from exc
    if resolve is None:
        raise ResolveError("Open DaVinci Resolve Studio and wait for it to finish starting. In Resolve Preferences > System > General, set External scripting to Local, then retry.")
    return resolve


def _callable(obj, name):
    method = getattr(obj, name, None)
    return method if callable(method) else None


def _identity(resolve):
    return {"version": str(resolve.GetVersionString() or "unknown"),
            "product": str(resolve.GetProductName() or "DaVinci Resolve")}


def probe() -> dict:
    """Read-only availability check. Never creates a project, timeline or job."""
    result = {"available": False, "status": "unavailable", "version": None,
              "resolve_version": None, "product": None, "project": None,
              "info": "", "error": None, "provider": "resolve", "ai": True}
    try:
        resolve = _connect()
        result.update(_identity(resolve))
        result["resolve_version"] = result["version"]
        manager = resolve.GetProjectManager()
        if manager is None:
            raise ResolveError("Resolve is still starting. Wait until its Project Manager is ready, then retry.")
        project = manager.GetCurrentProject()
        name = str(project.GetName()) if project is not None else None
        result["project"] = name
        if name and name != PROJECT_NAME:
            result["status"] = "project_in_use"
            result["error"] = _foreign_project_message(name)
        elif project is not None and project.IsRenderingInProgress():
            result["status"] = "rendering"
            result["error"] = "Resolve is already rendering. Wait for that render to finish, then retry."
        elif "studio" not in result["product"].lower():
            result["status"] = "studio_required"
            result["error"] = "Voice Isolation requires DaVinci Resolve Studio. Open the Studio edition, then retry."
        else:
            _assert_local_database(manager)
            result.update(available=True, status="ready", info="Local Resolve Studio connection ready. Cleanup uses AI Voice Isolation on this computer.")
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _is_resolve_running():
    if os.name != "nt":
        raise ResolveError("Automatic Resolve startup is currently available on Windows only.")
    try:
        result = subprocess.run(["tasklist.exe", "/FI", "IMAGENAME eq Resolve.exe", "/FO", "CSV", "/NH"],
                                capture_output=True, text=True, timeout=8,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResolveError(f"Could not check whether Resolve is already running: {exc}") from exc
    if result.returncode:
        raise ResolveError("Windows could not check whether Resolve is already running. Open Resolve Studio manually and retry.")
    return any(row and row[0].casefold() == "resolve.exe" for row in csv.reader(io.StringIO(result.stdout)))


def _launch_resolve():
    executable = RESOLVE_DIR / "Resolve.exe"
    if not executable.is_file():
        raise ResolveError("DaVinci Resolve Studio is not installed in its usual Windows location. Open it manually and retry.")
    try:
        # Keep Resolve's normal UI available. A persistent -nogui instance
        # prevents the user's desktop shortcut from opening the editor.
        # Start-Process also avoids inheriting the worker's redirected handles.
        quoted_executable = str(executable).replace("'", "''")
        command = f"Start-Process -FilePath '{quoted_executable}' -WindowStyle Minimized -ErrorAction Stop"
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                                cwd=str(RESOLVE_DIR), capture_output=True, text=True,
                                timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
                                env=dict(os.environ))
        if result.returncode:
            raise ResolveError("Resolve Studio could not start. Open it manually and retry.")
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResolveError(f"Resolve Studio could not start: {exc}") from exc


def _scripting_port_ready():
    # The installed Local-mode Fusion server opens this listener when ready.
    # Never load its SDK during Resolve's own Fusion initialization.
    try:
        with socket.create_connection(("127.0.0.1", 1144), timeout=.2):
            return True
    except OSError:
        return False


def _probe_for_startup(timeout):
    """Bound native SDK connection attempts in disposable child processes."""
    command = [sys.executable, str(Path(__file__).parent / "tools/probe_resolve.py")]
    try:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, env=dict(os.environ),
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            stdout, _ = child.communicate(timeout=max(.1, timeout))
        except subprocess.TimeoutExpired:
            # Windows venv python.exe may redirect to a second Python process.
            # Kill only this owned probe tree, not just its redirector wrapper.
            if os.name == "nt":
                subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                               capture_output=True, timeout=5,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                child.kill()
            stdout, _ = child.communicate(timeout=3)
            stdout = ""
        for line in reversed(stdout.splitlines()):
            if line.startswith("VS_RESULT:"):
                return json.loads(line[len("VS_RESULT:"):])
    except subprocess.TimeoutExpired:
        pass
    except (OSError, ValueError) as exc:
        raise ResolveError(f"Could not check Resolve's local scripting connection: {exc}") from exc
    return {"available": False, "status": "unavailable", "error": "Resolve's local scripting connection is not ready. Wait for startup to finish, and check that External scripting is set to Local."}


def ensure_started(timeout: float = 75) -> dict:
    """Attach or start local Resolve once, preserving any open user project.

    Unlike probe(), this permits launching Resolve and opening the dedicated
    cleanup project from its verified-empty Untitled startup placeholder. Actual
    user projects and scripting preferences remain protected.
    """
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ResolveError("Resolve startup timeout must be a positive number of seconds.")
    deadline = time.monotonic() + timeout
    launched = False
    # Loading fusionscript before starting Resolve can contaminate the native
    # child's script-server environment. Launch first; SDK probes run separately.
    if not _is_resolve_running():
        _launch_resolve()
        launched = True
    while True:
        result = (_probe_for_startup(min(8, max(.1, deadline - time.monotonic())))
                  if _scripting_port_ready() else
                  {"available": False, "status": "unavailable", "error": "Resolve's local scripting server is not ready. Wait for startup to finish, and check that External scripting is set to Local."})
        if result["available"]:
            result["started"] = launched
            return result
        if result["status"] == "project_in_use" and result.get("project") == "Untitled Project":
            if _open_cleanup_if_empty_startup(_connect()):
                continue
        if result["status"] in ("project_in_use", "studio_required", "rendering"):
            raise ResolveError(result["error"])
        if time.monotonic() >= deadline:
            raise ResolveError(result["error"] or "Resolve Studio did not become ready before the startup timeout.")
        time.sleep(min(.35, max(.001, deadline - time.monotonic())))


def _open_cleanup_if_empty_startup(resolve):
    """Handle Resolve's empty startup placeholder, never an edited project.

    Resolve opens 'Untitled Project' on some starts. Only the exact placeholder
    name with no timelines, no recursive media, no jobs and no active render may
    be replaced. There is deliberately no SaveProject/CloseProject operation.
    """
    manager = resolve.GetProjectManager()
    project = manager.GetCurrentProject() if manager is not None else None
    if project is None or str(project.GetName()) != "Untitled Project":
        return False
    _assert_local_database(manager)
    try:
        if project.GetTimelineCount() != 0 or project.GetRenderJobList() or project.IsRenderingInProgress():
            return False
        pending = [project.GetMediaPool().GetRootFolder()]
        visited = 0
        while pending:
            folder = pending.pop()
            visited += 1
            if folder is None or visited > 1000 or folder.GetClipList():
                return False
            children = folder.GetSubFolderList() or []
            pending.extend(children.values() if isinstance(children, dict) else children)
        # Recheck immediately before switching, in case the UI changed during
        # the recursive read. Missing/unsupported checks fail closed.
        current = manager.GetCurrentProject()
        if current is None or str(current.GetName()) != "Untitled Project" or current.GetTimelineCount() != 0 or current.IsRenderingInProgress():
            return False
    except Exception:
        return False
    names = manager.GetProjectListInCurrentFolder() or []
    opened = manager.LoadProject(PROJECT_NAME) if PROJECT_NAME in names else manager.CreateProject(PROJECT_NAME)
    if opened is None or str(opened.GetName()) != PROJECT_NAME:
        raise ResolveError(f"Resolve could not open its dedicated '{PROJECT_NAME}' project after its empty startup screen.")
    return True


def _foreign_project_message(name):
    return (f"Resolve has '{name}' open. Voice Studio will not change or close it. "
            f"Save your work in Resolve, then open '{PROJECT_NAME}' or return to the Project Manager and retry.")


def _assert_project(manager):
    current = manager.GetCurrentProject()
    if current is None or str(current.GetName()) != PROJECT_NAME:
        name = str(current.GetName()) if current is not None else "no project"
        raise ResolveError(_foreign_project_message(name))


def _dedicated_project(manager):
    _assert_local_database(manager)
    current = manager.GetCurrentProject()
    if current is not None:
        if str(current.GetName()) != PROJECT_NAME:
            raise ResolveError(_foreign_project_message(str(current.GetName())))
        return current
    names = manager.GetProjectListInCurrentFolder() or []
    project = manager.LoadProject(PROJECT_NAME) if PROJECT_NAME in names else manager.CreateProject(PROJECT_NAME)
    if project is None or str(project.GetName()) != PROJECT_NAME:
        raise ResolveError(f"Resolve could not open its dedicated '{PROJECT_NAME}' project. Create or open that local project in Resolve, then retry.")
    return project


def _assert_local_database(manager):
    getter = _callable(manager, "GetCurrentDatabase")
    database = getter() if getter else None
    if not isinstance(database, dict) or str(database.get("DbType", "")).lower() != "disk":
        raise ResolveError("Select a local Disk project library in Resolve's Project Manager. Voice Studio cleanup does not use cloud or network project libraries.")


@contextmanager
def _exclusive_session():
    """An OS-released cross-process lock; never truncate somebody else's lock."""
    directory = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "VoiceStudio"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "resolve-cleanup.lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ResolveError("Another Voice Studio cleanup is using Resolve. Wait for it to finish, then retry.") from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _set_fps(project):
    setter = _callable(project, "SetSettings")
    accepted = setter({"timelineFrameRate": "24"}) if setter else project.SetSetting("timelineFrameRate", "24")
    getter = _callable(project, "GetSettings")
    values = getter() if getter else {"timelineFrameRate": project.GetSetting("timelineFrameRate")}
    try:
        fps = float(values.get("timelineFrameRate", 0))
    except (ValueError, TypeError):
        fps = 0
    if fps not in (24, 25):
        raise ResolveError("Resolve could not configure the dedicated cleanup project's frame rate to 24 or 25 fps.")
    return fps


def _voice_isolation(timeline, clips, amount):
    desired = {"isEnabled": True, "amount": amount}
    # Prefer the imported clip, so other audio tracks can never be affected.
    targets = [(clip, (), "clip") for clip in clips]
    targets.append((timeline, (1,), "track"))
    for target, prefix, scope in targets:
        setter = _callable(target, "SetVoiceIsolationState")
        getter = _callable(target, "GetVoiceIsolationState")
        if setter is None or getter is None:
            continue
        if not setter(*prefix, desired):
            continue
        state = getter(*prefix) or {}
        if state.get("isEnabled") is True and state.get("amount") == amount:
            return {"scope": scope, "isEnabled": True, "amount": amount}
    raise ResolveError("Resolve did not confirm that AI Voice Isolation was enabled. Check that this is Resolve Studio and that its Voice Isolation feature is installed. No unprocessed substitute was exported.")


def _wav_settings(project):
    audio_formats = _callable(project, "GetAudioRenderFormats")
    audio_codecs = _callable(project, "GetAudioRenderCodecs")
    modern = audio_formats is not None and audio_codecs is not None
    formats = (audio_formats() if modern else project.GetRenderFormats()) or {}
    candidates = [(name, str(ext)) for name, ext in formats.items() if str(ext).lower().lstrip(".") in ("wav", "wave")]
    if not candidates:
        raise ResolveError("This Resolve installation does not expose WAV audio rendering to its scripting API.")
    extension = candidates[0][1].lstrip(".")
    codecs = (audio_codecs(extension) if modern else project.GetRenderCodecs(extension)) or {}
    choices = [(name, code) for name, code in codecs.items() if "pcm" in f"{name} {code}".lower() and "float" not in f"{name} {code}".lower()]
    if not choices:
        raise ResolveError("Resolve did not offer a linear PCM WAV codec. No lossy substitute was selected.")
    choices.sort(key=lambda item: ("linear" not in str(item[0]).lower(), str(item[0])))
    codec = str(choices[0][1])
    if not modern and not project.SetCurrentRenderFormatAndCodec(extension, codec):
        raise ResolveError("Resolve refused the WAV/linear PCM render format.")
    return extension, codec


def _wait_for_job(project, manager, job_id, timeout):
    deadline = time.monotonic() + timeout
    while True:
        _assert_project(manager)
        state = project.GetRenderJobStatus(job_id) or {}
        status = str(state.get("JobStatus", "")).lower()
        if status in ("complete", "render complete"):
            return state
        if any(word in status for word in ("fail", "cancel")):
            raise ResolveError(f"Resolve Voice Isolation render {status}: {state.get('Error') or 'no additional detail'}")
        if time.monotonic() >= deadline:
            # We started just this job after checking that no render was active.
            # Do not stop a render if our job is no longer the active one.
            if status == "rendering" and project.IsRenderingInProgress():
                project.StopRendering()
            raise ResolveError(f"Resolve Voice Isolation timed out after {timeout:g} seconds. Retry after checking Resolve's render queue.")
        if not project.IsRenderingInProgress():
            raise ResolveError(f"Resolve stopped before its cleanup render completed (status: {state.get('JobStatus') or 'unknown'}).")
        time.sleep(min(.25, max(.001, deadline - time.monotonic())))


def _validate_render(rendered, source_audio, fps):
    try:
        audio, rate = sf.read(str(rendered), dtype="float32", always_2d=True)
        info = sf.info(str(rendered))
    except Exception as exc:
        raise ResolveError(f"Resolve's rendered WAV could not be read: {exc}") from exc
    if rate != SAMPLE_RATE or info.format not in ("WAV", "WAVEX") or info.subtype != "PCM_24":
        raise ResolveError(f"Resolve returned {rate} Hz {info.format}/{info.subtype}; cleanup requires 48 kHz, 24-bit PCM WAV.")
    if not len(audio) or not np.isfinite(audio).all():
        raise ResolveError("Resolve returned empty or invalid audio.")
    # Resolve's video-framed timeline can round an audio-only tail by one frame.
    tolerance = math.ceil(SAMPLE_RATE / fps) + 2
    if abs(len(audio) - len(source_audio)) > tolerance:
        raise ResolveError("Resolve returned audio with an unexpected duration. The result was not exported.")
    # Mono sources may render through a stereo bus. Pick the most energetic
    # channel instead of summing channels that could cancel one another.
    channel = int(np.argmax(np.mean(np.square(audio.astype(np.float64)), axis=0)))
    audio = audio[:, channel]
    if float(np.sqrt(np.mean(source_audio.astype(np.float64) ** 2))) > 1e-4 and float(np.max(np.abs(audio))) < 1e-7:
        raise ResolveError("Resolve returned silence even though the source contains sound. Reduce Voice Isolation or check the clip in Resolve.")
    audio = audio[:len(source_audio)]
    if len(audio) < len(source_audio):
        audio = np.pad(audio, (0, len(source_audio) - len(audio)))
    return audio, len(audio) != info.frames


def isolate_audio(source: Path, output: Path, amount: int = 75, timeout: float = 180) -> dict:
    """AI-clean a mono WAV through local Resolve Studio, or raise ResolveError.

    The caller supplies a new output path. Input and existing output files are
    never overwritten. A failure never falls back to non-AI cleanup.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ResolveError("Resolve cleanup needs a separate output file; the original recording is protected.")
    if output.exists():
        raise ResolveError("The Resolve cleanup output already exists. Choose a new output filename.")
    if output.suffix.lower() != ".wav":
        raise ResolveError("Resolve cleanup output must use a .wav filename.")
    if isinstance(amount, bool) or not isinstance(amount, int) or not 0 <= amount <= 100:
        raise ResolveError("Voice Isolation amount must be an integer from 0 to 100.")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ResolveError("Resolve cleanup timeout must be a positive number of seconds.")
    try:
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        source_audio, rate = sf.read(str(source), dtype="float32", always_2d=True)
        source_info = sf.info(str(source))
    except Exception as exc:
        raise ResolveError(f"The cleanup source could not be read: {exc}") from exc
    if source_info.format != "WAV" or rate != SAMPLE_RATE or source_audio.shape[1] != 1 or not len(source_audio) or not np.isfinite(source_audio).all():
        raise ResolveError("Resolve cleanup expects a valid, non-empty mono WAV at 48 kHz.")
    with _exclusive_session():
        resolve = _connect()
        identity = _identity(resolve)
        if "studio" not in identity["product"].lower():
            raise ResolveError("AI Voice Isolation requires DaVinci Resolve Studio.")
        manager = resolve.GetProjectManager()
        if manager is None:
            raise ResolveError("Resolve's Project Manager is not ready. Wait for startup to finish, then retry.")
        project = _dedicated_project(manager)
        if project.IsRenderingInProgress():
            raise ResolveError("Resolve is already rendering. Wait for that render to finish, then retry.")
        _assert_project(manager)
        fps = _set_fps(project)
        pool = project.GetMediaPool()
        # Plain paths work on Resolve 20/21 and are required for ordinary WAVs
        # by 21.1.1 despite the new type stubs advertising only dict imports.
        imported = pool.ImportMedia([str(source)]) or []
        if len(imported) != 1:
            raise ResolveError("Resolve could not import the cleanup source WAV.")
        timeline_name = "VoiceStudio-" + uuid.uuid4().hex
        timeline = pool.CreateEmptyTimeline(timeline_name)
        if timeline is None or not project.SetCurrentTimeline(timeline):
            raise ResolveError("Resolve could not create its private audio cleanup timeline.")
        if timeline.GetTrackCount("audio") == 0 and not timeline.AddTrack("audio", "mono"):
            raise ResolveError("Resolve could not create a mono audio track.")
        clips = pool.AppendToTimeline([{"mediaPoolItem": imported[0], "mediaType": 2,
                                        "trackIndex": 1, "recordFrame": timeline.GetStartFrame()}]) or []
        if len(clips) != 1:
            raise ResolveError("Resolve could not place the recording on its cleanup timeline.")
        if timeline.GetItemListInTrack("video", 1):
            raise ResolveError("Resolve unexpectedly created a video clip in the audio cleanup timeline.")
        isolation = _voice_isolation(timeline, clips, amount)
        extension, codec = _wav_settings(project)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".resolve-cleanup-", dir=str(output.parent)) as scratch:
            render_name = "voice-" + uuid.uuid4().hex
            settings = {"SelectAllFrames": True, "TargetDir": scratch,
                        "CustomName": render_name, "ExportVideo": False,
                        "ExportAudio": True, "AudioFormat": extension,
                        "AudioCodec": codec, "AudioSampleRate": SAMPLE_RATE,
                        "AudioBitDepth": 24,
                        "UseUniqueFilenames": False}
            _assert_project(manager)
            if not project.SetCurrentRenderMode(1) or not project.SetRenderSettings(settings):
                raise ResolveError("Resolve refused the audio-only 48 kHz/24-bit WAV render settings.")
            if project.IsRenderingInProgress():
                raise ResolveError("Another render started in Resolve. Wait for it to finish, then retry.")
            job_id = project.AddRenderJob()
            if not job_id:
                raise ResolveError("Resolve could not add the cleanup render job.")
            jobs = project.GetRenderJobList() or []
            own_job = next((job for job in jobs if str(job.get("JobId")) == str(job_id)), None)
            if own_job is None or own_job.get("IsExportVideo") is not False or own_job.get("IsExportAudio") is not True:
                raise ResolveError("Resolve did not confirm the cleanup job is audio-only. It was not started.")
            if not project.StartRendering([job_id], isInteractiveMode=False):
                raise ResolveError("Resolve could not start the cleanup render. Check its render queue.")
            _wait_for_job(project, manager, job_id, float(timeout))
            files = list(Path(scratch).glob(render_name + "*.wav"))
            if len(files) != 1:
                raise ResolveError("Resolve reported completion but did not produce the expected WAV file.")
            audio, aligned = _validate_render(files[0], source_audio[:, 0], fps)
            if hashlib.sha256(source.read_bytes()).hexdigest() != before:
                raise ResolveError("The original recording changed during cleanup. The result was not exported.")
            # Exclusive creation prevents even a late competing export overwrite.
            created = False
            try:
                with output.open("xb") as destination:
                    created = True
                    sf.write(destination, audio, SAMPLE_RATE, subtype="PCM_24", format="WAV")
            except FileExistsError as exc:
                raise ResolveError("The Resolve cleanup output was created by another export. Choose a new output filename.") from exc
            except Exception as exc:
                if created:
                    output.unlink(missing_ok=True)
                raise ResolveError(f"The cleaned WAV could not be saved: {exc}") from exc
            return {"audio": str(output), "provider": "resolve", "ai": True,
                    "amount": amount, "resolve_version": identity["version"],
                    "product": identity["product"], "project": PROJECT_NAME,
                    "timeline": timeline_name, "job_id": str(job_id),
                    "isolation": isolation, "sample_rate": SAMPLE_RATE,
                    "tail_aligned": aligned, "warning": None}
