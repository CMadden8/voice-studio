"""Resolve API contract and isolation safety, without opening native Resolve."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import resolve_bridge as bridge


class Clip:
    def __init__(self):
        self.state = {}
        self.accept = True
        self.confirm = True

    def SetVoiceIsolationState(self, state):
        self.state = dict(state)
        return self.accept

    def GetVoiceIsolationState(self):
        return self.state if self.confirm else {"isEnabled": False, "amount": 0}


class Timeline(Clip):
    def GetTrackCount(self, kind):
        return 1 if kind == "audio" else 0

    def GetStartFrame(self):
        return 86400

    def GetItemListInTrack(self, kind, index):
        return []

    def SetVoiceIsolationState(self, index, state):
        return super().SetVoiceIsolationState(state)

    def GetVoiceIsolationState(self, index):
        return super().GetVoiceIsolationState()


class Pool:
    def __init__(self, project):
        self.project = project
        self.source = None
        self.clip = Clip()
        self.timeline = Timeline()
        self.calls = []
        self.root_folder = Folder()

    def GetRootFolder(self):
        return self.root_folder

    def ImportMedia(self, clips):
        self.calls.append(("import", clips))
        self.source = clips[0]
        return [object()]

    def CreateEmptyTimeline(self, name):
        self.calls.append(("timeline", name))
        return self.timeline

    def AppendToTimeline(self, clips):
        self.calls.append(("append", clips))
        return [self.clip]


class Project:
    def __init__(self, name=bridge.PROJECT_NAME):
        self.name = name
        self.pool = Pool(self)
        self.calls = []
        self.settings = {}
        self.job_status = "Complete"
        self.rendering = False
        self.render_write = True
        self.render_subtype = "PCM_24"
        self.render_rate = 48000
        self.render_samples = None
        self.render_silent = False
        self.audio_only = True
        self.settings_ok = True
        self.start_ok = True
        self.codecs = {"Linear PCM": "lpcm"}
        self.formats = {"Wave": "wav"}
        self.timeline_count = 0
        self.jobs = None

    def GetTimelineCount(self):
        return self.timeline_count

    def GetName(self):
        return self.name

    def IsRenderingInProgress(self):
        return self.rendering

    def SetSettings(self, settings):
        self.calls.append(("project_settings", settings))
        return True

    def GetSettings(self):
        return {"timelineFrameRate": 24}

    def GetMediaPool(self):
        return self.pool

    def SetCurrentTimeline(self, timeline):
        self.calls.append(("current_timeline", timeline))
        return True

    def GetAudioRenderFormats(self):
        return self.formats

    def GetAudioRenderCodecs(self, extension):
        assert extension == "wav"
        return self.codecs

    def SetCurrentRenderMode(self, mode):
        self.calls.append(("mode", mode))
        return True

    def SetRenderSettings(self, settings):
        self.calls.append(("render_settings", settings))
        self.settings = settings
        return self.settings_ok

    def AddRenderJob(self):
        self.calls.append(("add_job",))
        return "owned-job"

    def GetRenderJobList(self):
        if self.jobs is not None:
            return self.jobs
        return [{"JobId": "some-other-job", "IsExportVideo": True, "IsExportAudio": True},
                {"JobId": "owned-job", "IsExportVideo": not self.audio_only, "IsExportAudio": True}]

    def StartRendering(self, job_ids, isInteractiveMode=False):
        self.calls.append(("start", job_ids, isInteractiveMode))
        if self.render_write:
            audio, _ = sf.read(self.pool.source, dtype="float32")
            if self.render_samples is not None:
                audio = np.pad(audio, (0, max(0, self.render_samples - len(audio))))[:self.render_samples]
            if self.render_silent:
                audio[:] = 0
            target = Path(self.settings["TargetDir"]) / (self.settings["CustomName"] + ".wav")
            sf.write(target, audio, self.render_rate, subtype=self.render_subtype)
        self.rendering = self.job_status == "Rendering"
        return self.start_ok

    def GetRenderJobStatus(self, job_id):
        assert job_id == "owned-job"
        return {"JobStatus": self.job_status, "Error": "test render failure"}

    def StopRendering(self):
        self.calls.append(("stop",))
        self.rendering = False


class Manager:
    def __init__(self, project):
        self.project = project
        self.calls = []
        self.db_type = "Disk"
        self.existing = []

    def GetCurrentDatabase(self):
        return {"DbType": self.db_type, "DbName": "Local Database"}

    def GetCurrentProject(self):
        return self.project

    def GetProjectListInCurrentFolder(self):
        return self.existing

    def CreateProject(self, name):
        self.calls.append(("create", name))
        self.project = Project(name)
        return self.project

    def LoadProject(self, name):
        self.calls.append(("load", name))
        self.project = Project(name)
        return self.project


class Resolve:
    def __init__(self, project):
        self.manager = Manager(project)
        self.product = "DaVinci Resolve Studio"

    def GetVersionString(self):
        return "21.1.1.10"

    def GetProductName(self):
        return self.product

    def GetProjectManager(self):
        return self.manager


class Folder:
    def __init__(self, clips=None, children=None):
        self.clips = clips or []
        self.children = children or []

    def GetClipList(self):
        return self.clips

    def GetSubFolderList(self):
        return self.children


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "original.wav"
    audio = .12 * np.sin(2 * np.pi * 230 * np.arange(48000) / 48000)
    sf.write(source, audio, 48000, subtype="FLOAT")
    resolve = Resolve(Project())
    monkeypatch.setattr(bridge, "_connect", lambda: resolve)
    monkeypatch.setattr(bridge, "_exclusive_session", nullcontext)
    monkeypatch.setattr(bridge, "_is_resolve_running", lambda: True)
    monkeypatch.setattr(bridge, "_probe_for_startup", lambda _: bridge.probe())
    monkeypatch.setattr(bridge, "_scripting_port_ready", lambda: True)
    return source, tmp_path / "clean.wav", resolve


def test_probe_reads_only_and_reports_ready(setup):
    _, _, resolve = setup
    result = bridge.probe()
    assert result["available"] and result["ai"]
    assert result["version"] == "21.1.1.10"
    assert result["project"] == bridge.PROJECT_NAME
    assert resolve.manager.calls == []
    assert resolve.manager.project.calls == []
    assert resolve.manager.project.pool.calls == []


def test_foreign_project_protected_in_probe_and_cleanup(setup):
    source, output, resolve = setup
    resolve.manager.project.name = "A film with unsaved edits"
    result = bridge.probe()
    assert result["status"] == "project_in_use"
    assert "will not change or close" in result["error"]
    with pytest.raises(bridge.ResolveError, match="A film with unsaved edits"):
        bridge.isolate_audio(source, output)
    assert resolve.manager.calls == []
    assert resolve.manager.project.calls == []
    assert resolve.manager.project.pool.calls == []
    assert not output.exists()


def test_disconnect_actionable_and_no_fallback(setup, monkeypatch):
    source, output, _ = setup
    def disconnected():
        raise bridge.ResolveError("Open DaVinci Resolve Studio")
    monkeypatch.setattr(bridge, "_connect", disconnected)
    assert "Open DaVinci" in bridge.probe()["error"]
    with pytest.raises(bridge.ResolveError, match="Open DaVinci"):
        bridge.isolate_audio(source, output)
    assert not output.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_only_named_project_created_or_loaded(setup, existing):
    source, output, resolve = setup
    resolve.manager.project = None
    resolve.manager.existing = [bridge.PROJECT_NAME] if existing else []
    assert bridge.probe()["available"]
    assert resolve.manager.calls == []
    bridge.isolate_audio(source, output)
    assert resolve.manager.calls == [("load" if existing else "create", bridge.PROJECT_NAME)]


def test_cloud_or_network_library_refused_without_project_mutation(setup):
    source, output, resolve = setup
    resolve.manager.db_type = "PostgreSQL"
    assert not bridge.probe()["available"]
    with pytest.raises(bridge.ResolveError, match="local Disk"):
        bridge.isolate_audio(source, output)
    assert resolve.manager.project.calls == []


def test_real_wav_output_and_verified_isolation_metadata(setup):
    source, output, resolve = setup
    before = source.read_bytes()
    result = bridge.isolate_audio(source, output, amount=63)
    assert result["audio"] == str(output)
    assert result["provider"] == "resolve" and result["ai"] is True
    assert result["isolation"] == {"scope": "clip", "isEnabled": True, "amount": 63}
    assert result["resolve_version"] == "21.1.1.10"
    assert result["project"] == bridge.PROJECT_NAME
    assert source.read_bytes() == before
    info = sf.info(output)
    assert (info.samplerate, info.channels, info.frames, info.subtype) == (48000, 1, 48000, "PCM_24")
    project = resolve.manager.project
    assert ("start", ["owned-job"], False) in project.calls
    assert project.settings["ExportVideo"] is False
    assert project.settings["ExportAudio"] is True
    assert project.settings["AudioFormat"] == "wav"
    assert project.settings["AudioCodec"] == "lpcm"
    assert project.pool.calls[-1][1][0]["mediaType"] == 2


def test_clip_readback_failure_uses_verified_track_api(setup):
    source, output, resolve = setup
    resolve.manager.project.pool.clip.confirm = False
    result = bridge.isolate_audio(source, output)
    assert result["isolation"]["scope"] == "track"


def test_isolation_never_assumed_from_a_success_return(setup):
    source, output, resolve = setup
    pool = resolve.manager.project.pool
    pool.clip.confirm = pool.timeline.confirm = False
    with pytest.raises(bridge.ResolveError, match="did not confirm"):
        bridge.isolate_audio(source, output)
    assert not output.exists()
    assert not any(call[0] == "start" for call in resolve.manager.project.calls)


@pytest.mark.parametrize("status", ["Failed", "Cancelled", "Background Render Cancelled"])
def test_render_failure_no_output(setup, status):
    source, output, resolve = setup
    resolve.manager.project.job_status = status
    with pytest.raises(bridge.ResolveError, match="render"):
        bridge.isolate_audio(source, output)
    assert not output.exists()


def test_timeout_stops_only_owned_active_render(setup):
    source, output, resolve = setup
    resolve.manager.project.job_status = "Rendering"
    with pytest.raises(bridge.ResolveError, match="timed out"):
        bridge.isolate_audio(source, output, timeout=.001)
    assert ("stop",) in resolve.manager.project.calls
    assert not output.exists()


def test_preexisting_render_is_never_started_or_stopped(setup):
    source, output, resolve = setup
    resolve.manager.project.rendering = True
    assert bridge.probe()["status"] == "rendering"
    with pytest.raises(bridge.ResolveError, match="already rendering"):
        bridge.isolate_audio(source, output)
    assert resolve.manager.project.calls == []


@pytest.mark.parametrize("problem", ["render_write", "audio_only", "settings_ok", "start_ok"])
def test_missing_file_unverified_job_or_settings_fail_explicitly(setup, problem):
    source, output, resolve = setup
    setattr(resolve.manager.project, problem, False)
    with pytest.raises(bridge.ResolveError):
        bridge.isolate_audio(source, output)
    assert not output.exists()


@pytest.mark.parametrize("problem,value", [("render_samples", 96000), ("render_rate", 44100),
                                           ("render_subtype", "PCM_16"), ("render_silent", True)])
def test_bad_render_audio_rejected(setup, problem, value):
    source, output, resolve = setup
    setattr(resolve.manager.project, problem, value)
    with pytest.raises(bridge.ResolveError):
        bridge.isolate_audio(source, output)
    assert not output.exists()


def test_frame_rounded_tail_aligned_to_source(setup):
    source, output, resolve = setup
    resolve.manager.project.render_samples = 49000
    result = bridge.isolate_audio(source, output)
    assert sf.info(output).frames == 48000
    assert result["tail_aligned"] is True
    assert result["warning"] is None


def test_resolves_wav_extensible_output_is_valid(setup):
    source, output, resolve = setup
    # Resolve Studio 21 emits PCM WAVEX even for mono-source audio-only jobs.
    source_audio, _ = sf.read(source, dtype="float32")
    rendered = output.with_name("native-render.wav")
    sf.write(rendered, source_audio, 48000, subtype="PCM_24", format="WAVEX")
    audio, aligned = bridge._validate_render(rendered, source_audio, 24)
    assert len(audio) == len(source_audio) and not aligned


def test_lossy_codec_not_used_as_fallback(setup):
    source, output, resolve = setup
    resolve.manager.project.codecs = {"AAC": "aac"}
    with pytest.raises(bridge.ResolveError, match="linear PCM"):
        bridge.isolate_audio(source, output)


def test_original_and_existing_output_protected(setup):
    source, output, resolve = setup
    before = source.read_bytes()
    with pytest.raises(bridge.ResolveError, match="separate output"):
        bridge.isolate_audio(source, source)
    output.write_bytes(b"existing export")
    with pytest.raises(bridge.ResolveError, match="already exists"):
        bridge.isolate_audio(source, output)
    assert source.read_bytes() == before
    assert output.read_bytes() == b"existing export"
    assert resolve.manager.project.calls == []


@pytest.mark.parametrize("amount", [-1, 101, True, 3.5, "75"])
def test_invalid_amount_no_project_writes(setup, amount):
    source, output, resolve = setup
    with pytest.raises(bridge.ResolveError, match="integer"):
        bridge.isolate_audio(source, output, amount=amount)
    assert resolve.manager.project.calls == []


def test_ensure_started_ready_instance_never_launches(setup, monkeypatch):
    monkeypatch.setattr(bridge, "_launch_resolve", lambda: pytest.fail("must not launch"))
    assert bridge.ensure_started()["started"] is False


def test_ensure_started_foreign_project_never_launches(setup, monkeypatch):
    setup[2].manager.project.name = "User's work"
    monkeypatch.setattr(bridge, "_launch_resolve", lambda: pytest.fail("must not launch"))
    with pytest.raises(bridge.ResolveError, match="User's work"):
        bridge.ensure_started()


def test_ensure_started_launches_once_then_polls(monkeypatch):
    unavailable = {"available": False, "status": "unavailable", "error": "not ready"}
    states = iter([unavailable, unavailable, {"available": True, "status": "ready"}])
    calls = []
    def startup_probe(_):
        calls.append("probe")
        return next(states)
    monkeypatch.setattr(bridge, "_probe_for_startup", startup_probe)
    monkeypatch.setattr(bridge, "_scripting_port_ready", lambda: True)
    monkeypatch.setattr(bridge, "_is_resolve_running", lambda: False)
    monkeypatch.setattr(bridge, "_launch_resolve", lambda: calls.append("launch"))
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    assert bridge.ensure_started()["started"] is True
    assert calls == ["launch", "probe", "probe", "probe"]


def test_ensure_started_running_but_unavailable_never_launches(monkeypatch):
    monkeypatch.setattr(bridge, "_probe_for_startup", lambda _: {"available": False, "status": "unavailable", "error": "Set scripting to Local"})
    monkeypatch.setattr(bridge, "_is_resolve_running", lambda: True)
    monkeypatch.setattr(bridge, "_scripting_port_ready", lambda: True)
    monkeypatch.setattr(bridge, "_launch_resolve", lambda: pytest.fail("must not launch"))
    with pytest.raises(bridge.ResolveError, match="scripting to Local"):
        bridge.ensure_started(timeout=.001)


@pytest.mark.parametrize("existing", [False, True])
def test_empty_startup_placeholder_can_open_dedicated_project(setup, existing):
    _, _, resolve = setup
    resolve.manager.project.name = "Untitled Project"
    resolve.manager.project.jobs = []
    resolve.manager.existing = [bridge.PROJECT_NAME] if existing else []
    result = bridge.ensure_started()
    assert result["available"] and result["project"] == bridge.PROJECT_NAME
    assert resolve.manager.calls == [("load" if existing else "create", bridge.PROJECT_NAME)]


@pytest.mark.parametrize("content", ["timeline", "root_media", "nested_media", "job", "rendering"])
def test_nonempty_untitled_project_never_switched(setup, content):
    _, _, resolve = setup
    project = resolve.manager.project
    project.name = "Untitled Project"
    project.jobs = []
    if content == "timeline":
        project.timeline_count = 1
    elif content == "root_media":
        project.pool.root_folder.clips = [object()]
    elif content == "nested_media":
        project.pool.root_folder.children = [Folder(children=[Folder(clips=[object()])])]
    elif content == "job":
        project.jobs = [{"JobId": "user-render"}]
    else:
        project.rendering = True
    with pytest.raises(bridge.ResolveError, match="Untitled Project"):
        bridge.ensure_started()
    assert resolve.manager.calls == []
    assert project.calls == []


def test_startup_never_imports_sdk_before_local_server_is_ready(monkeypatch):
    ports = iter([False, False, True])
    calls = []
    monkeypatch.setattr(bridge, "_is_resolve_running", lambda: True)
    monkeypatch.setattr(bridge, "_scripting_port_ready", lambda: next(ports))
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    def startup_probe(_):
        calls.append("probe")
        return {"available": True, "status": "ready"}
    monkeypatch.setattr(bridge, "_probe_for_startup", startup_probe)
    assert bridge.ensure_started()["available"]
    assert calls == ["probe"]


def test_startup_timeout_never_probes_sdk_before_listener_ready(monkeypatch):
    elapsed = iter(range(100))
    calls = []
    monkeypatch.setattr(bridge.time, "monotonic", lambda: next(elapsed))
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    monkeypatch.setattr(bridge, "_is_resolve_running", lambda: False)
    monkeypatch.setattr(bridge, "_launch_resolve", lambda: calls.append("launch"))
    monkeypatch.setattr(bridge, "_scripting_port_ready", lambda: False)
    monkeypatch.setattr(bridge, "_probe_for_startup", lambda _: pytest.fail("SDK must not be loaded during startup"))
    with pytest.raises(bridge.ResolveError, match="server is not ready"):
        bridge.ensure_started()
    assert calls == ["launch"]


def test_bounded_probe_returns_json_without_loading_parent_sdk(monkeypatch):
    class Child:
        pid = 123456
        def communicate(self, timeout):
            return ('some native output\nVS_RESULT:{"available":true,"status":"ready"}\n', '')
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(bridge, "_connect", lambda: pytest.fail("parent must not load SDK"))
    assert bridge._probe_for_startup(1)["available"]


def test_timed_out_probe_kills_its_owned_windows_process_tree(monkeypatch):
    calls = []
    class Child:
        pid = 123456
        count = 0
        def communicate(self, timeout):
            self.count += 1
            if self.count == 1:
                raise bridge.subprocess.TimeoutExpired("probe", timeout)
            return ('', '')
    monkeypatch.setattr(bridge.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(bridge.subprocess, "run", lambda command, **kwargs: calls.append(command))
    assert not bridge._probe_for_startup(.001)["available"]
    assert calls == [["taskkill.exe", "/PID", "123456", "/T", "/F"]]


def test_resolve_launch_keeps_normal_editor_available_and_minimized(tmp_path, monkeypatch):
    (tmp_path / "Resolve.exe").write_bytes(b"not executed")
    monkeypatch.setattr(bridge, "RESOLVE_DIR", tmp_path)
    calls = []
    class Result:
        returncode = 0
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()
    monkeypatch.setattr(bridge.subprocess, "run", run)
    bridge._launch_resolve()
    command, options = calls[0]
    assert command[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    assert "Start-Process" in command[4] and "-WindowStyle Minimized" in command[4]
    assert "-nogui" not in command[4]
    assert "-WindowStyle Hidden" not in command[4]
    assert options["env"] == dict(bridge.os.environ)
