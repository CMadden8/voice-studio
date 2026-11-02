"""Local character voice effects with optional Resolve AI noise isolation.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QToolButton,
    QVBoxLayout, QWidget)

from audio_ui import AudioRecorder, Waveform
from engine import DEFAULT_SETTINGS, MAX_SECONDS, SAMPLE_RATE, read_audio
from library import VoiceLibrary, safe_name, unique_copy
from persistence import replace_with_retry
from preset_catalog import PRESETS
from consistency import CharacterProfiles, verify_reference
from character_ui import CharacterSetup

APP = Path(__file__).resolve().parent
DATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LocalAudioTools" / "Voice Studio"
ROOT = DATA
(ROOT / "Logs").mkdir(parents=True, exist_ok=True)
LOG = logging.getLogger("voice-studio")
LOG.setLevel(logging.INFO)
handler = RotatingFileHandler(ROOT / "Logs" / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
LOG.addHandler(handler)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as data:
        for chunk in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RenderJob(QThread):
    completed = Signal(dict)

    def __init__(self, job, context, cache, parent=None):
        super().__init__(parent)
        self.specification, self.context, self.cache = job, context, cache
        self.process = None
        self.cancelled = False

    @staticmethod
    def _stop_process(process):
        # Windows venv launchers spawn a second Python process. Stop this
        # worker's tree so closing a preview cannot leave an SDK call orphaned.
        poll = getattr(process, "poll", None)
        if callable(poll) and poll() is not None:
            return
        pid = getattr(process, "pid", None)
        if isinstance(pid, int):
            try:
                stopped = subprocess.run(["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, timeout=4, creationflags=subprocess.CREATE_NO_WINDOW)
                if stopped.returncode == 0:
                    return
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            process.kill()
        except OSError:
            pass

    def run(self):
        job_file = self.cache / ("job-" + uuid.uuid4().hex + ".json")
        try:
            if self.cancelled:
                return
            write_json(job_file, self.specification)
            if self.cancelled:
                return
            interpreter = Path(sys.executable).with_name("python.exe")
            self.process = subprocess.Popen([str(interpreter), str(APP / "engine.py"), "--job", str(job_file)],
                cwd=str(APP), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
            if self.cancelled:
                self._stop_process(self.process)
                self.process.communicate()
                return
            stdout, stderr = self.process.communicate(timeout=300)
            if self.cancelled:
                return
            lines = [line[10:] for line in stdout.splitlines() if line.startswith("VS_RESULT:")]
            if not lines:
                LOG.error("Audio worker failed: %s %s", stderr[-5000:], stdout[-1500:])
                result = {"ok": False, "error": "The voice engine stopped. Please try again."}
            else:
                result = json.loads(lines[-1])
                if not result.get("ok"):
                    LOG.error("Voice conversion failed: %s", stderr[-5000:])
            result["context"] = self.context
            self.completed.emit(result)
        except Exception:
            LOG.exception("Voice worker failed")
            if self.process and self.process.poll() is None:
                self._stop_process(self.process)
                self.process.communicate()
            if not self.cancelled:
                self.completed.emit({"ok": False, "error": "I couldn't finish that voice. Please try again.", "context": self.context})
        finally:
            try:
                job_file.unlink(missing_ok=True)
            except OSError:
                LOG.exception("Could not remove private audio job file")

    def cancel(self):
        self.cancelled = True
        if self.process and self.process.poll() is None:
            self._stop_process(self.process)


class ExportDialog(QDialog):
    def __init__(self, folder, voice, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save your voice recording")
        self.setMinimumWidth(570)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{voice['name']} · version {voice['version']}"))
        form = QFormLayout()
        self.line_name = QLineEdit("Voice line")
        self.line_name.setAccessibleName("Recording name")
        self.line_name.selectAll()
        form.addRow("Name", self.line_name)
        row = QHBoxLayout()
        self.folder = QLineEdit(str(folder))
        self.folder.setAccessibleName("Save folder")
        row.addWidget(self.folder, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        form.addRow("Folder", row)
        layout.addLayout(form)
        self.original = QCheckBox("Include the original recording")
        self.original.setChecked(True)
        layout.addWidget(self.original)
        note = QLabel("Earlier saved files are kept. Reusing a name adds a numbered version.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a save folder", self.folder.text())
        if path:
            self.folder.setText(path)


class VoiceManager(QDialog):
    def __init__(self, studio):
        super().__init__(studio)
        self.studio = studio
        self.setWindowTitle("Manage voices")
        self.resize(540, 460)
        layout = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setAccessibleName("Saved voices")
        layout.addWidget(self.list, 1)
        self.history = QComboBox()
        self.history.setAccessibleName("Saved voice versions")
        layout.addWidget(self.history)
        self.list.currentItemChanged.connect(self.selection)
        row = QHBoxLayout()
        for text, callback in [("Duplicate…", self.duplicate), ("Rename…", self.rename), ("Archive…", self.archive)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        restore = QPushButton("Restore selected version as a new version")
        restore.clicked.connect(self.restore_version)
        layout.addWidget(restore)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self, chosen=None):
        self.list.clear()
        for voice in self.studio.library.list_voices():
            item = QListWidgetItem(f"{voice['name']}   ·   v{voice['version']}")
            item.setData(Qt.ItemDataRole.UserRole, voice["id"])
            self.list.addItem(item)
            if voice["id"] == chosen:
                self.list.setCurrentItem(item)
        if self.list.currentRow() < 0 and self.list.count():
            self.list.setCurrentRow(0)
        self.studio._refresh_voice_list()
        if chosen:
            self.studio._select_voice(chosen)

    def voice_id(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def selection(self, *_):
        self.history.clear()
        if self.voice_id():
            for voice in self.studio.library.history(self.voice_id()):
                self.history.addItem(f"Version {voice['version']} — {voice['name']}", voice["version"])

    def duplicate(self):
        if not self.voice_id():
            return
        name, ok = QInputDialog.getText(self, "Duplicate voice", "New voice name")
        if ok:
            try:
                voice = self.studio.library.duplicate(self.voice_id(), name)
                self.refresh(voice["id"])
            except (ValueError, OSError) as error:
                QMessageBox.information(self, "Couldn't save voice", str(error))

    def rename(self):
        if not self.voice_id():
            return
        current = self.studio.library.get(self.voice_id())
        name, ok = QInputDialog.getText(self, "Rename voice", "Voice name", text=current["name"])
        if ok:
            try:
                self.studio.library.update(current["id"], current["settings"], name=name)
                self.refresh(current["id"])
            except (ValueError, OSError) as error:
                QMessageBox.information(self, "Couldn't rename voice", str(error))

    def archive(self):
        if not self.voice_id():
            return
        if self.list.count() <= 1:
            QMessageBox.information(self, "Keep a voice", "Create another voice before archiving the last one.")
            return
        answer = QMessageBox.question(self, "Archive voice?", "Hide this voice from the library? Its saved versions and exported recordings are kept.",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.studio.library.archive(self.voice_id())
                self.refresh()
            except (ValueError, OSError) as error:
                QMessageBox.information(self, "Couldn't archive voice", str(error))

    def restore_version(self):
        if not self.voice_id() or self.history.currentData() is None:
            return
        old = self.studio.library.get(self.voice_id(), self.history.currentData())
        try:
            voice = self.studio.library.update(old["id"], old["settings"])
            self.refresh(voice["id"])
        except (ValueError, OSError) as error:
            QMessageBox.information(self, "Couldn't restore voice", str(error))


class VoiceStudio(QMainWindow):
    def __init__(self, root=None, restore=True):
        super().__init__()
        self.root = Path(root) if root is not None else ROOT
        self.recordings = self.root / "Recordings"
        self.cache = self.root / "Cache"
        self.state_path = self.root / "State" / "session.json"
        self.preferences_path = self.root / "State" / "preferences.json"
        self.export_history_path = self.root / "Export history"
        for folder in (self.recordings, self.cache, self.export_history_path):
            folder.mkdir(parents=True, exist_ok=True)
        self.export_root = self.root / "Exports"
        if root is None:
            self.export_root = Path.home() / "Music" / "Voice Studio"
            try:
                self.export_root = Path(json.loads((self.root / "settings.json").read_text(encoding="utf-8"))["export_root"])
            except (OSError, KeyError, ValueError):
                pass
        self.last_output_folder = None
        self.library = VoiceLibrary(self.root / "Voices")
        self.profiles = CharacterProfiles(self.root)
        self.calibrations = {}  # Deliberately expires when the app closes.
        self.recorder = AudioRecorder()
        self.source = None
        self.original_audio = None
        self.original_rate = SAMPLE_RATE
        self.rendered = None
        self.rendered_path = None
        self.rendered_snapshot = None
        self.cleaned_path = None
        self.selected_voice = None
        self.job = None
        self.jobs = []
        self.revision = 0
        self.playing = None
        self.play_source = None
        self.play_buffer = None
        self.closing = False
        self.loading = True
        self.setWindowTitle("Voice Studio")
        self.setWindowIcon(QIcon(str(APP / "icon.svg")))
        self.resize(810, 865)
        self.setMinimumSize(690, 650)
        self._build_ui()
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(350)
        self.debounce.timeout.connect(lambda: self._start_render(True))
        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self._tick)
        self._refresh_voice_list()
        preferred = next((v for v in self.library.list_voices() if v["name"] == "Clear narrator"), self.library.list_voices()[0])
        self._select_voice(preferred["id"], render=False)
        self._load_preferences()
        self._refresh_devices()
        self.loading = False
        self._update_buttons()
        self.timer.start()
        if restore:
            QTimer.singleShot(120, self._restore_session)
        # Keep only private preview renders bounded; recordings and exports persist.
        for pattern in ("voice-*.wav", "cleaned-*.wav", "calibration-*.wav"):
            old = sorted(self.cache.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in old[10:]:
                try:
                    path.unlink()
                except OSError:
                    pass

    @staticmethod
    def card():
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(9)
        return card, layout

    def _build_ui(self):
        shell = QWidget()
        shell.setObjectName("shell")
        self.setCentralWidget(shell)
        outer = QVBoxLayout(shell)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)
        page = QWidget()
        scroll.setWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 18, 28, 16)
        layout.setSpacing(14)
        heading = QHBoxLayout()
        title = QLabel("Voice Studio")
        title.setObjectName("title")
        heading.addWidget(title, 1)
        self.ai_badge = QLabel("LOCAL · NO AI")
        self.ai_badge.setObjectName("badge")
        heading.addWidget(self.ai_badge)
        layout.addLayout(heading)
        subtitle = QLabel("Your performance. A different character. Saved voices you can reuse.")
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        record_card, record_layout = self.card()
        layout.addWidget(record_card)
        label = QLabel("1   Record your voice")
        label.setObjectName("heading")
        record_layout.addWidget(label)
        row = QHBoxLayout()
        self.record_button = QPushButton("●   Record voice")
        self.record_button.setObjectName("record")
        self.record_button.setMinimumSize(220, 50)
        self.record_button.clicked.connect(self.toggle_recording)
        row.addWidget(self.record_button)
        self.duration_label = QLabel("Ready when you are")
        self.duration_label.setObjectName("muted")
        row.addWidget(self.duration_label, 1, Qt.AlignmentFlag.AlignRight)
        record_layout.addLayout(row)
        self.waveform = Waveform()
        self.waveform.setMinimumHeight(68)
        self.waveform.setMaximumHeight(76)
        record_layout.addWidget(self.waveform)
        row = QHBoxLayout()
        self.original_button = QPushButton("Hear original")
        self.original_button.setObjectName("secondary")
        self.original_button.clicked.connect(self.play_original)
        row.addWidget(self.original_button)
        self.cleaned_button = QPushButton("Hear cleaned")
        self.cleaned_button.setObjectName("secondary")
        self.cleaned_button.setAccessibleName("Hear cleaned original")
        self.cleaned_button.clicked.connect(self.play_cleaned)
        row.addWidget(self.cleaned_button)
        row.addStretch()
        self.import_button = QPushButton("Open a recording…")
        self.import_button.setObjectName("link")
        self.import_button.clicked.connect(self.open_recording)
        row.addWidget(self.import_button)
        record_layout.addLayout(row)
        voice_card, voice_layout = self.card()
        layout.addWidget(voice_card)
        row = QHBoxLayout()
        label = QLabel("2   Choose a voice")
        label.setObjectName("heading")
        row.addWidget(label, 1)
        self.manage_button = QPushButton("Manage voices…")
        self.manage_button.setObjectName("link")
        self.manage_button.clicked.connect(lambda: VoiceManager(self).exec())
        row.addWidget(self.manage_button)
        voice_layout.addLayout(row)
        row = QHBoxLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.setAccessibleName("Character voice")
        self.voice_combo.setMinimumHeight(45)
        self.voice_combo.currentIndexChanged.connect(self._voice_changed)
        row.addWidget(self.voice_combo, 1)
        self.listen_button = QPushButton("▶   Listen")
        self.listen_button.setObjectName("listen")
        self.listen_button.setAccessibleName("Listen to changed voice")
        self.listen_button.setMinimumSize(140, 45)
        self.listen_button.clicked.connect(self.toggle_listen)
        row.addWidget(self.listen_button)
        voice_layout.addLayout(row)
        self.voice_caption = QLabel()
        self.voice_caption.setObjectName("muted")
        self.voice_caption.setWordWrap(True)
        voice_layout.addWidget(self.voice_caption)
        self.voice_description = QLabel()
        self.voice_description.setObjectName("muted")
        self.voice_description.setWordWrap(True)
        voice_layout.addWidget(self.voice_description)
        row = QHBoxLayout()
        self.reference_button = QPushButton("Hear reference")
        self.reference_button.setAccessibleName("Hear character reference")
        self.reference_button.clicked.connect(self._hear_reference)
        row.addWidget(self.reference_button)
        self.character_button = QPushButton("Character setup…")
        self.character_button.clicked.connect(self.character_setup)
        row.addWidget(self.character_button)
        row.addStretch()
        self.lock_voice = QCheckBox("Lock voice settings")
        self.lock_voice.setChecked(True)
        self.lock_voice.toggled.connect(self._lock_changed)
        row.addWidget(self.lock_voice)
        voice_layout.addLayout(row)
        self.character_status = QLabel()
        self.character_status.setWordWrap(True)
        self.character_status.setObjectName("muted")
        voice_layout.addWidget(self.character_status)
        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("Noise cleanup"))
        self.cleanup_provider = QComboBox()
        self.cleanup_provider.setAccessibleName("Noise cleanup method")
        self.cleanup_provider.addItem("Standard · no AI", "conventional")
        self.cleanup_provider.addItem("Resolve Voice Isolation · AI", "resolve")
        self.cleanup_provider.currentIndexChanged.connect(self._sound_changed)
        provider_row.addWidget(self.cleanup_provider, 1)
        voice_layout.addLayout(provider_row)
        cleanup_row = QHBoxLayout()
        cleanup_row.addWidget(QLabel("Clean up"))
        self.cleanup = QSlider(Qt.Orientation.Horizontal)
        self.cleanup.setRange(0, 100)
        self.cleanup.setValue(0)
        self.cleanup.setAccessibleName("Noise cleanup strength")
        self.cleanup.valueChanged.connect(self._sound_changed)
        cleanup_row.addWidget(self.cleanup, 1)
        self.cleanup_value = QLabel("Off")
        self.cleanup_value.setMinimumWidth(40)
        cleanup_row.addWidget(self.cleanup_value)
        voice_layout.addLayout(cleanup_row)
        self.cleanup_hint = QLabel("Reduce steady hiss and rumble before changing the voice. Compare using Hear cleaned.")
        self.cleanup_hint.setObjectName("muted")
        self.cleanup_hint.setWordWrap(True)
        voice_layout.addWidget(self.cleanup_hint)
        row = QHBoxLayout()
        row.addWidget(QLabel("Listen volume"))
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(75)
        self.volume.setMaximumWidth(175)
        self.volume.setAccessibleName("Listening volume")
        self.volume.valueChanged.connect(self._volume_changed)
        row.addWidget(self.volume)
        row.addStretch()
        self.adjust_button = QToolButton()
        self.adjust_button.setText("Adjust voice")
        self.adjust_button.setCheckable(True)
        self.adjust_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adjust_button.setArrowType(Qt.ArrowType.RightArrow)
        self.adjust_button.toggled.connect(self._toggle_adjustments)
        row.addWidget(self.adjust_button)
        voice_layout.addLayout(row)
        self.adjustments = QWidget()
        controls = QVBoxLayout(self.adjustments)
        controls.setContentsMargins(0, 5, 0, 0)
        self.slider_labels = {}
        for key, label, low, high, default in [("pitch", "Depth / pitch", -80, 80, 0),
            ("resonance", "Resonance", 75, 135, 100), ("tone", "Warm / bright", -100, 100, 0),
            ("texture", "Texture", 0, 100, 0), ("room", "Room", 0, 50, 0), ("speed", "Speed", 75, 125, 100)]:
            line = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setMinimumWidth(165)
            self.slider_labels[key] = label_widget
            line.addWidget(label_widget)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setAccessibleName(label)
            slider.setRange(low, high)
            slider.setValue(default)
            slider.valueChanged.connect(self._sound_changed)
            setattr(self, key, slider)
            line.addWidget(slider, 1)
            controls.addLayout(line)
        row = QHBoxLayout()
        self.reset_button = QPushButton("Reset to saved voice")
        self.reset_button.setObjectName("link")
        self.reset_button.clicked.connect(self.reset_voice)
        row.addWidget(self.reset_button)
        row.addStretch()
        self.voice_save_button = QToolButton()
        self.voice_save_button.setText("Save voice…")
        self.voice_save_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        voice_menu = QMenu(self.voice_save_button)
        voice_menu.addAction("Save as a new voice…", lambda: self.save_new_voice())
        self.update_voice_action = voice_menu.addAction("Update this voice (new version)", self.update_voice)
        self.voice_save_button.setMenu(voice_menu)
        row.addWidget(self.voice_save_button)
        controls.addLayout(row)
        self.adjustments.hide()
        voice_layout.addWidget(self.adjustments)
        mic = QHBoxLayout()
        mic.addWidget(QLabel("Microphone"))
        self.microphone = QComboBox()
        self.microphone.setAccessibleName("Microphone")
        self.microphone.currentIndexChanged.connect(self._microphone_changed)
        mic.addWidget(self.microphone, 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_devices)
        mic.addWidget(refresh)
        controls.addLayout(mic)
        self.message = QLabel()
        self.message.setObjectName("message")
        self.message.setWordWrap(True)
        self.message.hide()
        layout.addWidget(self.message)
        layout.addStretch()
        footer = QWidget()
        footer.setObjectName("footer")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(34, 6, 34, 18)
        outer.addWidget(footer)
        row = QHBoxLayout()
        self.status = QLabel("Ready to record")
        self.status.setWordWrap(True)
        self.status.setObjectName("status")
        row.addWidget(self.status, 1)
        self.save_button = QPushButton("Save audio…")
        self.save_button.setObjectName("listen")
        self.save_button.clicked.connect(self.save_audio)
        row.addWidget(self.save_button)
        footer_layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.hide()
        footer_layout.addWidget(self.progress)
        row = QHBoxLayout()
        note = QLabel("Only audio you save is copied to your output folder.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        row.addWidget(note, 1)
        folder_button = QPushButton("Open output folder")
        folder_button.setObjectName("link")
        folder_button.clicked.connect(self.open_output_folder)
        row.addWidget(folder_button)
        footer_layout.addLayout(row)

    def settings(self):
        return {"pitch": self.pitch.value() / 10, "resonance": self.resonance.value() / 100,
                "tone": self.tone.value() / 100, "texture": self.texture.value() / 100,
                "room": self.room.value() / 100, "speed": self.speed.value() / 100,
                "cleanup": self.cleanup.value() / 100}

    def _apply_settings(self, settings):
        settings = {**DEFAULT_SETTINGS, "cleanup": 0., **settings}
        values = {"pitch": settings["pitch"] * 10, "resonance": settings["resonance"] * 100,
                  **{key: settings[key] * 100 for key in ("tone", "texture", "room", "speed", "cleanup")}}
        for key, value in values.items():
            slider = getattr(self, key)
            slider.blockSignals(True)
            slider.setValue(round(value))
            slider.blockSignals(False)
        self._update_caption()

    def _refresh_voice_list(self):
        current = self.selected_voice["id"] if self.selected_voice else None
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for voice in self.library.list_voices():
            self.voice_combo.addItem(voice["name"], voice["id"])
        index = self.voice_combo.findData(current)
        self.voice_combo.setCurrentIndex(max(0, index))
        self.voice_combo.blockSignals(False)
        if current and index < 0 and self.voice_combo.count():
            self._select_voice(self.voice_combo.currentData())

    def _select_voice(self, voice_id, version=None, render=True):
        self.selected_voice = self.library.get(voice_id, version)
        self.voice_combo.blockSignals(True)
        self.voice_combo.setCurrentIndex(self.voice_combo.findData(voice_id))
        self.voice_combo.blockSignals(False)
        self._apply_settings(self.selected_voice["settings"])
        if render and not self.loading:
            self._sound_changed()

    def _voice_changed(self, *_):
        if not self.loading and self.voice_combo.currentData():
            self._select_voice(self.voice_combo.currentData())

    def _update_caption(self):
        if self.selected_voice:
            edited = self.settings() != self.selected_voice["settings"]
            self.voice_caption.setText(f"{self.selected_voice['name']} · version {self.selected_voice['version']}" + (" · edited settings" if edited else " · saved voice"))
            description = self.selected_voice.get("description") or next(
                (p["description"] for p in PRESETS if p["name"] == self.selected_voice["name"]), "Your saved character voice.")
            self.voice_description.setText(description)
            self.update_voice_action.setEnabled(edited)
            self._refresh_character_ui()
        amount = self.cleanup.value()
        self.cleanup_value.setText(f"{amount}%" if amount else "Off")
        use_resolve = self.cleanup_provider.currentData() == "resolve"
        self.ai_badge.setText("LOCAL · AI CLEANUP" if use_resolve and amount else "LOCAL · NO AI")
        self.cleanup_hint.setText(
            "Resolve removes background sound locally using AI. Starts in the background when needed. Compare using Hear cleaned."
            if use_resolve else "Strong cleanup can soften quiet speech. Compare with Hear original." if amount > 75
            else "Reduce steady hiss and rumble before changing the voice. Compare using Hear cleaned.")
        values = self.settings()
        for key, label in [("pitch", "Depth / pitch"), ("resonance", "Resonance"), ("tone", "Warm / bright"),
                           ("texture", "Texture"), ("room", "Room"), ("speed", "Speed")]:
            value = f"{values[key]:+.1f} st" if key == "pitch" else f"{round(values[key] * 100)}%" if key in ("resonance", "texture", "room", "speed") else f"{round(values[key] * 100):+}%"
            self.slider_labels[key].setText(label + "  " + value)

    def reference_matches(self):
        reference = self.profiles.reference(self.selected_voice)
        return bool(reference and self.settings() == reference["snapshot"]["settings"]
                    and self.cleanup_provider.currentData() == reference["snapshot"].get("requested_cleanup_provider", "conventional"))

    def calibration_key(self):
        reference = self.profiles.reference(self.selected_voice)
        return json.dumps([self.selected_voice["id"], self.selected_voice["version"], self.settings(),
            self.cleanup_provider.currentData(), self.microphone.currentText(), self.microphone.currentData(),
            reference["id"] if reference else None], sort_keys=True)

    def current_calibration(self):
        return self.calibrations.get(self.calibration_key()) if self.selected_voice else None

    def calibration_text(self):
        value = self.current_calibration()
        if value:
            return f"Session calibrated · pitch {value['pitch']:+.1f} st · level {value['gain_db']:+.1f} dB. Expression is preserved."
        return "Session calibration off. Calibrate once per session using the neutral reference phrase."

    def _refresh_character_ui(self):
        profile = self.profiles.get(self.selected_voice["id"])
        reference = self.profiles.reference(self.selected_voice)
        self.lock_voice.blockSignals(True)
        self.lock_voice.setChecked(profile["locked"])
        self.lock_voice.blockSignals(False)
        if self.current_calibration():
            text = self.calibration_text()
        elif reference:
            text = "Reference saved · session calibration off. Open Character setup to match this session."
        else:
            text = "No reference yet. Character setup saves a neutral reference and acting notes."
        if profile.get("notes"):
            text += "  " + profile["notes"]
        self.character_status.setText(text)
        self.reference_button.setEnabled(bool(reference) and not self.recorder.stream and self.job is None)
        for key in DEFAULT_SETTINGS:
            getattr(self, key).setEnabled(not profile["locked"] and not self.recorder.stream)
        self.cleanup_provider.setEnabled(not profile["locked"] and not self.recorder.stream)

    def _lock_changed(self, locked):
        if not self.selected_voice:
            return
        try:
            self.profiles.update(self.selected_voice["id"], locked=locked)
            self._refresh_character_ui()
        except (ValueError, OSError) as error:
            self._error(str(error))

    def character_setup(self):
        if self.job or self.recorder.stream:
            return
        self.stop_playback()
        CharacterSetup(self, RenderJob).exec()
        self._refresh_character_ui()
        self._update_buttons()

    def play_reference(self, original=False):
        reference = self.profiles.reference(self.selected_voice)
        if not reference:
            raise ValueError("Set a neutral reference in Character setup first.")
        verify_reference(reference)
        key = "source" if original else "output"
        audio, rate = read_audio(reference[key])
        self._play(audio, rate, "reference-original" if original else "reference")

    def _hear_reference(self):
        try:
            self.play_reference()
        except (ValueError, OSError) as error:
            self._error(str(error))

    def _microphone_changed(self, *_):
        if not self.loading:
            self.calibrations.clear()
            self._sound_changed()

    def _sound_changed(self, *_):
        if self.loading:
            return
        self.revision += 1
        self.rendered = self.rendered_path = self.rendered_snapshot = None
        self.cleaned_path = None
        self.stop_playback()
        self._update_caption()
        self._save_preferences()
        self._save_session()
        if self.source and not self.recorder.stream:
            self.debounce.start()
        self._update_buttons()

    def reset_voice(self):
        self._apply_settings(self.selected_voice["settings"])
        self._sound_changed()

    def save_new_voice(self, name=None):
        if name is None:
            name, ok = QInputDialog.getText(self, "Save a new voice", "Character or voice name")
            if not ok:
                return None
        try:
            voice = self.library.create(name, self.settings(), description=self.selected_voice.get("description", ""))
            self._refresh_voice_list()
            self._select_voice(voice["id"])
            self.status.setText(f"Saved voice {voice['name']}")
            return voice
        except (OSError, ValueError) as error:
            self._error(str(error))
            return None

    def update_voice(self):
        try:
            voice = self.library.update(self.selected_voice["id"], self.settings())
            self._refresh_voice_list()
            self._select_voice(voice["id"])
            self.status.setText(f"Saved {voice['name']} version {voice['version']}")
            return voice
        except (OSError, ValueError) as error:
            self._error(str(error))
            return None

    def _toggle_adjustments(self, shown):
        self.adjustments.setVisible(shown)
        self.adjust_button.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def _refresh_devices(self):
        old = self.microphone.currentText() or getattr(self, "preferred_microphone", "System default")
        self.microphone.blockSignals(True)
        self.microphone.clear()
        self.microphone.addItem("System default", None)
        try:
            devices, hosts = sd.query_devices(), sd.query_hostapis()
            preferred = [i for i, d in enumerate(devices) if d["max_input_channels"] and "WASAPI" in hosts[d["hostapi"]]["name"]]
            seen = set()
            for index in preferred or [i for i, d in enumerate(devices) if d["max_input_channels"]]:
                name = devices[index]["name"]
                if name not in seen:
                    self.microphone.addItem(name, index)
                    seen.add(name)
            self.microphone.setCurrentIndex(max(0, self.microphone.findText(old)))
        except Exception:
            LOG.exception("Microphone enumeration failed")
        self.microphone.blockSignals(False)
        if not self.loading and self.calibrations:
            self.calibrations.clear()
            self._sound_changed()

    def toggle_recording(self):
        if self.recorder.stream:
            audio, rate = self.recorder.stop()
            self.waveform.live = False
            if len(audio) < rate * .15:
                self._error("Record a short line before stopping.")
                self._update_buttons()
                return
            path = self.recordings / (datetime.now().strftime("Take-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6] + ".wav")
            try:
                sf.write(path, audio, rate, subtype="PCM_16")
                self._open_source(path)
            except Exception as error:
                LOG.exception("Could not save microphone take")
                self._error(str(error))
                self._update_buttons()
            return
        if self.job:
            return
        self.debounce.stop()
        self.stop_playback()
        self.message.hide()
        try:
            self.recorder.start(self.microphone.currentData())
            self.waveform.has_audio = False
            self.waveform.peaks[:] = 0
            self.waveform.live = True
            self.status.setText("Recording — speak your line, then press Stop recording")
        except Exception:
            LOG.exception("Microphone could not start")
            self._error("I couldn't open the microphone. Under Adjust voice, choose a microphone and try again.")
        self._update_buttons()

    def open_recording(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open a voice recording", str(Path.home() / "Downloads"),
                    "Audio recordings (*.wav *.mp3 *.flac *.ogg);;All files (*)")
        if path:
            self.load_recording(path)

    def load_recording(self, path):
        if self.job or self.recorder.stream:
            return
        try:
            path = Path(path)
            # Read first so invalid/unsupported files do not become stored takes.
            audio, rate = read_audio(path)
            imported = self.recordings / (datetime.now().strftime("Imported-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6] + path.suffix.lower())
            local = unique_copy(path, self.recordings, name=imported.name)
            self._open_source(local, audio, rate)
        except Exception as error:
            LOG.exception("Could not open voice recording")
            self._error(str(error))

    def _open_source(self, path, audio=None, rate=None, autoplay=True):
        self.stop_playback()
        self.debounce.stop()
        if audio is None:
            audio, rate = read_audio(path)
        self.source = Path(path)
        self.original_audio, self.original_rate = audio, rate
        self.rendered = self.rendered_path = self.rendered_snapshot = None
        self.cleaned_path = None
        self.revision += 1
        self.waveform.set_audio(audio)
        self.duration_label.setText(f"{len(audio) / rate:.1f} seconds recorded")
        self.message.hide()
        self._save_session()
        self._start_render(autoplay)

    def _start_render(self, autoplay=False):
        if self.job or self.recorder.stream or not self.source:
            return
        self.debounce.stop()
        try:
            source_hash = sha256(self.source)
            if self.current_calibration():
                verify_reference(self.profiles.reference(self.selected_voice))
        except (OSError, ValueError) as error:
            self._error(str(error) if isinstance(error, ValueError) else
                        "I couldn't read the original recording. Open it again before changing the voice.")
            self._update_buttons()
            return
        output = self.cache / ("voice-" + uuid.uuid4().hex + ".wav")
        cleanup_output = self.cache / ("cleaned-" + uuid.uuid4().hex + ".wav")
        context = {"revision": self.revision, "autoplay": autoplay,
                   "voice": copy.deepcopy(self.selected_voice), "settings": self.settings(),
                   "cleanup_provider": self.cleanup_provider.currentData(), "source_sha256": source_hash,
                   "session_calibration": copy.deepcopy(self.current_calibration())}
        worker = RenderJob({"kind": "render", "input": str(self.source), "output": str(output),
                            "cleanup_output": str(cleanup_output), "settings": self.settings(),
                            "calibration": context["session_calibration"],
                            "cleanup_provider": context["cleanup_provider"]}, context, self.cache, self)
        self.job = worker
        self.jobs.append(worker)
        worker.completed.connect(self._render_completed)
        worker.finished.connect(lambda: self.jobs.remove(worker) if worker in self.jobs else None)
        worker.finished.connect(worker.deleteLater)
        self.status.setText("Cleaning in Resolve, then preparing your voice…" if
            context["cleanup_provider"] == "resolve" and self.cleanup.value() else f"Preparing {self.selected_voice['name']}…")
        self.progress.show()
        self._update_buttons()
        worker.start()

    def _render_completed(self, response):
        if self.closing:
            return
        self.job = None
        self.progress.hide()
        context = response["context"]
        if context["revision"] != self.revision:
            self._start_render(True)
            return
        if not response.get("ok"):
            self._error(response.get("error", "I couldn't prepare that voice."))
            self._update_buttons()
            return
        try:
            result = response["result"]
            source_hash = result.get("source_sha256", context.get("source_sha256"))
            if source_hash != context.get("source_sha256"):
                self._error("The original recording changed. Open it again before changing the voice.")
                self._update_buttons()
                return
            self.rendered_path = Path(result["audio"])
            self.rendered, self.rendered_rate = sf.read(self.rendered_path, dtype="float32")
            self.rendered_snapshot = {"voice": context["voice"], "settings": context["settings"],
                                      "backend": result.get("backend", "Conventional DSP"),
                                      "warning": result.get("warning", ""), "ai": bool(result.get("ai", False)),
                                      "cleanup_provider": result.get("cleanup_provider", context.get("cleanup_provider", "conventional")),
                                      "requested_cleanup_provider": context.get("cleanup_provider", "conventional"),
                                      "source_sha256": source_hash,
                                      "cleanup": result.get("cleanup", {})}
            if result.get("resolve"):
                self.rendered_snapshot["resolve"] = copy.deepcopy(result["resolve"])
            if context.get("session_calibration"):
                if result.get("session_calibration") != context["session_calibration"]:
                    raise ValueError("The worker did not confirm the session calibration.")
                self.rendered_snapshot["session_calibration"] = copy.deepcopy(result["session_calibration"])
                self.rendered_snapshot["effective_pitch"] = result["effective_pitch"]
            cleaned = result.get("cleaned_audio")
            self.cleaned_path = Path(cleaned) if cleaned and Path(cleaned).is_file() else None
            self.rendered_snapshot["voice"]["settings"] = context["settings"]
            self.rendered_snapshot["modified_preset"] = context["settings"] != self.selected_voice["settings"]
            warning = result.get("warning")
            if warning:
                self.message.setText(warning)
                self.message.show()
            else:
                self.message.hide()
            self.status.setText("Ready to listen")
            self._update_buttons()
            if context["autoplay"]:
                self._play(self.rendered, self.rendered_rate, "changed")
        except Exception:
            LOG.exception("Could not read changed voice")
            self.rendered = self.rendered_path = self.rendered_snapshot = None
            self.cleaned_path = None
            self._error("I couldn't prepare playback. Try choosing the voice again.")
            self._update_buttons()

    def _play(self, audio, rate, kind):
        self.stop_playback()
        try:
            self.play_source = np.ascontiguousarray(audio, dtype=np.float32)
            self.play_buffer = np.ascontiguousarray(audio * (self.volume.value() / 100), dtype=np.float32)
            sd.play(self.play_buffer, rate, blocking=False)
            self.playing = kind
            self.status.setText("Playing original recording" if kind == "original" else
                                "Playing cleaned original" if kind == "cleaned" else
                                "Playing original reference" if kind == "reference-original" else
                                "Playing character reference" if kind == "reference" else f"Playing {self.selected_voice['name']}")
            self._update_buttons()
        except Exception:
            LOG.exception("Playback failed")
            self._error("I couldn't play sound. Check your speakers or headphones, then press Listen.")

    def play_original(self):
        if self.playing == "original":
            self.stop_playback()
        elif self.original_audio is not None:
            self._play(self.original_audio, self.original_rate, "original")

    def play_cleaned(self):
        if self.playing == "cleaned":
            self.stop_playback()
        elif self.cleaned_path and not self.job:
            try:
                audio, rate = sf.read(self.cleaned_path, dtype="float32")
                self._play(audio, rate, "cleaned")
            except Exception:
                LOG.exception("Could not play cleaned original")
                self._error("I couldn't play that comparison. Try changing the cleanup amount again.")

    def toggle_listen(self):
        if self.playing == "changed":
            self.stop_playback()
        elif self.rendered is not None:
            self._play(self.rendered, self.rendered_rate, "changed")

    def stop_playback(self):
        sd.stop()
        self.playing = self.play_source = self.play_buffer = None
        if self.rendered is not None and not self.job:
            self.status.setText("Ready to listen")
        self._update_buttons()

    def _volume_changed(self, *_):
        if self.loading:
            return
        self._save_preferences()
        if self.playing and self.play_source is not None and self.play_buffer is not None:
            np.multiply(self.play_source, self.volume.value() / 100, out=self.play_buffer)

    def _tick(self):
        if self.recorder.stream:
            elapsed = time.monotonic() - self.recorder.started
            self.duration_label.setText(f"Recording   {int(elapsed) // 60:02}:{int(elapsed) % 60:02}")
            self.waveform.live_level(self.recorder.peak)
            self.status.setText("The sound is loud — move a little farther from the microphone" if self.recorder.peak >= .98 else "Recording — speak your line, then press Stop recording")
            if elapsed >= MAX_SECONDS:
                self.toggle_recording()
        if self.playing:
            try:
                active = sd.get_stream().active
            except Exception:
                active = False
            if not active:
                self.playing = self.play_source = self.play_buffer = None
                self.status.setText("Ready to listen")
                self._update_buttons()

    def _update_buttons(self):
        recording, busy = bool(self.recorder.stream), self.job is not None
        self.record_button.setEnabled(not busy)
        self.record_button.setText("■   Stop recording" if recording else "●   Record again" if self.source else "●   Record voice")
        self.record_button.setAccessibleName("Stop recording" if recording else "Record voice")
        self.import_button.setEnabled(not busy and not recording)
        self.original_button.setEnabled(self.original_audio is not None and not recording)
        self.original_button.setText("■   Stop original" if self.playing == "original" else "Hear original")
        self.cleaned_button.setEnabled(self.cleaned_path is not None and self.cleanup.value() > 0 and not busy and not recording)
        self.cleaned_button.setText("■   Stop cleaned" if self.playing == "cleaned" else "Hear cleaned")
        self.listen_button.setEnabled(self.rendered is not None and not busy and not recording)
        self.listen_button.setText("■   Stop" if self.playing == "changed" else "▶   Listen")
        self.save_button.setEnabled(self.rendered is not None and not busy and not recording)
        self.microphone.setEnabled(not recording)
        self.voice_combo.setEnabled(not recording)
        self.manage_button.setEnabled(not recording)
        self.adjust_button.setEnabled(not recording)
        self.character_button.setEnabled(not recording and not busy)
        self.lock_voice.setEnabled(not recording and not busy)
        locked = self.lock_voice.isChecked()
        for key in DEFAULT_SETTINGS:
            getattr(self, key).setEnabled(not recording and not locked)
        self.cleanup_provider.setEnabled(not recording and not locked)
        reference = self.profiles.reference(self.selected_voice) if self.selected_voice else None
        self.reference_button.setEnabled(bool(reference) and not recording and not busy)

    def _error(self, text):
        self.message.setText(str(text))
        self.message.show()
        self.status.setText("Ready to try again")

    def save_audio(self):
        if self.rendered_path is None:
            return
        folder = self.export_root / safe_name(self.selected_voice["name"])
        dialog = ExportDialog(folder, self.selected_voice, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.export_audio(dialog.line_name.text(), Path(dialog.folder.text()), dialog.original.isChecked())
            except (OSError, ValueError) as error:
                LOG.exception("Export failed")
                self._error(str(error))

    def export_audio(self, name, folder, include_original=True):
        if not self.rendered_path or not self.rendered_snapshot or self.job or self.recorder.stream:
            raise ValueError("Wait for the changed voice to finish before saving.")
        if not str(name).strip():
            raise ValueError("Give this recording a name.")
        folder = Path(folder).expanduser()
        if not folder.is_absolute():
            raise ValueError("Choose a full folder path for your saved recording.")
        base = safe_name(str(name))
        if base.lower().endswith(".wav"):
            base = base[:-4]
        snapshot = copy.deepcopy(self.rendered_snapshot)
        expected_source = snapshot.get("source_sha256")
        if not expected_source or sha256(self.source) != expected_source:
            raise ValueError("The original recording changed. Open it again and prepare the voice before saving.")
        voice_name = safe_name(snapshot["voice"]["name"])
        files = [unique_copy(self.rendered_path, folder, name=f"{base} - {voice_name}.wav")]
        if include_original:
            files.append(unique_copy(self.source, folder, name=f"{base} - original{self.source.suffix.lower()}"))
        if sha256(self.source) != expected_source or (include_original and sha256(files[-1]) != expected_source):
            for path in files:
                path.unlink(missing_ok=True)
            raise ValueError("The original recording changed while saving. Open it again and prepare the voice.")
        report = {**snapshot, "source": str(self.source), "source_sha256": expected_source,
                  "files": [str(path) for path in files], "sha256": {str(path): sha256(path) for path in files},
                  "created": datetime.now().astimezone().isoformat()}
        write_json(self.export_history_path / (uuid.uuid4().hex + ".json"), report)
        self.last_output_folder = folder
        self._save_preferences()
        self.status.setText("Saved changed voice and original" if include_original else "Saved changed voice")
        return files

    def open_output_folder(self):
        try:
            folder = self.last_output_folder or self.export_root
            folder.mkdir(parents=True, exist_ok=True)
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
                raise OSError("I couldn't open that output folder.")
        except OSError as error:
            self._error(str(error))

    def _save_preferences(self, *_):
        if self.loading:
            return
        try:
            write_json(self.preferences_path, {"volume": self.volume.value(), "microphone": self.microphone.currentText(),
                "cleanup_provider": self.cleanup_provider.currentData(),
                "voice_id": self.selected_voice["id"], "last_output_folder": str(self.last_output_folder) if self.last_output_folder else None})
        except OSError:
            LOG.exception("Could not save preferences")

    def _load_preferences(self):
        try:
            settings = json.loads(self.preferences_path.read_text(encoding="utf-8"))
            self.volume.setValue(settings.get("volume", 75))
            self.preferred_microphone = settings.get("microphone", "System default")
            self.cleanup_provider.blockSignals(True)
            self.cleanup_provider.setCurrentIndex(max(0, self.cleanup_provider.findData(settings.get("cleanup_provider", "conventional"))))
            self.cleanup_provider.blockSignals(False)
            if settings.get("last_output_folder"):
                self.last_output_folder = Path(settings["last_output_folder"])
            if self.voice_combo.findData(settings.get("voice_id")) >= 0:
                self._select_voice(settings["voice_id"], render=False)
            self._update_caption()
        except (OSError, ValueError, KeyError):
            pass

    def _save_session(self):
        if self.source:
            try:
                write_json(self.state_path, {"source": str(self.source), "voice_id": self.selected_voice["id"],
                    "voice_version": self.selected_voice["version"], "settings": self.settings(),
                    "cleanup_provider": self.cleanup_provider.currentData()})
            except OSError:
                LOG.exception("Could not retain current recording session")

    def _restore_session(self):
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._select_voice(saved["voice_id"], saved["voice_version"], render=False)
            self.cleanup_provider.blockSignals(True)
            self.cleanup_provider.setCurrentIndex(max(0, self.cleanup_provider.findData(saved.get("cleanup_provider", "conventional"))))
            self.cleanup_provider.blockSignals(False)
            settings = dict(saved["settings"])
            if "cleanup" not in settings:
                settings["cleanup"] = 0. if self.selected_voice["name"] == "Original" else .45
            self._apply_settings(settings)
            self._open_source(Path(saved["source"]), autoplay=False)
        except Exception:
            LOG.info("No usable prior recording to restore")

    def closeEvent(self, event):
        self.closing = True
        self.timer.stop()
        self.debounce.stop()
        self.stop_playback()
        if self.recorder.stream:
            audio, rate = self.recorder.stop()
            if len(audio):
                path = self.recordings / ("Recovered-" + uuid.uuid4().hex[:10] + ".wav")
                sf.write(path, audio, rate, subtype="PCM_16")
                self.source = path
        self._save_session()
        self._save_preferences()
        for job in list(self.jobs):
            job.cancel()
            job.wait(4000)
        event.accept()


STYLE = """
QMainWindow, QWidget#shell, QWidget#footer, QScrollArea, QScrollArea > QWidget > QWidget { background: #edf2f0; }
QDialog { background: #edf2f0; }
QWidget { color: #243c3a; font-family: 'Segoe UI'; font-size: 14px; }
QLabel#title { font-size: 29px; font-weight: 700; }
QLabel#heading { font-size: 17px; font-weight: 650; }
QLabel#muted { color: #768b84; font-size: 12px; }
QLabel#badge { color: #397866; background: #dcebe3; border-radius: 8px; padding: 7px 10px; font-size: 11px; font-weight: 650; }
QLabel#status { color: #277467; font-size: 13px; }
QLabel#message { color: #9d493e; background: #fff0e9; padding: 12px; border-radius: 8px; }
QFrame#card { background: white; border: 1px solid #dfe7e2; border-radius: 14px; }
QPushButton, QToolButton { background: #f5f8f6; border: 1px solid #d9e3dd; border-radius: 7px; padding: 9px 13px; }
QPushButton:hover, QToolButton:hover { background: #e7f1eb; }
QPushButton:disabled, QToolButton:disabled { color: #a5b0a9; }
QPushButton#record { color: white; background: #bd5148; border: none; font-size: 16px; font-weight: 650; }
QPushButton#record:hover { background: #a84038; }
QPushButton#record:disabled { background: #d8ada7; }
QPushButton#listen { color: white; background: #247f73; border: none; font-weight: 650; }
QPushButton#listen:hover { background: #19685f; }
QPushButton#listen:disabled { background: #9cbcb2; }
QPushButton#link { background: transparent; border: none; color: #638c7e; font-size: 12px; padding: 6px; }
QComboBox, QLineEdit { background: #f8fbf9; border: 1px solid #d8e3dc; border-radius: 7px; padding: 8px 12px; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #dceee5; selection-color: #243c3a; padding: 5px; }
QListWidget { background: white; color: #243c3a; border: 1px solid #d8e3dc; border-radius: 7px; outline: none; }
QListWidget::item { padding: 7px 10px; }
QListWidget::item:selected { background: #dceee5; color: #243c3a; }
QSlider::groove:horizontal { height: 5px; background: #dce7e1; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #5b9b89; }
QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #2e7e6d; border-radius: 7px; }
QProgressBar { background: #e1ece6; border: none; }
QProgressBar::chunk { background: #5eaa92; }
QMenu { background: white; border: 1px solid #d3dfd7; padding: 5px; }
QMenu::item { padding: 8px 16px; }
QMenu::item:selected { background: #e2efe8; }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", type=Path)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Voice Studio")
    app.setOrganizationName("LocalAudioTools")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Segoe UI", 10))
    server = QLocalServer(app)
    server_name = "voice-studio-" + hashlib.sha256(str(DATA).casefold().encode()).hexdigest()[:16]
    if not server.listen(server_name):
        client = QLocalSocket(app)
        client.connectToServer(server_name)
        if client.waitForConnected(500):
            client.write(b"activate")
            client.waitForBytesWritten(200)
            return 0
        QLocalServer.removeServer(server_name)
        if not server.listen(server_name):
            return 1
    window = VoiceStudio(restore=not (args.fresh or args.open))
    def activate():
        connection = server.nextPendingConnection()
        if connection:
            connection.disconnectFromServer()
            connection.deleteLater()
        window.showNormal()
        window.raise_()
        window.activateWindow()
    server.newConnection.connect(activate)
    window.show()
    if args.open:
        QTimer.singleShot(200, lambda: window.load_recording(args.open))
    LOG.info("Voice Studio opened; conventional voices with optional local Resolve AI cleanup")
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        LOG.exception("Application failed")
        raise
