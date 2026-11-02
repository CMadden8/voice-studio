"""Small character setup and neutral-phrase capture dialogs.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import copy
from pathlib import Path
import time
import uuid

import soundfile as sf
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout)

from audio_ui import AudioRecorder, Waveform
from consistency import PHRASE, make_calibration, verify_reference


class CalibrationCapture(QDialog):
    """A separate capture leaves the dialogue take in the main window intact."""
    def __init__(self, studio, reference=False):
        super().__init__(studio)
        self.studio = studio
        self.recorder = AudioRecorder()
        self.path = None
        self.reference = reference
        self.setWindowTitle("Read the neutral phrase")
        self.setMinimumWidth(540)
        layout = QVBoxLayout(self)
        prompt = QLabel(PHRASE)
        prompt.setWordWrap(True)
        layout.addWidget(prompt)
        hint = QLabel("Use a comfortable neutral delivery you can repeat, with the microphone at your usual distance. Keep emotion for your dialogue lines."
                      if reference else "Use the same neutral delivery and microphone distance as your reference. Speak normally; keep emotion for your dialogue lines.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.waveform = Waveform()
        layout.addWidget(self.waveform)
        self.status = QLabel("Your current dialogue take stays in the main window.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.record_label = "Record neutral reference" if reference else "Record calibration phrase"
        self.record = QPushButton(self.record_label)
        self.record.setAccessibleName(self.record_label)
        self.record.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.timer = QTimer(self)
        self.timer.setInterval(80)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def toggle_recording(self):
        if self.recorder.stream:
            audio, rate = self.recorder.stop()
            self.record.setText(self.record_label)
            if len(audio) < 2 * rate:
                self.status.setText("Read the whole phrase before stopping (at least two seconds).")
                return
            self.path = self.studio.cache / ("calibration-" + uuid.uuid4().hex + ".wav")
            try:
                sf.write(self.path, audio, rate, subtype="PCM_16")
            except OSError:
                self.status.setText("Could not save the calibration sample. Check free disk space and try again.")
                self.path = None
                return
            self.accept()
            return
        self.studio.stop_playback()
        try:
            self.recorder.start(self.studio.microphone.currentData())
            self.record.setText("Stop and check reference" if self.reference else "Stop and check calibration")
        except Exception:
            self.status.setText("Could not open the microphone. Choose a microphone under Adjust voice and try again.")

    def tick(self):
        if self.recorder.stream:
            elapsed = time.monotonic() - self.recorder.started
            self.waveform.live_level(self.recorder.peak)
            self.status.setText(f"Recording · {elapsed:.0f} seconds. Read the phrase, then stop.")
            if elapsed >= 30:
                self.toggle_recording()

    def done(self, result):
        self.timer.stop()
        if self.recorder.stream:
            self.recorder.stop()
        super().done(result)


class CharacterSetup(QDialog):
    def __init__(self, studio, worker_class):
        super().__init__(studio)
        self.studio, self.worker_class = studio, worker_class
        self.worker = None
        self.owned_workers = []
        self.closing = False
        self.voice = copy.deepcopy(studio.selected_voice)
        self.setWindowTitle(f"Character setup · {self.voice['name']}")
        self.resize(610, 510)
        layout = QVBoxLayout(self)
        heading = QLabel(f"{self.voice['name']} · version {self.voice['version']}")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        instructions = QLabel("Record this phrase in your chosen neutral character delivery. Your current dialogue take stays in the main window. Save any edited voice settings first.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        phrase = QLabel('“' + PHRASE + '”')
        phrase.setWordWrap(True)
        layout.addWidget(phrase)
        row = QHBoxLayout()
        self.hear_original = QPushButton("Hear original reference")
        self.hear_original.clicked.connect(lambda: self.play_reference(True))
        row.addWidget(self.hear_original)
        self.hear_changed = QPushButton("Hear character reference")
        self.hear_changed.clicked.connect(lambda: self.play_reference(False))
        row.addWidget(self.hear_changed)
        layout.addLayout(row)
        self.record_reference = QPushButton("Record neutral reference…")
        self.record_reference.clicked.connect(self.capture_reference)
        layout.addWidget(self.record_reference)
        self.use_take = QPushButton("Use current take instead")
        self.use_take.clicked.connect(self.save_reference)
        layout.addWidget(self.use_take)
        self.reference_status = QLabel()
        self.reference_status.setWordWrap(True)
        layout.addWidget(self.reference_status)
        layout.addWidget(QLabel("Acting notes · accent, pace, placement, characteristic words"))
        self.notes = QLineEdit(studio.profiles.get(self.voice["id"])["notes"])
        self.notes.setMaxLength(360)
        self.notes.setAccessibleName("Character acting notes")
        layout.addWidget(self.notes)
        self.save_notes = QPushButton("Save acting notes")
        self.save_notes.clicked.connect(self.store_notes)
        layout.addWidget(self.save_notes)
        hint = QLabel("For each new session: hear the original reference, then record the same phrase below. A small fixed pitch/level correction carries across your lines. Whispering and shouting keep their expression.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.calibrate = QPushButton("Calibrate this session…")
        self.calibrate.clicked.connect(self.capture_calibration)
        row.addWidget(self.calibrate)
        self.reset = QPushButton("Reset session calibration")
        self.reset.clicked.connect(self.reset_calibration)
        row.addWidget(self.reset)
        layout.addLayout(row)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setObjectName("status")
        layout.addWidget(self.message)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self):
        reference = self.studio.profiles.reference(self.voice)
        busy = self.worker is not None
        self.hear_original.setEnabled(bool(reference) and not busy)
        self.hear_changed.setEnabled(bool(reference) and not busy)
        ready = self.studio.rendered_snapshot is not None and self.studio.job is None
        self.use_take.setEnabled(ready and not busy)
        self.record_reference.setEnabled(not busy)
        self.calibrate.setEnabled(bool(reference) and not busy)
        self.reset.setEnabled(self.studio.current_calibration() is not None and not busy)
        self.save_notes.setEnabled(not busy)
        self.reference_status.setText(
            "Reference saved for this voice version. Replacing it keeps the earlier reference files."
            if reference else "No neutral reference yet. Record the phrase above to set one.")
        if not busy:
            self.message.setText(self.studio.calibration_text())

    def store_notes(self):
        try:
            self.studio.profiles.update(self.voice["id"], notes=self.notes.text().strip())
            self.message.setText("Acting notes saved.")
        except (ValueError, OSError) as error:
            self.message.setText(str(error))

    def play_reference(self, original=False):
        try:
            self.studio.play_reference(original)
        except (ValueError, OSError) as error:
            self.message.setText(str(error))

    def save_reference(self):
        studio = self.studio
        if studio.settings() != self.voice["settings"]:
            self.message.setText("Save your edited voice as a new version, then set its reference.")
            return
        if studio.current_calibration():
            self.message.setText("Reset session calibration first, then wait for the take to finish preparing.")
            return
        if not studio.rendered_snapshot or studio.job:
            return
        context = {"purpose": "reference", "source": str(studio.source),
                   "output": str(studio.rendered_path), "snapshot": copy.deepcopy(studio.rendered_snapshot)}
        self.start_analysis(studio.source, context)

    def capture_calibration(self):
        if not self.studio.reference_matches():
            self.message.setText("Reset to the saved voice and reference cleanup method before calibrating.")
            return
        self.studio.stop_playback()
        capture = CalibrationCapture(self.studio)
        if capture.exec() == QDialog.DialogCode.Accepted and capture.path:
            self.calibrate_from_path(capture.path)

    def capture_reference(self):
        if self.studio.settings() != self.voice["settings"]:
            self.message.setText("Save your edited voice as a new version, then record its reference.")
            return
        self.studio.stop_playback()
        capture = CalibrationCapture(self.studio, reference=True)
        if capture.exec() == QDialog.DialogCode.Accepted and capture.path:
            self.reference_from_path(capture.path)

    def reference_from_path(self, path):
        self.start_analysis(path, {"purpose": "reference_capture", "source": str(path),
            "settings": self.studio.settings(), "cleanup_provider": self.studio.cleanup_provider.currentData()})

    def calibrate_from_path(self, path):
        if not self.studio.reference_matches():
            self.message.setText("Use the saved reference settings before calibrating.")
            return
        try:
            reference = verify_reference(self.studio.profiles.reference(self.voice))
        except ValueError as error:
            self.message.setText(str(error))
            return
        self.start_analysis(path, {"purpose": "calibration", "reference": reference})

    def start_analysis(self, path, context):
        self.start_worker({"kind": "analyse_neutral", "input": str(path)}, context)

    def start_worker(self, specification, context):
        if self.worker:
            return
        self.studio.stop_playback()
        worker = self.worker_class(specification, context, self.studio.cache, self.studio)
        self.worker = worker
        self.owned_workers.append(worker)
        self.studio.jobs.append(worker)
        worker.completed.connect(self.completed)
        worker.finished.connect(lambda: self.studio.jobs.remove(worker) if worker in self.studio.jobs else None)
        worker.finished.connect(lambda: self.owned_workers.remove(worker) if worker in self.owned_workers else None)
        worker.finished.connect(worker.deleteLater)
        self.refresh()
        self.message.setText("Preparing the character reference…" if specification["kind"] == "render" else "Checking the neutral phrase…")
        worker.start()

    def completed(self, response):
        self.worker = None
        if self.closing:
            return
        try:
            if not response.get("ok"):
                raise ValueError(response.get("error", "Could not check that phrase. Record it again."))
            result, context = response["result"], response["context"]
            if context["purpose"] == "reference_capture":
                output = self.studio.cache / ("voice-" + uuid.uuid4().hex + ".wav")
                self.start_worker({"kind": "render", "input": context["source"], "output": str(output),
                    "settings": context["settings"], "cleanup_provider": context["cleanup_provider"]},
                    {**context, "purpose": "reference_render", "measurement": result["measurement"],
                     "source_sha256": result["source_sha256"]})
                return
            if context["purpose"] == "reference_render":
                if result["source_sha256"] != context["source_sha256"]:
                    raise ValueError("The original changed. Record a new reference.")
                snapshot = {"voice": self.voice, "settings": context["settings"],
                    "source_sha256": result["source_sha256"], "backend": result["backend"],
                    "warning": result.get("warning", ""), "ai": result["ai"],
                    "cleanup_provider": result.get("cleanup_provider", "conventional"),
                    "requested_cleanup_provider": context["cleanup_provider"], "cleanup": result.get("cleanup", {})}
                if result.get("resolve"):
                    snapshot["resolve"] = result["resolve"]
                self.studio.profiles.save_reference(self.voice, Path(context["source"]), Path(result["audio"]),
                    context["measurement"], snapshot)
                self.studio.profiles.update(self.voice["id"], notes=self.notes.text().strip(), locked=True)
                # A replacement reference invalidates any correction based on the old one.
                self.studio._sound_changed()
                message = "Reference saved. Use Hear character reference to check the sound; Calibrate this session matches later takes."
            elif context["purpose"] == "reference":
                if result["source_sha256"] != context["snapshot"]["source_sha256"]:
                    raise ValueError("The original changed. Record a new reference.")
                self.studio.profiles.save_reference(self.voice, Path(context["source"]), Path(context["output"]),
                    result["measurement"], context["snapshot"])
                self.studio.profiles.update(self.voice["id"], notes=self.notes.text().strip(), locked=True)
                message = "Neutral reference saved. Voice settings are locked. You can now calibrate a session."
            else:
                calibration = make_calibration(context["reference"], result["measurement"],
                    sample_sha256=result["source_sha256"], microphone=self.studio.microphone.currentText())
                self.studio.calibrations[self.studio.calibration_key()] = calibration
                self.studio._sound_changed()
                message = self.studio.calibration_text()
            self.studio._refresh_character_ui()
            self.refresh()
            self.message.setText(message)
        except (ValueError, OSError) as error:
            self.refresh()
            self.message.setText(str(error))

    def reset_calibration(self):
        self.studio.calibrations.pop(self.studio.calibration_key(), None)
        self.studio._sound_changed()
        self.refresh()

    def done(self, result):
        self.closing = True
        self.studio.stop_playback()
        for worker in list(self.owned_workers):
            worker.cancel()
            worker.wait(4000)
        super().done(result)
