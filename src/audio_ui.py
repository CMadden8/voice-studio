"""Microphone capture and waveform UI, shared by the local audio tools.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import time
import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

class AudioRecorder:
    """The microphone is opened only when the Record button is pressed."""
    def __init__(self):
        self.stream = None
        self.frames = []
        self.peak = 0.
        self.rate = 48000
        self.started = 0.
        self.overflows = 0

    def start(self, device=None):
        if self.stream:
            raise RuntimeError("Already recording")
        information = sd.query_devices(device, "input")
        self.rate = round(information["default_samplerate"])
        self.frames, self.peak, self.overflows = [], 0., 0

        def callback(data, _frames, _time, status):
            if status.input_overflow:
                self.overflows += 1
            self.frames.append(data[:, 0].copy())
            self.peak = float(np.max(np.abs(data)))

        self.stream = sd.InputStream(device=device, samplerate=self.rate, channels=1,
                                     dtype="float32", callback=callback, latency="high")
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            self.stream = None
            raise
        self.started = time.monotonic()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        frames, self.frames = self.frames, []
        return (np.concatenate(frames) if frames else np.empty(0, dtype=np.float32)), self.rate

class Waveform(QWidget):
    def __init__(self):
        super().__init__()
        self.peaks = np.zeros(120)
        self.has_audio = False
        self.live = False
        self.setMinimumHeight(100)
        self.setAccessibleName("Recording waveform")

    def set_audio(self, audio):
        self.has_audio = len(audio) > 0
        bins = np.array_split(np.asarray(audio), 120) if self.has_audio else []
        self.peaks = np.array([np.max(np.abs(b)) if len(b) else 0 for b in bins]) if bins else np.zeros(120)
        maximum = max(float(np.max(self.peaks)), .01)
        self.peaks = np.minimum(1., self.peaks / maximum)
        self.live = False
        self.update()

    def live_level(self, level):
        self.live = True
        self.peaks = np.roll(self.peaks, -1)
        self.peaks[-1] = min(1., level * 3)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f4f8f7"))
        if not self.has_audio and not self.live:
            painter.setPen(QColor("#7d8b8b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Your recording will appear here")
            return
        width = self.width() - 32
        center = self.height() / 2
        painter.setPen(QPen(QColor("#d35f50" if self.live else "#25867d"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for index, peak in enumerate(self.peaks):
            x = round(16 + (index + .5) * width / len(self.peaks))
            height = max(2., float(peak) ** .6 * (self.height() - 22) / 2)
            painter.drawLine(x, round(center - height), x, round(center + height))
