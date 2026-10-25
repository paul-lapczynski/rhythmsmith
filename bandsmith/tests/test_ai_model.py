"""The AI model loader: found only when its files are there, and how it turns
the model's per-slot output into chart notes (with a stand-in ONNX session,
so no model file is needed)."""
import numpy as np
import pytest

from bandsmith.analyze import ai_model
from bandsmith.chart import Tempo, TempoMap

TM = TempoMap([Tempo(0, 120.0)])
SR = 22050


def test_not_available_without_files(tmp_path, monkeypatch):
    monkeypatch.setenv("BANDSMITH_MODEL_DIR", str(tmp_path))
    assert ai_model.model_dir() == tmp_path
    assert not ai_model.available("drums")
    (tmp_path / "chart_drums.onnx").write_bytes(b"")
    assert not ai_model.available("drums")                 # the .json has to be there too


class FakeSession:
    def __init__(self, out):
        self.out = out
        self.inputs = None

    def run(self, _, inputs):
        self.inputs = inputs
        return [self.out[None, : inputs["x"].shape[1]]]


META = {"sr": SR, "hop": 256, "n_fft": 2048, "n_mels": 80, "fmin": 27.5, "win": 5,
        "thresholds": {"expert": 0.5}, "lane_threshold": 0.5, "chord_threshold": 0.4}


def run(monkeypatch, inst, out):
    sess = FakeSession(out)
    monkeypatch.setattr(ai_model, "_session", lambda path: (sess, META))
    y = np.zeros(SR * 4, np.float32)
    notes = ai_model.predict(inst, y, y, SR, TM, 0.0, 480 * 6, path="x.onnx")
    return notes, sess


def test_drum_notes(monkeypatch):
    n = 480 * 6 // 40 + 1
    out = np.zeros((n, 11), np.float32)
    out[12, 0] = 0.9                                         # an onset on beat 1 (slot 12 = 480 ticks)
    out[13, 0] = 0.8                                         # next to it, weaker: dropped
    out[12, [1, 2, 3, 4, 5]] = [0.9, 0.8, 0.7, 0.6, 0.1]      # kick + snare, yellow, blue: two hands at most
    out[12, 8:11] = [0.9, 0.1, 0.1]                          # yellow is a cymbal
    out[36, 0] = 0.7
    out[36, 1:6] = [0.1, 0.1, 0.1, 0.1, 0.2]                 # nothing over the cut: the likeliest lane
    notes, sess = run(monkeypatch, "drums", out)
    assert sess.inputs["x"].shape == (1, n, 2, 5, 80) and sess.inputs["diff"].tolist() == [3]
    assert [(x.tick, x.lanes, x.cymbals) for x in notes] == [(480, (0, 1, 2), (2,)), (1440, (4,), ())]


def test_fretted_chords_and_sustains(monkeypatch):
    n = 480 * 6 // 40 + 1
    out = np.zeros((n, 8), np.float32)
    out[12, 0], out[12, 1:6] = 0.9, [0.9, 0.55, 0.1, 0.1, 0.1]     # a chord: two frets over the lane cut
    out[12, 6] = np.log1p(24)                                       # rings 24 slots = 2 beats
    out[72, 0], out[72, 1:6] = 0.9, [0.1, 0.1, 0.3, 0.1, 0.1]       # one note
    notes, _ = run(monkeypatch, "guitar", out)
    assert notes[0].lanes == (0, 1) and notes[0].length == 960
    assert notes[1].lanes == (2,) and notes[1].length == 0


def test_window_switch_needs_a_model(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from bandsmith import app
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("BANDSMITH_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(app, "settings", lambda: QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat))
    w = app.MainWindow()
    assert not w.ai_model.isEnabled() and not w.ai_model.isChecked()
    w.close()
    (tmp_path / "chart_drums.onnx").write_bytes(b"x")
    (tmp_path / "chart_drums.json").write_text("{}")
    pytest.importorskip("onnxruntime")
    w = app.MainWindow()
    assert w.ai_model.isEnabled() and not w.ai_model.isChecked()   # there, but off until ticked
    w.close()
