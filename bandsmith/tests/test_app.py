"""The window: defaults, the parts it asks for, the remembered output folder, and
the background job (with make() stood in for, so no audio is needed)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from bandsmith import app  # noqa: E402
from bandsmith.analyze import make as mk  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(app, "settings", lambda: QSettings(ini, QSettings.IniFormat))
    w = app.MainWindow()
    yield w
    w.close()


def test_defaults(window):
    assert window.windowTitle() == "BandSmith 0.0.1"
    assert window.out_dir() == app.default_out_dir()
    assert window.out_dir().parts[-3:] == ("Clone Hero", "Songs", "BandSmith Test")
    assert window.parts() == ("guitar", "bass", "drums", "vocals")
    assert not window.art_online.isChecked()                        # online lookups are opt-in
    assert not window.checks["Rock Band 3 compatible"].isEnabled()


def test_parts_follow_the_checkboxes(window):
    window.checks["Vocals"].setChecked(False)
    window.checks["Bass"].setChecked(False)
    assert window.parts() == ("guitar", "drums")


def test_output_folder_is_remembered(window, tmp_path):
    window.set_out_dir(tmp_path / "songs")
    again = app.MainWindow()
    assert again.out_dir() == tmp_path / "songs"
    again.close()


def run_job(qapp, job):
    got = {}
    job.finished_ok.connect(lambda f: got.setdefault("done", f))
    job.failed.connect(lambda m: got.setdefault("failed", m))
    job.cancelled.connect(lambda: got.setdefault("cancelled", True))
    job.log.connect(lambda s: got.setdefault("log", []).append(s))
    job.start()
    assert job.wait(20000)
    qapp.processEvents()
    return got


def test_job_runs_make_off_the_window_thread(qapp, tmp_path, monkeypatch):
    seen = {}

    def fake_make(song, out, parts, art_online=False, log=print, cancel=None):
        seen.update(song=song, out=out, parts=parts, art_online=art_online)
        log("progress: drums")
        return mk.Result(out, mk.Song(), 0.0, {"total": 1.0}, {"drums": 10})
    monkeypatch.setattr(mk, "make", fake_make)
    song = tmp_path / "Artist - Title.mp3"
    song.write_bytes(b"")
    got = run_job(qapp, app.MakeJob(song, tmp_path / "out", ("drums",), False))
    assert got["done"] == str(tmp_path / "out" / "Artist - Title")
    assert seen["parts"] == ("drums",) and seen["art_online"] is False
    assert "progress: drums" in got["log"]


def test_job_cancel_and_errors(qapp, tmp_path, monkeypatch):
    def cancelled_make(song, out, parts, art_online=False, log=print, cancel=None):
        assert cancel()                  # the job was cancelled before it started
        raise mk.Cancelled()
    monkeypatch.setattr(mk, "make", cancelled_make)
    song = tmp_path / "x.mp3"
    song.write_bytes(b"")
    job = app.MakeJob(song, tmp_path, ("drums",), False)
    job.cancel()
    assert run_job(qapp, job) == {"cancelled": True, "log": [f"charting x.mp3 into {tmp_path / 'x'}"]}

    def broken_make(*a, **k):
        raise ValueError("no audio")
    monkeypatch.setattr(mk, "make", broken_make)
    got = run_job(qapp, app.MakeJob(song, tmp_path, ("drums",), False))
    assert got["failed"] == "ValueError: no audio"
