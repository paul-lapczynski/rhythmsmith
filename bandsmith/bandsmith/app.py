"""The BandSmith window: pick a song, and it becomes an Expert Clone Hero / YARG
song folder in the output folder (by default Documents\\Clone Hero\\Songs\\BandSmith Test).

    python -m bandsmith          # from bandsmith/, with core/ installed or on PYTHONPATH

The chart is made in a background thread (analyze.make), so the window keeps
responding; Cancel stops it after the step it's on.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)

from bandsmith import __version__

TITLE = f"BandSmith {__version__}"
TARGETS = ("Clone Hero / YARG", "Rock Band 3 compatible")
PARTS = ("Guitar", "Bass", "Drums", "Vocals")
AUDIO = "Audio (*.mp3 *.ogg *.opus *.wav *.flac *.m4a *.wma);;All files (*)"


def default_out_dir() -> Path:
    docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or str(Path.home() / "Documents")
    return Path(docs) / "Clone Hero" / "Songs" / "BandSmith Test"


def settings() -> QSettings:
    return QSettings("BandSmith", "BandSmith")


class MakeJob(QThread):
    """analyze.make() for one song, off the window's thread."""
    log = Signal(str)
    finished_ok = Signal(str)          # the song folder
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, song: Path, out_dir: Path, parts: tuple[str, ...], art_online: bool):
        super().__init__()
        self.song, self.out_dir, self.parts, self.art_online = song, out_dir, parts, art_online
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._stop.set()

    def run(self) -> None:
        from bandsmith.analyze import make as mk
        try:
            title, artist, _, _ = mk.tags(self.song)
            folder = self.out_dir / mk.folder_name(title, artist)
            self.log.emit(f"charting {self.song.name} into {folder}")
            res = mk.make(self.song, folder, self.parts, art_online=self.art_online,
                          log=self.log.emit, cancel=self._stop.is_set)
            notes = ", ".join(f"{k} {v} notes" for k, v in res.notes.items())
            self.log.emit(f"done in {res.timings['total']:.0f} s: {notes}")
            self.finished_ok.emit(str(res.folder))
        except mk.Cancelled:
            self.cancelled.emit()
        except Exception as e:                 # shown in the window, never a crash
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.job: MakeJob | None = None
        self.last_folder: Path | None = None
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.addWidget(QLabel("Make Clone Hero / YARG charts (Expert) from songs."))

        self.add_song = QPushButton("Add song")
        self.add_song.clicked.connect(self.pick_song)
        lay.addWidget(self.add_song)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output folder:"))
        self.out_edit = QLineEdit(settings().value("out_dir", str(default_out_dir())))
        self.out_edit.setReadOnly(True)
        row.addWidget(self.out_edit, 1)
        self.change_out = QPushButton("Change...")
        self.change_out.clicked.connect(self.pick_out_dir)
        row.addWidget(self.change_out)
        lay.addLayout(row)

        self.checks: dict[str, QCheckBox] = {}
        for title, names in (("Output", TARGETS), ("Parts", PARTS)):
            box = QGroupBox(title)
            col = QVBoxLayout(box)
            for name in names:
                cb = QCheckBox(name)
                cb.setChecked(True)
                col.addWidget(cb)
                self.checks[name] = cb
            lay.addWidget(box)
        rb = self.checks["Rock Band 3 compatible"]
        rb.setChecked(False)
        rb.setEnabled(False)
        rb.setToolTip("Coming in a later version")
        self.checks["Clone Hero / YARG"].setEnabled(False)

        from rhythmsmith_core import artwork
        self.art_online = QCheckBox("Look up album art online when the song has none")
        self.art_online.setChecked(False)
        self.art_online.setToolTip("Only when ticked. It asks:\n" + "\n".join(
            f"{what}: {where}" for what, where in artwork.ONLINE_SOURCES))
        lay.addWidget(self.art_online)

        bar = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        bar.addWidget(self.progress, 1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        bar.addWidget(self.cancel_btn)
        lay.addLayout(bar)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)

        done = QHBoxLayout()
        self.status = QLabel("")
        done.addWidget(self.status, 1)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setVisible(False)
        self.open_btn.clicked.connect(self.open_folder)
        done.addWidget(self.open_btn)
        lay.addLayout(done)

        self.setCentralWidget(body)
        self.resize(560, 620)

    # ------------------------------------------------------------------ state

    def parts(self) -> tuple[str, ...]:
        return tuple(n.lower() for n in PARTS if self.checks[n].isChecked())

    def out_dir(self) -> Path:
        return Path(self.out_edit.text())

    def set_out_dir(self, path: Path) -> None:
        self.out_edit.setText(str(path))
        settings().setValue("out_dir", str(path))

    def busy(self, on: bool) -> None:
        self.add_song.setEnabled(not on)
        self.change_out.setEnabled(not on)
        self.cancel_btn.setEnabled(on)
        self.progress.setRange(0, 0 if on else 1)          # 0..0: a moving "busy" bar

    # ------------------------------------------------------------------ actions

    def pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_edit.text())
        if d:
            self.set_out_dir(Path(d))

    def pick_song(self) -> None:
        start = settings().value("song_dir", str(Path.home()))
        f, _ = QFileDialog.getOpenFileName(self, "Add song", start, AUDIO)
        if f:
            settings().setValue("song_dir", str(Path(f).parent))
            self.start(Path(f))

    def start(self, song: Path) -> None:
        if not self.parts():
            self.status.setText("Pick at least one part.")
            return
        self.out_dir().mkdir(parents=True, exist_ok=True)
        self.open_btn.setVisible(False)
        self.status.setText(f"Working on {song.name}...")
        self.log.clear()
        self.job = MakeJob(song, self.out_dir(), self.parts(), self.art_online.isChecked())
        self.job.log.connect(self.log.appendPlainText)
        self.job.finished_ok.connect(self.on_done)
        self.job.failed.connect(self.on_failed)
        self.job.cancelled.connect(self.on_cancelled)
        self.busy(True)
        self.job.start()

    def cancel(self) -> None:
        if self.job:
            self.job.cancel()
            self.cancel_btn.setEnabled(False)
            self.status.setText("Stopping after this step...")

    def on_done(self, folder: str) -> None:
        self.busy(False)
        self.last_folder = Path(folder)
        self.status.setText(f"Done: {Path(folder).name}")
        self.open_btn.setVisible(True)

    def on_failed(self, msg: str) -> None:
        self.busy(False)
        self.log.appendPlainText(msg)
        self.status.setText("That didn't work (see above).")

    def on_cancelled(self) -> None:
        self.busy(False)
        self.status.setText("Cancelled.")

    def open_folder(self) -> None:
        if self.last_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_folder)))

    def closeEvent(self, e) -> None:
        if self.job and self.job.isRunning():
            self.job.cancel()
            self.job.wait(30000)
        super().closeEvent(e)


def main(argv=None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
