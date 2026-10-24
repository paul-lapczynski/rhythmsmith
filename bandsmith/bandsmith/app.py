"""The BandSmith window. Phase 0: the layout only, nothing is generated yet.

    python -m bandsmith          # from bandsmith/, with core/ installed or on PYTHONPATH
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import (QApplication, QCheckBox, QGroupBox, QLabel, QMainWindow,
                               QPushButton, QVBoxLayout, QWidget)

from bandsmith import __version__

TITLE = f"BandSmith {__version__}"
TARGETS = ("Clone Hero / YARG", "Rock Band 3 compatible")
PARTS = ("Guitar", "Bass", "Drums", "Vocals")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.addWidget(QLabel("Make Clone Hero / Rock Band charts from songs. (Coming soon.)"))

        self.add_song = QPushButton("Add song")       # TODO: pick songs (phase 1)
        lay.addWidget(self.add_song)

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
        lay.addStretch(1)
        self.setCentralWidget(body)
        self.resize(420, 360)


def main(argv=None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
