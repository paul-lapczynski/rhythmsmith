"""Imports BandSmith and the shared core and checks the versions. No audio,
no network, no window."""
import re

import bandsmith
import rhythmsmith_core
from rhythmsmith_core import accel, addons, artwork, lyrics, stems


def test_version():
    assert bandsmith.__version__ == "0.0.1"
    assert re.fullmatch(r"0\.\d+\.\d+", rhythmsmith_core.__version__)    # below 1.0


def test_core_modules():
    assert accel.wanted() in accel.CHOICES
    assert stems.stems_for(stems.MODEL) == ("vocals", "drums", "bass", "other")
    assert set(stems.stems_for(stems.MODEL_6S)) >= {"guitar", "piano"}
    assert set(addons.DEMUCS_MODELS) == set(stems.MODELS)
    assert addons.sources("stems")
    assert lyrics.MODEL_URL.startswith("https://")
    assert artwork.file_names("x")["BANNER"] == "x-bn.png"


def test_app_imports():
    from bandsmith import app
    assert app.TITLE == "BandSmith 0.0.1"
