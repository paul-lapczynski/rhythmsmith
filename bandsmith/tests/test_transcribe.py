"""Transcription and the Expert charting rules, on the synth demo song (exact ground truth)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bandsmith.analyze import charting
from bandsmith.analyze.drums import Hit, limb_limit
from bandsmith.analyze.pitched import Event
from bandsmith.chart import Tempo, TempoMap, read_mid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import demo_song  # noqa: E402
from eval_vs_reference import compare  # noqa: E402

TM = TempoMap([Tempo(0, 120.0)])          # a beat is 0.5 s


def test_quantize_prefers_sixteenths():
    assert charting.quantize(118) == 120
    assert charting.quantize(158) == 160                 # clearly a triplet
    assert charting.quantize(140) == 120                 # in between: the 16th


def test_frets_follow_pitch():
    evs = [Event(1.0 + 0.25 * i, 1.2 + 0.25 * i, [p]) for i, p in enumerate([40, 42, 44, 47, 44, 42])]
    frets = [n.lanes[0] for n in charting.fret_notes(evs, TM)]
    assert frets[:4] == sorted(frets[:4]) and frets[3] > frets[0]
    assert frets[4] < frets[3] and frets[5] < frets[4]
    assert max(frets) - min(frets) <= 3                  # a 3-4 fret window per phrase


def test_power_chords_and_sustains():
    evs = [Event(1.0, 2.0, [40, 47]), Event(2.5, 2.6, [45])]
    notes = charting.fret_notes(evs, TM)
    assert len(notes[0].lanes) == 2 and notes[0].lanes[1] - notes[0].lanes[0] == 1
    assert notes[0].length >= charting.MIN_SUSTAIN       # rang a whole 2 beats: a sustain
    assert notes[0].tick + notes[0].length <= notes[1].tick - charting.SUSTAIN_GAP
    assert notes[1].length == 0


def test_limb_limit():
    chord = [Hit(1.0, 0, False, 1), Hit(1.0, 1, False, 0.9), Hit(1.0, 2, True, 0.5), Hit(1.0, 4, True, 0.8)]
    assert [h.lane for h in limb_limit(chord)] == [0, 1, 4]


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    _, mix = demo_song.make(d / "song", seconds=70)
    return d, mix


@pytest.fixture(scope="module")
def made(demo):
    from bandsmith.analyze.make import make
    d, mix = demo
    # the demo's own stems stand in for separation; no add-ons (as in CI)
    return make(mix, d / "out", stems_dir=d / "song", ai_notes=False, whisper=False,
                title="Demo", artist="BandSmith", log=lambda s: None)


def test_make_matches_the_demo(demo, made):
    d, _ = demo
    cmp = compare(made.song, read_mid(d / "song" / "notes.mid"), made.pad)
    assert cmp["drums"]["f1_near"] > 0.85, cmp["drums"]
    assert cmp["guitar"]["f1_near"] > 0.85, cmp["guitar"]
    assert cmp["bass"]["f1_near"] > 0.85, cmp["bass"]
    assert cmp["vocals"]["f1_near"] > 0.7, cmp["vocals"]
    assert cmp["bass"]["direction"] > 0.6


def test_make_writes_a_song_folder(made):
    for f in ("notes.mid", "song.ini", "guitar.ogg", "bass.ogg", "drums.ogg", "vocals.ogg"):
        assert (made.folder / f).is_file(), f
    first = made.song.tempo_map().seconds(made.song.first_tick())
    assert first >= 2.0                                  # lead-in for Clone Hero


def test_make_scans_clean(made):
    from conftest import scan_chart_cmd
    import os
    cmd = scan_chart_cmd()
    if cmd is None:
        if os.environ.get("BANDSMITH_REQUIRE_SCAN_CHART"):
            pytest.fail("scan-chart isn't installed")
        pytest.skip("scan-chart isn't installed")
    r = subprocess.run(cmd + [str(made.folder)], capture_output=True, text=True, check=True)
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["playable"]
    assert set(res["instruments"]) == {"guitar", "bass", "drums"} and res["hasVocals"]
    assert [i for i in res["chartIssues"]] == []
    assert [i["folderIssue"] for i in res["folderIssues"]] in ([], ["noAlbumArt"])


def test_syllables():
    from bandsmith.analyze.vocals import syllables
    assert syllables("alone") == ["a", "lone"]            # a silent e stays on its syllable
    assert syllables("like") == ["like"] and syllables("singing") == ["sin", "ging"]


def test_lyrics_with_curly_quotes_write(tmp_path):
    from bandsmith.chart import Song, VocalNote, VocalPart, read_mid, write_mid
    from bandsmith.chart.midi import midi_text
    assert midi_text("don’t “stop” — naïve") == "don't \"stop\" - naïve"
    s = Song(name="Café 日本")
    s.vocals = VocalPart([VocalNote(480 * 4, 240, 60, "don’t")])
    write_mid(s, tmp_path / "notes.mid")
    assert read_mid(tmp_path / "notes.mid").vocals.notes[0].lyric == "don't"
