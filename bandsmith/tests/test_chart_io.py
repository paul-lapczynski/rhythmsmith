"""The chart model, notes.mid round trips, song.ini and the tempo map."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from bandsmith.chart import (Note, Part, Phrase, Section, Song, Tempo, TimeSig, VocalNote, VocalPart,
                             from_midi, ini_text, read_mid, tempo_map, to_midi, write_mid)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import demo_song  # noqa: E402

R = 480


def small_song() -> Song:
    s = Song(name="T", artist="A", tempos=[Tempo(0, 120.0), Tempo(8 * R, 150.5)],
             time_sigs=[TimeSig(0, 4, 4), TimeSig(16 * R, 3, 4)],
             sections=[Section(4 * R, "Verse 1"), Section(12 * R, "Chorus")])
    s.parts["guitar"] = Part({"expert": [Note(4 * R, (0, 1)), Note(5 * R, (2,), forced="hopo"),
                                         Note(6 * R, (4,), length=R, forced="strum")]},
                             [Phrase(4 * R, 4 * R)])
    s.parts["guitar"].solos = [Phrase(5 * R, 2 * R)]
    s.parts["bass"] = Part({"expert": [Note(4 * R, (0,)), Note(4 * R + R // 4, (1,))],
                            "hard": [Note(4 * R, (0,))]})
    s.parts["drums"] = Part({"expert": [Note(4 * R, (0, 2), cymbals=(2,)), Note(5 * R, (1,)),
                                        Note(6 * R, (3, 4), cymbals=(4,)), Note(7 * R, (0,), double_kick=True)]},
                            activations=[Phrase(6 * R, 2 * R)])
    s.vocals = VocalPart([VocalNote(4 * R, R // 2, 60, "Hel-"), VocalNote(5 * R, R, 62, "lo"),
                          VocalNote(7 * R, R // 2, None, "hey")], [Phrase(4 * R, 4 * R)])
    s.end = 20 * R
    return s


def test_round_trip():
    s = small_song()
    back = from_midi(to_midi(s))
    assert [(t.tick, t.bpm) for t in back.tempos] == [(0, 120.0), (8 * R, 150.5)]
    assert [(t.tick, t.numerator) for t in back.time_sigs] == [(0, 4), (16 * R, 3)]
    assert [(x.tick, x.name) for x in back.sections] == [(4 * R, "Verse 1"), (12 * R, "Chorus")]
    for name in ("guitar", "bass", "drums"):
        assert back.parts[name].notes == s.parts[name].notes, name
    assert back.parts["guitar"].star_power == s.parts["guitar"].star_power
    assert back.parts["drums"].activations == s.parts["drums"].activations
    assert back.parts["guitar"].solos == s.parts["guitar"].solos
    assert back.vocals.notes == s.vocals.notes
    assert back.vocals.phrases == s.vocals.phrases
    assert back.end == s.end


def test_write_read_file(tmp_path):
    s = small_song()
    write_mid(s, tmp_path / "notes.mid")
    assert read_mid(tmp_path / "notes.mid").parts["drums"].notes == s.parts["drums"].notes


def test_midi_layout():
    names = [t.name for t in to_midi(small_song()).tracks]
    assert names == ["T", "PART DRUMS", "PART BASS", "PART GUITAR", "PART VOCALS", "EVENTS", "BEAT"]


def test_ini():
    text = ini_text(small_song(), 61000, {"guitar": 4})
    assert text.startswith("[song]\n")
    assert "song_length = 61000" in text and "diff_guitar = 4" in text and "pro_drums = True" in text


def test_tempo_map_follows_a_change():
    # 2.3 s of silence, 16 beats at 100 BPM, then 16 at 130 BPM, with a little jitter
    rng = np.random.default_rng(0)
    beats = list(2.3 + np.arange(16) * 0.6)
    beats += list(beats[-1] + 0.6 + np.arange(16) * 60 / 130)
    beats = np.array(beats) + rng.normal(0, 0.002, len(beats))
    tempos, sigs, grid = tempo_map(beats, downbeats=beats[::4])
    assert 2 <= len(tempos) <= 6                       # the lead-in, 100 and 130 BPM (maybe split once)
    assert any(abs(t.bpm - 100) < 1 for t in tempos) and any(abs(t.bpm - 130) < 1.5 for t in tempos)
    lead = len(grid) - len(beats)
    err = np.abs(np.array(grid[lead:]) - beats)
    assert err.max() < 0.012
    assert sum(s.numerator for s in sigs[:1]) in (1, 2, 3, 4)


def test_demo_song(tmp_path):
    song, mix = demo_song.make(tmp_path / "demo", seconds=20)
    folder = tmp_path / "demo"
    for f in ("notes.mid", "song.ini", "guitar.ogg", "bass.ogg", "drums.ogg", "vocals.ogg"):
        assert (folder / f).is_file(), f
    assert mix.is_file()
    back = read_mid(folder / "notes.mid")
    assert back.parts["drums"].notes["expert"] == song.parts["drums"].notes["expert"]
    assert back.vocals.notes == song.vocals.notes


@pytest.fixture(scope="module")
def scan_chart():
    from conftest import scan_chart_cmd
    cmd = scan_chart_cmd()
    if cmd is None:
        msg = "scan-chart isn't installed (npm ci --prefix bandsmith/tools/scan-chart)"
        if os.environ.get("BANDSMITH_REQUIRE_SCAN_CHART"):         # CI: a missing checker is a failure
            pytest.fail(msg)
        pytest.skip(msg)
    return cmd


def test_scan_chart_clean(tmp_path, scan_chart):
    import json
    import subprocess
    demo_song.make(tmp_path / "demo", seconds=70)
    r = subprocess.run(scan_chart + [str(tmp_path / "demo")], capture_output=True, text=True, check=True)
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["playable"]
    assert set(res["instruments"]) == {"guitar", "bass", "drums"}
    assert res["hasVocals"] and res["hasLyrics"]
    assert res["drumType"] == 1                       # 4-lane pro drums
    folder = [i["folderIssue"] for i in res["folderIssues"] if i["folderIssue"] != "noAlbumArt"]
    assert folder == [] and res["metadataIssues"] == []
    issues = sorted({i["noteIssue"] for i in res["chartIssues"]})
    assert issues == [], res["chartIssues"][:10]


def test_sixteenth_notes_are_not_sustains():
    """Rock Band writes plain notes a 16th long; Clone Hero only sustains past 1/3 beat."""
    import mido
    from bandsmith.chart import from_midi
    mf = mido.MidiFile(type=1, ticks_per_beat=R)
    mf.tracks.append(mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000, time=0)]))
    tr = mido.MidiTrack([mido.MetaMessage("track_name", name="PART GUITAR", time=0)])
    for pitch, length in ((96, R // 4), (97, R // 3 + 1), (98, R // 3 + 2)):
        tr += [mido.Message("note_on", note=pitch, velocity=100, time=0),
               mido.Message("note_off", note=pitch, velocity=0, time=length),
               mido.Message("note_on", note=0, velocity=0, time=R)]
    mf.tracks.append(tr)
    notes = from_midi(mf).parts["guitar"].notes["expert"]
    assert [n.length for n in notes] == [0, 0, R // 3 + 2]
