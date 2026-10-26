"""The arranger's decisions as the chart uses them (no model file needed):
sticky solo sections, chord sizes never invented, HOPOs only where they differ."""
import numpy as np

from bandsmith.analyze import arranger, music
from bandsmith.analyze.music import Played
from bandsmith.chart import Tempo, TempoMap

TM = TempoMap([Tempo(0, 120.0)])


def outputs(solo_probs):
    out = np.zeros((len(solo_probs), 12), np.float32)
    out[:, 11] = np.log(np.array(solo_probs) / (1 - np.array(solo_probs)))
    return out


def test_solo_sections_are_sticky():
    ticks = [i * 240 for i in range(40)]
    probs = [0.1] * 8 + [0.9] * 6 + [0.2, 0.8, 0.9, 0.2, 0.9] + [0.9] * 6 + [0.1] * 15
    spans = arranger.solo_sections(ticks, outputs(probs))
    assert len(spans) == 1                                   # one solo, not flickering in and out
    a, b = spans[0]
    assert a == ticks[8] and b >= ticks[24]


def test_chord_sizes_never_invent_a_chord():
    out = np.zeros((3, 12), np.float32)
    out[:, 6] = 5.0                                          # the model says "two frets" everywhere
    assert arranger.chord_sizes(out, [1, 2, 3]) == [1, 2, 2]


def test_chart_uses_the_arranged_sizes_and_hopos():
    ps = [Played(1.0, 1.2, [40, 47], 1.0, 1.0), Played(1.5, 1.7, [45], 1.0, 1.0), Played(1.6, 1.7, [47], 1.0, 1.0)]
    ticks = [music.quantize(TM.tick(p.start)) for p in ps]
    notes, _ = music.chart(ps, TM, arranged={ticks[0]: (3, False), ticks[1]: (1, True), ticks[2]: (1, False)})
    assert len(notes[0].lanes) == 2 or len(notes[0].lanes) == 3
    assert notes[1].forced == "hopo"                         # the arranger wants a HOPO Clone Hero wouldn't make
    assert notes[2].forced == "strum"                        # and a strum where Clone Hero would make a HOPO
