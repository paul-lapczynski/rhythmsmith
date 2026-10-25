"""The music layer: only played notes, held notes kept whole, chords by
interval, frets from pitch, riffs charted the same, slides and bends."""
import numpy as np

from bandsmith.analyze import music
from bandsmith.analyze.music import Contour, Played
from bandsmith.chart import Tempo, TempoMap

TM = TempoMap([Tempo(0, 120.0)])          # a beat is 0.5 s


def p(start, end, *pitches, attack=1.0):
    return Played(start, end, list(pitches), 1.0, attack)


def test_chord_kinds():
    assert music.chord_kind([40]) == "single"
    assert music.chord_kind([40, 47]) == "power"                 # root and fifth
    assert music.chord_kind([40, 47, 52]) == "power"             # with the octave
    assert music.chord_kind([40, 44, 47]) == "full"              # a major triad
    assert music.chord_kind([40, 52]) == "single"                # octaves of one note
    assert music.chord_kind([40, 44, 47, 52, 55]) == "dense"     # a big open chord


def test_held_note_is_one_note():
    ps = music.merge_held([p(1.0, 1.4, 45), p(1.42, 2.0, 45, attack=0.1), p(2.5, 2.7, 45)])
    assert [(x.start, x.end) for x in ps] == [(1.0, 2.0), (2.5, 2.7)]   # the re-hit at 2.5 is a new note
    notes, _ = music.chart(ps, TM)
    assert len(notes) == 2 and notes[0].length >= 480               # sustained through


def test_same_pitch_same_fret_and_contour():
    pitches = [40, 43, 45, 43, 40, 47, 45, 40]
    ps = [p(1.0 + 0.25 * i, 1.2 + 0.25 * i, x) for i, x in enumerate(pitches)]
    notes, _ = music.chart(ps, TM)
    fret = {}
    for x, n in zip(pitches, notes):
        assert fret.setdefault(x, n.lanes) == n.lanes                # same pitch, same fret
    order = sorted(fret, key=lambda x: x)
    assert [fret[x][0] for x in order] == sorted(fret[x][0] for x in order)   # higher pitch, further right
    assert music.same_pitch_same_fret(notes, ps, TM) == 1.0


def test_chord_shapes():
    ps = [p(1.0, 1.2, 40, 47), p(1.5, 1.7, 40, 44, 47), p(2.0, 2.2, 45), p(2.5, 2.7, 40, 44, 47, 52, 55)]
    notes, info = music.chart(ps, TM)
    assert [len(n.lanes) for n in notes] == [2, 3, 1, 4]
    assert all(n.lanes == tuple(range(n.lanes[0], n.lanes[0] + len(n.lanes))) for n in notes)
    assert info["kinds"] == {"single": 1, "power": 1, "full": 1, "dense": 1}
    capped, _ = music.chart(ps, TM, max_chord=3)
    assert max(len(n.lanes) for n in capped) == 3
    bass, _ = music.chart([p(1.0, 1.2, 28, 35)], TM, bass=True)
    assert len(bass[0].lanes) == 1


def test_riffs_repeat_identically():
    riff = [40, 43, 45, 43]
    ps = [p(1.0 + 0.25 * i, 1.1 + 0.25 * i, x) for i, x in enumerate(riff)]
    ps += [p(4.0 + 0.25 * i, 4.1 + 0.25 * i, x) for i, x in enumerate(riff)]
    ps += [p(2.5, 2.6, 52), p(2.75, 2.85, 55)]                       # something else in between
    notes, info = music.chart(ps, TM)
    first = [n.lanes for n in notes if n.tick < 960 * 2]
    again = [n.lanes for n in notes if n.tick >= 960 * 4]
    assert first == again and info["repeated_phrases"] >= 1


def contour_of(segments):
    """[(start, end, midi start, midi end)] -> a 10 ms Contour."""
    t = np.arange(0, 4, 0.01)
    m = np.full(len(t), np.nan)
    for a, b, x, y in segments:
        sel = (t >= a) & (t < b)
        m[sel] = np.linspace(x, y, sel.sum())
    return Contour(t, m)


def test_bend_stays_one_note():
    c = contour_of([(1.0, 1.3, 45, 45), (1.3, 1.5, 45, 46.8), (1.5, 1.7, 46.8, 45), (1.7, 2.0, 45, 45)])
    ps, counts = music.gestures([p(1.0, 1.35, 45), p(1.4, 2.0, 47, attack=0.1)], c)
    assert counts["bend"] == 1 and len(ps) == 1 and ps[0].end == 2.0


def test_slide_is_held_then_hopo():
    c = contour_of([(1.0, 1.4, 45, 45), (1.4, 1.5, 45, 50), (1.5, 2.0, 50, 50)])
    ps, counts = music.gestures([p(1.0, 1.5, 45), p(1.5, 2.0, 50, attack=0.1)], c)
    assert counts["slide"] == 1 and ps[1].hopo
    notes, _ = music.chart(ps, TM)
    assert notes[1].forced == "hopo" and notes[1].lanes[0] > notes[0].lanes[0]
    assert music.held_changes(notes, ps, TM) == 0


def test_new_attack_is_a_normal_note():
    c = contour_of([(1.0, 1.5, 45, 45), (1.5, 2.0, 50, 50)])
    ps, counts = music.gestures([p(1.0, 1.5, 45), p(1.5, 2.0, 50, attack=0.9)], c)
    assert not ps[1].hopo and sum(counts.values()) == 0


def test_model_confirms_notes():
    ps = [p(1.0, 1.2, 45), p(1.5, 1.7, 47)]
    probs = np.zeros(200)
    probs[int(TM.tick(1.0) / 40)] = 0.9
    assert [x.start for x in music.confirm(ps, TM, probs, 0.5)] == [1.0]


def test_easier_difficulties_only_take_away():
    from bandsmith.analyze import charting
    from bandsmith.chart import Note
    rng = np.random.default_rng(0)
    expert = []
    for k in range(64):
        size = int(rng.choice([1, 1, 2, 3, 4]))
        base = int(rng.integers(0, 6 - size))
        expert.append(Note(480 * 4 + k * 120, tuple(range(base, base + size)), 480 if k % 7 == 0 else 0))
    diffs = charting.all_difficulties(expert)
    ticks = {d: [n.tick for n in ns] for d, ns in diffs.items()}
    assert set(ticks["easy"]) <= set(ticks["medium"]) <= set(ticks["hard"]) <= set(ticks["expert"])
    assert len(ticks["expert"]) > len(ticks["hard"]) > len(ticks["medium"]) > len(ticks["easy"]) > 0
    by_tick = {n.tick: n for n in expert}
    for d, width, chord in (("hard", 5, 3), ("medium", 4, 2), ("easy", 3, 1)):
        for n in diffs[d]:
            assert max(n.lanes) < width and len(n.lanes) <= min(chord, len(by_tick[n.tick].lanes))
    drums = charting.all_difficulties([Note(480 * 4 + k * 240, (0, 1, 2), cymbals=(2,)) for k in range(32)], drums=True)
    assert all(len(n.lanes) <= 1 for n in drums["easy"])
    assert all(set(n.tick for n in drums[d]) <= set(n.tick for n in drums["expert"]) for d in ("hard", "medium", "easy"))
