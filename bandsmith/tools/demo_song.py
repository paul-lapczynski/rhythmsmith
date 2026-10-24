"""A short made-up song with an exact chart, for tests: guitar, bass, drums and
vocals charted on Expert, rendered to audio with a plain synth (oscillators and
noise; no samples), so what's in the audio is exactly what's in the chart.

    python bandsmith/tools/demo_song.py OUTDIR [--seconds 40] [--seed 1]

Writes OUTDIR/ as a Clone Hero song folder (notes.mid, song.ini, guitar.ogg,
bass.ogg, drums.ogg, vocals.ogg) and OUTDIR.mix.ogg beside it: everything mixed,
the input a chart generator would get. The chart is built first, written to
notes.mid, and the audio is rendered from that notes.mid read back.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run from a checkout

from bandsmith.chart import (Note, Part, Phrase, Section, Song, Tempo, TimeSig, VocalNote,  # noqa: E402
                             VocalPart, read_mid, write_ini, write_mid)

SR = 44100
R = 480
LEAD_BEATS = 5                      # silence before the music (Clone Hero wants 2 s or more)
GUITAR_PITCH = [40, 43, 45, 47, 50]           # E2 G2 A2 B2 D3 (pentatonic), one per fret
BASS_PITCH = [28, 31, 33, 35, 38]
LYRICS = ["la", "da", "na", "hey", "oh", "yeah", "go", "now"]


def build(seconds: float = 40.0, seed: int = 1) -> Song:
    rng = np.random.default_rng(seed)
    song = Song(name="BandSmith Demo", artist="BandSmith", album="Tests", genre="Test", year="2026",
                charter="BandSmith demo", tempos=[Tempo(0, 120.0)], time_sigs=[TimeSig(0, 4, 4)])
    # 120 BPM, then 140 halfway: the tempo map has to follow a change
    bars = max(8, int(seconds / 2.0))
    change_bar = bars // 2
    song.tempos.append(Tempo((LEAD_BEATS + change_bar * 4) * R, 140.0))
    song.time_sigs = [TimeSig(0, 1, 4), TimeSig(R, 4, 4)]        # a 1-beat pickup, then 4/4 bars from beat 1
    # the lead-in is 5 beats: one pickup beat and one empty bar before the music
    start = (1 + 4) * R
    guitar, bass, drums, vocals, phrases, fills = [], [], [], [], [], []
    names = ["Intro", "Verse 1", "Chorus 1", "Verse 2", "Chorus 2", "Outro"]
    per = max(2, bars // len(names))
    for i in range(0, bars, per):
        song.sections.append(Section(start + i * 4 * R, names[min(len(names) - 1, i // per)]))
    fret = 2
    for bar in range(bars):
        b0 = start + bar * 4 * R
        # drums: kick 1 and 3 (and the and of 3), snare 2 and 4, hi-hat 8ths, crash on section starts
        for e in range(8):
            t = b0 + e * R // 2
            lanes, cym = [2], [2]                                   # hi-hat: yellow cymbal
            if e in (0, 4, 5):
                lanes.append(0)
            if e in (2, 6):
                lanes.append(1)
            if e == 0 and bar % per == 0:
                lanes, cym = [0, 4], [4]                            # crash: green cymbal
            drums.append(Note(t, tuple(sorted(lanes)), cymbals=tuple(cym)))
        if bar % per == per - 1:                                    # a tom fill to end the section
            drums = [n for n in drums if n.tick < b0 + 2 * R]
            fills.append(Phrase(b0 + 2 * R, 2 * R))
            for k, lane in enumerate([2, 2, 3, 3, 4, 4, 4, 4]):
                drums.append(Note(b0 + 2 * R + k * R // 4, (lane,)))
        # guitar: 8th notes walking the pentatonic, a power chord on beat 1, a sustain to end the bar
        for e in range(7):
            t = b0 + e * R // 2
            fret = int(np.clip(fret + rng.choice([-1, 0, 1]), 0, 4))
            if e == 0:
                lo = min(fret, 3)
                guitar.append(Note(t, (lo, lo + 1)))
            else:
                guitar.append(Note(t, (fret,)))
        guitar.append(Note(b0 + 7 * R // 2, (fret,), length=R // 2 - R // 8))
        # bass: quarter notes on the root
        root = int(rng.integers(0, 3))
        for q in range(4):
            bass.append(Note(b0 + q * R, (root if q < 3 else min(4, root + 2),)))
        # vocals: a phrase every other bar, from the second section on
        if bar >= per and bar % 2 == 0:
            notes = []
            for k in range(6):
                t = b0 + k * R // 2
                pitch = int(60 + rng.choice([0, 2, 4, 5, 7]))
                ln = R // 2 - R // 16 if k < 5 else R * 2
                notes.append(VocalNote(t, ln, pitch, str(rng.choice(LYRICS))))
            vocals += notes
            phrases.append(Phrase(b0, 4 * R + 2 * R))
    song.parts["guitar"] = Part({"expert": guitar}, [Phrase(start + 4 * 4 * R, 4 * 4 * R)])
    song.parts["bass"] = Part({"expert": bass}, [Phrase(start + 4 * 4 * R, 4 * 4 * R)])
    song.parts["drums"] = Part({"expert": sorted(drums, key=lambda n: n.tick)},
                               [Phrase(start + 4 * 4 * R, 4 * 4 * R)],
                               [f for f in fills if f.tick > start + 8 * 4 * R])
    song.vocals = VocalPart(vocals, phrases, [phrases[1]] if len(phrases) > 1 else [])
    song.end = start + bars * 4 * R + 4 * R
    return song


# ------------------------------------------------------------------ synth

def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    a = np.exp(-2 * np.pi * cutoff / SR)
    from scipy.signal import lfilter
    return lfilter([1 - a], [1, -a], x)


def saw(f: float, n: int) -> np.ndarray:
    ph = (np.arange(n) * f / SR) % 1.0
    return 2 * ph - 1


def add(buf: np.ndarray, start: float, sig: np.ndarray) -> None:
    i = int(start * SR)
    j = min(len(buf), i + len(sig))
    if j > i:
        buf[i:j] += sig[: j - i]


def render(song: Song, seconds: float) -> dict[str, np.ndarray]:
    tm = song.tempo_map()
    n = int(seconds * SR)
    out = {k: np.zeros(n) for k in ("guitar", "bass", "drums", "vocals")}
    rng = np.random.default_rng(7)
    for name, pitches, level in (("guitar", GUITAR_PITCH, 0.25), ("bass", BASS_PITCH, 0.4)):
        for note in song.parts[name].notes["expert"]:
            t = tm.seconds(note.tick)
            dur = max(0.18, tm.seconds(note.tick + max(note.length, R // 3)) - t)
            m = int(dur * SR)
            sig = np.zeros(m)
            for lane in note.lanes:
                f = hz(pitches[lane] + (12 if name == "guitar" else 0))
                sig += saw(f, m) + 0.5 * saw(f * 1.005, m)
                if name == "guitar" and len(note.lanes) == 1:
                    sig += 0.6 * saw(f * 1.5, m)                    # a fifth on top: power-chord tone
            sig = lowpass(sig, 2500 if name == "guitar" else 900)
            add(out[name], t, level * sig * env(m, 0.003, dur * 0.6 + 0.05))
    for note in song.parts["drums"].notes["expert"]:
        t = tm.seconds(note.tick)
        for lane in note.lanes:
            if lane == 0:
                m = int(0.25 * SR)
                f = 50 + 100 * np.exp(-np.arange(m) / SR / 0.03)
                sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(m, 0.001, 0.12)
                add(out["drums"], t, 0.9 * sig)
            elif lane == 1:
                m = int(0.2 * SR)
                sig = (rng.standard_normal(m) * 0.6 + np.sin(2 * np.pi * 190 * np.arange(m) / SR)) * env(m, 0.001, 0.07)
                add(out["drums"], t, 0.5 * sig)
            elif lane in note.cymbals:
                d = 0.05 if lane == 2 else 0.6
                m = int((d * 4) * SR)
                noise = rng.standard_normal(m)
                sig = (noise - lowpass(noise, 6000)) * env(m, 0.001, d)
                add(out["drums"], t, 0.35 * sig)
            else:
                m = int(0.3 * SR)
                f0 = {2: 160, 3: 120, 4: 90}[lane]
                f = f0 * (1 + 0.5 * np.exp(-np.arange(m) / SR / 0.02))
                sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(m, 0.001, 0.15)
                add(out["drums"], t, 0.6 * sig)
    for v in song.vocals.notes:
        t = tm.seconds(v.tick)
        dur = tm.seconds(v.tick + v.length) - t
        m = int(dur * SR)
        tt = np.arange(m) / SR
        f = hz(v.pitch) * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * tt))
        ph = 2 * np.pi * np.cumsum(f) / SR
        sig = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
        e = np.minimum(1, tt / 0.02) * np.minimum(1, (dur - tt) / 0.04)
        add(out["vocals"], t, 0.3 * sig * e)
    return out


def write_ogg(path: Path, y: np.ndarray) -> None:
    import soundfile as sf
    y = np.clip(y, -1, 1).astype(np.float32)
    with sf.SoundFile(str(path), "w", samplerate=SR, channels=1, format="OGG", subtype="VORBIS") as f:
        for i in range(0, len(y), SR * 10):                 # in blocks: big Vorbis writes can crash
            f.write(y[i:i + SR * 10])


def make(out: Path, seconds: float = 40.0, seed: int = 1) -> tuple[Song, Path]:
    """Write the song folder and the full mix; returns (the chart, the mix's path)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    song = build(seconds, seed)
    write_mid(song, out / "notes.mid")
    chart = read_mid(out / "notes.mid")                     # render what the file says
    chart.name, chart.artist = song.name, song.artist
    length = chart.tempo_map().seconds(chart.end) + 1.0
    stems = render(chart, length)
    for name, y in stems.items():
        write_ogg(out / f"{name}.ogg", y)
    mix = out.with_name(out.name + ".mix.ogg")
    write_ogg(mix, sum(stems.values()) * 0.7)
    write_ini(song, out / "song.ini", int(length * 1000),
              difficulties={"guitar": 2, "bass": 1, "drums": 2, "vocals": 1})
    return song, mix


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    _, mix = make(a.out, a.seconds, a.seed)
    print(f"song folder {a.out}, full mix {mix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
