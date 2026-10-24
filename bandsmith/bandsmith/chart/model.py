"""The song model BandSmith charts into, independent of the file format.

Times are in ticks at RESOLUTION ticks per beat (480, as Rock Band and Clone
Hero .mid files use); TempoMap turns them into seconds and back.

Lanes:
    guitar, bass   0 green, 1 red, 2 yellow, 3 blue, 4 orange
    drums          0 kick, 1 red (snare), 2 yellow, 3 blue, 4 green;
                   2-4 are toms unless listed in Note.cymbals (pro drums)
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

RESOLUTION = 480
DIFFICULTIES = ("expert", "hard", "medium", "easy")
GUITAR, BASS, DRUMS, VOCALS = "guitar", "bass", "drums", "vocals"
FRETTED = (GUITAR, BASS)
INSTRUMENTS = (GUITAR, BASS, DRUMS)


@dataclass
class Tempo:
    tick: int
    bpm: float


@dataclass
class TimeSig:
    tick: int
    numerator: int = 4
    denominator: int = 4


@dataclass
class Section:
    tick: int
    name: str


@dataclass
class Phrase:
    """A span: star power, a vocal phrase."""
    tick: int
    length: int


@dataclass
class Note:
    tick: int
    lanes: tuple[int, ...]
    length: int = 0                     # sustain in ticks; 0 for none
    forced: str | None = None           # 5-fret: "hopo" or "strum"
    cymbals: tuple[int, ...] = ()       # drums: which of lanes 2-4 are cymbals
    double_kick: bool = False           # drums: the kick is Expert+ (2x kick) only


@dataclass
class VocalNote:
    tick: int
    length: int
    pitch: int | None                   # MIDI note 36-84; None for unpitched (spoken)
    lyric: str                          # a syllable; "-" at the end joins it to the next


@dataclass
class Part:
    notes: dict[str, list[Note]] = field(default_factory=dict)      # difficulty -> notes
    star_power: list[Phrase] = field(default_factory=list)
    activations: list[Phrase] = field(default_factory=list)       # drums: fills that activate star power


@dataclass
class VocalPart:
    notes: list[VocalNote] = field(default_factory=list)
    phrases: list[Phrase] = field(default_factory=list)
    star_power: list[Phrase] = field(default_factory=list)


@dataclass
class Song:
    name: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: str = ""
    charter: str = "BandSmith"
    resolution: int = RESOLUTION
    tempos: list[Tempo] = field(default_factory=lambda: [Tempo(0, 120.0)])
    time_sigs: list[TimeSig] = field(default_factory=lambda: [TimeSig(0)])
    sections: list[Section] = field(default_factory=list)
    parts: dict[str, Part] = field(default_factory=dict)            # guitar, bass, drums
    vocals: VocalPart | None = None
    music_start: int | None = None      # ticks; None: the first note
    end: int | None = None              # ticks; None: after the last note

    def tempo_map(self) -> "TempoMap":
        return TempoMap(self.tempos, self.resolution)

    def last_tick(self) -> int:
        last = 0
        for part in self.parts.values():
            for notes in part.notes.values():
                for n in notes:
                    last = max(last, n.tick + n.length)
        if self.vocals:
            for v in self.vocals.notes:
                last = max(last, v.tick + v.length)
        return last

    def first_tick(self) -> int | None:
        ticks = [n.tick for p in self.parts.values() for ns in p.notes.values() for n in ns[:1]]
        if self.vocals and self.vocals.notes:
            ticks.append(self.vocals.notes[0].tick)
        return min(ticks) if ticks else None


class TempoMap:
    """Ticks to seconds and back, from tempo changes."""

    def __init__(self, tempos: list[Tempo], resolution: int = RESOLUTION):
        self.res = resolution
        ts = sorted(tempos, key=lambda t: t.tick)
        if not ts or ts[0].tick != 0:
            ts.insert(0, Tempo(0, 120.0))
        self.ticks = [t.tick for t in ts]
        self.spt = [60.0 / t.bpm / resolution for t in ts]           # seconds per tick
        self.secs = [0.0]
        for i in range(1, len(ts)):
            self.secs.append(self.secs[-1] + (self.ticks[i] - self.ticks[i - 1]) * self.spt[i - 1])

    def seconds(self, tick: float) -> float:
        i = max(0, bisect.bisect_right(self.ticks, tick) - 1)
        return self.secs[i] + (tick - self.ticks[i]) * self.spt[i]

    def tick(self, seconds: float) -> float:
        """The (fractional) tick at a time."""
        i = max(0, bisect.bisect_right(self.secs, seconds) - 1)
        return self.ticks[i] + (seconds - self.secs[i]) / self.spt[i]

    def bpm_at(self, tick: float) -> float:
        i = max(0, bisect.bisect_right(self.ticks, tick) - 1)
        return 60.0 / (self.spt[i] * self.res)
