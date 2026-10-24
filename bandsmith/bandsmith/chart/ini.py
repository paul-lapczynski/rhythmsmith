"""song.ini for a Clone Hero / YARG song folder."""
from __future__ import annotations

from pathlib import Path

from .model import Song

APP_ICON = ""           # no icon of our own yet


def ini_text(song: Song, song_length_ms: int, difficulties: dict[str, int] | None = None,
             preview_ms: int | None = None, delay_ms: int = 0) -> str:
    """difficulties: instrument -> 0-6 tier ("guitar", "bass", "drums", "vocals");
    parts missing from the song get -1."""
    d = difficulties or {}
    has = {name: bool(p.notes.get("expert")) for name, p in song.parts.items()}
    has["vocals"] = bool(song.vocals and song.vocals.notes)
    # Clone Hero lists songs by these, and checkers flag empty ones
    lines = ["[song]",
             f"name = {song.name or 'Untitled'}",
             f"artist = {song.artist or 'Unknown Artist'}",
             f"album = {song.album or 'Unknown Album'}",
             f"genre = {song.genre or 'Unknown Genre'}",
             f"year = {song.year or 'Unknown Year'}",
             f"charter = {song.charter}",
             f"song_length = {int(song_length_ms)}",
             f"preview_start_time = {int(preview_ms if preview_ms is not None else song_length_ms * 0.35)}",
             f"delay = {int(delay_ms)}"]
    for inst in ("guitar", "bass", "drums", "vocals"):
        lines.append(f"diff_{inst} = {d.get(inst, 3) if has.get(inst) else -1}")
    lines.append(f"diff_band = {max([d.get(i, 3) for i in has if has[i]] or [-1])}")
    if has.get("drums"):
        lines.append("pro_drums = True")
    lines.append("loading_phrase = Charted by BandSmith")
    return "\n".join(lines) + "\n"


def write_ini(song: Song, path: Path, song_length_ms: int, **kw) -> None:
    Path(path).write_text(ini_text(song, song_length_ms, **kw), encoding="utf-8")
