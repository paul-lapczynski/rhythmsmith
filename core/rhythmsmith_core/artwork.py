"""Song artwork for the StepMania song folder: banner, background and jacket.

Where the picture comes from, in order:
  1. an image next to the song: the same name as the song (Song.jpg for
     Song.mp3), else cover/folder/front/album .jpg or .png
  2. art made on an earlier run, kept as it is (so hand edits survive and
     nothing is fetched twice; delete the files to make them again)
  3. the cover art inside the song file's own tags (MP3, FLAC, Ogg, M4A)
  4. only when asked (--art-online): looked up online by title and artist:
     the song's cover, its album's, another by the artist, or a photo of the
     artist, from MusicBrainz / Cover Art Archive, iTunes and Deezer.
     Each address is printed before it's fetched.

From the one picture it writes <title>-bn.png (418x164 banner),
<title>-bg.png (640x480 background, dimmed so the arrows stay readable)
and <title>-jacket.png (512x512). Resizing uses Qt (in the app) or Pillow;
with neither, the picture is copied as it is and StepMania stretches it.

    python -m rhythmsmith_core.artwork song.mp3 [--online] [--artist A] [--title T] [--out DIR]

Each app names itself to the web services with set_app() (BeatSmith by default).
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BANNER = (418, 164)
BACKGROUND = (640, 480)
JACKET = (512, 512)
NEXT_TO_SONG = ("cover", "folder", "front", "album", "albumart")
IMAGE_EXT = (".jpg", ".jpeg", ".png")
MIN_SIZE = 200          # smaller than this looks like mush on a 640x480 background

def app_version() -> str:
    """The version built into the app (version.txt, written by the build),
    else the newest release tag of a source checkout, else "dev"."""
    here = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    try:
        return (here / "version.txt").read_text().strip() or "dev"
    except OSError:
        pass
    try:
        import subprocess
        tag = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], cwd=here, capture_output=True,
                             text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        return tag.lstrip("v") or "dev"
    except (OSError, subprocess.SubprocessError):
        return "dev"


# MusicBrainz requires one naming the application, its version and a contact
USER_AGENT = f"BeatSmith/{app_version()} ( https://github.com/paul-lapczynski/beatsmith )"


def set_app(name: str, version: str, url: str) -> None:
    """Name the app using this module in what it tells web services about itself."""
    global USER_AGENT
    USER_AGENT = f"{name}/{version} ( {url} )"
MUSICBRAINZ = "https://musicbrainz.org/ws/2/recording/"
COVER_ART_ARCHIVE = "https://coverartarchive.org"
ITUNES = "https://itunes.apple.com/search"
DEEZER = "https://api.deezer.com/search"
# for the app's tooltip and the README: every place an online lookup reaches
ONLINE_SOURCES = [
    ("Song search by title and artist", "https://musicbrainz.org (MusicBrainz)"),
    ("Cover pictures for what it finds", "https://coverartarchive.org (Cover Art Archive; "
                                         "the files come from archive.org)"),
    ("When MusicBrainz has none", "https://itunes.apple.com (iTunes Search API; "
                                  "the pictures come from Apple's mzstatic.com)"),
    ("Then, and for a photo of the artist when no cover is found", "https://api.deezer.com (Deezer API; "
                                                                   "the pictures come from dzcdn.net)"),
]


def file_names(title: str) -> dict[str, str]:
    """The .sm tag for each picture and its file name in the song folder."""
    return {"BANNER": f"{title}-bn.png", "BACKGROUND": f"{title}-bg.png", "JACKET": f"{title}-jacket.png"}


# ------------------------------------------------------------------ finding

def next_to(src: Path) -> Path | None:
    """An image beside the song: its own name first, then an album cover."""
    for stem in (src.stem, *NEXT_TO_SONG):
        for ext in IMAGE_EXT:
            for name in (stem + ext, stem.capitalize() + ext, stem.upper() + ext):
                p = src.with_name(name)
                if p.is_file():
                    return p
    return None


def embedded(src: Path) -> bytes | None:
    """The front cover from the song file's tags, else any picture in them."""
    try:
        import mutagen
        f = mutagen.File(str(src))
    except Exception:
        return None
    if f is None:
        return None
    pics = []                                   # (is the front cover, bytes)
    tags = getattr(f, "tags", None) or {}
    try:
        for key in list(tags.keys()):
            v = tags[key]
            if key.startswith("APIC"):                          # MP3 (ID3)
                pics.append((v.type == 3, v.data))
            elif key == "covr":                                 # M4A
                pics += [(True, bytes(c)) for c in v]
            elif key.lower() == "metadata_block_picture":       # Ogg / Opus
                from mutagen.flac import Picture
                for b64 in v:
                    p = Picture(base64.b64decode(b64))
                    pics.append((p.type == 3, p.data))
    except Exception:
        pass
    for p in getattr(f, "pictures", []) or []:                  # FLAC
        pics.append((p.type == 3, p.data))
    pics.sort(key=lambda x: not x[0])
    return pics[0][1] if pics else None


def _norm(s: str) -> str:
    s = re.sub(r"\s*[\(\[].*?[\)\]]", "", s.casefold())         # "(Radio Edit)", "[feat. X]"
    s = re.sub(r"\b(feat|ft)\..*$", "", s)
    return re.sub(r"[^\w]+", "", s)


def _same(a: str, b: str) -> bool:
    """Names that read the same, counting what's in brackets: "타샤니 (TASHANNIE)"
    is TASHANNIE."""
    return _title_is(a, b)


_MB_LAST = Path(__import__("tempfile").gettempdir()) / "beatsmith-musicbrainz-last"
MB_GAP = 1.1         # seconds between MusicBrainz calls: their rule is at most one a second


def _musicbrainz_turn():
    """MusicBrainz allows at most one call a second from an application,
    retries included, and several songs can be looking at once (the app and
    each chart being made). They take turns under a lock file, and a time
    stamp file says when the last call went out."""
    import os
    lock = _MB_LAST.with_name(_MB_LAST.name + ".lock")
    start = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 30:     # left behind by a crash
                    lock.unlink()
            except OSError:
                pass
        except OSError:
            fd = None                      # no temp folder to share: this process alone keeps the gap
            break
        if time.time() - start > 120:
            fd = None
            break
        time.sleep(0.1)
    try:
        try:
            gap = time.time() - _MB_LAST.stat().st_mtime
        except OSError:
            gap = MB_GAP
        if gap < MB_GAP:
            time.sleep(MB_GAP - gap)
        try:
            _MB_LAST.touch()
        except OSError:
            pass
    finally:
        if fd is not None:
            os.close(fd)
            try:
                lock.unlink()
            except OSError:
                pass


def _get(url: str, log, accept: str = "application/json") -> bytes | None:
    log(f"  artwork: fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    mb = urllib.parse.urlparse(url).hostname == "musicbrainz.org"
    for attempt in range(4):
        if mb:
            _musicbrainz_turn()
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 3:       # too many requests: wait and try again
                try:
                    wait = min(15.0, float(e.headers.get("Retry-After") or 0)) or 2.0 * 2 ** attempt
                except ValueError:
                    wait = 2.0 * 2 ** attempt
                log(f"  artwork: {urllib.parse.urlparse(url).hostname} is busy ({e.code}), "
                    f"trying again in {wait:g} s")
                time.sleep(wait)
                continue
            if e.code != 404:
                log(f"  artwork: {e.code} {e.reason}")
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log(f"  artwork: couldn't reach it ({getattr(e, 'reason', e)})")
            return None
    return None


def _clean(s: str) -> str:
    """A title as a search wants it: no "(CAUTION)", "[Official Video]",
    "feat. X", track numbers or underscores."""
    s = re.sub(r"\s*[\(\[\{].*?[\)\]\}]", " ", s.replace("_", " "))
    s = re.sub(r"\s+\b(feat|ft|featuring)\b\.?\s.*$", "", s, flags=re.I)
    s = re.sub(r"^\s*\d{1,3}\s*[.\-)]?\s+(?=\D)", "", s)         # "01 - ", "07. "
    return re.sub(r"\s+", " ", s).strip(" -")


def guesses(title: str, artist: str) -> list[tuple[str, str]]:
    """(title, artist) pairs to search for, best first. Without an artist,
    one is guessed from the name ("Artist - Title", "ARTIST-TITLE"), and
    last of all the title is searched on its own ("" for the artist)."""
    t, a = _clean(title), _clean(artist)
    out = []
    if a:
        out.append((t, a))
    for sep in (r"\s+[-\u2013\u2014]\s+", r"\s*[-\u2013\u2014]\s*"):
        parts = re.split(sep, t, maxsplit=1)
        if len(parts) == 2 and all(parts):
            if not a:
                out += [(parts[1], parts[0]), (parts[0], parts[1])]    # "Artist - Title", then the other way
            elif _same(parts[0], a):
                out.append((parts[1], a))                              # the artist written in the title too
            break
    # what's in brackets may be the real title: "TASHANNIE-DONT BOTHER ME (CAUTION)" is
    # Tashannie's "경고 (Caution)"
    extra = [x.strip() for x in re.findall(r"[\(\[](.*?)[\)\]]", title) if x.strip()]
    for x in extra:
        for _, who in list(out):
            if who:
                out.append((x, who))
    if not a:
        out.append((t, ""))
    seen, res = set(), []
    for g in out:
        k = (_norm(g[0]), _norm(g[1]))
        if k[0] and k not in seen:
            seen.add(k)
            res.append(g)
    return res[:6]


def _lucene(s: str) -> str:
    return re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/\']', " ", s).strip()


def _exact(a: str, b: str) -> bool:
    return bool(_norm(a)) and _norm(a) == _norm(b)


def _forms(title: str) -> set[str]:
    """A title and the names in its brackets: a song in another language is
    often listed as "건들지마 (Don't Bother Me)" or "Don't Bother Me (건들지마)"."""
    out = {_norm(title)} | {_norm(x) for x in re.findall(r"[\(\[](.*?)[\)\]]", title)}
    return {f for f in out if f}


def _title_is(found: str, wanted: str, exact: bool = False) -> bool:
    for a in _forms(found):
        for b in _forms(wanted):
            if a == b or (not exact and (a.startswith(b) or b.startswith(a))):
                return True
    return False


_searched: dict[str, bytes | None] = {}     # MusicBrainz answers during one online(), asked once


def musicbrainz(title: str, artist: str, log, loose: bool = False) -> tuple[bytes, str, str | None] | None:
    """loose: when no title matches, the artist's best match (a song listed
    under another language's title, say), flagged to check."""
    q = f"recording:({_lucene(title)})" + (f" AND artist:({_lucene(artist)})" if artist else "")
    url = f"{MUSICBRAINZ}?{urllib.parse.urlencode({'query': q, 'fmt': 'json', 'limit': 15})}"
    if url not in _searched:
        _searched[url] = _get(url, log)
    data = _searched[url]
    try:
        recs = json.loads(data)["recordings"] if data else []
    except (ValueError, KeyError):
        return None
    groups, releases, artist_only = [], [], []
    for rec in recs:
        credit = "".join(c.get("name", "") + c.get("joinphrase", "") for c in rec.get("artist-credit", []))
        if artist:
            ok = rec.get("score", 0) >= 60 and _title_is(rec.get("title", ""), title) and _same(credit, artist)
            if not ok and _title_is(credit, artist, exact=True) and rec.get("releases"):
                # the artist for sure, but a title that doesn't read the same (another
                # language, another spelling): kept as a last resort
                artist_only.append((credit, rec))
        else:                       # the title alone: only an exact, confident match
            ok = rec.get("score", 0) >= 95 and _title_is(rec.get("title", ""), title, exact=True)
        if not ok:
            continue
        for rel in rec.get("releases", []):
            rg = rel.get("release-group", {})
            # the artist's own albums and singles before compilations and live records
            plain = rel.get("status") == "Official" and not rg.get("secondary-types")
            rank = (not plain, rg.get("primary-type") not in ("Album", "Single", "EP"))
            what = f"{credit} - {rec.get('title', '')}"
            if rg.get("id"):
                groups.append((rank, rg["id"], what))
            releases.append((rank, rel["id"], what))
        if not artist and (groups or releases):
            break                   # without an artist, only the best match's covers
    note = None
    if loose and not groups and not releases and artist_only:
        credit, rec = artist_only[0]           # MusicBrainz's best guess for the artist
        note = "a cover by the same artist, not this song's: check it"
        for rel in rec["releases"]:
            what = f"{credit} - {rec.get('title', '')}"
            if rel.get("release-group", {}).get("id"):
                groups.append(((False, False), rel["release-group"]["id"], what))
            releases.append(((False, False), rel["id"], what))
    for kind, ids in (("release-group", groups), ("release", releases)):
        seen = []
        for _, i, what in sorted(ids, key=lambda x: x[0]):
            if i not in [x for x, _ in seen]:
                seen.append((i, what))
        for i, what in seen[:3]:
            img = _get(f"{COVER_ART_ARCHIVE}/{kind}/{i}/front-500", log, accept="image/*")
            if img:
                return img, what, note
    return None


ITUNES_STORES = ("us", "kr", "jp")     # US first; Korean and Japanese stores list K-pop and J-pop in full


def itunes(title: str, artist: str, log) -> tuple[bytes, str, str | None] | None:
    term = f"{artist} {title}".strip()
    for store in ITUNES_STORES:
        data = _get(f"{ITUNES}?{urllib.parse.urlencode({'term': term, 'entity': 'song', 'limit': 15, 'country': store})}",
                    log)
        try:
            results = json.loads(data)["results"] if data else []
        except (ValueError, KeyError):
            continue
        for r in results:
            url = r.get("artworkUrl100", "")
            track, by = r.get("trackName", ""), r.get("artistName", "")
            ok = (_title_is(track, title) and _same(by, artist)) if artist else _title_is(track, title, exact=True)
            if url and ok:
                img = _get(url.replace("100x100bb", "600x600bb"), log, accept="image/*")
                if img:
                    return img, f"{by} - {track}", None
    return None


def _json(url: str, log, key: str) -> list:
    data = _get(url, log)
    try:
        return json.loads(data).get(key) or [] if data else []
    except ValueError:
        return []


def deezer_song(title: str, artist: str, log):
    q = f'artist:"{artist}" track:"{title}"' if artist else title
    for r in _json(f"{DEEZER}?{urllib.parse.urlencode({'q': q, 'limit': 15})}", log, "data"):
        track, by, cover = r.get("title", ""), r.get("artist", {}).get("name", ""), r.get("album", {}).get("cover_xl")
        ok = (_title_is(track, title) and _same(by, artist)) if artist else _title_is(track, title, exact=True)
        if cover and ok:
            img = _get(cover, log, accept="image/*")
            if img:
                return img, f"{by} - {track}", None
    return None


def musicbrainz_album(album: str, artist: str, log):
    q = f"releasegroup:({_lucene(album)}) AND artist:({_lucene(artist)})"
    for rg in _json(f"https://musicbrainz.org/ws/2/release-group/?"
                    f"{urllib.parse.urlencode({'query': q, 'fmt': 'json', 'limit': 10})}", log, "release-groups"):
        credit = "".join(c.get("name", "") + c.get("joinphrase", "") for c in rg.get("artist-credit", []))
        if rg.get("score", 0) >= 60 and _title_is(rg.get("title", ""), album) and _same(credit, artist):
            img = _get(f"{COVER_ART_ARCHIVE}/release-group/{rg['id']}/front-500", log, accept="image/*")
            if img:
                return img, f"{credit} - {rg.get('title', '')} (album)", None
    return None


def itunes_album(album: str, artist: str, log):
    for store in ITUNES_STORES:
        query = {"term": f"{artist} {album}", "entity": "album", "limit": 15, "country": store}
        for r in _json(f"{ITUNES}?{urllib.parse.urlencode(query)}", log, "results"):
            name, by, url = r.get("collectionName", ""), r.get("artistName", ""), r.get("artworkUrl100", "")
            if url and _title_is(name, album) and _same(by, artist):
                img = _get(url.replace("100x100bb", "600x600bb"), log, accept="image/*")
                if img:
                    return img, f"{by} - {name} (album)", None
    return None


def deezer_album(album: str, artist: str, log):
    q = 'artist:"%s" album:"%s"' % (artist, album)
    for r in _json(f"{DEEZER}/album?{urllib.parse.urlencode({'q': q, 'limit': 15})}", log, "data"):
        name, by, cover = r.get("title", ""), r.get("artist", {}).get("name", ""), r.get("cover_xl")
        if cover and _title_is(name, album) and _same(by, artist):
            img = _get(cover, log, accept="image/*")
            if img:
                return img, f"{by} - {name} (album)", None
    return None


def deezer_artist(artist: str, log):
    for r in _json(f"{DEEZER}/artist?{urllib.parse.urlencode({'q': artist, 'limit': 10})}", log, "data"):
        pic = r.get("picture_xl") or ""
        # an artist without a photo gets Deezer's blank one, with no picture id in its address
        if pic and "/artist//" not in pic and _title_is(r.get("name", ""), artist, exact=True):
            img = _get(pic, log, accept="image/*")
            if img:
                return img, f"{r['name']} (artist photo)", "no cover found for the song or its album"
    return None


def online(title: str, artist: str, log=print, album: str = "") -> tuple[bytes, dict] | None:
    """Search by title and artist, trying a few readings of the name: the
    song's cover, else its album's (from the album tag), else another cover by
    the same artist, else a photo of the artist. The record says what it
    matched, to check in the Details tab."""
    _searched.clear()
    MB, IT, DZ = "MusicBrainz / Cover Art Archive", "iTunes", "Deezer"
    tries = guesses(title, artist)
    artists = list(dict.fromkeys(a for _, a in tries if a))
    album = _clean(album)
    steps = []
    for t, a in tries:                                   # the song itself
        steps += [(f"{a + ' - ' if a else ''}{t}" + ("" if a else " (title only)"), MB, musicbrainz, (t, a)),
                  ("", IT, itunes, (t, a)), ("", DZ, deezer_song, (t, a))]
    for a in artists if album else []:                   # its album
        steps += [(f"the album {a} - {album}", MB, musicbrainz_album, (album, a)),
                  ("", IT, itunes_album, (album, a)), ("", DZ, deezer_album, (album, a))]
    for t, a in tries:                                   # another cover by the same artist
        if a:
            steps.append((f"any cover by {a}", MB, musicbrainz, (t, a, None, True)))
    for a in artists:                                    # a photo of the artist
        steps.append((f"a photo of {a}", DZ, deezer_artist, (a,)))
    for what, name, fn, args in steps:
        if what:
            log(f"  artwork: looking for {what}")
        if len(args) == 4:                               # the loose MusicBrainz search
            found = fn(args[0], args[1], log, loose=True)
        else:
            found = fn(*args, log)
        if found and _size(found[0])[0] >= MIN_SIZE:
            where = {"from": name, "match": found[1]}
            if found[2]:
                where["note"] = found[2]
            elif args[-1] == "":
                where["note"] = "matched by title only: check it's the right song"
            return found[0], where
    log(f"  artwork: nothing found online for {title}")
    return None


# ------------------------------------------------------------------ drawing

def _size(data: bytes) -> tuple[int, int]:
    """Width and height of a PNG or JPEG without decoding it; (0, 0) if neither."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    i = 2
    if data[:2] != b"\xff\xd8":
        return 0, 0
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker, n = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + n
    return 0, 0


def _layouts():
    """For each picture: (size, the height the sharp picture gets in it,
    brightness of the blurred fill behind it, brightness of the picture)."""
    return {"BANNER": (BANNER, BANNER[1], 0.55, 1.0),
            "BACKGROUND": (BACKGROUND, 360, 0.35, 0.8),
            "JACKET": (JACKET, JACKET[1], 1.0, 1.0)}


def _plan(iw: int, ih: int, w: int, h: int, side: int) -> tuple[bool, tuple[int, int, int, int]]:
    """Whether the picture fills the frame by itself (it's about the frame's
    shape already, e.g. a wide picture for the banner, so little is cut
    off), and else where the whole picture goes, centered, with nothing
    cropped (no cutting through text on a cover): (x, y, width, height)."""
    if 0.8 <= (iw / ih) / (w / h) <= 1.25:
        return True, (0, 0, w, h)
    k = min(side / ih, w / iw)
    pw, ph = max(1, round(iw * k)), max(1, round(ih * k))
    return False, ((w - pw) // 2, (h - ph) // 2, pw, ph)


def _draw_qt(data: bytes, out: dict[str, Path]) -> bool:
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    img = QImage.fromData(data)
    if img.isNull():
        return False
    img = img.convertToFormat(QImage.Format_RGB32)
    for tag, ((w, h), side, fill, front) in _layouts().items():
        canvas = QImage(w, h, QImage.Format_RGB32)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        # the picture filling the frame (cropped to its middle)
        big = img.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        big = big.copy((big.width() - w) // 2, (big.height() - h) // 2, w, h)
        whole, (x, y, pw, ph) = _plan(img.width(), img.height(), w, h, side)
        if whole:
            p.drawImage(QRect(0, 0, w, h), big)
            if front < 1:
                p.fillRect(0, 0, w, h, QColor(0, 0, 0, round(255 * (1 - front))))
        else:
            # behind: that, blurred (shrunk, then grown back) and dimmed; in front: the whole picture, centered
            blur = big.scaled(max(1, w // 24), max(1, h // 24), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            p.drawImage(QRect(0, 0, w, h), blur)
            if fill < 1:
                p.fillRect(0, 0, w, h, QColor(0, 0, 0, round(255 * (1 - fill))))
            p.drawImage(QRect(x, y, pw, ph), img.scaled(pw, ph, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            if front < 1:
                p.fillRect(x, y, pw, ph, QColor(0, 0, 0, round(255 * (1 - front))))
        p.end()
        canvas.save(str(out[tag]), "PNG")
    return True


def _draw_pil(data: bytes, out: dict[str, Path]) -> bool:
    import io
    from PIL import Image, ImageEnhance, ImageOps
    img = Image.open(io.BytesIO(data)).convert("RGB")
    for tag, ((w, h), side, fill, front) in _layouts().items():
        big = ImageOps.fit(img, (w, h), Image.LANCZOS)
        whole, (x, y, pw, ph) = _plan(img.width, img.height, w, h, side)
        if whole:
            canvas = ImageEnhance.Brightness(big).enhance(front) if front < 1 else big
        else:
            canvas = big.resize((max(1, w // 24), max(1, h // 24)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
            if fill < 1:
                canvas = ImageEnhance.Brightness(canvas).enhance(fill)
            pic = img.resize((pw, ph), Image.LANCZOS)
            if front < 1:
                pic = ImageEnhance.Brightness(pic).enhance(front)
            canvas.paste(pic, (x, y))
        canvas.save(out[tag], "PNG")
    return True


def draw(data: bytes, out: dict[str, Path]) -> None:
    """Write the banner, background and jacket from one picture."""
    for fn in (_draw_qt, _draw_pil):
        try:
            if fn(data, out):
                return
        except ImportError:
            continue
    # no way to resize: the picture as it is, under each name
    for path in out.values():
        path.write_bytes(data)


# ------------------------------------------------------------------ all of it

KEPT = "already in the song folder"


def describe(where: dict) -> str:
    """Where a picture came from, in words."""
    out = where.get("from") or "nowhere"
    if where.get("file"):
        out += f" ({where['file']})"
    if where.get("match"):
        out += f", matched to \u201c{where['match']}\u201d"
    if where.get("note"):
        out += f" ({where['note']})"
    return out


def album_of(src: Path) -> str:
    try:
        import mutagen
        f = mutagen.File(str(src), easy=True)
        return ((f.tags or {}).get("album") or [""])[0].strip() if f else ""
    except Exception:
        return ""


def find(src: Path, title: str, artist: str, song: Path | None = None, look_online: bool = False,
         log=print, name: str | None = None) -> tuple[bytes | None, dict]:
    """The picture for a song and where it came from. With the song folder
    given, art already in it is kept: (None, {"from": KEPT})."""
    pic = next_to(src)
    if pic is not None:
        # put there by hand for this song: always used
        return pic.read_bytes(), {"from": "image next to the song", "file": pic.name}
    if song is not None and all((song / n).is_file() for n in file_names(name or title).values()):
        # an earlier run's, or hand-made: kept, and nothing fetched again
        return None, {"from": KEPT}
    data = embedded(src)
    if data is not None:
        return data, {"from": "the song file's cover art"}
    if look_online:
        print("progress: finding artwork", file=sys.stderr, flush=True)
        found = online(title, artist, log, album_of(src))
        if found:
            return found
    return None, {"from": None}


def make(src: Path, song: Path, title: str, artist: str, look_online: bool = False,
         log=print, image: Path | None = None, image_from: str | None = None,
         name: str | None = None) -> tuple[dict[str, str], dict]:
    """Find a picture for the song and write its art into the song folder.
    `image` is one already chosen (the app finds it when the song is added).

    Returns the .sm tags to fill ({"BANNER": file name, ...}, empty when
    there is none) and a record of where it came from."""
    names = file_names(name or title)          # name: the song's file name (layout.file_name)
    if image is not None:
        data, where = image.read_bytes(), {"from": image_from or "chosen in the app"}
    else:
        data, where = find(src, title, artist, song, look_online, log, name)
    if where["from"] == KEPT:
        log("  artwork: kept what's in the song folder (delete it to make it again)")
        return names, where
    if data is None:
        return {}, where
    draw(data, {tag: song / n for tag, n in names.items()})
    log(f"  artwork: from {describe(where)}")
    return names, where


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--title")
    ap.add_argument("--artist", default="")
    ap.add_argument("--online", action="store_true", help="look it up online when the song has none")
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    names, where = make(args.audio, args.out, args.title or args.audio.stem, args.artist, args.online)
    print(json.dumps({"tags": names, **where}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
