# Continue here

A handoff for any coding agent (Claude Code, OpenAI Codex, ...) picking up
this repo. Read this first, then the [README](README.md). Last updated
2026-10-25.

**If you can read Paul's private repo `paul-lapczynski/rhythm-agent`, start
there:** its `AGENTS.md` and its issues are the up-to-date task list,
and when you finish, update the issue and its history. This file is the
fallback for when you can't.

## What this repo is

RhythmSmith is a monorepo for rhythm-game chart tools. The work here is
**BandSmith**: it takes a song and writes a Clone Hero / Rock Band song folder
(`notes.mid` and `song.ini`) with guitar, bass, drums and vocals charts on
Expert, Hard, Medium and Easy. Python, PySide6 app. BeatSmith (the StepMania
generator) lives in its own repo, `paul-lapczynski/beatsmith`.

## Hard rules

1. **Never push to `main`.** Every change goes on its own branch and through a
   pull request. Paul reviews and squash-merges; never merge a PR yourself.
2. **Never delete a repository**, and never force-push `main` or delete
   someone else's branch.
3. **Never commit songs, audio, artwork, commercial (Rock Band, Guitar Hero,
   Konami) charts or fan charts**, or anything from `training-data/`. That
   includes generated charts and eval reports, since they name songs.
   `.gitignore` covers most of this; check `git status` before every commit
   and stage files by name, not with `git add -A`.
4. **Charting rules for guitar and bass:**
   - **Expert matches what the instrument plays, literally.** The
     transcription is the truth; Expert never adds notes the instrument
     doesn't play (no filler, no invented notes to make it "feel" busier).
   - **Lower difficulties only remove notes** from the one above (and may
     simplify chords). They never add or move notes in time.
   - **The same note repeated keeps the same button**, in every difficulty
     and across phrase breaks. The same chord keeps the same frets. A test
     enforces this (`repeat_switches` must be 0); never weaken it.
5. Write docs, PRs and release notes in plain, short sentences for a
   non-technical reader. Start each PR body with the attribution block the
   existing PRs use.
6. Heavy jobs (stem separation, transcription, training) on Paul's PC must
   leave at least 2.5 GB of RAM free during the day.

## Current state

- PRs #1 to #8 are merged. #8 is "transcription as the truth" for guitar and
  bass.
- **Three stacked PRs are open, all green, to be merged in order:**
  - #9 Phrase pass (solos, fret paths with resets, quick notes, held
    sustains), base `main`. It **conflicts with `main`**, because #8 was
    squash-merged under it: rebase `phase1/phrase-pass` onto `main`
    (keep #8's version of shared code), push, and wait for green.
  - #10 v4 (rhythm chords stay, held notes, solo sections, experimental
    stereo split), base `phase1/phrase-pass`.
  - #11 v5 (the same note never switches buttons, slides, quick notes in
    fast runs), base `phase1/v4`.
  After Paul squash-merges each one, retarget the next PR to `main`, rebase
  it onto `main`, and check it is green again before he merges it.
- Paul is play-testing v5 in Clone Hero ("Milwaukee" and "Rock'n Me" in his
  "BandSmith Test" song folder). Ask him how the slides and fast parts feel.
- v5 on the 26 held-out songs, Expert: guitar precision 0.832, recall 0.874;
  bass precision 0.828, recall 0.914 (details in PR #11).
- No BandSmith release yet (versions stay below 1.0).

## Next tasks, in order

1. **Get #9, #10 and #11 merged**: fix the #9 conflict, then rebase each
   follow-up as above. Don't change behaviour while doing this.
2. **Hybrid arranger.** Today the rules (`bandsmith/analyze/music.py` and
   `charting.py`) decide frets, chord size, HOPOs and sustains. The plan:
   keep the transcription deciding *which notes exist and when* (rule 4),
   and let a small learned model decide *chord size and HOPO* for each note,
   with the rules as the fallback when no model is installed (like
   `analyze/ai_model.py` does for the opt-in chart model). Paul has
   early work for this on his PC, outside git. The model must never add or
   drop notes on Expert, and the same-note rule still applies on top of it.
   Compare against v5 on the held-out songs with
   `bandsmith/tools/eval_vs_reference.py` and only ship it if it is better.
3. Act on Paul's v5 play-test notes.
4. The stereo lead trick (`pan_solos`) stays off by default: it was right on
   Milwaukee but wrong more often than right elsewhere.
5. BandSmith still uses BeatSmith's icon (`bandsmith/packaging/BUILD.md`).

## Layout

See the README's "Layout" section. In short: `core/` is shared code (stems,
GPU choice, add-ons, lyrics, artwork, beats, notes); `bandsmith/bandsmith/`
is the app (`chart/` reads and writes charts, `analyze/` does the music);
`bandsmith/tests/` makes its own synthetic songs; `bandsmith/tools/` has
`demo_song.py`, `eval_vs_reference.py`, `inventory.py` and the `scan-chart`
format checker (Node).

## Run and test

```bash
python -m venv .venv
.venv/Scripts/activate        # macOS/Linux: . .venv/bin/activate
pip install -r bandsmith/requirements.txt -e core pytest
npm ci --prefix bandsmith/tools/scan-chart     # for the chart format tests
cd bandsmith
python -m pytest tests
python -m bandsmith make "Some Song.mp3" -o "out/Some Song"
python -m bandsmith          # the app window
```

All tests must pass (46 on v5), and generated charts must pass scan-chart.
Evaluation against reference charts runs locally only (the songs can't be in
git): `python bandsmith/tools/eval_vs_reference.py SONGS_DIR --out REPORTS_DIR`.
Keep `REPORTS_DIR` out of git.

## Building and releasing

`.github/workflows/bandsmith.yml` builds Windows and macOS for changes to
`bandsmith/` or `core/`, on PRs and on `main`. Installers: see
`bandsmith/packaging/BUILD.md`. A release is a tag named `bandsmith-vX.Y.Z`
on a green `main` commit (for example `bandsmith-v0.1.0`); only tag when
Paul asks for a release.

## Working style

- One PR per change, on a branch named like `phase1/...`, `fix/...`,
  `docs/...`. Stack PRs only when one really needs the other, and say so in
  the body.
- Back changes to the charting with numbers from the held-out songs and a
  test for any new rule.
- When unsure what Paul wants, ask before building; propose a plan for
  anything large.
