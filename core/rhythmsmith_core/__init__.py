"""Shared parts of the RhythmSmith apps (BandSmith, BeatSmith):

    stems    split a song into stems with Demucs (htdemucs, or htdemucs_6s
             for guitar and piano too)
    accel    which processor the neural networks run on (CUDA, MPS, CPU)
    addons   the opt-in downloads, each showing where it comes from
    lyrics   read the words off a vocals stem with Whisper
    artwork  a song's cover art, from beside it, its tags or (when asked) online
"""
__version__ = "0.0.1"
