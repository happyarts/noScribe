# Diarization: the measured baseline, and four ideas that did not survive it

Written after a run with automatic speaker counting invented a third speaker on
a two-person recording and put a word from the middle of a sentence under it.
What shipped in the end is documented where it lives — the surplus-label report
in the comment above `GHOST_SPEAKER_MAX_SHARE` in `noScribe/main.py`, the cue fix
in `_segments_from_words` in `noScribe/voxtral_engine.py`, both pinned by
`tests/test_ghost_speaker.py` and `tests/test_cut_and_cue_quality.py`. This file
keeps only what the code cannot carry: the baseline that licenses those numbers,
and the approaches that were measured and rejected, so they are not re-had.

## Does our setup reproduce the published numbers?

VoxConverse v0.3 (free, CC-BY-4.0; audio from robots.ox.ac.uk, RTTM from
github.com/joonson/voxconverse). Scored with `pyannote.metrics` under pyannote's
own conditions: `collar=0`, `skip_overlap=False`, aggregated over the sums of all
files.

    Split   Files   Speech      our DER   pyannote (community-1)
    dev       216    20.3 h      7.18 %   --
    test      232    40.2 h     11.14 %   11.2 %

The model card names no split; it is **test** (dev is the easier one). That
validates the setup (`pyannote/`, community-1 with VBx+PLDA, on MPS) — from here
on, our own diarization measurements are worth something. Runtime was
14 s/file on dev, ~22 s on test; since the per-chunk embedding fast path
(`noScribe/pyannote_fast_embeddings.py`, 2026-08-27) it is ~6 s/file on dev
at a DER identical on every file.

VoxConverse rather than AMI because the question was the speaker *count*: AMI has
exactly four participants per session, so the axis under test is constant there,
while VoxConverse ranges from 1 to 21 speakers per recording. Its YouTube audio
is also closer to the Zoom recordings and podcasts this fork is used on.

**Automatic counting errs downwards.** Too few speakers in 62 of 216 dev and 93
of 232 test recordings, against too many in 16 and 30 — three to four times more
often. Anything that merges speakers therefore pushes in the direction the
counting already overshoots, which is why the surplus-label finding shipped as a
report rather than a repair.

## Four knobs that look like the fix and are not

- **`clustering.Fb` from 0.8 to 1.5.** Cured the recording that started this
  (3 → 2 clusters) and left three clean files bit-identical, but collapsed 14
  detected speakers to 12 on a 25-minute question round. `Fb` is the speaker
  regularization coefficient of the VB loop ("high values result in the VB
  inference dropping more speakers", Diez et al.); the 0.07/0.8/0.6 in
  `pyannote/config.yaml` are pyannote's tuned optima within search ranges that
  are only findable in the source. Wrong error direction, rejected.
- **Raising `clustering.threshold`.** 0.70 changed nothing; 0.80 (the top of the
  search range) left the spurious label in place and merely renamed it. Under VBx
  the threshold only initialises the agglomeration — the VB loop decides the
  final count.
- **`AgglomerativeClustering` with `min_cluster_size`.** The parameter counts
  embeddings, not length, and the spurious label is spread over many chunks. Even
  at 15 (maximum 20) the file stayed at 3 clusters, and leaving VBx gives up the
  tuned 4.x pipeline.
- **Voxtral's audio encoder as a second identifier.** Nearest-centroid over 109
  confident segments ≥ 3 s: `audio_tower` 58.2 % ±11.1, projector 82.3 % ±7.8 on
  a two-class problem. cos(speaker A, speaker B) = 0.993 against
  cos(segment, own centroid) = 0.93/0.96 — the speaker axis is buried in the
  content, which is what an ASR encoder is trained for.

Rejected for the same reason but documented in the code, because the numbers
belong next to the decision: **relabelling the spurious cluster in noScribe**
(see the comment above `GHOST_SPEAKER_MAX_SHARE` in `noScribe/main.py`).

## Two MLX candidates, surveyed and not pursued

mlx-audio carries `vad/sortformer` — NVIDIA's end-to-end diarization, natively on
MLX, where noScribe runs pyannote through torch on MPS and the embedding stage was
measured to dominate. (The cost argument has since weakened: the embedding stage
lost two thirds of its work to the per-chunk fast path in
`noScribe/pyannote_fast_embeddings.py`.) It has a **hard cap of four speakers**,
so it is not a replacement for pyannote, which has none; at best a fast path for
the two- and three-speaker recordings that make up most of this material. Against
the counts above — automatic counting already guesses too few three to four times
more often than too many — a hard cap is the wrong failure direction again.

`vad/silero_vad` is there too. This pipeline has no VAD at all, and one would bear
on the leading speech Voxtral sometimes drops rather than on diarization.

## Open

Surface `min_speakers` / `max_speakers` — pyannote accepts both, noScribe offers
only "auto" or an exact number. By the counts above, the **lower** bound is the
more valuable half. The caveat is noted at the call site in
`noScribe/pyannote_mp_worker.py`: a requested count that differs from the
automatically found one drops `VBxClustering` into KMeans, so this is an
algorithm change and not a pure win.
