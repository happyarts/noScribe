# Diarization: a measured baseline on VoxConverse (29.-30.07.2026)

Why: a run with automatic speaker counting invented a third speaker on a
two-person recording, and a word in the middle of a sentence ended up under it.
The question was whether that can be repaired inside the program.

## Does our setup reproduce the published numbers?

VoxConverse v0.3 (free, CC-BY-4.0; audio from robots.ox.ac.uk, RTTM from
github.com/joonson/voxconverse). Scored with `pyannote.metrics` under pyannote's
own conditions: `collar=0`, `skip_overlap=False`, aggregated over the sums of all
files.

    Split   Files   Speech      our DER   pyannote (community-1)
    dev       216    20.3 h      7.18 %   --
    test      232    40.2 h     11.14 %   11.2 %

The model card names no split; it is **test** (dev is the easier one). That
validates the setup (`pyannote/`, community-1 with VBx+PLDA, on MPS) -- from here
on, our own diarization measurements are worth something. Runtime 14 s/file on
dev, ~22 s on test.

VoxConverse rather than AMI because the question was the speaker *count*: AMI has
exactly four participants per session, so the axis under test is constant there,
while VoxConverse ranges from 1 to 21 speakers per recording. Its YouTube audio
is also closer to the Zoom recordings and podcasts this fork is used on.

## The finding that sets the direction

    Speaker count against RTTM truth
    dev :  exact 138 | too many 16 | too few 62
    test:  exact 109 | too many 30 | too few 93

**Too few speakers is three to four times more common than too many.** Anything
that merges speakers therefore pushes in the direction automatic counting already
overshoots.

## What was tried and rejected

- **`clustering.Fb` from 0.8 to 1.5.** Cured the recording that started this
  (3 -> 2 clusters) and left three clean files bit-identical, but collapsed 14
  detected speakers to 12 on a 25-minute question round. `Fb` is the speaker
  regularization coefficient of the VB loop ("high values result in the VB
  inference dropping more speakers", Diez et al.); 0.07/0.8/0.6 are pyannote's
  tuned optima within the search ranges found in the source, documented nowhere.
  Wrong error direction, rejected.
- **Raising `clustering.threshold`.** 0.70 changed nothing; 0.80 (the top of the
  search range) left the spurious label in place and merely renamed it. Under VBx
  the threshold only initialises the agglomeration -- the VB loop decides the
  final count.
- **Relabelling the spurious cluster in noScribe.** Against the fixed-count run
  as a reference, folding each of the 56 turns into its nearest neighbour placed
  46 right and 10 wrong (82 %). Those 10 would be silent misattributions to real
  speakers in place of a visible phantom speaker -- the worse kind of error.
  Besides, the label held material from *both* real speakers (24 turns of one, 32
  of the other), so there is no single speaker to merge it into.
- **`AgglomerativeClustering` with `min_cluster_size`.** The parameter counts
  embeddings, not length, and the spurious label is spread over many chunks. Even
  at 15 (maximum 20) the file stayed at 3 clusters, and leaving VBx gives up the
  tuned 4.x pipeline.
- **Voxtral's audio encoder as a second identifier.** Nearest-centroid over 109
  confident segments >= 3 s: `audio_tower` 58.2 % +-11.1, projector 82.3 % +-7.8
  on a two-class problem. cos(speaker A, speaker B) = 0.993 against
  cos(segment, own centroid) = 0.93/0.96 -- the speaker axis is buried in the
  content, which is what an ASR encoder is trained for.

## What was kept

Two things. The actual fix is not in the diarization but in cue building: a lone
stretched word was allowed to keep its own cue and took its speaker from whatever
the diarization put under it (`voxtral_engine._segments_from_words`). Re-run over
all seven recordings of the series: word counts unchanged, one 26-minute file
bit-identical, exactly two words changed speaker -- both repairs of sentences cut
in half.

Plus a report when a label never holds the floor. Scored against ground truth at
label level ("does a longer label already cover the same reference speaker?"):

    share < 0.05, turn < 2000 ms   dev 71 %   test 67 %
    share < 0.02, turn < 2000 ms   dev 83 %   test 69 %   <- shipped
    share < 0.01, turn < 2000 ms   dev 100 %  test 67 %

0.02 is the only value better than the original on both splits; 0.01 is
overfitting to 32 positives. Recall is 9 of 77 surplus labels on test -- silence
means nothing. A third axis (a minimum number of short turns) is out: the
recording that started this had 56, surplus labels on VoxConverse typically have
few, and requiring a minimum cut hits from 5 to 1.

Note on scope: this report lives in `main.py` and is engine-independent, so it
applies to the Whisper path as well -- unlike the cue fix above, which is
Voxtral-only.

## Open

Surface `min_speakers` / `max_speakers` -- pyannote accepts both, noScribe offers
only "auto" or an exact number. By the counts above, the **lower** bound is the
more valuable half. Caveat for that change: when the requested number differs
from the automatically found one, `VBxClustering` switches to KMeans, so a fixed
or bounded count is an algorithm change and not a pure win.
