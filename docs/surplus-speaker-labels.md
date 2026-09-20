# Surplus speaker labels: what can be detected, and what cannot

pyannote sometimes returns one label more than there are people. This file records
a study (2026-09-19/20) of how well such labels can be recognised after the fact,
which data streams carry the signal, and where the limit is. It complements
`docs/diarization.md`, which holds the DER baseline and the first ghost-speaker
report; the numbers here replace that report's threshold table as the better
evidence. Nothing in this file is built into noScribe yet.

The scripts and caches are local only (`benchmarks-local/whisper-split-ab/`,
data in `benchmarks-local/ghostsets/` and `benchmarks-local/voxconverse/`);
the file names are given where a result comes from, so a run can be repeated.

## Two kinds of surplus label

- **Garbage cluster**: short fragments, overlap, material of several people. Costly
  for the user, because every passage has to be moved by hand.
- **Split**: one real person on two labels (other room sound, phone, agitation),
  clean in itself and often not small. Cheap for the user: one search-and-replace in
  the editor.

A label counts as surplus when a longer label maps to the same reference speaker
(or it covers no reference speech); it counts as garbage when less than 80 % of the
reference speech under it belongs to one speaker.

## Material

802 recordings, 3629 labels, 177 of them surplus; in the 489 recordings with at most
four true speakers 1459 labels, 70 surplus (5 %), spread over 66 recordings (13 %).

    corpus                       files  labels  surplus   kind
    VoxConverse dev / test        448    2197     116     broadcast, 1-21 speakers, en
    AMI (whole corpus)            171     708      27     meetings of four, en
    ICSI                           75     491      27     meetings, 3-10 speakers, en
    CallHome                       77     173       4     phone calls of two, de en es ja zh
    AliMeeting (Eval + Test)       28      86       3     meetings, 2-4 speakers, zh
    two-track interviews            3       6       0     de, exact reference

The last row is three interviews recorded with one track per speaker: the tracks
are mixed to mono for the pipeline, and each track's own energy is the reference
(`podcast_prep.py`). pyannote finds exactly two speakers in all three, and also in
the published, edited mixes of the same recordings.

## Data streams and markers

One diarization pass per file keeps what the pipeline normally discards
(`ghost_cache2.py`): the chunk embeddings and binarised segmentations handed to the
clustering, the hard cluster assignment and the centroids. From that and from
per-turn embeddings (`ghost_embed.py`) one row per label is built (`ghost_table.py`):

    D   diarization   share of speech, longest and median turn, seconds, overlap with other labels
    C   clustering    embeddings in the cluster, share of them clean enough to train VBx on,
                      cohesion of those, survival of the cluster when VBx is run again with
                      another threshold / Fa / Fb (a re-clustering of the cached embeddings
                      costs 0.0 s -- no second diarization)
    S   segmentation  mean active seconds per chunk track, share of overlapped frames
                      (the soft activations are already 0/1 at this point: no confidence to read)
    E1  first ear     per-turn embeddings of pyannote's own voice model: cohesion of the label,
                      fit to the best other speaker, similarity to the nearest other centroid
    E2  second ear    the same from two independent ONNX speaker models (CAM++ trained on
                      200k zh/en speakers, ERes2Net), run through onnxruntime, which
                      faster-whisper already depends on -- so this would work on every platform
    T   text          faster-whisper on the 204 files holding candidate labels: passages and
                      words written under the label, words per passage, word probability,
                      avg_logprob, no_speech_prob

Single markers separate well and consistently across corpora and languages. Pooled
AUC, recordings with at most four speakers: share 0.97, survival under
re-clustering 0.95, longest turn 0.92, cohesion in either ear 0.90-0.91. A naive
voice test tried first -- "the label's turns fit another speaker as well as its
own" -- was right in only 12 % of its reports on VoxConverse; a garbage cluster
fits nobody, itself included, so cohesion is the question to ask, not fit.

## What a rule achieves

Rules are chosen on all corpora but one and applied only to the held-out corpus
(`ghost_eval.py` for AND-rules of up to three thresholds, `ghost_eval2.py` for a
regularised logistic model per stream set; both agree). Recordings with at most
four true speakers, 70 surplus labels:

    streams                         reports right      finds
    D                               21 / 28   75 %      30 %
    D + C                           22 / 26   85 %      31 %
    D + E1                          18 / 22   82 %      26 %
    D + E2                          16 / 19   84 %      23 %
    D + C + E1 + E2                 14 / 20   70 %      20 %
    old report (share < 2 %,
      longest turn < 2 s)           12 / 14   86 %      17 %

- **The ceiling is about 85 % right at a third found.** It comes from the base
  rate (5 % of labels are surplus) and from the material: a label made of a few
  turns under a second carries too little information in any stream.
- **Neither ear adds anything on unseen material**, the second no more than the
  first. All voice models fail at the same place. A second speaker model is not
  worth shipping for this.
- **More streams make it worse**: with 70 positives every further threshold fits
  noise. The most economical combination is the best one.
- **The gain over the old report is recall**, about twice as many labels found at
  the same precision, and it comes from the clustering markers, which cost nothing.
- Of the four wrong reports of D + C, two are real speakers with a single turn of
  5-8 s (a cap on the longest turn excludes them); the other two look like garbage
  and count as real only under the strict definition above.
- **Text adds nothing** (41 files of AMI, ICSI and CallHome, `t4_eval.py`): words
  per passage AUC 0.68, word probability 0.71, no_speech_prob 0.51. Only the
  *amount* of text separates (0.96-0.97), which is the speech share again.
  Whisper's segments are sentence-shaped under a surplus label too, so "starts in
  mid-sentence" is no marker.
- A typical surplus label puts a median of **13 words in 2 passages** into the
  transcript (a real speaker: 900 in 124). Most are blemishes; the large ones that
  hurt are the ones the amount-based markers cannot tell from a minor real speaker.

Consequences: automatic correction is out -- 85 % is not enough to move passages
unasked, and moving a garbage label's turns to the nearest voice placed only about
55 % of them right. What the numbers do support is a visible hint after the
transcription, rare (about five per hundred recordings) and mostly right, built from
D and C alone inside the diarization worker. Silence would still mean little: two
thirds of the surplus labels stay unreported.

## Telling pyannote the number of speakers

VoxConverse dev, 216 files (`bounds_vox.py`; confirmed on two files with the stock
pipeline on the CPU). Where the automatic count is wrong, the true number was given
as `num_speakers`, or as `min_speakers` / `max_speakers` (same code path, same result):

    automatic count   true speakers   files   DER auto   DER with the number   better / worse
    too few           1-4               16     13.1 %        25.1 %               1 / 15
    too few           5-8               32      9.2 %        22.7 %               3 / 29
    too few           9+                14      8.6 %        26.5 %               1 / 13
    too many          1-4               11      8.3 %         7.7 %               9 /  1
    too many          5+                 5      13.6 %        19.9 %               2 /  3

A requested count that differs from the one VBx finds drops the clustering into
KMeans, which splits a large speaker instead of finding the one that was missed. So
**the number helps only against too many labels in recordings of few speakers** --
the interview case, and exactly the case a surplus-label hint would fire in -- and
harms when someone was overlooked, which is the three to four times more frequent
error. A lower bound, once thought the more valuable half, is therefore not worth
surfacing; bounds that do not bite change nothing.

## Stereo balance

Would the left/right balance of a stereo file be an independent speaker marker,
given that noScribe mixes everything to mono? Simulated from the two-track
interviews with the voices panned 1 dB apart, the balance of a turn names the
speaker without error at every turn length (`stereo_sim.py`) -- trivially, since
clean tracks have no crosstalk. But the published mixes of the same recordings
carry no usable balance: the level difference is 0.00 dB in every turn, and 95 % of
all 10 s blocks have no side signal at all; only the music is stereo
(`stereo_test.py`). The marker exists only where the voices really are panned in
the delivered file, which is rare. The stronger form of the same idea would be to
accept one track per speaker, which makes diarization unnecessary.

## Related work

Standard systems filter small clusters (pyannote's older agglomerative clustering
reassigned clusters under 12 embeddings; NeMo and Kaldi recipes do the like) -- the
old report's rule with automatic correction, at the price of swallowing real minor
speakers. VBx itself is built to let unneeded speakers die out; surplus labels are
what survives it. Voting between independent systems (DOVER-Lap) removes labels
only one system sees, at twice the diarization time. Classifying frames with a
trained head on a speech foundation model (arXiv 2406.07890) beats clustering
clearly when the roles are known and labelled training data exists, which is not
noScribe's situation. Speaker counting remains an open weakness in the literature;
progress comes from better models, not from post-hoc detectors.
