# Brief: replace torchaudio's forced_align with a numpy Viterbi

Hand this file to a fresh session. Everything below was verified on 2026-08-22; the
verification method is stated so you can re-check rather than trust.

## Read this before writing any code

**This task is optional, and there is a cheaper thing to do first.** Do not start
until you have understood why, because the obvious motivation for it is wrong.

The motivation people reach for is "get torch out of the alignment path". That is
unreachable: `environments/requirements_macOS_arm64.txt` pins `torch==2.13` and
`pyannote.audio>=4` in the **base** requirements, because the diarizer is torch-based.
Removing torchaudio does not remove torch.

The speed motivation is real but **separable, and far cheaper than this task.** The
aligner currently runs on the **CPU** — there is no `.to(device)` in the file at all.
Moving it to MPS takes it from 20.2x to 65.2x realtime with an identical argmax on
every frame, which is about 130 seconds per hour of audio for one line of code.
Emissions are 97 % of the aligner's runtime, so that captures nearly all of the
available benefit.

**That is already done** — the aligner runs on MPS as of 2026-08-22, with every
timestamp bit-identical to the CPU path. See the *aligner moved to the GPU* section
of `docs/voxtral-quantisierung.md`. So the speed argument for this task is spent;
what remains is the code-simplification argument below.

Also note that `forced_align` is not going away. TorchAudio is in maintenance, but
its v2.10 release notes name `forced_align` explicitly as one of five C++ extensions
"preserved and will remain in torchaudio", and 2.11 is documented as compatible with
future torch versions.

## What this task actually buys

Three things, none of them dependency removal:

1. **It deletes a workaround.** `torchaudio`'s CPU `forced_align` indexes its
   `frames × (2·tokens+1)` DP buffer with 32-bit integers and segfaults past that.
   `voxtral_engine.py` carries a hard cap and a recursive word/audio splitting path
   purely to stay under it. A numpy DP with 64-bit indexing has no such limit, so the
   cap and the split can go — real simplification of a load-bearing routine.
2. **It drops the `torchaudio==2.11` pin**, which exists for the same kernel.
3. **It removes the dependency on an unmerged upstream fix.** Our
   [pytorch/audio#4209](https://github.com/pytorch/audio/pull/4209) fixes exactly
   this overflow, was approved by a maintainer on 2026-08-05, and is still unmerged.

If none of those matter today, close the task.

## Everything that was checked so it need not be built

Searched 2026-08-22. Nothing usable was found, and the reasons differ:

| candidate | verdict |
|---|---|
| `ctc-segmentation` | no torch, right domain — but **source distribution only**, so installing needs a C compiler, which a click-and-run app cannot ask of users; last release 2022-10-11; and it solves utterance segmentation robust to imperfect text, not exact-target frame alignment |
| `ctc-forced-aligner` | requires torch — trading torchaudio for another torch consumer |
| `speechbrain`, `whisperx`, `stable-ts`, `nemo-toolkit` | all require torch, all far heavier than the problem |
| `onnx-asr` | contains no alignment code at all (repository search: zero hits) |
| `kaldialign` | edit distance between token sequences, not Viterbi over emissions |
| `pyctcdecode` | beam-search decoding, not forced alignment |
| mlx-audio `qwen3_forced_aligner` | measured: collapses at its documented 5-minute cap, 20 % zero-duration spans, last 30 s of a 300 s clip on one timestamp |
| mlx-audio generally | no `forced_align` or Viterbi anywhere; MMS's `_ctc_decode` is greedy argmax |

**Do not re-run this survey.** One candidate did survive it, though, and it
deserves its own section.

## The one that nearly makes this unnecessary: `ctc-forced-aligner`

PyPI's `ctc-forced-aligner` 1.0.2 exposes

```python
forced_align(log_probs, targets, input_lengths=None, target_lengths=None, blank=0)
    -> (paths, scores)
```

— the same signature as `torchaudio.functional.forced_align`, over numpy arrays,
backed by a compiled C++ extension. It also ships `merge_repeats`, the counterpart
to `merge_tokens`. **Measured against torchaudio on 40 random cases** (varying T
and target length, including tight `T == L + repeats` fits): **40 identical, 0
divergent**. So the algorithm question is answered — if you adopt it, there is
nothing to write.

Four things to weigh before you do, none of them fatal on its own:

* **Source distribution only.** A 22 KB tarball that builds a C++ extension at
  install time. That is the same objection that ruled out `ctc-segmentation`: a
  click-and-run desktop app cannot ask users for a compiler, and PyInstaller has to
  be shown to bundle the built extension.
* **Custom licence.** "Deskpai Open Source License (DOSL) 1.0" — BSD-shaped, but
  clause 3 requires any distribution to carry a README that *clearly displays*
  "This software is distributed with permission from https://www.deskpai.com." in
  a prominent, visible location, and clause 6 requires the licence file in the
  distribution root. Attribution terms of that kind are plausibly permitted by
  GPL-3.0 §7(b), but noScribe is GPL-3.0 and this is exactly the class of question
  that cost this project a week already. Get it read properly before depending on
  it.
* **Name confusion.** The PyPI name belongs to `deskpai/ctc_forced_aligner` (11
  stars, one release burst in 2025-02, untouched since). It is **not**
  MahmoudAshraf's `ctc-forced-aligner` (552 stars, actively maintained), whose
  *model* this project uses as its multilingual aligner. Anyone reaching for "the
  ctc forced aligner package" will assume the wrong one.
* It was found sitting in the project venv, installed deliberately and listed in no
  requirements file, imported by nothing. It has since been removed.

**Whatever you decide, install it temporarily as a second oracle.** It is an
independent implementation that agrees with torchaudio, so checking a hand-written
DP against two references rather than one costs one `pip install` and no thought.

If you do write the DP instead, torchaudio's own "Forced Alignment with Wav2Vec2"
tutorial carries it in a few dozen lines of readable Python — adapt that rather
than deriving it from the paper.

## Writing it is easy. Making it fast is not.

Both halves of that were measured, so plan around the numbers rather than the
intuition that 151 lines of C++ cannot be doing much.

**Correctness is a short afternoon.** A vectorised numpy DP — state axis
vectorised, time axis looped, ~30 lines — reproduced `torchaudio.functional.forced_align`
on **40 of 40 random cases** including tight `T == L + repeats` fits, first try.
The algorithm is not where the risk is.

**Speed is the problem.** On a real 300 s chunk (T = 14 985 frames, L = 4 886
tokens, N = 9 773 states, **146 million DP cells**):

| implementation | time | vs C++ |
|---|---:|---:|
| `torchaudio.functional.forced_align` (C++) | 264 ms | — |
| `ctc-forced-aligner` (C++) | 253 ms | 1.0x |
| vectorised numpy | **3 144 ms** | **11.9x** |

All three returned identical paths.

The gap is not C++ being clever. It is that the recurrence is sequential in time,
so numpy runs ~15 000 iterations of six array operations on 9 773-element arrays,
and pays dispatch and allocation overhead on every one of them; C++ runs one tight
loop with neither. Vectorising harder does not help, because the axis that is long
is the one that cannot be vectorised.

**And 3.1 s is the wrong kind of slow here.** The emission step for that same chunk
costs about 2.9 s on the GPU, so a numpy DP would roughly double alignment — giving
back about half of the ~130 s per hour of audio that moving the aligner to the GPU
just won.

**The shape that does work: keep C++ for the normal path and use numpy only where
the cap would otherwise force a split.** `FORCED_ALIGN_MAX_CELLS` is 2^30 cells;
the real chunk above uses 146 million, so a ten-minute pass sits around 300 million
and the cap fires only on unusually dense speech. A numpy DP with 64-bit indexing
has no such limit, so wiring it in as the fallback removes the recursive
word/audio split — the actual simplification this task is for — while the
common path keeps the C++ speed. That is a much smaller and safer change than
replacing the DP outright.

## The task

Replace `torchaudio.functional.forced_align` and `torchaudio.functional.merge_tokens`
in `noScribe/voxtral_engine.py` with a numpy implementation.

**Algorithm.** Standard CTC forced alignment. The target token sequence of length S
is blank-interleaved to length 2S+1. Viterbi over `T × (2S+1)` with three
transitions into each state — stay, advance one, advance two — where advance-by-two
is legal only when it skips a blank between two *different* labels. Backtrace yields
one label per frame; `merge_tokens` collapses equal runs into spans carrying a mean
score.

**Shape of the implementation.** Vectorise over the state axis; loop over time. T is
around 50 frames per second of audio, so a ten-minute pass is ~30 000 iterations of a
vector update — fine in numpy, and note that MLX is the *wrong* tool here because the
recurrence is strictly sequential in time and would mean one kernel launch per frame.
The backtrace table is `T × (2S+1)` of small integers; allocate it as `uint8` or
`int8`, not `int64`.

**The oracle already exists.** `tests/test_forced_align_stability.py` pins
torchaudio's behaviour against a recorded reference in `tests/data/`. That is the
acceptance test: the new implementation must reproduce it. Read that test and
`tests/test_forced_align_cap.py` and `tests/test_forced_align_density.py` before
starting — their docstrings describe defects that were expensive to find.

## Acceptance

* Frame-for-frame identical output to `torchaudio.functional.forced_align` on the
  recorded reference, and on the two hand-corrected references in
  `Audiotest2/referenz/` (gitignored; ask if absent).
* `venv/bin/python3 -m pytest tests/ -q` green.
* Word timestamps unchanged end to end — compare against a stored run, do not eyeball.
* No `torchaudio` left in `noScribe/`: `grep -rn torchaudio noScribe/` empty except
  comments you have updated.
* The DP cap and the recursive split are removed **only** after a pass that would
  previously have tripped the cap is shown to work without them.

## Traps

* **Blank index.** The CTC blank is `config.pad_token_id`, not the token spelled
  `<pad>`. `_Aligner.__init__` documents why: the multilingual MMS aligner's vocab
  has `{"<blank>": 0, "<pad>": 1, …}` and emits blanks on 0, so reading `<pad>`
  picked an index the model never produces, and every word stretched to meet its
  neighbour. Keep that logic exactly.
* **Repeated tokens.** Two identical labels in a row require a blank between them —
  advance-by-two must not skip it. torchaudio enforces this; so must you.
* **Scores.** `merge_tokens` returns a score per span. Check what the existing code
  does with it before changing its meaning; note that what the engine stores as
  `prob` is `exp()` of that score, so it matches `whisper_mp_worker`'s field.
* **Span ends are exclusive.** Verified against torchaudio: for a token occupying
  frames 0, 1, 2, `merge_tokens` returns `start=0, end=3`, so `end - start` is the
  duration and `sp.end / fps` is the time after the last frame. The engine relies on
  this. Do not add a `+1` — MahmoudAshraf's aligner needed one
  ([c344f5b](https://github.com/MahmoudAshraf97/ctc-forced-aligner/commit/c344f5bc900323aa434a7cb200b7c629d463bd02))
  because *its own* `merge_repeats` returns an inclusive end; the conventions differ.
* **The frame rate is derived, and that is fine.** `fps = n_frames / duration`
  rather than the model's `inputs_to_logits_ratio` (320, i.e. exactly 50 fps).
  Measured on a 300 s chunk the derived value is 49.95, and because it comes from
  the concatenated total the resulting timestamp error stays within **10-20 ms and
  oscillates rather than accumulating**. Not worth changing, and changing it would
  move every timestamp the tests pin.
* **Do not touch the surrounding logic** — the emission windowing, the tokenizer, the
  prefix-salvage path. Swap the DP, nothing else.
* **A generic HMM Viterbi is the wrong shape, and searching for "Viterbi in numpy"
  will hand you one.** Those implementations take a dense `N x N` transition matrix
  and an emission *table* indexed by a discrete observation alphabet, then loop over
  both time and states in Python. Every part of that fights this problem. CTC's
  transitions are a narrow band — from state `s` only to `s`, `s+1` or `s+2` — so
  for a ten-minute pass the dense matrix is 2401 x 2401 entries of which **0.12 %
  are not `-inf`**, rebuilt on every call because the target sequence changes per
  chunk. The emissions are already a `[T, vocab]` array from the network, so what
  you need is a gather `emissions[t, target[s]]`, not a table lookup. And the double
  loop is **72 million Python iterations** where vectorising over the state axis
  leaves 30 000. Worse, none of the actual difficulty — the blank interleaving, the
  advance-by-two rule, the backtrace, `merge_tokens` — appears in a generic
  implementation at all. Take the torchaudio tutorial instead.
* **numpy is not a new dependency.** `voxtral_engine.py` already imports it, and a
  dozen installed packages require it transitively. Nothing to weigh there.
* Read the docstrings in `voxtral_engine.py` before changing any constant.

## Rollback

The current state is committed and pushed on `local/main`. Nothing here is urgent;
if it turns out messier than it looks, revert and leave torchaudio in place.
