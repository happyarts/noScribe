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

**Do not re-run this survey.** If you want a reference implementation to follow
rather than invent, torchaudio's own "Forced Alignment with Wav2Vec2" tutorial
carries the DP in a few dozen lines of readable Python; adapt that rather than
deriving it.

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
  does with it before changing its meaning; note that what the engine calls `prob` is
  a log probability.
* **Do not touch the surrounding logic** — the emission windowing, the tokenizer, the
  prefix-salvage path. Swap the DP, nothing else.
* Read the docstrings in `voxtral_engine.py` before changing any constant.

## Rollback

The current state is committed and pushed on `local/main`. Nothing here is urgent;
if it turns out messier than it looks, revert and leave torchaudio in place.
