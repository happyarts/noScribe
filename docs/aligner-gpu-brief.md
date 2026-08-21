# Brief: run the aligner on the GPU

Hand this file to a fresh session. Everything below was measured on 2026-08-22 on an
M1 Max; the method is stated so you can re-check rather than trust.

## The finding

`_Aligner.__init__` in `noScribe/voxtral_engine.py` does:

```python
self.model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
```

There is no `.to(device)` anywhere in the file. **The forced-alignment model runs on
the CPU**, in a module that is Apple-Silicon-exclusive, while `pyannote_mp_worker.py`
already selects MPS for the diarizer a few files away.

Measured on the 300 s reference, 20 s windows, identical `(14985, 38)` output:

| | time | realtime | argmax vs CPU |
|---|---:|---:|---:|
| CPU fp32 — today | 14.85 s | 20.2x | — |
| **MPS fp32** | **4.60 s** | **65.2x** | **100.000 %** |
| MPS fp16 | 3.52 s | 85.1x | 99.933 % |

Emissions are ~97 % of the aligner's runtime, so this moves alignment from roughly
19x to roughly 59x realtime — about **130 seconds saved per hour of audio**, ten
minutes on a four-hour job.

**This was already known.** A review on 2026-07-27 recorded it as "the largest single
open win", measured at 3.86x on 120 s of audio with a maximum log-prob deviation of
4.8e-4. It fell outside that review's scope and has been sitting since. The numbers
above are an independent re-measurement and agree.

**Use fp32, not fp16.** fp16 is faster still, but 0.067 % of frames pick a different
argmax and the maximum deviation is 1.68 — enough to move a word boundary. The whole
point of this change is that it is free of consequences; do not trade that away for
20 more seconds per hour.

## Do this at the same time, not after

A second finding from the same review, and it becomes more important once the aligner
is on the GPU:

> `vox._mx.clear_cache()` runs **after** the alignment, not before — so the decode
> pass's MLX buffers are still resident while the wav2vec2 forward runs. That is
> exactly the co-residency `MIN_HEADROOM_GB` exists to prevent. One-line fix.

Today those buffers merely coexist with a CPU forward, so the cost is address space.
Once the aligner is on the same GPU, they compete for the same memory. On a 32 GB
machine there is room; on the 16 GB machine the model menu still allows for
`voxtral-mini-8bit` there may not be. Move the `clear_cache()` call before the
alignment as part of this change, and measure peak memory on a long pass.

## What was considered instead, and rejected

An earlier version of this brief proposed computing the emissions with `mlx-audio`
instead. That works — its `mms` module wraps the same wav2vec2 encoder with a CTC
head, our German checkpoint loads through it with only a config key changed, and the
emissions are numerically identical to torch at 103.8x realtime.

**It is not worth it.** Against `.to("mps")` it buys about 20 more seconds per hour
of audio, and costs: a new dependency in the Voxtral extra, which hard-requires
`sounddevice` and `miniaudio` and therefore needs PyInstaller excludes proven with a
throwaway build; a safetensors conversion for **15 per-language models** that ship
only `pytorch_model.bin`, none of which exist pre-converted on the Hub; and a second
loading path to maintain. The full measurement is in
`docs/voxtral-quantisierung.md` under *What mlx-audio holds beyond ASR* — it stays on
record because it may matter later, not because it should be done now.

Also checked and rejected: replacing the aligner model. No pre-converted variants of
the `jonatasgrosman/wav2vec2-large-xlsr-53-*` family exist, and no better German CTC
model was found — that family is the de-facto standard for this job, and the choice
was already made deliberately in an aligner audit on 2026-07-23 against WhisperX's
own German default and against MMS, which a published benchmark puts *last* on
German.

## The task

1. Pick the device the way `pyannote_mp_worker.py` does: MPS when available on
   Darwin, else CPU. Do not invent a new mechanism — read that file first and match
   it, including its guard on the macOS version.
2. Move the model and each input window to that device; bring the log-probs back to
   the CPU before they reach `align_words`, so nothing downstream changes.
3. Keep a CPU fallback and use it if the MPS forward raises. This model is old
   enough that an unsupported op is conceivable; failing the whole job for a
   timestamp is not acceptable.
4. Move `vox._mx.clear_cache()` to before the alignment.
5. Consider whether a config key belongs here. `pyannote_xpu` and
   `force_whisper_cpu` already exist in `config.yml` for exactly this kind of
   escape hatch — look at how they are read before adding a third.

## Acceptance

* **Emissions identical**: argmax equal on every frame against the CPU path, on both
  hand-corrected references in `Audiotest2/referenz/` (gitignored; ask if absent).
  A maximum absolute deviation around 5e-3 is expected and fine; a changed argmax is
  not.
* **Word timestamps unchanged end to end.** Store a run before and after and diff per
  word. Equal emissions should give equal alignment; if they do not, something else
  moved and you need to know what.
* `venv/bin/python3 -m pytest tests/ -q` green — `test_forced_align_stability.py`,
  `test_forced_align_cap.py`, `test_forced_align_density.py` and
  `test_salvage_prefix_alignment.py` are the ones that bear on this.
* Peak memory measured on a long pass with the decode buffers freed first, and
  compared against `MEM_MODEL`'s expectation. Watch **free RAM via `vm_stat`, not
  RSS** — RSS badly under-reports MLX unified memory.
* Speedup measured and written into `docs/voxtral-quantisierung.md`.
* A throwaway PyInstaller build starts and transcribes. Never infer frozen behaviour
  from source.

## Traps

* **Do not touch the blank-index logic.** The CTC blank is `config.pad_token_id`, not
  the token spelled `<pad>`; `_Aligner.__init__` documents what goes wrong otherwise
  — every word stretches to meet its neighbour.
* **Keep the windowing exactly as it is**, including the `< 0.025 s` guard on the
  trailing remainder: wav2vec2's conv frontend raises on inputs shorter than its
  receptive field.
* `_AlignerPool` keeps an LRU of two aligners. Two models on the GPU at once is the
  normal case, not the exception — account for it when measuring memory.
* Benchmarks only on an idle machine. A previous round of measurements was distorted
  by GPU tests running in parallel, and the wrong numbers were believed for a while.
* Read the docstrings in `voxtral_engine.py` before changing any constant.

## Rollback

The current state is committed and pushed on `local/main`. This is a speed
improvement to a step that already works; if the device move misbehaves, revert and
leave it on the CPU.
