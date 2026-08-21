# Brief: compute alignment emissions with mlx-audio instead of torch

Hand this file to a fresh session. Everything below was verified on 2026-08-22 and
the verification method is stated, so re-check rather than trust.

## What this is

Word timestamps come from CTC forced alignment: a wav2vec2 model produces a
`[frames, vocab]` matrix of log probabilities (the *emissions*), and a Viterbi pass
walks that matrix against the known words. This task replaces **only the first
half** — the emissions move to mlx-audio, and `torchaudio.functional.forced_align`
keeps doing the Viterbi exactly as today.

Measured on the same 300 s clip, the same 20 s windows, and the same `(14985, 38)`
output:

| | time | realtime factor |
|---|---:|---:|
| torch `_emission` | 15.64 s | 19.2x |
| mlx-audio via the MMS path | **2.89 s** | **103.8x** |

Emissions are **97 %** of the aligner's runtime — the full aligner takes 16.1 s on
that clip, so the Viterbi and everything around it is about half a second. The swap
therefore moves alignment from ~19x to roughly ~88x, worth about **150 seconds per
hour of audio**; on a four-hour job, ten minutes of roughly a hundred.

## Why it works without any porting

mlx-audio's `mms/mms.py` wraps the same `Wav2Vec2Model` encoder that
`wav2vec/wav2vec.py` provides and adds the CTC head:

```python
self.wav2vec2 = Wav2Vec2Model(config)
self.lm_head  = nn.Linear(config.hidden_size, config.vocab_size)
```

That is exactly the structure of any `Wav2Vec2ForCTC` checkpoint, which is what all
of this project's aligner models are. Loading one through that path needs **no code
change in mlx-audio** — only `model_type: "mms"` in the config it reads.

Verified: `jonatasgrosman/wav2vec2-large-xlsr-53-german` loaded this way reproduces
the torch emissions with an **identical argmax on 100 % of frames** and a maximum
absolute difference of **0.00068**. That is float noise, not a different result.

## Scope

In: `_Aligner.__init__` and `_Aligner._emission` in `noScribe/voxtral_engine.py`
(around lines 1842 and 1863), the model-preparation step described below, and
`environments/requirements_voxtral_macOS_arm64.txt`.

Out: `_tokenize`, `align_words`, `align_prefix`, the DP cap, the recursive split,
`_AlignerPool`, and anything to do with which language picks which model. **The
Viterbi stays torchaudio.** A separate brief covers replacing it
(`docs/viterbi-numpy-brief.md`) — do not do both at once, or a change in output
cannot be attributed.

## The one real problem: getting the models into a loadable shape

This is the bulk of the work, not the code.

`resolve_align_model()` picks from **15 per-language models** plus the multilingual
fallback. Their situation differs:

* `MahmoudAshraf/mms-300m-1130-forced-aligner` (the fallback) **already ships
  `model.safetensors`.** It needs only the config key patched.
* The 15 `jonatasgrosman/wav2vec2-large-xlsr-53-*` models ship **`pytorch_model.bin`
  only** — no safetensors. Each needs converting.

**No MLX conversions of any of these exist on the Hub** (checked). So decide one of:

1. **Convert on first use, cache locally.** Load with transformers, write
   safetensors plus a patched config beside the HF cache entry. torch is present
   anyway (see below), so this costs nothing in dependencies — but it adds a
   one-time delay per language and a cache location to manage.
2. **Publish converted models**, the way the Voxtral builds are published. Cleaner
   at runtime, but it is 16 uploads to maintain and pin.
3. **Convert only German and the fallback**, keep torch emissions for the rest.
   `_Aligner` would choose its backend per model. Least work, covers almost all
   real use, and the fallback path stays exercised.

Option 3 is probably right for a first pass; measure before committing to 2.

## What does not change, and why torch stays

`torch` and `pyannote.audio>=4` are pinned in the **base** requirements because the
diarizer needs them. This task does not remove torch and is not meant to — it makes
one step faster. `torchaudio` also stays, because it still does the Viterbi.

So `mlx-audio` becomes an **additional** dependency in the Voxtral extra. Note that
it hard-requires `sounddevice` and `miniaudio` (native, PortAudio). The STT import
path does not pull them, but pip installs them, so the frozen build likely needs
excludes. **Prove that with a throwaway PyInstaller build — never infer frozen
behaviour from source.**

## Steps

1. Work in a **throwaway venv** until the approach is settled. Verify with
   `pip freeze` before and after that the project venv comes back byte-identical.
2. Reproduce the measurement above before changing anything, so you have your own
   baseline rather than this document's.
3. Write the conversion: load the checkpoint with `Wav2Vec2ForCTC.from_pretrained`,
   `save_file(state_dict)` to safetensors, dump `config.to_dict()` with
   `model_type` set to `"mms"`. The state dict's top-level keys are `wav2vec2` and
   `lm_head`, which is what the MMS module expects.
4. Add an MLX emission path to `_Aligner`. Keep the **20 s windowing exactly as it
   is** — `EMISSION_WINDOW_SEC` exists because the aligner's self-attention is
   O(T²) in memory, and that is as true on MLX as on torch.
5. Return the same thing `_emission` returns today: a torch tensor of log
   probabilities, `[frames, vocab]`, so `align_words` needs no change. Converting
   the numpy array to a torch tensor at the boundary is cheap and keeps the change
   contained.
6. Keep the torch path as a fallback for any model that has not been converted, and
   for the case where mlx-audio is not installed.

## Acceptance

* Emissions equal to the torch path: **identical argmax on every frame**, max
  absolute difference below 1e-3, on both hand-corrected references in
  `Audiotest2/referenz/` (gitignored; ask if absent).
* **Word timestamps unchanged end to end.** Store a run before and after and compare
  per word — do not eyeball. This is the real acceptance test; equal emissions
  should give equal alignment, and if they do not, something else moved.
* `venv/bin/python3 -m pytest tests/ -q` green. `tests/test_forced_align_stability.py`
  and `tests/test_salvage_prefix_alignment.py` are the ones that matter here.
* The measured speedup reproduced and written into
  `docs/voxtral-quantisierung.md` next to the existing table.
* A throwaway PyInstaller build starts and transcribes.

## Traps

* **Do not touch the blank-index logic.** The CTC blank is `config.pad_token_id`,
  not the token spelled `<pad>`; `_Aligner.__init__` documents what goes wrong
  otherwise — every word stretches to meet its neighbour. The vocab and blank come
  from the processor and stay on the torch side unless you deliberately move them.
* **Windowing must stay identical**, including the `< 0.025 s` guard on the trailing
  remainder. wav2vec2's conv frontend raises on inputs shorter than its receptive
  field, and the guard is `< min_win`, not just the empty case.
* **Equal emissions are necessary, not sufficient.** Check the timestamps too.
* Read the docstrings in `voxtral_engine.py` before changing any constant — several
  encode a defect that was expensive to find.
* mlx-audio's `load_model` defaults to `strict=False` and will silently return a
  randomly initialised model if keys do not match (reported as
  [Blaizzy/mlx-audio#902](https://github.com/Blaizzy/mlx-audio/issues/902)).
  **Assert something about the loaded model** — that `lm_head` exists and that a
  known clip gives the expected argmax — rather than trusting that loading worked.

## Rollback

The current state is committed and pushed on `local/main`. Nothing here is urgent:
it is a speed improvement to a step that already works. If it gets messy, revert and
leave the torch emissions in place.
