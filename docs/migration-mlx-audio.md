# Brief: move the Voxtral engine from mlx-voxtral to mlx-audio

Hand this file to a fresh session. It is a work order, not documentation — everything
in it was verified on 2026-08-21 and the verification method is stated so you can
re-check rather than trust.

## Why this is happening

`mlx-voxtral` ships under a **"Personal Use License"**: MIT plus a ban on commercial
use, including "using the Software to provide commercial services". **noScribe is
GPL-3.0**, whose section 7 forbids further restrictions, and most of noScribe's users
transcribe for paid work. `mlx-audio` is **MIT**.

So this is not a maintenance upgrade. It removes a licence conflict that would
otherwise block the Voxtral engine from going upstream. Background:
`VOXTRAL.md` (licence section) and `docs/voxtral-quantisierung.md`
(*Moving off mlx-voxtral*).

> **STOP — this migration is on hold as of 2026-08-22.**
>
> [mzbac/mlx.voxtral#9](https://github.com/mzbac/mlx.voxtral/issues/9) asked the
> copyright holder to relicense, and he did: the repository is **plain MIT** from
> v0.0.5, he merged our stop-token fix, and he cut the first release in a year.
> The licence conflict that made this migration necessary no longer exists.
>
> Do not work through this file unless something *else* makes the move
> worthwhile — mlx-voxtral going quiet again, an MLX bump breaking it, or a
> decision to ship a second engine, where mlx-audio's model collection pays for
> the work in one go. The findings below were measured and remain valid; only
> the reason to act on them is gone.

## Scope

In: `noScribe/voxtral_engine.py`, `tools/quantize_voxtral.py`,
`environments/requirements_voxtral_macOS_arm64.txt`, the affected tests, and
re-publishing the two quantised builds.

Out: chunking, loop detection, the temperature ladder, forced alignment, prefix
salvage. None of that touches the library. **Do not refactor them while you are in
there.**

## The coupling is five calls

Verified with `grep -rn "mlx_voxtral\|mlx_lm" noScribe/ tools/ tests/`:

| where | what |
|---|---|
| `voxtral_engine.py:1601` | `load_voxtral_model`, `VoxtralProcessor` |
| `voxtral_engine.py:1779` | `proc.apply_transcrition_request(...)` (note the upstream typo) |
| `voxtral_engine.py:1809` | `model.generate(...)` — the non-greedy fallback only |
| `voxtral_engine.py:1813` | `proc.decode(...)` |
| `tools/quantize_voxtral.py` | `mlx_voxtral.quantization`, `utils.model_loading.download_model` |

Plus `_merged_embeddings` and `_LMAdapter`, both of which likely become unnecessary —
see below.

## Verified facts you must not re-derive

**1. mlx-audio will not load the published builds, and it says nothing.**

Loading `models/voxtral-mini-8bit` through `mlx_audio.stt.utils.load_model` yields
**0 quantised modules and 405 dense `Linear`**, 211 of them in the language model,
with no exception. Mechanism, all in `mlx_audio/utils.py`:

* `apply_quantization`'s `get_class_predicate` looks up `p in quantization` and
  otherwise falls back to `f"{p}.scales" in weights`.
* Our config and weights use `language_model.layers.0…`; mlx-audio's module tree is
  `language_model.model.layers.0…` (its `LanguageModel.__init__` does
  `self.model = LlamaModel(config)`). Neither lookup matches.
* `sanitize()` only transposes conv weights — it does not remap the prefix.
* `load_model(..., strict=False)` is the **default**, so the mismatched keys are
  skipped silently and the language model stays at its initial values.

The audio tower does match and does load, which is why a naive smoke test looks
half-plausible.

**2. Re-quantising is format-only. It cannot change quality or speed.**

MLX's affine quantisation is data-free — scale and bias come from each group's own
min and max, no calibration set, no randomness. Verified: `mx.quantize` on the same
tensor twice returns bit-identical results, and so does a fresh copy of it.
mlx-audio's predicate is `not p.startswith("audio_tower")`, which selects exactly
the set the current builds carry: **213 modules, all 8 bit, group size 64, affine,
encoder dense**. Same weights in, same tensors out; only the keys change.

So the measurement tables in `docs/voxtral-quantisierung.md` still describe a
re-quantised build. **Do not re-run the bit sweeps.**

**3. What could change output, and therefore must be checked:**

* mlx-audio's `_merge_input_embeddings` scatters without promoting dtype.
  `_merged_embeddings` promotes deliberately. Measured on the current build both
  sides are bfloat16 so nothing is lost — re-check on whatever build you produce,
  because the failure is invisible in the text and shows up only as different logits.
* `_VOXTRAL_EOS_TOKEN_IDS = [2, 4, 32000]`. 32000 is the text token `" Capital"`,
  and stopping on it silently truncates. Fixed in
  [Blaizzy/mlx-audio#901](https://github.com/Blaizzy/mlx-audio/pull/901) — **check
  whether that merged**. If not, resolve stop ids from the processor as
  `_resolve_stop_tokens` already does; do not rely on the library default.
* mlx-audio vendors its own `generate_step` (`mlx_audio.lm.generate`) rather than
  using `mlx_lm`'s. Same chunked prefill (`prefill_step_size=2048`), so the ~18 %
  peak saving survives — but `MEM_MODEL` is calibrated against the current path and
  **must be re-measured**.
* Voxtral is the only STT model in mlx-audio that uses `AutoProcessor`, which
  resolves to transformers' `VoxtralProcessor` and **requires torch**. noScribe has
  torch for the aligner, so this is not a blocker, but it changes the worker's
  import graph — re-check `tests/test_worker_import_lightweight.py` and the
  PyInstaller specs. mlx-audio also hard-requires `sounddevice` and `miniaudio`
  (native, PortAudio); the STT import path does not pull them, but pip installs
  them, so the frozen build likely needs excludes. **Prove that with a throwaway
  PyInstaller build — never infer frozen behaviour from source.**

## The alignment path is not affected — but know this before you touch it

Out of scope here, and stated so you do not go looking: **the German forced-aligner
model already loads through mlx-audio today**, via `mms/mms.py`, which wraps the same
`Wav2Vec2Model` encoder and adds the `lm_head` a `Wav2Vec2ForCTC` checkpoint carries.
It needs no code change, only `model_type: "mms"` in a converted config, and it
reproduces the torch emissions exactly — identical argmax on every frame, max |Δ|
0.00068 — at 103.8x realtime against torch's 19.2x on the same 300 s clip and the
same 20 s windows.

This is **not** part of the migration, for two reasons. It is independent of which
package loads Voxtral, so it can be done before, after, or never. And it still needs
a replacement for `torchaudio.functional.forced_align`, which mlx-audio does not
have. Do not bundle it in; a migration that also rewrites alignment cannot be shown
to have changed nothing.

## Do this first: try to avoid re-publishing at all

Before re-quantising 6 GB and 25 GB and making every user re-download, evaluate a
key remap in mlx-audio's Voxtral `sanitize()` that accepts the mlx-voxtral layout
(`language_model.X` → `language_model.model.X`). If that works it is a small
upstream contribution, it fixes the same problem for every other published
mlx-voxtral build, and it removes the largest single cost from this migration.

Send it as a PR to mlx-audio and see. If it is rejected or takes too long, fall back
to re-quantising.

## Steps

1. **Check `mzbac/mlx.voxtral#9`.** A yes means stop.
2. Install `mlx-audio[stt]` **in a throwaway venv first** and confirm the current
   behaviour above still holds against whatever version is current. Do not touch the
   project venv until the approach is settled — verify with `pip freeze` before and
   after that it comes back byte-identical.
3. Try the `sanitize()` remap route. If it works, the build story is solved.
4. Rewrite the five call sites. Expect `_LMAdapter` to become unnecessary —
   mlx-audio's `Model.__call__(input_ids, input_features, cache)` already returns
   logits — and `_merged_embeddings` likewise, since its merge is already a single
   scatter. Delete them only after the equality check in step 6 passes.
5. If re-quantising is needed: rewrite `tools/quantize_voxtral.py` against
   mlx-audio's loader, rebuild `voxtral-mini-8bit` and `voxtral-small-8bit`,
   re-publish, update the model URLs.
6. **Equality check before anything else is believed:** the same audio through the
   old and new paths must produce the same transcript. `tests/test_voxtral_smoke.py`
   asserts fast == library today; extend it to assert **the number of quantised
   modules is non-zero**, because the failure mode in fact 1 is silent.
7. Re-measure `MEM_MODEL` (see `docs/voxtral-quantisierung.md`, *Decode path and
   memory*) and update the entries.
8. Re-run the two hand-corrected references and FLEURS with
   `docs/skripte/wer.py` and `docs/skripte/fleurs.py`. Expect the numbers to match
   the tables. **If they do not, something in fact 3 is biting — find it, do not
   update the tables.**
9. Update `VOXTRAL.md` (drop the licence warning, rewrite the dependency note),
   `environments/requirements_voxtral_macOS_arm64.txt` (drop the warning block, swap
   the pin), and the *Moving off mlx-voxtral* section of the measurement doc.

## Acceptance

* Transcript identical to the current engine on the same audio, or the difference
  explained and measured.
* `venv/bin/python3 -m pytest tests/ -q` green, including the new
  quantised-module-count assertion.
* WER/CER on both references and FLEURS within noise of the published tables.
* A throwaway PyInstaller build starts and transcribes.
* No `mlx_voxtral` left anywhere: `grep -rn mlx_voxtral . --include="*.py"` empty.

## Traps

* **Never trust a silent load.** Assert quantised-module count, not just "it loaded".
* **Do not re-run the bit sweeps.** Fact 2 says they still hold; re-running them is
  days of compute for a known answer.
* **Do not touch the pinned venv** until the approach is settled.
* **Do not infer frozen behaviour.** Build it.
* **Do not delete `_merged_embeddings`' dtype guard reflexively.** It costs nothing
  and protects against a build whose dtypes disagree.
* Read the docstrings in `voxtral_engine.py` before changing a constant. Several
  encode a defect that was expensive to find.

## Rollback

The current state is committed and pushed on `local/main`. If the migration stalls,
the pins are stable and nothing is broken — revert and leave the licence warning in
place. The measurement documents describe the pre-migration state accurately.
