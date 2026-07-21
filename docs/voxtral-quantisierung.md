# Voxtral: which build, and why

Everything here was measured on an M1 Max (32 GB) with `mlx-voxtral` 0.0.4.
The scripts are in `docs/skripte/`, the builds are made with
`tools/quantize_voxtral.py`.

## The short answer

Ship **`voxtral-mini-8bit`**: the 3B weights at 8 bit with the **audio encoder
left in bf16**. On hard German interview audio it reproduces the unquantised
transcript word for word at 4.5x the speed, in 7.7 GB.

## How the measurements are made

Two yardsticks, because they disagree — and the disagreement is the finding.

**A hand-corrected passage of real podcast audio** (`Audiotest2/referenz/`):
two minutes, 422 words, chosen for what makes transcription hard — brand names,
foreign words, two speakers talking over each other. Corrected by ear against
the recording.

**FLEURS German** (`docs/skripte/fleurs.py`): the benchmark the Voxtral report
uses (arXiv:2507.13264), so the numbers can be held against a published figure.
Read-aloud, clean, one speaker.

Both are scored with the same metric (`docs/skripte/wer.py`):

- **WER** — word error rate, the usual measure.
- **CER** — character error rate on the text with all spaces removed. German
  lets the writer choose between "Balanceöl" and "Balance Öl", or "Hokuspokus"
  and "Hocus Pocus"; identical speech, different typing. WER charges those as
  errors, CER cannot see them. **Where WER and CER disagree, the difference was
  orthographic; where both move together, the model misheard.**

The reference marks overlapping speech as `//text//`. A transcript that drops
those words is not wrong — the model is asked for the dominant voice — so they
are counted separately rather than as errors.

Generation is deterministic (`temperature=0.0`): the same build on the same
audio produces byte-identical output across runs. Differences between builds
are real, not run-to-run noise.

## What each part of the model is worth

A Voxtral build has three parts that can carry different precision:

| Part | Size | Runs | Precision matters for |
|---|---|---|---|
| audio encoder + projector | ~0.6B in **both** model sizes | once per pass | accuracy, not speed |
| `lm_head` | 0.4B (3B model) / 0.7B (24B) | once per **generated token** | speed, not accuracy |
| language model body | the rest | per token | memory, mostly |

That the encoder is *the same size in both models* is why it is cheap to
protect in the 24B build and merely cheap in the 3B one.

### The encoder wants 8 bit (measured, isolated)

24B at 4 bit, `lm_head` and language model held fixed, only the encoder varied:

| Encoder | WER | CER |
|---|---:|---:|
| 4 bit | 10.19 % | 4.87 % |
| 6 bit | 9.72 % | 4.28 % |
| **8 bit** | **8.29 %** | **2.46 %** |
| bf16 | 8.29 % | 2.41 % |

Monotone, and the step from 6 to 8 bit halves the character error rate. WER and
CER move together, so these are genuine mishearings. **Above 8 bit nothing more
is gained** — bf16 matches 8 bit.

This matters because the ready-made `voxtral-small-4bit` build ships its encoder
at 6 bit, and the locally-converted 6-bit build had it at 6 bit too.

### `lm_head` wants to stay quantised

Same experiment, `lm_head` varied instead:

| `lm_head` | WER (24B) | WER (3B) | Speed (3B) |
|---|---:|---:|---:|
| 6 bit | 8.29 % | 4.98 % | 6.74x |
| 8 bit | 8.29 % | 4.27 % | 6.72x |
| bf16 | 8.29 % | 4.27 % | **4.93x** |

Leaving it dense buys nothing and costs 27 % throughput, because it runs on
every generated token. 8 bit it is.

### The resulting 3B build

| Build | WER | Speed | Peak (120 s) |
|---|---:|---:|---:|
| 8 bit uniform | 4.74 % | 6.68x | 7.2 GB |
| **8 bit, encoder bf16** | **4.27 %** | **6.60x** | 7.7 GB |
| bf16 | 4.27 % | 1.46x | 13.2 GB |

The recommended build is byte-identical to bf16 on the reference passage. The
encoder costs 0.5 GB and no measurable speed, because it runs once per pass.

## The two yardsticks disagree — and that is the point

| | hard podcast passage | FLEURS (clean, read) |
|---|---:|---:|
| voxtral-mini-8bit | **4.27 %** | 4.81 % |
| voxtral-mini-8bit uniform | 4.74 % | **4.81 %** |
| voxtral-small-4bit | 7.82 %* | **2.82 %** |

*best 24B configuration found (encoder bf16, `lm_head` 4 bit)

Two things flip:

1. **On clean audio the 24B model wins decisively** (2.82 % vs 4.81 %); on the
   hard passage the 3B model wins by nearly as much. Model choice follows the
   material, not a ranking.
2. **On FLEURS the encoder precision is invisible** — both 3B builds score
   exactly 4.81 % — while on the hard passage it is worth 0.5 points. This
   matches the published finding that quantisation damage concentrates on
   difficult audio and hides on clean test sets. A clean-audio A/B would have
   concluded the encoder does not matter.

**Weight of evidence:** the FLEURS numbers rest on 25 minutes and 100
recordings; the podcast numbers on a single two-minute passage of 422 words.
The "3B wins on hard audio" conclusion is the thinner of the two and deserves a
second hand-corrected passage.

## The 24B model has a ceiling here

Sweeping encoder (4/6/8/bf16), `lm_head` (4/6/8/bf16) and language model (4/6)
puts every 24B configuration between **7.8 % and 10.2 %** on the hard passage,
against 4.27 % for the 3B build — at 5x the runtime and 3x the memory. More bits
in the language model do not help: 4-bit scores 7.82 %, 6-bit 8.06 %.

The 8-bit 24B build needs 26.4 GB and runs at 0.80x realtime — slower than the
recording. Not usable on 32 GB.

## Against Whisper

Same passage, same metric, noScribe's own Whisper settings:

| Model | WER | CER | Sub | Del | **Ins** | Speed |
|---|---:|---:|---:|---:|---:|---:|
| **voxtral-mini-8bit** | **4.27 %** | 3.39 % | 10 | 8 | **0** | 6.79x |
| voxtral-small-4bit (best) | 7.82 % | 2.11 % | 20 | 5 | 8 | 1.77x |
| whisper-fast | 8.06 % | 3.34 % | 22 | 4 | 8 | 2.43x |
| whisper-precise | 14.22 % | 8.75 % | 22 | 4 | **34** | 2.65x |

The error *profiles* differ more than the totals:

- **Voxtral omits.** Eight deletions, zero insertions. It dropped a whole
  parenthetical ("bitte höre das jetzt in Anführungszeichen") but invented
  nothing.
- **Whisper invents.** `precise` transcribed that parenthetical correctly, and
  got "Tohuwabohu" and "Hokuspokus" right where Voxtral failed — then
  hallucinated 27 consecutive words of nonsense at the end of the clip,
  including a product that does not exist ("Omega 2"). Its loop detector fired
  at every temperature step.

For interview and podcast work this favours Voxtral: an omission is visible
when proof-reading, a fluent hallucination is not. Note also that the larger
Whisper model is the *worse* one here — `precise` loops where `fast` does not.

### The same four models on clean audio

| FLEURS German (100 recordings, 25 min) | WER | CER | Speed |
|---|---:|---:|---:|
| voxtral-small-4bit | **2.82 %** | 0.76 % | 2.03x |
| whisper-precise | 4.05 % | 1.40 % | 4.11x |
| whisper-fast | 4.05 % | 1.40 % | 4.01x |
| voxtral-mini-8bit | 4.81 % | 1.44 % | 7.64x |

The ranking inverts almost completely against the hard passage. What the two
tables say together is not "model X is best" but **how far each model falls
when the audio gets hard**:

| | clean | hard | change |
|---|---:|---:|---|
| voxtral-mini-8bit | 4.81 % | 4.27 % | **improves** |
| voxtral-small-4bit | 2.82 % | 7.82 % | 2.8x worse |
| whisper-fast | 4.05 % | 8.06 % | 2.0x worse |
| whisper-precise | 4.05 % | 14.22 % | **3.5x worse** |

For interview work the second column is the one that matters, and the model
with the worst clean-audio score is the one that holds up.

(The two Whisper builds scoring identically is not a measurement error: they
are different models — 819 MB and 1618 MB — but produce byte-identical output
on all 15 clips checked. On read-aloud speech both saturate; the larger model
can only show a difference on difficult audio, where its failure mode is
hallucination.)

## Reproducing

```bash
# the recommended build
python tools/quantize_voxtral.py mistralai/Voxtral-Mini-3B-2507 \
    models/voxtral-mini-8bit 8 64 dense-encoder --lm-head-bits 8

# score it
python docs/skripte/wer.py Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav models/voxtral-mini-8bit
python docs/skripte/fleurs.py 100 models/voxtral-mini-8bit whisper:precise
```

Each build is ~20 seconds to make and 5–21 GB on disk. On macOS, remember that
hourly Time Machine snapshots keep deleted builds alive: reclaim with
`tmutil thinlocalsnapshots / <bytes> 4`.

## Decode path and memory (what moves peak, what doesn't)

Decoding runs through Apple's maintained `mlx_lm.generate_step` (a thin adapter
bridges our LM), which chunks the audio prompt instead of one forward. That is
byte-identical to the reference decode but ~7% faster and ~18% lower peak on long
passes. Four memory levers were then measured and mostly rejected:

- **Peak is set during *prefill*** (processing the audio prompt), not generation:
  prefill+5 tokens already peaks at 7.7 GB vs 8.7 GB for a full 600 s pass. The
  floor is model weights + full KV cache, both irreducible for a bidirectional,
  full-context model.
- **`prefill_step_size` is not a lever** — 512/1024/2048 give an identical peak.
- **KV-cache quantization makes it *worse*, not better.** Because our "context" is
  a huge audio prompt, quantizing the cache adds a dequant transient during
  prefill attention that scales with prompt length and exceeds the
  generation-phase saving (16-bit slope 4.3 MB/s vs 8-bit 5.7 MB/s). It wins only
  for short-prompt/long-generation (chat), the opposite of transcription.
- **KV-cache *size* cannot unlock the 24B model on 32 GB either.** small-8bit is
  25 GB of weights and small-6bit ~19 GB; after weights + headroom there is ~0 GB
  left for the cache on 32 GB. Only small-4bit (~13 GB) fits, and its accuracy was
  already rejected. The weights are the wall.

Net: `MEM_MODEL` for `mini8` is recalibrated to the generate_step path
(`peak ~= 6.5 + 0.0060*s`, from a 180-1200 s fresh-process sweep with margin),
which lets 16-24 GB machines run longer single passes; 32 GB is unchanged
(already capped by the context limit).

The streaming **Voxtral-Mini-4B-Realtime** model (Awni Hannun's
[voxmlx](https://github.com/awni/voxmlx) runs it with a bounded rotating KV cache)
was considered as a low-memory option, then ruled out on Mistral's own published
numbers: on German FLEURS it scores 6.19% WER at its 480ms setting and 4.15% even
at 2.4s delay — worse than the offline Voxtral Mini 3B (3.54%), a smaller model.
The causal/streaming architecture trades look-ahead for latency, and on hard
conversational audio the gap would only widen. Its advantages (sub-500ms latency,
bounded memory) are irrelevant to offline file transcription. Not adopted.

## What is still open

- Voxtral drops parentheticals. Whether that is steerable is untested.
- The measurements are tied to this MLX version and macOS release; re-run the
  scripts after an upgrade before trusting the memory model.
