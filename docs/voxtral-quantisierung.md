# Voxtral: which build, and why

Everything here was measured on an M1 Max (32 GB) with `mlx-voxtral` 0.0.4.
The scripts are in `docs/skripte/`, the builds are made with
`tools/quantize_voxtral.py`.

## The short answer

Ship **`voxtral-mini-8bit`**: the 3B weights at 8 bit with the **audio encoder
left in bf16**. On hard German interview audio it reproduces the unquantised
transcript word for word at 4.5x the speed, in 7.7 GB.

That recommendation rests on speed and on readable output, **not** on
recognition accuracy. Measured by character error the 24B model hears better,
and at 4 bit it does fit on a 32 GB machine. Both are worked out below.

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
is gained** — bf16 matches 8 bit, and the 0.05-point CER difference between them
is one character out of 2034.

This matters because the ready-made `voxtral-small-4bit` build ships its encoder
at 6 bit, and the locally-converted 6-bit build had it at 6 bit too.

**On the 24B model the threshold is 8 bit. On the 3B model there is no
threshold in this range at all** — see "Does the 3B encoder need the bits?"
below. The two models do not behave the same way, and the 24B result must not
be carried over to the 3B one.

### `lm_head` wants to stay quantised

Same experiment, `lm_head` varied instead:

| `lm_head` | WER (24B) | CER (24B) | WER (3B) | CER (3B) | Speed (3B) |
|---|---:|---:|---:|---:|---:|
| 4 bit | 7.82 % | 2.11 % | — | — | — |
| 6 bit | 8.29 % | 2.41 % | 4.98 % | 3.83 % | 6.74x |
| 8 bit | 8.29 % | 2.46 % | 4.27 % | 3.39 % | 6.72x |
| bf16 | 8.29 % | 2.46 % | 4.27 % | 3.39 % | **4.93x** |

Leaving it dense buys nothing and costs 27 % throughput, because it runs on
every generated token. 8 bit it is.

**On the 24B model this column is noise, and it runs backwards.** More bits
score slightly *worse* (2.11 → 2.41 → 2.46), and the whole spread is seven
characters out of 2034. Under greedy decoding a single token choice flips and
the text diverges from there on — the same effect the similarity comparison in
`docs/messungen/` warns about. So `lm_head` precision is simply irrelevant on
the 24B model; pick 4 bit for the speed. On the 3B model the 6→8 step is nine
characters plus two extra deletions, marginally above that noise floor, and 8
bit is free anyway.

### The resulting 3B build

| Build | WER | CER | Speed | Peak (120 s) |
|---|---:|---:|---:|---:|
| 8 bit uniform | 4.74 % | **3.34 %** | 6.68x | 7.2 GB |
| **8 bit, encoder bf16** | **4.27 %** | 3.39 % | **6.60x** | 7.7 GB |
| bf16 | 4.27 % | 3.39 % | 1.46x | 13.2 GB |

The recommended build is byte-identical to bf16 on the reference passage. The
encoder costs 0.5 GB and no measurable speed, because it runs once per pass.

**But the two metrics disagree about whether it is worth anything.** The uniform
build differs from the recommended one by exactly two insertions — same
substitutions, same deletions — and its CER is a hair *lower*. Two insertions at
roughly zero character cost means two compound words were written apart
("Balance Öl" for "Balanceöl"). By the rule stated at the top of this document
that is orthography, not recognition. FLEURS says the same thing on 100
recordings: 4.81 % for both, CER 1.40 % uniform against 1.44 % with the bf16
encoder.

Neither scoring run, though, has the resolution to settle what the encoder's
precision is worth. The next chapter takes that question to 26 619 words.

## Does the 3B encoder need the bits?

Three builds, identical except for the audio encoder — language model and
`lm_head` at 8 bit throughout, only `audio_tower` and `projector` vary. Scored
on the hand-corrected passage, with bootstrap intervals over 10 000 resamples
(`docs/skripte/bootstrap_cer.py`):

| Encoder | WER | CER | paired difference vs bf16 |
|---|---:|---:|---|
| bf16 (shipped) | 4.27 % | 3.39 % | — |
| 8 bit | 4.74 % | 3.34 % | dWER −0.47 [−1.73, +0.50], dCER +0.05 [−0.14, +0.29] |
| **6 bit** | **4.03 %** | **3.29 %** | dWER +0.24 [+0.00, +0.74], dCER +0.10 [+0.00, +0.30] |

**No paired interval excludes zero, and the 6-bit encoder is nominally the best
of the three.** The 24B model's clean 8-bit threshold does not transfer.

These intervals are also what the passage's resolution actually is, and it is
worth writing down: a single build's WER is 4.27 % **[1.90, 7.43]**, i.e. ±3
points. Comparing two builds paired on the same blocks is far sharper (±0.3),
because the shared difficulty of the material cancels — but a lone score from
this passage carries an error bar three points wide, and half-point differences
between builds are below what it can see.

422 words therefore cannot settle the encoder question, so the same question
went to 26 619 words of hard German — the Swiss-German interview, a podcast, and
the pathological window — with no reference at all
(`docs/skripte/encoder_diff.py`). Both builds transcribe identical fixed
windows; every difference between the two transcripts is classified by character
similarity, on the theory that a build which only *spells* differently produces
near-identical spans:

| vs the bf16 build | 8-bit encoder | 6-bit encoder |
|---|---:|---:|
| differing spans (unique) | 104 (88) | 139 (109) |
| of those acoustic (different word / omission) | 58 = 56.9 % | 77 = 56.2 % |
| acoustic per 1000 words | 2.18 | 2.89 |
| total words against bf16 | +7 | +27 |
| omission balance | −8 words | +8 words |

**The differences are not merely orthographic.** Roughly 57 % of them are a
different word heard, or a word one build emitted and the other did not — and
the divergence scales with the bit reduction, 104 spans at 8 bit against 139 at
6. The encoder's precision does something acoustic.

**Which direction it cuts is not established.** The diff is symmetric: bf16 is
the reference point, not the truth. Omissions come out balanced both ways (−8
and +8 words), the lower-precision builds emit slightly *more* text rather than
dropping speech, and where ground truth does exist — the hand-corrected passage
— the 6-bit build scored best. Individual cases point both ways: the 8-bit
encoder produced "buttercroissants" where the bf16 one produced "buttercrosons".

**Hence the shipped build keeps bf16.** Not because it measurably hears better —
it does not — but because the encoder's precision demonstrably changes what is
heard while the direction is unknown, and 0.5 GB is a cheap price for holding
the highest precision until that is resolved.

**And it stays unresolved, deliberately.** Resolving it would mean listening to
the ~109 localised disagreements and judging each by ear. That is the cheapest
route there is — far cheaper than a second hand-corrected reference passage —
and it is still an evening of someone's attention for a question whose answer
changes nothing: bf16 is kept either way, because it is the safe side of an
open question and costs 0.5 GB. Do not re-propose it.

## Clean audio and hard audio rank the models differently

| | hard podcast passage | FLEURS (clean, read) |
|---|---:|---:|
| voxtral-mini-8bit | **4.27 %** | 4.81 % |
| voxtral-mini-8bit uniform | 4.74 % | **4.81 %** |
| voxtral-small-4bit | 7.82 %* | **2.82 %** |

*best 24B configuration found (encoder bf16, `lm_head` 4 bit)

Read by word error rate the ranking inverts between the two sets: the 24B model
wins decisively on clean audio, the 3B model on the hard passage. That inversion
is an artefact of the word metric — it is a statement about spelling read as one
about hearing. The next chapter separates the two.

**Weight of evidence:** the FLEURS numbers rest on 25 minutes and 100
recordings; the podcast numbers on a single two-minute passage of 422 words and
2034 characters. One character is 0.049 CER points, one word 0.237 WER points,
and the bootstrap above puts a single build's error bar at ±3 WER points. Only
the paired comparisons on this passage are sharp enough to carry weight.

## 24B against 3B: one hears better, the other writes better

Sweeping encoder (4/6/8/bf16) and `lm_head` (4/6/8/bf16) puts every 24B
configuration between **7.8 % and 10.2 %** WER on the hard passage, against
4.27 % for the 3B build. Read that way there is a wall, and the sweep says
nothing about where it comes from: seventeen builds, no gradient.

Ranked by CER the same seventeen builds separate cleanly into two groups, and
the 24B model is the better half:

| Rank | Build | CER | WER | Char. errors | Word errors |
|---|---|---:|---:|---:|---:|
| 1–2 | small, encoder bf16, `lm_head` 4 bit | **2.11 %** | 7.82 % | 43 | 33 |
| 3–4 | small, encoder ≥ 8 bit, `lm_head` 6 bit | 2.41 % | 8.29 % | 49 | 35 |
| 5–7 | small, encoder ≥ 8 bit, `lm_head` 8/bf16 | 2.46 % | 8.29 % | 50 | 35 |
| 8 | mini 8 bit uniform | 3.34 % | 4.74 % | 68 | 20 |
| 9–13 | **mini 8 bit, encoder bf16** (shipped), dense, bf16 | 3.39 % | 4.27 % | 69 | 18 |
| 14 | mini 8 bit, `lm_head` 6 bit | 3.83 % | 4.98 % | 78 | 21 |
| 15–17 | small, encoder < 8 bit | 4.03–4.87 % | 9.48–10.19 % | 82–99 | 40–43 |

**Every 24B build with an 8-bit-or-better encoder beats every 3B build**, with a
gap of 0.88 CER points between the groups — far outside the noise floor. The
24B model also has the lowest CER of anything measured here, Whisper included.

The clearest single statistic is **characters per word error**: 3.8 for the 3B
build, 1.3 for the 24B one. The 24B model's errors are on average a third the
size of the 3B model's. It misspells; the 3B model mishears.

Note which way that measurement is biased. CER runs on space-free text, so the
24B model's eight insertions — the verbatim false starts and filler the
hand-corrected reference smooths away — are charged against it at full character
length. It wins anyway.

In short: **the 24B model hears this passage distinctly better, and the 3B model
writes it down better.** Choosing the 3B build is right on readability and on
6.6x versus 1.8x realtime — not on recognition accuracy.

What the sweep establishes about precision is narrow: **on the 24B model the
audio encoder is the only part whose bit width matters, and its threshold is 8
bit.** By CER the encoder row is cleanly monotone (4.87 → 4.28 → 2.46 → 2.41);
the 6→8 step is 37 characters, the 8→bf16 step is one. Every other column in the
sweep is noise. None of this carries over to the 3B model, which shows no such
threshold.

The 8-bit 24B build needs 26.4 GB and runs at 0.80x realtime — slower than the
recording. Not usable on 32 GB. The 4-bit build is a different matter: see
"Running the 24B model on 32 GB" below.

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

The CER column ranks these four differently from the WER column, and the gap
between `whisper-fast` and the 24B Voxtral build is where it matters most: near
identical word error (8.06 % vs 7.82 %), but 3.34 % against 2.11 % in
characters. Those two transcripts are not of comparable quality; one of them is
mostly misspelling what it heard correctly.

### The same four models on clean audio

| FLEURS German (100 recordings, 25 min) | WER | CER | Speed |
|---|---:|---:|---:|
| voxtral-small-4bit | **2.82 %** | 0.76 % | 2.03x |
| whisper-precise | 4.05 % | 1.40 % | 4.11x |
| whisper-fast | 4.05 % | 1.40 % | 4.01x |
| voxtral-mini-8bit | 4.81 % | 1.44 % | 7.64x |

The interesting question the two tables answer together is **how far each model
falls when the audio gets hard**. By word error the 3B model appears to get
*better* on hard audio, which is not a thing that happens — a good sign the word
metric is measuring something other than recognition. By character error the
picture is ordinary:

| | clean CER | hard CER | change | (clean WER | hard WER) |
|---|---:|---:|---|---:|---:|
| voxtral-mini-8bit | 1.44 % | 3.39 % | 2.4x worse | 4.81 % | 4.27 % |
| voxtral-small-4bit | 0.76 % | 2.11 % | 2.8x worse | 2.82 % | 7.82 % |
| whisper-fast | 1.40 % | 3.34 % | 2.4x worse | 4.05 % | 8.06 % |
| whisper-precise | 1.40 % | 8.75 % | **6.3x worse** | 4.05 % | 14.22 % |

Hard audio costs every sane model roughly 2.5x in character error, and the
ranking from the clean set survives into the hard one. The outlier is
`whisper-precise`, which does not degrade but breaks: its 6.3x is hallucination,
a different failure altogether. That — not a reversal of model quality — is the
finding that should drive the engine choice for interview work.

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

# score it (also leaves the transcripts in /tmp/wer_<build>.txt)
python docs/skripte/wer.py Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav models/voxtral-mini-8bit
python docs/skripte/fleurs.py 100 models/voxtral-mini-8bit whisper:precise

# how much of a build-to-build difference the passage can actually resolve
python docs/skripte/bootstrap_cer.py Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    voxtral-mini-8bit voxtral-mini-8bit-enc6

# compare two builds over hours of audio, no reference needed
python docs/skripte/encoder_diff.py models/voxtral-mini-8bit \
    models/voxtral-mini-8bit-enc6 --audio <file> [file ...]

# the Parakeet comparison (needs `pip install parakeet-mlx`, which is NOT in
# the requirements -- it pulls only dacite on top of what is already installed,
# and mlx 0.32.0 satisfies its floor, so it can be added and removed without
# disturbing the pinned Voxtral stack)
python docs/skripte/engines/parakeet_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav 5      # trailing 5 = beam width
python docs/skripte/engines/parakeet_wer.py fleurs 100

# the Qwen3-ASR comparison (no extra dependency -- transformers 5.13+ has it)
python docs/skripte/engines/qwen_asr_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav chunk60
python docs/skripte/engines/qwen_asr_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav "vocab:Xtend, BalanceOil"
python docs/skripte/engines/qwen_asr_wer.py fleurs 100

# VibeVoice-ASR: same yardsticks, plus its own diarization
python docs/skripte/engines/vibevoice_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav
python docs/skripte/engines/vibevoice_wer.py fleurs 100

# Cohere Transcribe (gated repo -- accept on the model page first; its Xet
# transfer is broken, so HF_HUB_DISABLE_XET=1 for the download)
python docs/skripte/engines/cohere_asr_wer.py tokens
python docs/skripte/engines/cohere_asr_wer.py ref \
    Audiotest2/referenz/hart_780-900_REFERENZ.txt \
    Audiotest2/referenz/hart_780-900.wav plain
python docs/skripte/engines/cohere_asr_wer.py fleurs 100
```

The four scripts under `docs/skripte/engines/` share a README and the same three
yardsticks; see [`docs/skripte/engines/README.md`](skripte/engines/README.md).

Each build is ~20 seconds to make and 5–21 GB on disk. On macOS, remember that
hourly Time Machine snapshots keep deleted builds alive: reclaim with
`tmutil thinlocalsnapshots / <bytes> 4`.

`encoder_diff.py` writes its raw event list outside the repository
(`NOSCRIBE_MESS_DIR`, otherwise the temp directory): it quotes the audio
verbatim, and the audio is real interview material. Only the aggregate counts
belong in `docs/`.

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
- **KV-cache *size* cannot unlock the 8-bit 24B model on 32 GB either.**
  small-8bit is 25 GB of weights and small-6bit ~19 GB; after weights + headroom
  there is ~0 GB left for the cache on 32 GB. The weights are the wall. What
  *does* fit is small-4bit — see the next chapter.

Net: `MEM_MODEL` for `mini8` is recalibrated to the generate_step path
(`peak ~= 6.5 + 0.0060*s`, from a 180-1200 s fresh-process sweep with margin),
which lets 16-24 GB machines run longer single passes; 32 GB is unchanged
(already capped by the context limit).

## Running the 24B model on 32 GB

Only the 8-bit and 6-bit 24B builds are out of reach on 32 GB. The 4-bit one is
not, and it is the configuration the measurements point to anyway.

The bit sweep, read by CER, says the language-model body's precision does not
matter — only the encoder's does, and only up to 8 bit. The build that follows
from that is **`small-4bit-denc`: a 4-bit language model with a bf16 audio
encoder**, which is also the best-scoring 24B configuration measured (CER
2.11 %). It weighs ~13 GB and peaked at 15.6 GB on a 30 s pass and 17.0 GB on a
120 s one.

`_auto_chunk_sec()` sizes it automatically on a 32 GB machine:

| Build | auto pass length on 32 GB |
|---|---:|
| mini-8bit | 600 s (capped at the longest measured window) |
| **small-4bit** | **576 s (9.6 min)** |
| small-6bit | 303 s (5.0 min) |
| small-8bit | refused — needs ~28 GB for its shortest pass |

So the 24B model does run, in ten-minute passes, at ~1.8x realtime — a four-hour
recording in something over two hours. Two caveats worth knowing before choosing
it:

- The `small` entry in `MEM_MODEL` still carries **one-shot-prefill** numbers.
  Switching to `generate_step` cut the 3B model's slope by ~2.5x; nobody has
  re-measured the 24B build since. The real pass length is likely well above
  576 s, so this table is conservative in the safe direction.
- It is the more verbatim of the two models. That is a style choice, not an
  accuracy one — see the chapter above.

Reproduce the build with:

```bash
python tools/quantize_voxtral.py mistralai/Voxtral-Small-24B-2507 \
    models/voxtral-small-4bit-denc 4 64 dense-encoder --lm-head-bits 4
```

## Other options considered

### Voxtral-Mini-4B-Realtime

The streaming **Voxtral-Mini-4B-Realtime** model (Awni Hannun's
[voxmlx](https://github.com/awni/voxmlx) runs it with a bounded rotating KV cache)
was considered as a low-memory option, then ruled out on Mistral's own published
numbers: on German FLEURS it scores 6.19% WER at its 480ms setting and 4.15% even
at 2.4s delay — worse than the offline Voxtral Mini 3B (3.54%), a smaller model.
The causal/streaming architecture trades look-ahead for latency, and on hard
conversational audio the gap would only widen. Its advantages (sub-500ms latency,
bounded memory) are irrelevant to offline file transcription. Not adopted.

### Parakeet-TDT (measured, rejected)

`nvidia/parakeet-tdt-0.6b-v3` is the obvious structural alternative: a
FastConformer encoder with a Token-and-Duration Transducer decoder, 0.6B
parameters, CC-BY-4.0, 25 European languages including German, and — unlike
anything else here — distributed as MLX, ONNX, CoreML and GGUF, so it would run
on Windows and Linux CPUs as well. A transducer also cannot do several things
this engine has to defend against: it emits tokens bound to audio frames, so
repetition loops, whole-chunk language drift and a dropped head are structurally
impossible, and word timestamps fall out of the predicted durations instead of
needing a CTC forced aligner.

Measured with `docs/skripte/engines/parakeet_wer.py` (which reuses `norm` and `wer` from
`wer.py` unchanged, so the metric is identical), via `parakeet-mlx` 0.5.2 at its
default bf16:

| Hard passage (422 words) | WER | CER | Sub | Del | Ins | Speed |
|---|---:|---:|---:|---:|---:|---:|
| voxtral-mini-8bit | **4.27 %** | **3.39 %** | 10 | 8 | 0 | 6.79x |
| whisper-fast | 8.06 % | 3.34 % | 22 | 4 | 8 | 2.43x |
| parakeet-tdt-0.6b-v3, beam 5 | 14.69 % | 6.49 % | 49 | 6 | 7 | 9.13x |
| parakeet-tdt-0.6b-v3, greedy | 18.01 % | 9.88 % | 49 | 20 | 7 | 26.47x |

The second hand-corrected reference was scored alongside it — 859 words, five
minutes, a video-call recording, cleaner than the hard passage but still real
conversation:

| Second reference (859 words) | WER | CER | Sub | Del | Ins | Speed |
|---|---:|---:|---:|---:|---:|---:|
| voxtral-mini-8bit | **0.81 %** | **0.64 %** | 2 | 4 | 1 | 7.58x |
| parakeet-tdt-0.6b-v3, beam 5 | 8.50 % | 5.18 % | 34 | 15 | 24 | 10.30x |
| parakeet-tdt-0.6b-v3, greedy | 10.59 % | 6.81 % | 36 | 14 | 41 | 41.75x |

| FLEURS German (100 recordings) | WER | CER | Speed |
|---|---:|---:|---:|
| whisper-precise | **4.05 %** | **1.40 %** | 4.11x |
| voxtral-mini-8bit | 4.81 % | 1.44 % | 7.64x |
| parakeet-tdt-0.6b-v3, greedy | 4.81 % | 2.15 % | **44.45x** |

**The benchmark is worthless here, and that is the finding.** On FLEURS the two
models tie to the second decimal. On real conversation Parakeet is an order of
magnitude behind — 0.81 % against 8.50 % on 859 words, far outside any noise
floor this document works with.

**The failure has a name: uncontrolled code-switching.** Parakeet v3 is
multilingual with no language conditioning — the model has no language token, so
`transcribe()` has no `language` argument to pass. When the acoustics get
ambiguous it writes German function words as their English homophones: *und* as
"and", *wenn* as "when", *es ist* as "it is", *gut* as "good", *ja* as "yeah".
Counted: **14 English function-word tokens in 407 words** on the hard passage,
**0 in 882 words** on the cleaner second reference. It is triggered by audio
quality, and there is no input that suppresses it.

The error *profile* is the second problem. 49 substitutions against Voxtral's 10
on the same passage: Parakeet replaces where Voxtral omits, and a wrong word
survives proof-reading in a way a missing one does not — the same argument that
decided this engine against `whisper-precise`.

Beam search (width 5) is worth having if the model is ever revisited: it cuts
deletions from 20 to 6 and roughly 3 WER points, for two thirds of the
throughput. It does not touch the code-switching.

Not adopted. The result says nothing about Parakeet in English or in the other
24 languages, and nothing about the ONNX build's speed on a CPU — only that it
is the wrong engine for German conversational audio.

### Qwen3-ASR-1.7B (measured, not adopted)

`Qwen/Qwen3-ASR-1.7B-hf` is the Voxtral principle at half the size: an audio
encoder in front of a Qwen3-Omni language model, Apache-2.0, 30 languages. Two
things make it easier to try than anything else here — **transformers supports it
natively** (`AutoModelForMultimodalLM`, no new dependency at all in this venv),
and it takes an explicit language as well as a free-form context prompt.

Measured with `docs/skripte/engines/qwen_asr_wer.py`, bf16 on MPS, same metric:

| Hard passage (422 words) | WER | CER | Sub | Del | Ins |
|---|---:|---:|---:|---:|---:|
| voxtral-mini-8bit | **4.27 %** | **3.39 %** | 10 | 8 | 0 |
| whisper-fast | 8.06 % | 3.34 % | 22 | 4 | 8 |
| qwen3-asr-1.7b, 60 s chunks | 10.90 % | 4.33 % | 29 | 5 | 12 |
| qwen3-asr-1.7b, one pass | 11.14 % | 4.08 % | 31 | 4 | 12 |
| qwen3-asr-1.7b, auto language | 11.85 % | 4.33 % | 32 | 5 | 13 |

| Second reference (859 words) | WER | CER | Sub | Del | Ins |
|---|---:|---:|---:|---:|---:|
| voxtral-mini-8bit | **0.81 %** | **0.64 %** | 2 | 4 | 1 |
| qwen3-asr-1.7b, 60 s chunks | 10.83 % | 6.69 % | 36 | 16 | 41 |
| qwen3-asr-1.7b, one pass | 12.11 % | 7.01 % | 43 | 17 | 44 |

| FLEURS German (100 recordings) | WER | CER |
|---|---:|---:|
| **qwen3-asr-1.7b** | **3.96 %** | **1.23 %** |
| whisper-precise | 4.05 % | 1.40 % |
| voxtral-mini-8bit | 4.81 % | 1.44 % |
| parakeet-tdt-0.6b-v3 | 4.81 % | 2.15 % |

**It wins the benchmark outright and still loses the job.** On FLEURS German it
is the best model this document has measured — ahead of Whisper and of the
shipped Voxtral build, in words and in characters. On the two hand-corrected
conversational references it is two to thirteen times behind Voxtral. That is
the same lesson the Parakeet section teaches, and it is worth stating once more
in the strongest form available: **a model can top the read-aloud benchmark and
be unusable for interview work.**

Unlike Parakeet the cause is not code-switching — there is none, and forcing the
language buys only 0.7 WER points over auto-detect. It simply hears this
material less well.

Three findings from the run that outlive the verdict:

**Punctuation collapses on long input, and chunking fixes it.** Fed the whole
120 s or 300 s clip, the model returns text with *zero* commas and *zero* full
stops. Cut into ~60 s windows it punctuates normally — 8.73 and 10.33 commas per
100 words, against Whisper's 10.62 and Voxtral's 10.87. A length sweep on the
same audio puts the usable band at roughly 30–90 s. The card advertises long
audio; for German prose output it does not hold, and any engine built on this
model would have to chunk far more aggressively than Voxtral does.

**The context prompt is a real hotword mechanism — the thing Voxtral lacks.**
With `Vocabulary: …` in the system message, "Extent" becomes "Xtend", "Balance
Oil" becomes "BalanceOil" and a mangled compound comes back correct, on a
controlled 30 s clip with no other change to the text. Over the full passage it
costs nothing in word error (10.90 % either way) and 0.05 points of character
error. This is exactly what `voxtral_corrections.yml` exists to work around, and
it is the one capability that would argue for the model.

**But it is paid for in punctuation:** the same vocabulary hint halves comma
density, 8.73 to 4.28 per 100 words. A term list behaves as a decode
perturbation here too, just a cheaper one than in Voxtral — same phenomenon as
the `repetition_penalty` default, and a reminder to measure punctuation
whenever a decode-level knob is turned.

Two implementation notes for anyone who picks this up. The `prompt=` argument
documented on `apply_transcription_request` **does not exist** in transformers
5.15.0.dev0 — it lands in `**kwargs` and is dropped with a warning; the context
has to go into a system message next to the language, which is what the chat
template concatenates anyway. And the speed figures on MPS are not worth quoting:
the same 120 s clip measured 2.04x cold and 4.45x warm in the same session. An
MLX port would be the honest place to measure throughput.

### VibeVoice-ASR (measured — the architecture works, the recognition does not)

`microsoft/VibeVoice-ASR-HF` is not another engine behind the same seam. It does
ASR, diarization and timestamping in **one pass** and emits speaker-attributed
segments directly — noScribe's whole pipeline collapsed into one model. MIT
licence, 16.7 GB in bf16, natively supported by transformers (again no new
dependency), 4-bit and 8-bit MLX ports published by mlx-community. Measured with
`docs/skripte/engines/vibevoice_wer.py`, bf16 on MPS.

| Hard passage (422 words) | WER | CER | Sub | Del | Ins | Commas/100w |
|---|---:|---:|---:|---:|---:|---:|
| voxtral-mini-8bit | **4.27 %** | **3.39 %** | 10 | 8 | 0 | 10.87 |
| whisper-fast | 8.06 % | 3.34 % | 22 | 4 | 8 | 10.62 |
| qwen3-asr-1.7b, 60 s chunks | 10.90 % | 4.33 % | 29 | 5 | 12 | 8.73 |
| vibevoice-asr | 13.03 % | 6.98 % | 34 | 2 | 19 | **11.34** |

| Second reference (859 words) | WER | CER | Sub | Del | Ins | Commas/100w |
|---|---:|---:|---:|---:|---:|---:|
| voxtral-mini-8bit | **0.81 %** | **0.64 %** | 2 | 4 | 1 | — |
| qwen3-asr-1.7b, 60 s chunks | 10.83 % | 6.69 % | 36 | 16 | 41 | 10.33 |
| vibevoice-asr | 12.34 % | 7.58 % | 35 | 10 | 61 | **12.57** |

| FLEURS German (100 recordings) | WER | CER |
|---|---:|---:|
| qwen3-asr-1.7b | **3.96 %** | **1.23 %** |
| whisper-precise | 4.05 % | 1.40 % |
| voxtral-mini-8bit | 4.81 % | 1.44 % |
| vibevoice-asr | 8.26 % | 5.84 % |

FLEURS is arguably the wrong test for this model — its design point is an hour
of multi-speaker audio, not a 15-second single-utterance clip, and one of the
100 clips came back as nothing but a `[Silence]` tag. Take the 8.26 % as a lower
bound on what it can do there rather than a verdict.

**The diarization, however, is the real thing.** Compared frame by frame against
noScribe's own pyannote pipeline on the same two clips, 10 ms resolution, best
speaker permutation:

| | speakers | segments | timeline labelled | agreement with pyannote |
|---|---:|---:|---:|---:|
| hard passage | 2 vs 2 | 16 vs 22 | 100 % vs 96.4 % | **98.2 %** |
| second reference | 2 vs 2 | 19 vs 52 | 97.1 % vs 93.3 % | **97.5 %** |

Same speaker count, and near-total agreement on who is speaking — from a model
that produced the transcript in the same forward pass. The segmentation is much
coarser (16 and 19 turns against 22 and 52), which would be a problem for
subtitle cues but not for speaker attribution.

It also punctuates better than anything else measured here — 11.34 and 12.57
commas per 100 words, above Whisper's 10.62 and Voxtral's 10.87 — and it labels
non-speech explicitly (`[Silence]`, `[Human Sounds]`); those tags are stripped
before scoring, and leaving them in costs 3.3 WER points on FLEURS.

**Not adopted, and the reason is only recognition.** Three times Voxtral's word
error on the hard passage, fifteen times on the second reference, and 0.66–0.87x
realtime on MPS — slower than the two-stage pipeline it would replace, in which
pyannote is cheap and Voxtral runs at 6.6x. But the architecture is validated,
which is worth writing down: **a single model really can deliver speaker, time
and text at pyannote-grade diarization quality.** The thing to watch is a model
of this shape that hears German conversation as well as Voxtral does. Nothing
here suggests that is far off.

### Cohere Transcribe (measured — the best challenger so far, and still not close enough)

`CohereLabs/cohere-transcribe-03-2026` came out of the leaderboard below: ~2B
parameters, Apache-2.0, 3.9 GB, transformers-native, and the best open-weight
model on the leaderboard's English long-form tab. Measured with
`docs/skripte/engines/cohere_asr_wer.py`, bf16 on MPS, language forced to `de`.

| Hard passage (422 words) | WER | CER | Sub | Del | Ins | Speed | Commas/100w |
|---|---:|---:|---:|---:|---:|---:|---:|
| voxtral-mini-8bit | **4.27 %** | **3.39 %** | 10 | 8 | 0 | 6.79x | 10.87 |
| whisper-fast | 8.06 % | 3.34 % | 22 | 4 | 8 | 2.43x | 10.62 |
| **cohere-transcribe** | 10.43 % | 3.69 % | 25 | 4 | 15 | **16.58x** | 11.74 |
| qwen3-asr-1.7b, 60 s chunks | 10.90 % | 4.33 % | 29 | 5 | 12 | — | 8.73 |
| vibevoice-asr | 13.03 % | 6.98 % | 34 | 2 | 19 | 0.66x | 11.34 |
| parakeet-tdt-0.6b-v3, beam 5 | 14.69 % | 6.49 % | 49 | 6 | 7 | 9.13x | — |

| Second reference (859 words) | WER | CER | Speed |
|---|---:|---:|---:|
| voxtral-mini-8bit | **0.81 %** | **0.64 %** | 7.58x |
| **cohere-transcribe** | 9.20 % | 6.12 % | **27.93x** |
| qwen3-asr-1.7b, 60 s chunks | 10.83 % | 6.69 % | — |
| vibevoice-asr | 12.34 % | 7.58 % | 0.87x |

| FLEURS German (100 recordings) | WER | CER | Speed |
|---|---:|---:|---:|
| qwen3-asr-1.7b | **3.96 %** | **1.23 %** | — |
| whisper-precise | 4.05 % | 1.40 % | 4.11x |
| **cohere-transcribe** | 4.55 % | 1.85 % | 13.51x |
| voxtral-mini-8bit | 4.81 % | 1.44 % | 7.64x |

**On the hard passage its character error is within noise of Voxtral's** — 3.69
against 3.39, where this document's own noise floor is ~0.15 points and a single
build's word-error interval is ±3. By the measure that tracks what was *heard*
rather than how it was spelled, a 2B model at 16x realtime is level with the 3B
Voxtral build at 6.8x, on the hardest audio here. That is the best result any
challenger has produced.

**The second reference kills it anyway.** 9.20 % against 0.81 %, character error
6.12 against 0.64 — an order of magnitude, far outside anything the error bars
cover, on the larger of the two references. Whatever the hard passage suggests,
this model does not transcribe ordinary German conversation to the standard the
shipped engine does.

Three limitations from its own model card, all of which matter for an engine:

* **No timestamps, no diarization.** The tokenizer knows `<|timestamp|>` and
  `<|diarize|>` and the decoder prompt accepts them — they are leftovers of the
  training format. Setting them changes nothing useful (the diarize arm just
  truncates, 307 words against 426), and the card says plainly that the model
  does not feature either. Word timestamps would still need the CTC aligner.
* **No language detection**, and explicitly inconsistent on code-switched audio.
  noScribe's "Auto" would have to be resolved before the engine is called.
* **It hallucinates on silence** and wants a VAD or noise gate in front. Visible
  here: the first window opens with a header of its own, `Input transcript
  corrected:`, once per file, regardless of what is put in the context slot.
  That is stripped before scoring; leaving it in costs 0.7 WER points.

Not adopted. But it is the first challenger where the gap is about a specific
weakness rather than the whole model, and its ecosystem is the broadest of
anything here — transformers, vLLM, mlx-audio, ONNX, GGUF, a Rust port and a
WebGPU demo. Worth re-measuring when Cohere ships a successor.

*Practical note:* the repo is gated (click-through) and its Xet transfer fails
with `Unable to parse string as hex hash value`. `HF_HUB_DISABLE_XET=1` in front
of the download falls back to plain HTTP and works.

### Moving off mlx-voxtral (evaluated 2026-08-21 — and the licence changes the order)

**The reason to move is no longer maintenance.** `mlx-voxtral` ships under a
"Personal Use License": MIT plus a prohibition on commercial use, including
"using the Software to provide commercial services". noScribe is GPL-3.0, which
does not permit further restrictions, and much of its audience transcribes for
paid work. `mlx-audio` is MIT. That turns this section from a cleanup into a
prerequisite — see the licence note in [`../VOXTRAL.md`](../VOXTRAL.md).

The cheaper resolution is to ask the copyright holder to relicense; the licence
text invites exactly that, and the maintainer is active on GitHub even though
this repository is not. Everything below is what the move costs if that fails.

`mlx-voxtral` has had no release since 2025-08-19. The maintained alternative is
`mlx-audio`, whose `mlx_audio/stt/models/` holds Voxtral next to cohere_asr,
qwen3_asr, vibevoice_asr, parakeet and canary — every engine measured above,
behind one API. The exit route is written up in
[`../VOXTRAL.md`](../VOXTRAL.md); what belongs here is what it would do to the
builds and the numbers.

**Re-quantising changes the file format and nothing else.** MLX's affine
quantisation is data-free — scale and bias come from each group's own min and
max, there is no calibration set and no randomness. Quantising the same tensor
twice returns bit-identical results, and so does quantising a fresh copy of it.
mlx-audio's quantisation predicate is `not p.startswith("audio_tower")`, which
selects exactly the set these builds already carry: **213 modules, all at 8 bit,
group size 64, affine, encoder left dense**. Same weights in, same layer set,
same parameters, therefore the same tensors out. Only the keys change, from
`language_model.layers.0…` to `language_model.model.layers.0…`.

So a re-quantised build would be *the same build*, and the tables above would
still describe it. **Quality and speed would not move because of the
re-quantisation** — quantised matmuls are the same kernels either way.

What could move them is everything around it, and each is a separate check:

* mlx-audio's `_merge_input_embeddings` scatters without promoting dtype first,
  where `_merged_embeddings` promotes deliberately. Measured, this is currently
  moot: on the shipped 8-bit build `get_audio_embeds` and `embed_tokens` both
  return bfloat16, so there is nothing to round away. It stays worth re-checking
  because the failure would be invisible in the text and show up only as
  different logits.
* Its stop-token default includes 32000, an ordinary text token. Left as it is,
  that silently truncates any pass containing the word.
* It vendors its own `generate_step` rather than using `mlx_lm`'s. Same chunked
  prefill (`prefill_step_size=2048`), so the ~18 % peak saving survives — but
  `MEM_MODEL` is calibrated against the current path and would need
  re-measuring before the numbers in *Long audio & memory* could be trusted.

Two things that are *not* concerns, checked rather than assumed. mlx-audio does
not compute the log-Mel at all — it takes `input_features` straight from
transformers' `VoxtralProcessor`, i.e. the reference — so the per-30-s-block
normalisation defect reported against mlx-voxtral cannot occur there. And its
audio path mirrors the reference line for line: `audio_tower(x).reshape(-1,
intermediate_size)` then the projector, exactly as `VoxtralModel.get_audio_features`
does, with the embedding merge already a scatter rather than the quadratic walk.

None of that argues against the move; it argues that the move is a measurement
exercise, not a port. The trigger to start it is `mlx-voxtral` breaking against
a newer MLX, or a decision to ship a second engine — at which point mlx-audio's
model collection pays for the work in one go rather than one engine at a time.

### What the Open ASR Leaderboard says (German tab, checked 2026-08-20)

The leaderboard has a German tab of its own, fed by `hf-audio/multilingual_evals`
(`multilingual_de.csv`), and it is worth quoting because it is an independent
check on everything above. Ranked by FLEURS German WER, with Common Voice
alongside:

| Model | FLEURS | MCV | RTFx |
|---|---:|---:|---:|
| microsoft/azure-speech-06-2026 *(API)* | 1.93 | 1.88 | — |
| elevenlabs/scribe_v2 *(API)* | 2.30 | 2.19 | — |
| assemblyai/universal-3-pro *(API)* | 2.42 | 2.76 | — |
| reson8/resonant-1 *(API)* | 2.56 | 3.01 | — |
| **mistralai/Voxtral-Small-24B-2507** | **2.61** | 3.19 | 83 |
| openai/whisper-large-v3 | 3.20 | 4.79 | 328 |
| CohereLabs/cohere-transcribe-03-2026 | 3.33 | **2.87** | 607 |
| Qwen/Qwen3-ASR-1.7B-hf | 3.35 | 4.60 | 369 |
| nvidia/canary-1b-v2 | 3.43 | 4.69 | 1308 |
| **mistralai/Voxtral-Mini-3B-2507** | **3.64** | 5.29 | 150 |
| microsoft/Phi-4-multimodal-instruct | 3.99 | 4.25 | 123 |
| nvidia/parakeet-tdt-0.6b-v3 | 4.16 | 4.07 | 3363 |
| microsoft/VibeVoice-ASR-HF | 7.44 | 20.97 | 114 |

**Voxtral-Small is the best open-weight model on German here.** Everything above
it reports no RTFx, which on this leaderboard marks a proprietary API. That is
worth knowing: the engine this document settled on is not a compromise pick, it
is the top of the open field for this language.

It also cross-checks the harness. The ordering on FLEURS matches ours for every
model we measured ourselves — Qwen3-ASR ahead of Voxtral-Mini, Parakeet behind
it, VibeVoice far behind — and the absolute values sit within a few tenths, the
difference being normalisation. Our 4-bit Voxtral-Small scores 2.82 against the
leaderboard's 2.61 for the unquantised model, so the quantisation costs about
0.2 points on clean audio, which is the same story the sweeps above tell.

Two things the table adds that we had not seen. **`CohereLabs/cohere-transcribe-03-2026`** was the one candidate the table added
that we had not seen — best Common Voice German of any open model in the list at
2.87, ahead of Voxtral-Small's 3.19, and best open model outright on the English
long-form tab. It has since been measured; see the section above. Note how little
that predicted: two Common Voice points ahead of Voxtral-Small, and an order of
magnitude behind Voxtral-Mini on a real conversation. The long-form tab is
**English only**,
so it says nothing about German conversation — the gap this document keeps
running into has no public benchmark at all.

### Larger models in the same families (surveyed, not measured)

Scaling up within Parakeet's own family does not help: `parakeet-tdt-1.1b` and
`canary-qwen-2.5b` are English-only. `nvidia/canary-1b-v2` is the real step up —
25 languages, CC-BY-4.0, and 4.40 % against Parakeet's 5.04 % on NVIDIA's own
FLEURS German figure — but it is an attention encoder-decoder rather than a
transducer, so it gives back the structural guarantees that made the family
interesting, and a 0.6 FLEURS point says nothing about conversational audio.

`Qwen/Qwen3-ASR-1.7B` was the obvious next candidate and has now been measured;
see the section above. Its word timestamps, had it worked out, would have come
from a separate `Qwen3-ForcedAligner-0.6B` capped at five minutes of speech.

`microsoft/VibeVoice-ASR` has now been measured too; see the section above. Its
nearest relative, `OpenMOSS-Team/MOSS-Transcribe-Diarize`, does the same joint
trick and is more popular, but supports only Chinese and English.

## What is still open

- **Which encoder precision is better on the 3B model — closed as not worth
  answering.** Measured: the bit width changes what is heard, at ~2-3 acoustic
  differences per 1000 words, scaling with the reduction. The direction is
  unknown and will stay that way; the only route to it is listening to the
  disagreements one by one, and the answer would not change the build. bf16
  stays.
- **Why is the 24B model's encoder so much more sensitive?** Its sweep held the
  language model at 4 bit while the 3B sweep ran at 8. A coarsely quantised body
  may amplify encoder noise instead of averaging it out. **Not answerable on a
  32 GB machine:** the clean counter-test is the 24B encoder sweep repeated with
  an 8-bit body, and small-8bit is 25 GB of weights, which `_auto_chunk_sec()`
  refuses outright. It needs a bigger machine, and it would only matter if mixed
  builds are ever laid out differently.
- **`MEM_MODEL` for the 24B builds is stale.** The `small`, `small6` and
  `small8` entries still hold one-shot-prefill numbers; `generate_step` cut the
  3B model's slope by ~2.5x. Re-measuring would probably lengthen the 24B pass
  budget well beyond the 576 s currently allowed on 32 GB.
- Most of the hard-audio side of this document rests on one 422-word,
  2034-character passage. That is a deliberate decision, not an oversight — but
  it is why differences under ~0.15 CER points are treated as noise throughout.
  A second hand-corrected reference (859 words, five minutes, a video call)
  exists and is used for engine comparisons — the Voxtral build scores 0.81 %
  WER / 0.64 % CER on it. The build sweeps above have not been repeated against
  it, and are not going to be.
- Voxtral drops parentheticals. Whether that is steerable is untested.
- The measurements are tied to this MLX version and macOS release; re-run the
  scripts after an upgrade before trusting the memory model.
