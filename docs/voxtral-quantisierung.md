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

Resolving it needs ground truth on hard audio, and the diff makes that
affordable: it localises the disagreements to ~109 unique spans of a few seconds
each. Adjudicating those by ear is an evening, against the many days a second
hand-corrected reference passage would cost.

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
```

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
| mini-8bit | 1500 s (context-capped) |
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

The streaming **Voxtral-Mini-4B-Realtime** model (Awni Hannun's
[voxmlx](https://github.com/awni/voxmlx) runs it with a bounded rotating KV cache)
was considered as a low-memory option, then ruled out on Mistral's own published
numbers: on German FLEURS it scores 6.19% WER at its 480ms setting and 4.15% even
at 2.4s delay — worse than the offline Voxtral Mini 3B (3.54%), a smaller model.
The causal/streaming architecture trades look-ahead for latency, and on hard
conversational audio the gap would only widen. Its advantages (sub-500ms latency,
bounded memory) are irrelevant to offline file transcription. Not adopted.

## What is still open

- **Which encoder precision is actually better on the 3B model?** Measured: the
  bit width changes what is heard, at ~2-3 acoustic differences per 1000 words,
  scaling with the reduction. Unmeasured: the direction. The ~109 localised
  disagreement spans from `encoder_diff.py` are the cheap way in — adjudicate
  them by ear rather than building a second reference passage.
- **Why is the 24B model's encoder so much more sensitive?** Its sweep held the
  language model at 4 bit while the 3B sweep ran at 8. A coarsely quantised body
  may amplify encoder noise instead of averaging it out. Untested, and it would
  change how mixed builds are laid out if true.
- **`MEM_MODEL` for the 24B builds is stale.** The `small`, `small6` and
  `small8` entries still hold one-shot-prefill numbers; `generate_step` cut the
  3B model's slope by ~2.5x. Re-measuring would probably lengthen the 24B pass
  budget well beyond the 576 s currently allowed on 32 GB.
- The whole hard-audio side of this document rests on one 422-word, 2034-character
  passage. That is a deliberate decision, not an oversight — but it is why
  differences under ~0.15 CER points are treated as noise throughout.
- Voxtral drops parentheticals. Whether that is steerable is untested.
- The measurements are tied to this MLX version and macOS release; re-run the
  scripts after an upgrade before trusting the memory model.
