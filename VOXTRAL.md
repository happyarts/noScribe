# Voxtral transcription engine (Apple Silicon)

An optional alternative to faster-whisper, using Mistral's **Voxtral** models via
[`mlx-voxtral`](https://pypi.org/project/mlx-voxtral/) on the Apple Silicon GPU.

On German / Swiss-German interview and podcast audio it is, in our tests,
**more accurate and more readable than Whisper** — it gets technical terms right
where Whisper mis-hears them (e.g. *Wortfindungsstörungen*, not
*Gottfindungsstörungen*), spells consistently, and produces fluent, readable
sentences instead of literal disfluent strings. It runs **faster than realtime**
even for the 24B model, and never ran into the repetition loops Whisper can
produce on hard audio.

## Install (macOS, Apple Silicon)

```bash
pip install -r environments/requirements_voxtral_macOS_arm64.txt
```

The models `voxtral-mini` and `voxtral-small` then appear in the model dropdown.
They are downloaded on first use.

### A note on the pinned dependencies

`mlx`, `mlx-lm` and `mlx-voxtral` are pinned to exact versions.
[`mlx-voxtral`](https://github.com/mzbac/mlx.voxtral) went untouched between
2025-08 and 2026-08, so the pin started life as a deliberate freeze; the author
has since relicensed it to plain MIT and merged all four fixes this engine had
reported, so it is a normal version constraint again. `mlx-lm` is pinned because
decoding drives `generate_step` directly. `tests/test_voxtral_pin.py` checks that
the installed versions are the pinned ones.

The pin is `0.0.6`. One of those fixes moves this engine's output — the log-Mel
is now computed over the whole audio rather than per 30-second chunk, changing
the encoder input on most frames — and it was measured over 92 minutes of paired
audio to cost nothing at the transcript (dWER +0.02 [−0.19, +0.24]); see
[docs/voxtral-audio-vorverarbeitung.md](docs/voxtral-audio-vorverarbeitung.md),
section 6.

The engine also swaps the processor's feature extractor for its own
(`_PercentileFloorFeatures`): the library clamps the log-Mel at `log_max − 8`
with the maximum taken over the whole input, so one loud transient — a door
slam — raises the floor for the entire pass and costs a measured +1.52 WER
points at 28 dB; the engine takes the floor from a percentile of the
spectrogram instead, which removes that and costs nothing on clean material.
The spectrogram itself is the library's (built from its own window, STFT and
filter bank), and `tests/test_mel_floor.py` pins it bit for bit to the
library's path at percentile 100. Section 6b of the same document has the
measurements.

The pinning is a known liability, so here is the exit route, measured rather
than assumed. **The coupling is five calls** (plus a handful of attribute reaches —
`embed_tokens`, `get_audio_embeds`, `config.audio_token_id`, `language_model`,
`lm_head`) — `load_voxtral_model`,
`VoxtralProcessor`, `apply_transcrition_request`, `model.generate` (the
non-greedy fallback only) and `proc.decode`, plus the four audio primitives
`stft_mlx`, `hanning`, `get_mel_filters` and `pad_to_multiple` with their
frame constants behind the feature extractor above,
and `mlx_voxtral.quantization` and `download_model` in
`tools/quantize_voxtral.py`. Everything else in
`voxtral_engine.py` is this project's own: chunking, loop detection, the
temperature ladder, forced alignment, prefix salvage. None of it depends on
which package loads the weights.

The replacement, if the pin ever breaks against a newer MLX, is
[`mlx-audio`](https://github.com/Blaizzy/mlx-audio) — actively maintained, and
its `mlx_audio/stt/models/` carries Voxtral alongside a dozen other ASR models
behind one API. Three things to know before starting, all verified against
mlx-audio 0.5.0:

* **It will not load the published quantised builds, and it fails silently.**
  Loading `voxtral-mini-8bit` yields *zero* quantised modules and 405 dense
  `Linear` layers, 211 of them in the language model, with no exception raised.
  The cause is naming: these builds write `language_model.layers.0…` while
  mlx-audio's tree is `language_model.model.layers.0…`. Migration therefore
  means re-quantising and re-publishing both builds, and any smoke test has to
  assert the number of quantised modules, because the failure mode is quiet.
* **It carries the same stop-token defect this engine already works around** —
  `_VOXTRAL_EOS_TOKEN_IDS = [2, 4, 32000]`, and 32000 is the ordinary text
  token `" Capital"`, not a pad token. See `_resolve_stop_tokens`.
* **Its `_merge_input_embeddings` scatters without promoting dtype first.**
  `_merged_embeddings` promotes deliberately, so a build whose projector and
  token embeddings disagree would round every audio embedding away silently.
  Measured on the shipped 8-bit builds both sides are bfloat16 and nothing is
  lost, so this is a latent risk rather than a live defect — but it is the kind
  of thing that only shows up as different logits, so re-check it against
  whatever build is in use.

Re-quantising itself is safe: MLX's affine quantisation is deterministic and
data-free, and mlx-audio's own predicate (`not p.startswith("audio_tower")`)
selects exactly the layer set these builds already use — 213 modules at 8 bit,
group size 64, encoder left dense. Same weights in, same tensors out; only the
keys change.

## Language & word-timestamp quality

Word timestamps come from a CTC forced aligner: a wav2vec2 model produces
per-frame emissions (transformers, on the GPU), and a numpy Viterbi in
`noScribe/ctc_align.py` turns them into word boundaries — torchaudio's
`forced_align` kernel is no longer used, see
[docs/viterbi-numpy-brief.md](docs/viterbi-numpy-brief.md) for why. The
aligner model is chosen **per chunk from the transcribed text itself**: when one language dominates the
chunk (function-word analysis; non-Latin scripts are recognised directly),
the char-native model for that language is used -- so "Auto" gets the same
alignment quality as an explicit language choice. Mixed speech with a clear
majority language (e.g. German with English phrases) uses the majority
model, which also anchors the minority-language words; only text without a
dominant language falls back to the romanised multilingual aligner
(MMS-300M, 1130+ languages). If an explicitly selected language contradicts
what the transcript looks like, a warning is logged.

## Which model

Two builds, both quantised on Apple Silicon and published so they download on
first use. They appear in the model menu only on Apple Silicon Macs — MLX cannot
run anywhere else — and Whisper (`precise`) stays the default because it runs on
every platform. The menu shows what each build needs, because picking one that
does not fit does not fail loudly: the machine starts swapping and the run stops
making progress. Builds that cannot fit are refused before a run starts.

| Build | Size | Needs | Notes |
|---|---:|---:|---|
| `voxtral-mini-8bit` (3B) | 6 GB | ~13 GB | **recommended** — runs on a 16 GB Mac, faster than realtime, reproduces the bf16 transcript at ~4.5× the speed |
| `voxtral-small-8bit` (24B) | 25 GB | ~34 GB | quality ceiling for clean, read-aloud audio on 48 GB+; slower than realtime |

**Which of the two?** It depends on the recording, not on a ranking: on clean,
read-aloud speech the 24B model is clearly better (2.8 % vs 4.8 % word error
rate), on hard conversational German with crosstalk and brand names the 3B model
is (4.3 % vs 7.8 %). For interviews and podcasts, pick mini — it is also the only
one that runs on a 32 GB or smaller machine. The measurements, including the
comparison against Whisper, are in
[docs/voxtral-quantisierung.md](docs/voxtral-quantisierung.md).

Both builds keep the **audio encoder in bf16** and quantise the language model
and `lm_head` to 8 bit. The encoder runs once per pass, so its precision costs no
speed, but compressing it below 8 bit measurably costs accuracy on difficult
audio. `lm_head` runs once per generated token and is left quantised for that
reason.

They download on first use from Hugging Face
([mini](https://huggingface.co/MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder),
[small](https://huggingface.co/MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder));
the weights are Apache-2.0. Other bit widths (4/5/6-bit, or a bf16 encoder on a
lower-bit body) can be made locally in a few seconds with
`tools/quantize_voxtral.py` — see the script's header and
[docs/voxtral-quantisierung.md](docs/voxtral-quantisierung.md).

## How it works

Voxtral produces clean text but no timestamps, so noScribe uses two paths:

- **Fast path** – plain text only. Used only for `.txt` output without
  speakers, timestamps or pauses — ideal for the "just give me a clean
  transcript" case (e.g. course summaries). Its segment times are
  approximations, which is why formats that embed audio-sync anchors never
  use it.
- **Long path** – word timestamps are recovered by CTC forced alignment
  against a language-matched wav2vec2 model (the WhisperX approach) and
  grouped into subtitle-sized cues. Used automatically for `.html` (its
  anchors drive the editor's click-to-play audio sync), `.vtt` subtitles,
  visible timestamps, speaker detection and pause marking.

Note: Voxtral has no prompt/hotword hook, so the "Disfluencies" option cannot
steer it (a log line says so). Voxtral Mini naturally smooths most fillers;
Voxtral Small stays closer to the exact wording.

## Long audio & memory

Voxtral transcribes each pass in a single `generate()` call whose peak memory
grows roughly linearly with the pass length (flash attention + a KV cache, *not*
O(T²)). Files up to one pass long therefore go through in one piece; longer ones
are split.

A pass is capped at **10 minutes**, whatever the machine could hold. That is the
longest Voxtral has been *measured* on: the widely quoted 30/40 minutes is a
capacity calculation (12.5 Hz frame rate against a 32k context), while the
paper's own long-form ASR protocol segments one-hour earnings calls "into
shorter, 10 minute variants" ([arXiv:2507.13264](https://arxiv.org/abs/2507.13264)).
Past that this project has recorded two distinct failures — whole passes coming
back translated, and passes returning without their opening — so the cap is a
refusal to run the model twice as far out as anyone has measured it, not a fix
for either (both are near-ties that a slightly shorter window would not dodge).

Below the cap the per-pass length is chosen from installed RAM so the estimated
generate peak stays within physical memory (compute on swapped-out MLX buffers
would thrash and never finish). The one-off model-load spike is allowed to swap
— it frees before transcription starts. Measured peaks: mini ≈ 6.5 GB + ~0.4
GB/min, small ≈ 27 GB + ~0.8 GB/min. Rough per-pass lengths:

| RAM | mini-8bit | small-8bit |
|----:|:---------:|:----------:|
| 16 GB | ~7 min | won't run |
| 24 GB | 10 min | won't run |
| 32 GB | 10 min | won't run (refused) |
| 48 GB+ | 10 min | 10 min |

When a file is longer than one pass it is split into **equal, pause-aligned
passes**: each cut is snapped to a real speaker pause found in a wide window
(searching backward, since a shorter pass is always memory-safe), and a short
lead-in overlap is carried across the seam and de-duplicated by timestamp — so a
pass never splits a word and boundaries are effectively lossless.

The first pass is additionally checked for a dropped opening: a window
occasionally returns without its first seconds of speech, silently, so a short
head of the same audio is decoded and whatever is missing is spliced back. Later
passes need no check — they carry a lead-in overlap that the previous pass
already transcribed. When it finds something, the log says so.

This is not rare enough to skip: on a raw Zoom recording, 5 of 64 windows cut at
300 s and 600 s came back missing their opening, once losing 18 words of fluent
speech. It depends on the recording — read-aloud benchmark audio never shows it.
Measurements in [docs/voxtral-benchmarks.md](docs/voxtral-benchmarks.md), §5.

To pin the length yourself, set `voxtral_chunk_sec:` (seconds) in `config.yml`
(`0` = automatic). Lower it if other apps need RAM. Raising it past 10 minutes
is refused — see the cap above.

## Correcting brand / product / programme names

Voxtral has no hotword support, so it mis-hears proper names. Maintain a simple
find/replace list at:

```
<config dir>/voxtral_corrections.yml
```

(macOS: `~/Library/Application Support/noScribe/voxtral_corrections.yml`)

```yaml
- to: VitaFlor
  from: [vitaflor, "vita flor", "flor-öl", "flor-öle"]
- to: Sonvita
  from: [sonvida, sonvieda]
```

Matches are whole-word and case-insensitive. The file is created empty (with
commented examples) on first use — add your recurring brand, product and
programme names (e.g. from earlier podcast transcripts).

Why this and not a prompt: the transcription request has exactly one text slot
(`lang:xx`), and putting terms there was measured to act as a decode
perturbation rather than a vocabulary hint — on three clips it fixed one term,
ignored another and broke a third that the plain request had got right.
Voxtral's chat mode *does* use a term list, but it is not a verbatim
transcriber: one of four clips came back as a 12-word answer instead of an
84-word transcript, and another wrote the instruction into the text. A term
Voxtral does not know is therefore fixed after the fact, here.

## Author

The Voxtral integration for noScribe (engine, forced alignment, quantised
model builds) was created by **[Markus Kämmerer](https://markus-kaemmerer.de)**
· [Instagram @markuskaemmerer](https://www.instagram.com/markuskaemmerer/).
