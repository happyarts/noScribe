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

## Language & word-timestamp quality

Word timestamps come from a CTC forced aligner. Its model is chosen **per
chunk from the transcribed text itself**: when one language dominates the
chunk (function-word analysis; non-Latin scripts are recognised directly),
the char-native model for that language is used -- so "Auto" gets the same
alignment quality as an explicit language choice. Mixed speech with a clear
majority language (e.g. German with English phrases) uses the majority
model, which also anchors the minority-language words; only text without a
dominant language falls back to the romanised multilingual aligner
(MMS-300M, 1130+ languages). If an explicitly selected language contradicts
what the transcript looks like, a warning is logged.

## Which model?

Two builds, both quantised on Apple Silicon and published so they download on
first use (they appear in the model menu only on Apple Silicon Macs — MLX cannot
run anywhere else). Whisper (`precise`) stays the default because it runs on
every platform.

| Model | Size | Needs | Best for |
|-------|------|-------|----------|
| **voxtral-mini-8bit** (3B) | 6 GB | 16 GB | The everyday choice. More accurate than Whisper on hard German, faster than realtime, runs on any Mac with 16 GB. |
| **voxtral-small-8bit** (24B) | 25 GB | 48 GB | A quality ceiling for clean, read-aloud audio on a big machine. Slower than realtime; refused below ~34 GB. |

Model choice follows the **recording**, not a ranking: on clean read-aloud
speech the 24B model wins (2.8 % vs 4.8 % word error rate), on hard
conversational German with crosstalk and brand names the 3B model wins (4.3 % vs
7.8 %). For interviews and podcasts, use **mini** — it is also the only one that
runs on a 32 GB or smaller machine. Full measurements, including the comparison
against Whisper, are in [docs/voxtral-quantisierung.md](docs/voxtral-quantisierung.md).

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

## Which model to pick

The model menu shows how much RAM each build needs, because picking one that
does not fit does not fail loudly — the machine starts swapping and the run
stops making progress. Builds that cannot fit are refused before a run starts.

| Build | Size | Needs | Notes |
|---|---:|---:|---|
| `voxtral-mini-8bit` | 6 GB | ~13 GB | **recommended** — reproduces the bf16 transcript at ~4.5× the speed |
| `voxtral-small-8bit` | 25 GB | ~34 GB | quality ceiling for clean audio on 48 GB+; slower than realtime |

Both builds keep the **audio encoder in bf16** and quantise the language model
and `lm_head` to 8 bit. The encoder runs once per pass, so its precision costs no
speed, but compressing it below 8 bit measurably costs accuracy on difficult
audio. `lm_head` runs once per generated token and is left quantised for that
reason. The full sweep behind this is in
[docs/voxtral-quantisierung.md](docs/voxtral-quantisierung.md).

**Which model?** It depends on the recording, not on a ranking: on clean,
read-aloud speech the 24B model is clearly better (2.8 % vs 4.8 % word error
rate), on hard conversational German with crosstalk and brand names the 3B model
is (4.3 % vs 7.8 %). For interviews and podcasts, pick mini. The measurements,
including a comparison against Whisper, are in
[docs/voxtral-quantisierung.md](docs/voxtral-quantisierung.md).

Both builds download on first use from Hugging Face
([mini](https://huggingface.co/MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder),
[small](https://huggingface.co/MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder)).
The weights are Apache-2.0. Other bit widths (4/5/6-bit, or a bf16 encoder on a
lower-bit body) can be made locally in a few seconds with
`tools/quantize_voxtral.py` — see the script's header and
[docs/voxtral-quantisierung.md](docs/voxtral-quantisierung.md).

## Long audio & memory

Voxtral transcribes each pass in a single `generate()` call whose peak memory
grows roughly linearly with the pass length (flash attention + a KV cache, *not*
O(T²)). noScribe therefore feeds the **whole file in one pass** when it fits the
machine's RAM, and only splits longer files.

The per-pass length is chosen automatically from installed RAM so the estimated
generate peak stays within physical memory (compute on swapped-out MLX buffers
would thrash and never finish). The one-off model-load spike is allowed to swap
— it frees before transcription starts. Measured peaks: mini ≈ 6.4 GB + ~0.7
GB/min, small ≈ 27 GB + ~0.8 GB/min. Rough per-pass lengths:

| RAM | mini-8bit | small-8bit |
|----:|:---------:|:----------:|
| 16 GB | ~4 min | won't run |
| 24 GB | ~16 min | won't run |
| 32 GB | up to 25 min | won't run (refused) |
| 48 GB | up to 25 min | ~17 min |
| 64 GB+ | up to 25 min | up to 25 min |

When a file is longer than one pass it is split into **equal, pause-aligned
passes**: each cut is snapped to a real speaker pause found in a wide window
(searching backward, since a shorter pass is always memory-safe), and a short
lead-in overlap is carried across the seam and de-duplicated by timestamp — so a
pass never splits a word and boundaries are effectively lossless.

To pin the length yourself, set `voxtral_chunk_sec:` (seconds) in `config.yml`
(`0` = automatic). Lower it if other apps need RAM; raise it on a large machine.

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

## Author

The Voxtral integration for noScribe (engine, forced alignment, quantised
model builds) was created by **[Markus Kämmerer](https://markus-kaemmerer.de)**
· [Instagram @markuskaemmerer](https://www.instagram.com/markuskaemmerer/).
