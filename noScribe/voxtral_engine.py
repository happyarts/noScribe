"""
Voxtral transcription engine for noScribe (Apple Silicon / MLX).

Provides an alternative to the faster-whisper backend:

- Transcription is done by Mistral's Voxtral (via `mlx-voxtral`), which produces
  clean text but *no* timestamps.
- When timestamps are required (subtitles, speaker assignment, pause marking),
  word-level timestamps are recovered with CTC forced alignment against a German
  wav2vec2 model (the same approach WhisperX uses).

The public entry point `transcribe()` returns a list of segment dicts that are
shape-compatible with what `whisper_mp_worker` streams to the main app:

    {"start": float_seconds, "end": float_seconds, "text": str,
     "words": [{"word": str, "start": float, "end": float, "prob": float}] | None}

Long files are processed in chunks so memory and model context stay bounded.
"""

import functools
import importlib.resources as impres
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Rough count of text tokens Voxtral generates per second of audio, used ONLY to
# drive the intra-pass liveness estimate (German conversational speech ~2-2.5
# words/s x ~1.3 tokens/word). It is intentionally a lower-ish bound: the
# estimate is capped below 100% and snaps to the true position when a pass
# finishes, so under-estimating just makes the bar move a touch fast, never past
# the real chunk boundary.
_EST_TOKENS_PER_SEC = 3.0

# --- Long-audio chunking (reference approach: one pass up to the model's
# context, split only beyond that) -------------------------------------------
# Voxtral uses flash attention + a rolling KV cache, so a single generate()
# call's memory grows ~linearly with the audio length (NOT O(T^2)). We feed the
# whole file in one pass when it fits the memory budget, and only split longer
# files. The per-pass length is chosen from the machine's RAM by
# `_auto_chunk_sec()` -- see MEM_MODEL below -- unless the caller/config pins it.
PREF_MIN_CHUNK_SEC = 180     # preferred quality floor (below this we warn)
HARD_MIN_CHUNK_SEC = 60      # absolute floor; memory safety wins over this only with a warning
MAX_CHUNK_SEC = 1500         # 25 min: safely under Voxtral's ~30 min / 32k ctx
# The binding limit is the Voxtral *generate* working set: it must fit in
# physical RAM, because MLX compute on swapped-out buffers thrashes and never
# finishes. The one-off load transient (weights dict + model briefly duplicated)
# is NOT counted here -- it is allowed to swap since it frees before generate
# starts. So this reserve only holds back non-swappable kernel/wired memory, the
# forced aligner that runs alongside, and a margin; the OS/GUI may page out.
# The generate pass may use (total_RAM - this).
#
# Measured on a 32 GB M1 Max: a 21.8 GB pass ran with 2.7 GB still free, and a
# 26.4 GB pass completed as well, so 7 GB is defensible while leaving room for
# the aligner. Users who free up the machine can go further via the config key
# `voxtral_ram_reserve_gb`, which buys noticeably longer passes for the memory
# hungry builds (6-bit small: 229 s at 8 GB, 304 s at 7, 378 s at 6).
RAM_RESERVE_GB = 7
# Below this much free memory a run does not just get slow, it stops progressing:
# the forced aligner (~2 GB) and the OS still need room next to the model, and
# once MLX has to compute on swapped-out buffers nothing finishes. Used to refuse
# a model outright instead of starting a run that cannot succeed.
# Calibrated on a 32 GB machine: passes peaking at 25 GB (7 GB headroom) ran
# fine, while the 8-bit 24B build at 27.2 GB (4.8 GB headroom) exhausted the swap
# and stopped progressing once the aligner loaded next to it.
MIN_HEADROOM_GB = 6
# Measured peak unified-memory model per pass:  peak_GB ~= fixed + slope*seconds.
#   mini : anchored on a real measurement (600 s single pass = 17.4 GB peak on
#          an M1 Max; the full 1143 s pass exhausted 32 GB -> unsafe).
#   small: 4-bit 24B; measured 30 s = 15.6 GB and 120 s = 17.0 GB, i.e. the same
#          ~linear slope as mini but a ~15 GB fixed offset from its weights, so it
#          needs shorter passes / more RAM for the same length.
#   mini8 : the 3B weights at 8 bit with the audio encoder left in bf16
#          (tools/quantize_voxtral.py, mode dense-encoder). RECALIBRATED for the
#          generate_step decode path (which chunks the prompt): fresh-process
#          full-pass peaks on an M1 Max were 6.82 / 7.41 / 8.72 / 10.00 / 11.18 GB
#          at 180 / 300 / 600 / 900 / 1200 s (two podcast files, ~1.6% apart),
#          a clean line peak ~= 6.11 + 0.00427*s. The old one-shot-prefill path
#          measured ~2.5x that slope (120 s = 7.7, 1100 s = 18.7 GB); switching to
#          generate_step genuinely lowered it. The entry below keeps a safety
#          margin over the fit (6.5 + 0.0060*s over-predicts every point by
#          15-25%): the slope is dominated by the audio prompt-token rate (~12.5
#          tok/s, fixed by the recording), so speech density moves it only ~6%,
#          but machine/version variance warrants the cushion. On 32 GB mini8 is
#          capped by MAX_CHUNK_SEC (context) anyway; the recalibration mainly lets
#          16-24 GB machines run longer single passes. small/small6/small8 below
#          are still on old-path numbers (no local build to re-measure) -- safe,
#          just conservative, and they target 48 GB+ machines regardless.
#          Why the encoder stays dense: measured against a hand-corrected
#          German reference, quantising it costs real accuracy while costing
#          nothing to keep -- the encoder runs ONCE per pass, so its precision
#          does not affect speed (6.60x vs 6.68x for the fully-quantised
#          build), and this build reproduces the bf16 transcript word for word
#          at 4.5x the speed. lm_head, by contrast, runs once per generated
#          token: leaving it dense costs 27% throughput and measured no better,
#          so it stays quantised.
#   small8: the shipped 24B build (8-bit LM + lm_head, bf16 encoder). Needs
#          ~34 GB minimum, so it is refused below that and wants 48 GB+. Even
#          when it fits it runs at ~0.80x realtime -- slower than the recording.
#          On a 32 GB machine the 25 GB of weights cannot stay resident: a
#          benchmark swapped continuously and did not finish a 25 min pass in
#          4.5 h. This entry exists so auto-sizing refuses it up front there.
#   small6: the same 24B weights quantised to 6 bit instead of 4 (converted
#          locally from the original mistralai release). Measured 150 s =
#          22.9 GB and 410 s = 26.4 GB, i.e. ~6 GB more fixed than the 4-bit
#          build, which is why it only allows much shorter passes on 32 GB.
MEM_MODEL = {
    "mini":   {"fixed": 7.9, "slope": 0.016},
    "mini8":  {"fixed": 6.5, "slope": 0.0060},   # generate_step path; see note above
    "small":  {"fixed": 15.2, "slope": 0.017},
    "small6": {"fixed": 20.9, "slope": 0.0135},
    "small8": {"fixed": 27.4, "slope": 0.0135},
}

# Window size for the wav2vec2 alignment forward pass. Kept small so the
# O(T^2) self-attention memory of the *aligner* stays bounded regardless of
# chunk length (this is the aligner, not Voxtral).
EMISSION_WINDOW_SEC = 20
# Hard cap on the forced_align DP-buffer size, frames * (2*tokens + 1).
# torchaudio's CPU kernel indexes that buffer with 32-bit ints and segfaults
# once the product nears 2**31 (empirically: 2.10e9 cells fine, 2.17e9 crashes,
# torchaudio 2.11, int32 and int64 targets alike). 2**30 leaves a 2x safety
# margin and keeps a single align call under ~5 s.
FORCED_ALIGN_MAX_CELLS = 2**30
# When splitting a long file, snap each cut to the longest speaker pause found
# within a *wide* radius of the target boundary. Because the passes are long we
# have plenty of slack to hunt far for a real pause, so a cut lands between
# utterances and never mid-word. Widened well beyond a token's worth of audio.
SILENCE_SEARCH_SEC = 90
# Lead-in overlap (seconds) read from the previous chunk so the words right
# after a boundary have preceding audio context; the duplicated overlap is then
# dropped by timestamp. Combined with pause-aware cuts, seams are ~lossless.
OVERLAP_SEC = 15

# Known model repositories (MLX).
#   mini  ~ 3B bf16  (~9 GB download, ~1.5 GB resident quantised variants exist)
#   small ~ 24B 4-bit (~14 GB, fits 32 GB machines; runs at/above realtime on M1 Max)
VOXTRAL_MODELS = {
    # Two builds, both quantised on Apple Silicon and published so they download
    # on first use (the picker only shows them on arm64 Macs; elsewhere MLX
    # cannot run at all). Each keeps the audio encoder in bf16 while quantising
    # the language model and lm_head to 8 bit -- see docs/voxtral-quantisierung.md
    # for why (the encoder runs once per pass, so its precision is nearly free,
    # and compressing it below 8 bit measurably costs accuracy on hard audio).
    #
    # mini (3B): the everyday build. Reproduces the bf16 transcript word for
    # word on our hard-German reference at ~4.5x the speed, runs on any Mac with
    # 16 GB, and beats the 24B model on difficult conversational audio.
    "voxtral-mini-8bit": "MarkusKaemmerer/Voxtral-Mini-3B-2507-8bit-dense-encoder",
    # small (24B): a quality ceiling for clean, read-aloud audio on machines
    # with 48 GB+. Better than mini on clean speech but slower than realtime, and
    # refused below ~34 GB (it would swap forever). See MEM_MODEL / min_ram_gb.
    "voxtral-small-8bit": "MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder",
}

# The raw unquantised source releases: handing one of these to transcribe()
# would download tens of GB (24B: 48 GB) and, because the basename carries no
# bit width, _model_kind would meter it with a quantised profile and wave a
# too-long pass through. No model here points at them, but a direct caller might
# copy one in, so transcribe() refuses them (see below).
SOURCE_REPOS = frozenset({
    "mistralai/Voxtral-Mini-3B-2507", "mistralai/Voxtral-Small-24B-2507",
})


def _local_copy(name):
    """Path of a converted build under the package `models/` dir, or None.

    The one place that decides "this build exists locally": resolve_model()
    prefers it and has_local_build() gates the picker on it, and both must
    agree on the same filesystem fact.
    """
    try:
        local = impres.files("models") / str(name)
        if (local / "config.json").is_file():
            return str(local)
    except Exception:
        pass
    return None


def has_local_build(name):
    """True if `name` is usable on this machine. Both shipped builds download on
    first use, so this is always true for known models; kept as the picker's
    gate in case a future entry is local-only again."""
    return name in VOXTRAL_MODELS or _local_copy(name) is not None

# Forced-alignment (word timestamps) is language dependent. Use a wav2vec2 CTC
# model that matches the transcription language when we have one (best accuracy,
# native alphabet incl. umlauts/accents), and fall back to a multilingual MMS
# aligner for "auto"/"multilingual" or unmapped languages (handles code-switched
# audio like German+English; characters outside its vocabulary are skipped and
# their timing interpolated).
ALIGN_MODELS = {
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
}
ALIGN_MODEL_MULTILINGUAL = "MahmoudAshraf/mms-300m-1130-forced-aligner"


def min_ram_gb(repo):
    """Smallest amount of RAM in which this model can realistically run.

    Shown next to each model in the picker, so the choice is made before a run
    starts rather than after the machine has begun to swap.
    """
    m = MEM_MODEL[_model_kind(repo)]
    return m["fixed"] + m["slope"] * HARD_MIN_CHUNK_SEC + MIN_HEADROOM_GB


def max_safe_chunk_sec(repo):
    """Longest pass (seconds) whose estimated generate peak still fits physical
    RAM, i.e. the hard ceiling past which a run stops progressing (see
    MIN_HEADROOM_GB). This is THE memory-safety invariant: the picker hint
    (min_ram_gb), the automatic sizing and the pinned-value clamp must all
    agree with it, so it lives in exactly one place.

    Raises MemoryError when even the shortest pass (HARD_MIN_CHUNK_SEC) cannot
    fit -- proceeding would not fail, it would swap forever.
    """
    kind = _model_kind(repo)
    m = MEM_MODEL[kind]
    total = _total_ram_gb()
    if total < min_ram_gb(repo):
        est = m["fixed"] + m["slope"] * HARD_MIN_CHUNK_SEC
        raise MemoryError(
            f"Voxtral {kind} needs about {est:.0f} GB even for its shortest "
            f"pass, which does not fit in {total:.0f} GB. Pick a smaller model "
            f"(see the memory hint next to each model) or use a machine with more RAM."
        )
    if not m["slope"]:
        return MAX_CHUNK_SEC
    return (total - MIN_HEADROOM_GB - m["fixed"]) / m["slope"]


def resolve_align_model(language):
    """Pick the alignment model for a language code (e.g. "de"); fall back to the
    multilingual aligner for None/"auto"/"multilingual"/unmapped languages."""
    code = (language or "").strip().lower()[:2]
    return ALIGN_MODELS.get(code, ALIGN_MODEL_MULTILINGUAL)


def _frame_energy(window, frame):
    """Per-frame mean squared amplitude of `window` in `frame`-sample blocks --
    the energy curve used to find quiet spots. Trailing samples that don't fill
    a whole frame are dropped."""
    import numpy as np
    nf = len(window) // frame
    return (window[:nf * frame].reshape(nf, frame).astype(np.float32) ** 2).mean(axis=1)


def _quietest_frame_near(audio, target, radius_sec):
    """Sample index of the quietest 100 ms frame within +/- `radius_sec` of
    `target`, so a cut lands in a pause rather than mid-word. Returns `target`
    unchanged when the search window is too small to snap."""
    frame = max(1, int(0.1 * SAMPLE_RATE))
    lo = max(frame, target - int(radius_sec * SAMPLE_RATE))
    hi = min(len(audio) - frame, target + int(radius_sec * SAMPLE_RATE))
    if hi - lo <= frame:
        return target
    window = audio[lo:hi]
    if len(window) // frame < 2:
        return target
    return lo + int(_frame_energy(window, frame).argmin()) * frame + frame // 2


def _chunk_boundaries(audio, chunk_len, back_len, fwd_len, max_len=None):
    """Split `audio` into ~`chunk_len`-sample passes, snapping every cut to a real
    speaker pause found in the window `[target - back_len, target + fwd_len]`.

    The search leans *backward*: a shorter pass is always memory-safe, so we hunt
    far back (large `back_len`) for a genuine pause and allow only a small forward
    reach (`fwd_len`) to avoid growing the pass. Among clear pauses we take the one
    *closest to the target* (keeps passes near the intended length); if none is
    clearly a pause we fall back to the longest quiet run, then to the target
    itself. Cutting between utterances means a pass never splits a word, and with
    the lead-in overlap the seam is effectively lossless.

    `max_len` is a hard per-pass cap (the RAM-safe length): no pass — including
    the final one after tail-merging and backward snaps — ever exceeds it.
    """
    import numpy as np
    n = len(audio)
    max_len = max_len or chunk_len * 2
    if n <= min(chunk_len, max_len):
        return [0, n]
    frame = max(1, int(0.05 * SAMPLE_RATE))  # 50 ms resolution
    good_run = 8   # >= ~400 ms silence counts as a clear speaker pause
    min_run = 3    # >= ~150 ms silence is an acceptable fallback cut
    bounds = [0]
    target = chunk_len
    while True:
        remaining = n - bounds[-1]
        # Done when the tail fits one RAM-safe pass AND is not much more than
        # the balanced length (small tails merge into the final pass instead of
        # becoming a tiny extra chunk) — but a tail beyond max_len always gets
        # another cut, so backward snaps can't inflate the final pass past the
        # memory budget.
        if remaining <= max_len and target >= n - chunk_len // 2:
            break
        # never let a pass shrink below half the target, even hunting backward
        lo = max(bounds[-1] + max(frame, chunk_len // 2), target - back_len)
        # ...and never let a snap or tail-merge push this pass past max_len
        hi = min(n, target + fwd_len, bounds[-1] + max_len)
        if hi - lo <= 2 * frame:
            cut = min(target, n)
        else:
            window = audio[lo:hi]
            nf = len(window) // frame
            energy = _frame_energy(window, frame)
            # frames below 15% of the typical speech level count as "silence"
            thresh = float(np.percentile(energy, 75)) * 0.15 + 1e-9
            silent = energy <= thresh
            target_f = (min(target, hi) - lo) / frame  # target position in frames
            best_close, best_close_d = None, None       # clear pause nearest target
            best_long, best_long_len = None, 0           # longest quiet run (fallback)
            i = 0
            while i < nf:
                if silent[i]:
                    j = i
                    while j < nf and silent[j]:
                        j += 1
                    run, mid = j - i, (i + j) // 2
                    if run > best_long_len:
                        best_long_len, best_long = run, mid
                    if run >= good_run:
                        d = abs(mid - target_f)
                        if best_close_d is None or d < best_close_d:
                            best_close_d, best_close = d, mid
                    i = j
                else:
                    i += 1
            if best_close is not None:
                cut = lo + best_close * frame + frame // 2
            elif best_long_len >= min_run:
                cut = lo + best_long * frame + frame // 2
            else:
                cut = min(target, n)
        bounds.append(min(max(cut, bounds[-1] + frame), n))
        target = bounds[-1] + chunk_len
    bounds.append(n)
    return bounds


def resolve_model(name, default_repo=None):
    """Prefer a persistent local copy under the package `models/` dir (e.g.
    `models/voxtral-mini`) so models don't have to live in the HF cache and
    won't be re-fetched after an aborted run. Falls back to the Hugging Face
    repo id (download on first use) when no local copy is present.
    """
    default_repo = default_repo or VOXTRAL_MODELS.get(name, name)
    return _local_copy(name) or default_repo


# A pass that collapses into a repetition loop ("Jetzt. Jetzt. Jetzt. ...")
# repeats one word far more often than any real utterance: measured over real
# transcripts (Whisper and Voxtral, German) the longest run of identical words
# is 2, while an observed loop ran to 690. This is the primary, precise signal.
DEGENERATE_WORD_RUN = 12
# Compression ratio as a secondary net, calibrated on the same files: real
# transcripts measure 2.59-2.63, the loop measured 5.75. Whisper uses 2.4 for
# this, which would flag good German prose here, so the threshold sits in the
# wide gap between the two populations.
DEGENERATE_COMPRESSION_RATIO = 4.0
# Last-resort penalties for a pass that still loops after being split, tried in
# order so the gentlest effective one wins. A penalty cannot tell a degenerate
# loop from a meaningful repetition, and a strong one deletes doubled words: 1.1
# turned "wir können es dir nicht nicht erzählen" into "... nicht erzählen",
# inverting the statement. A sentence that reads fluently but says the opposite
# is worse than an obviously broken one. Measured on the pass that looped:
#
#   1.01 -> loop gone, "nicht nicht" kept, 10.17 commas/100 words
#   1.05 -> loop gone, "nicht nicht" kept, 10.01
#   1.10 -> loop gone, "nicht nicht" LOST,  9.55
#
# A barely-there nudge is enough to tip a self-reinforcing loop but too weak to
# override a repetition the model is confident about.
RETRY_REPETITION_PENALTIES = (1.01, 1.1)
# Splitting a looping pass is tried first, because the loop is a long-generation
# effect: the pass that produced 4099 identical words at 410 s transcribed
# cleanly as 2x205 s, with the doubled negation intact.
LOOP_SPLIT_MAX_DEPTH = 3
LOOP_SPLIT_MIN_SEC = 45


def _looks_degenerate(text):
    """True if a pass collapsed into a repetition loop.

    Voxtral is run without a repetition penalty because that is what Mistral's
    reference transcription does and because a penalty strips punctuation from
    verbatim speech. Rarely -- seen with the 24B model -- a pass still degenerates
    into repeating one word thousands of times, which must not reach the
    transcript.
    """
    import zlib
    if not text:
        return False
    words = text.split()
    if len(words) < 30:
        return False
    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best >= DEGENERATE_WORD_RUN:
        return True
    raw = text.encode("utf-8")
    return len(raw) / max(1, len(zlib.compress(raw))) > DEGENERATE_COMPRESSION_RATIO


def _quietest_split(audio):
    """Sample index nearest the middle that sits in the quietest 100 ms frame,
    so a pass is never split in the middle of a word."""
    return _quietest_frame_near(audio, len(audio) // 2, 5.0)


def _transcribe_guarded(vox, audio, language, log_cb, label, depth=0, token_cb=None):
    """Transcribe one pass and repair it if it collapses into a repetition loop.

    Repair order matters. Splitting is tried first because the loop is a
    long-generation effect -- the same audio in shorter pieces comes out clean --
    and it keeps repetition_penalty at 1.0, so genuine repetitions survive.
    A penalty is the last resort only, because it silently deletes meaningful
    repeated words (a doubled negation flips the meaning of the sentence).
    """
    dur = len(audio) / SAMPLE_RATE
    max_new = min(32768, int(dur * 20) + 512)
    text = vox.transcribe_array(audio, language, max_new_tokens=max_new,
                                token_cb=token_cb)
    if not _looks_degenerate(text):
        return text

    if depth < LOOP_SPLIT_MAX_DEPTH and dur >= 2 * LOOP_SPLIT_MIN_SEC:
        _log(log_cb, "warn", f"{label}: repetition loop, splitting it and retrying "
                             f"({dur:.0f}s -> 2x{dur / 2:.0f}s).")
        cut = _quietest_split(audio)
        left = _transcribe_guarded(vox, audio[:cut], language, log_cb, label, depth + 1, token_cb)
        right = _transcribe_guarded(vox, audio[cut:], language, log_cb, label, depth + 1, token_cb)
        joined = f"{left} {right}".strip()
        if not _looks_degenerate(joined):
            return joined
        text = joined

    # Splitting did not help. Fall back to the gentlest penalty that works,
    # because a stronger one starts deleting meaningful repeated words.
    for penalty in RETRY_REPETITION_PENALTIES:
        _log(log_cb, "warn", f"{label}: still looping after splitting; retrying with "
                             f"repetition_penalty={penalty}. Note that a penalty can drop "
                             f"meaningful repeated words.")
        retry = vox.transcribe_array(audio, language, max_new_tokens=max_new,
                                     repetition_penalty=penalty)
        if not _looks_degenerate(retry):
            return retry
        text = min((text, retry), key=len)
    _log(log_cb, "warn", f"{label}: could not resolve the loop; keeping the shorter result.")
    return text


def _model_kind(repo):
    """Classify a repo id / local path into a MEM_MODEL entry.

    Only the final path component (model dir / repo name) is matched, so a
    parent directory that happens to contain "small" cannot misclassify the
    model and shrink its passes for no reason. The bit width matters as much as
    the parameter count: a 6-bit 24B build needs ~6 GB more than the 4-bit one,
    and mistaking one for the other would size passes too long and exhaust RAM.

    A name with no recognisable size token falls back to the most conservative
    (highest-RAM) profile, never the cheap `mini` one, so an unrecognised build
    is sized safely-short rather than metered too generously and swapped.
    """
    name = os.path.basename(os.path.normpath(str(repo))).lower()
    eight = "8bit" in name or "8-bit" in name
    # The shipped builds keep the audio encoder in bf16 ("dense-encoder"); the
    # encoder is small, so this adds well under a GB and the bit-width profile
    # still fits (mini dense-encoder measured 6.35 GB fixed vs 6.0 for uniform
    # 8-bit -- within the mini8 entry). Classify by parameter count and bit
    # width; the MEM_MODEL entries already carry a safety margin.
    if "small" in name or "24b" in name:
        if eight:
            return "small8"
        return "small6" if ("6bit" in name or "6-bit" in name) else "small"
    if "mini" in name or "3b" in name:
        return "mini8" if eight else "mini"
    # Unrecognised build: the name carries no reliable size signal. Under-sizing
    # a pass only runs slower, but over-sizing swaps forever -- so fall back to
    # the most memory-hungry profile rather than optimistically metering an
    # unknown model as the cheap `mini` one (which would wave a too-long pass
    # through on a large model). The old code defaulted anything without "small"
    # to mini; that is the direction that can swap.
    logger.warning(
        "Unrecognised Voxtral build %r; sizing passes with the conservative "
        "small8 profile. Name a local build with mini/small and its bit width "
        "(e.g. voxtral-mini-8bit) to have it sized correctly.", name)
    return "small8"


@functools.lru_cache(maxsize=None)
def _total_ram_gb():
    """Total physical RAM in GB (sysctl on macOS; psutil fallback; else 16).

    Cached: total RAM is a process constant, but this is queried once per model
    on every model-dropdown open (via App.model_label) and again during a job,
    so without the cache each call would spawn a `sysctl` subprocess for a value
    that never changes."""
    try:
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / 1024**3
    except Exception:
        try:
            import psutil
            return psutil.virtual_memory().total / 1024**3
        except Exception:
            return 16.0  # conservative default


def _auto_chunk_sec(repo, log_cb=None, ram_reserve_gb=None):
    """Choose the per-pass length (seconds) from the machine's RAM.

    Voxtral feeds the whole pass into one generate() call, whose peak
    unified-memory is ~`fixed + slope*seconds` (MEM_MODEL). We pick the longest
    pass whose estimated peak still fits `(total_RAM - RAM_RESERVE_GB)`, clamped
    to [HARD_MIN, MAX]. So most files go through in a single pass on a roomy
    machine, while low-RAM machines (or the memory-hungry 24B `small` model)
    automatically get shorter, still-safe passes -- and a warning when RAM is the
    limiting factor.
    """
    kind = _model_kind(repo)
    m = MEM_MODEL[kind]
    total = _total_ram_gb()
    # Refuses outright when even the shortest pass cannot fit: proceeding would
    # not merely run slowly -- the working set no longer fits, MLX computes on
    # swapped-out buffers and the run stops making progress at all (observed
    # with the 8-bit 24B build, which sat at 25 GB while the aligner pushed the
    # machine into swap).
    ceiling = max_safe_chunk_sec(repo)
    reserve = RAM_RESERVE_GB if not ram_reserve_gb else float(ram_reserve_gb)
    budget = total - reserve
    raw = (budget - m["fixed"]) / m["slope"] if m["slope"] else MAX_CHUNK_SEC
    # A reserve below MIN_HEADROOM_GB must not turn into a refusal (the model
    # fits -- the *reserve* is what doesn't); it just can't buy passes beyond
    # the hard ceiling.
    chunk = int(max(HARD_MIN_CHUNK_SEC, min(MAX_CHUNK_SEC, raw, ceiling)))
    est_peak = m["fixed"] + m["slope"] * chunk
    if raw < HARD_MIN_CHUNK_SEC:
        _log(log_cb, "warn",
             f"Low RAM for Voxtral {kind} ({total:.0f} GB total): passes forced "
             f"to {chunk}s (~{est_peak:.0f} GB peak) and may still swap. Consider "
             f"the mini model or a machine with more RAM.")
    elif raw < PREF_MIN_CHUNK_SEC:
        _log(log_cb, "info",
             f"Voxtral {kind}: {total:.0f} GB RAM -> short {chunk}s passes "
             f"(~{est_peak:.0f} GB peak). More RAM allows longer, higher-context passes.")
    else:
        _log(log_cb, "info",
             f"Voxtral {kind}: {total:.0f} GB RAM -> ~{chunk // 60}m{chunk % 60:02d}s "
             f"passes (~{est_peak:.0f} GB peak).")
    return chunk


# A sentence is a run up to and including terminal punctuation; the final
# alternative captures a trailing run that has no terminal punctuation (the last
# pass of a file, or text Voxtral emits unpunctuated) as ONE fragment. An earlier
# `\S+$` fallback matched only the last whitespace token, silently dropping every
# word between the last period and the end of the text.
_SENT_SPLIT = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$", re.UNICODE)


def is_available():
    """True if the Voxtral backend and its dependencies can be imported.

    Deliberately NOT cached: _register_voxtral_models re-checks this on every
    dropdown open so a backend pip-installed while the app runs appears without a
    restart. The check is four cheap importlib.util.find_spec lookups (no heavy
    import), so re-running it per dropdown open is negligible."""
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("mlx_voxtral", "transformers", "torchaudio", "soundfile"))


def _log(cb, level, msg):
    if cb:
        try:
            cb(level, msg)
        except Exception:
            pass
    else:
        logger.log(logging.INFO if level == "info" else logging.DEBUG, msg)


# --------------------------------------------------------------------------- #
# Voxtral transcription
# --------------------------------------------------------------------------- #
def _cap_mlx_memory():
    """Nudge MLX to release its reusable buffer cache once usage nears physical
    RAM, so cached (not active) buffers don't push the working set into swap.
    MLX's limit is *soft* (it never hard-fails an allocation), so the real memory
    control is the per-pass length picked by _auto_chunk_sec; this only trims
    cache retention near the ceiling. Set just below total RAM so it doesn't
    throttle a normal generate pass (which is sized to stay under RAM anyway)."""
    try:
        import mlx.core as mx
        total = int(_total_ram_gb() * 1024**3)
        mx.set_memory_limit(max(8 * 1024**3, total - 4 * 1024**3))
    except Exception:
        pass


class _Voxtral:
    def __init__(self, repo):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_voxtral import load_voxtral_model, VoxtralProcessor
        self._mx = mx
        _cap_mlx_memory()
        self.model, _ = load_voxtral_model(repo, dtype=mx.bfloat16)
        self.proc = VoxtralProcessor.from_pretrained(repo)

        # Adapter so mlx_lm.generate_step can drive our LM: generate_step calls
        # model(tokens, cache=, input_embeddings=) and expects logits, while our
        # language_model takes `inputs_embeds` and returns hidden states (the
        # lm_head lives on the parent). This bridges both.
        class _LMAdapter(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.language_model = parent.language_model
                self.lm_head = parent.lm_head

            def __call__(self, inputs, cache=None, input_embeddings=None):
                h = self.language_model(inputs, cache=cache,
                                        inputs_embeds=input_embeddings)
                return self.lm_head(h)

        self._lm_adapter = _LMAdapter(self.model)

    # Default stop tokens, matching mlx_voxtral.generate_stream: </s>, [/INST],
    # and a potential padding token.
    _STOP_TOKENS = (2, 4, 32000)

    def _consume_tokens(self, token_stream, token_cb=None):
        """Collect ids from a greedy token stream up to the first stop token
        (which is dropped, matching decode(skip_special_tokens=True)). The stream
        is already length-bounded by generate_step's max_tokens.

        This deliberately does NOT replicate generate_stream's 10-identical-token
        backstop. That backstop cuts a single-token repetition loop off after only
        10 tokens, which lands *under* _looks_degenerate's thresholds (a 12-word
        run, or a >4 compression ratio) -- so the degenerate pass would slip
        through un-flagged, keeping truncated garbage and losing everything after
        the loop. Letting the loop run instead lets _looks_degenerate catch it
        (the compression-ratio net) and fire the split/penalty retry, which
        recovers clean text. In practice the backstop almost never fired anyway:
        real Voxtral loops repeat a *word* ("Jetzt. Jetzt.") whose tokens cycle,
        so consecutive-identical-token never reached 10. Output is therefore
        identical to model.generate() on every normal pass and better (retried
        rather than truncated) on the rare single-token loop."""
        stops = self._STOP_TOKENS
        out = []
        last_beat = 0.0
        for t in token_stream:
            t = int(t)
            if t in stops:
                break
            out.append(t)
            # Liveness heartbeat: Voxtral emits a whole pass at once, so without
            # this the log/progress sits silent for the entire (possibly minutes-
            # long) decode. Throttled to ~1.5 s so it never floods the queue.
            if token_cb is not None:
                now = time.monotonic()
                if now - last_beat >= 1.5:
                    last_beat = now
                    try:
                        token_cb(len(out))
                    except Exception:
                        pass
        return out

    def _fast_generate(self, mi, max_new_tokens, token_cb=None):
        """Greedy decode through the maintained mlx_lm.generate_step, returning the
        generated token ids ([1, n]).

        Output matches model.generate() (same greedy argmax) on every normal pass;
        it is faster and -- crucially -- lower peak memory on long passes, because
        generate_step processes the (large audio) prompt in prefill_step_size
        chunks instead of one forward. Measured on mini-8bit at a 600s prompt:
        ~7% faster and ~18% lower peak; near break-even on short prompts, where
        memory is not the constraint anyway. (The one intentional divergence is on
        a single-token repetition loop -- see _consume_tokens.)

        Greedy only. The rare retry path (repetition_penalty > 1) stays on the
        library implementation.
        """
        mx = self._mx
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import KVCache
        model = self.model

        # [seq, hidden] merged audio+text embeddings; generate_step adds the batch.
        embeds = model._merge_input_embeddings(
            input_ids=mi["input_ids"], input_features=mi.get("input_features"))[0]
        cache = [KVCache() for _ in range(len(model.language_model.layers))]
        # sampler=None -> greedy argmax, matching temperature 0.0.
        stream = generate_step(prompt=mx.array([], dtype=mx.int32),
                               input_embeddings=embeds, model=self._lm_adapter,
                               max_tokens=max_new_tokens, sampler=None,
                               prompt_cache=cache)
        toks = self._consume_tokens((t for t, _ in stream), token_cb=token_cb)
        return mx.array([toks], dtype=mx.uint32)

    def transcribe_array(self, audio, language, max_new_tokens=4096,
                         repetition_penalty=1.0, token_cb=None):
        inp = self.proc.apply_transcrition_request(audio=audio, language=language,
                                                   sampling_rate=SAMPLE_RATE)
        mi = {"input_ids": inp.input_ids, "input_features": inp.input_features}
        if getattr(inp, "attention_mask", None) is not None:
            mi["attention_mask"] = inp.attention_mask
        # temperature=0.0 and NO repetition penalty, matching Mistral's reference
        # transcription request. mlx_voxtral defaults repetition_penalty to 1.2,
        # which is a chat default: it divides the logit of every token seen in
        # the last 20 tokens, and in verbatim speech the most-repeated tokens are
        # punctuation and function words. Measured on a 10 min German podcast,
        # the default cost 27% of all commas (8.6 vs 10.9 per 100 words) and
        # swallowed real repetitions ("sehr, sehr" -> "sehr"), which is what made
        # transcripts read worse than Whisper's. Generation stays bounded by
        # max_new_tokens.
        if repetition_penalty == 1.0 and "attention_mask" not in mi:
            # Fast greedy path: identical tokens, faster and lower peak memory on
            # long passes. See _fast_generate. Returns the generated tokens only.
            # Skipped when a padding mask is present (the fast path assumes a
            # single unpadded sequence and the model's internal causal mask).
            gen = self._fast_generate(mi, max_new_tokens, token_cb=token_cb)
        else:
            out = self.model.generate(**mi, max_new_tokens=max_new_tokens,
                                      temperature=0.0,
                                      repetition_penalty=repetition_penalty)
            gen = out[:, inp.input_ids.shape[1]:]  # drop the prompt
        return self.proc.decode(gen[0], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# CTC forced alignment (text + audio -> word timestamps)
# --------------------------------------------------------------------------- #
class _Aligner:
    def __init__(self, model_name=ALIGN_MODEL_MULTILINGUAL):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self._torch = torch
        self.proc = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.vocab.get("<pad>", 0)
        self.delim = self.vocab.get("|", None)

    def _emission(self, audio):
        """Windowed wav2vec2 log-prob emissions [T, vocab] (bounded memory)."""
        import numpy as np
        torch = self._torch
        # Zero-copy view of the (already float32) audio buffer instead of a
        # duplicate allocation per alignment call.
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        win = int(EMISSION_WINDOW_SEC * SAMPLE_RATE)
        # wav2vec2's conv feature extractor raises on inputs shorter than its
        # receptive field (~400 samples / 25 ms). A trailing remainder that small
        # carries negligible alignment signal, so skip it rather than crash the
        # whole job -- the guard is `< min_win`, not just the empty case.
        min_win = int(0.025 * SAMPLE_RATE)
        parts = []
        with torch.inference_mode():
            for i in range(0, len(wav), win):
                seg = wav[i:i + win]
                if seg.numel() < min_win:
                    continue
                lg = self.model(seg.unsqueeze(0)).logits[0]
                parts.append(torch.log_softmax(lg, dim=-1))
        if not parts:
            return None
        return torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def _tokenize(self, words):
        tokens, tok_word = [], []
        for wi, w in enumerate(words):
            for ch in w.lower():
                if ch in self.vocab:
                    tokens.append(self.vocab[ch]); tok_word.append(wi)
            if self.delim is not None:
                tokens.append(self.delim); tok_word.append(-1)
        return tokens, tok_word

    def _spread(self, words, audio, t_offset):
        """Fallback: distribute words evenly (by length) over the audio when
        real alignment isn't possible (empty/too-dense text)."""
        if not words:
            return []
        dur = len(audio) / SAMPLE_RATE
        total = sum(len(w) for w in words) or len(words)
        out, pos = [], 0.0
        for w in words:
            s = t_offset + dur * pos
            pos += (len(w) or 1) / total
            e = t_offset + dur * pos
            out.append({"word": w, "start": s, "end": e, "prob": 0.0})
        return out

    def align_words(self, words, audio, t_offset=0.0, depth=0):
        """Return [{word,start,end,prob}] for `words` against `audio`.

        CTC forced alignment requires at least as many audio frames as target
        tokens. Dense speech (or a Voxtral over-generation) in a long chunk can
        have more character tokens than frames, which makes forced_align raise.
        When that happens we split the words in half, cut the audio at a nearby
        pause, and align each half recursively — so we always get a transcript
        instead of a crash. As a last resort the words are spread evenly.
        """
        import torchaudio
        torch = self._torch
        if not words:
            return []
        if len(audio) < int(0.1 * SAMPLE_RATE):
            return self._spread(words, audio, t_offset)

        emission = self._emission(audio)
        if emission is None:
            return self._spread(words, audio, t_offset)
        n_frames = emission.shape[0]
        fps = n_frames / (len(audio) / SAMPLE_RATE)
        tokens, tok_word = self._tokenize(words)
        if not tokens:
            return self._spread(words, audio, t_offset)

        too_dense = len(tokens) > n_frames * 0.95
        # torchaudio's CPU forced_align indexes its (frames x 2*tokens+1) DP
        # buffer with 32-bit ints; once the product nears 2**31 the index
        # wraps negative and the whole process dies with SIGSEGV (verified
        # empirically on torchaudio 2.11 -- int64 targets crash too, so no
        # dtype workaround exists). Split well below that limit; smaller
        # windows are also much faster to align.
        too_big = n_frames * (2 * len(tokens) + 1) > FORCED_ALIGN_MAX_CELLS
        if too_big and (len(words) == 1 or depth >= 12):
            # Cannot split further -- never risk the segfault.
            return self._spread(words, audio, t_offset)
        if not (too_dense or too_big) or len(words) == 1 or depth >= 12:
            try:
                targets = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0)
                aligned, scores = torchaudio.functional.forced_align(
                    emission.unsqueeze(0), targets, blank=self.blank)
                spans = torchaudio.functional.merge_tokens(aligned[0], scores[0])
            except Exception:
                return self._spread(words, audio, t_offset)

            out, ti = {}, 0
            for sp in spans:
                if sp.token == self.blank:
                    continue
                if ti < len(tok_word):
                    wi = tok_word[ti]
                    if wi >= 0:
                        s = t_offset + sp.start / fps
                        e = t_offset + sp.end / fps
                        if wi not in out:
                            out[wi] = [s, e, float(sp.score)]
                        else:
                            out[wi][1] = e
                    ti += 1

            # Fill words whose characters were all out-of-vocabulary (numbers,
            # symbols) by interpolating between aligned neighbours.
            n = len(words)
            idx = sorted(out)
            if not idx:
                return self._spread(words, audio, t_offset)
            full = [None] * n
            for wi in idx:
                full[wi] = out[wi]
            # Leading/trailing OOV words: spread them over the audio before the
            # first / after the last aligned word (capped at ~0.6 s per word)
            # instead of collapsing them to zero-width spans, which would turn
            # into zero-duration cues that subtitle players skip or reject.
            # A tiny per-word floor so a lead/tail OOV word never collapses to a
            # zero-duration cue when the first/last aligned word sits exactly on
            # the chunk boundary (step would otherwise be 0). The resulting
            # sub-frame overlap into neighbouring audio is harmless for cues.
            min_step = 0.02
            if idx[0] > 0:
                s1 = out[idx[0]][0]
                lead = max(t_offset, s1 - 0.6 * idx[0])
                step = max((s1 - lead) / idx[0], min_step)
                for i in range(idx[0]):
                    full[i] = [lead + step * i, lead + step * (i + 1), 0.0]
            n_tail = n - (idx[-1] + 1)
            if n_tail > 0:
                e0 = out[idx[-1]][1]
                tail_end = min(t_offset + len(audio) / SAMPLE_RATE, e0 + 0.6 * n_tail)
                step = max((tail_end - e0) / n_tail, min_step)
                for k, i in enumerate(range(idx[-1] + 1, n)):
                    full[i] = [e0 + step * k, e0 + step * (k + 1), 0.0]
            for a, b in zip(idx, idx[1:]):
                if b - a > 1:
                    s0, s1, gap = out[a][1], out[b][0], b - a
                    for k in range(a + 1, b):
                        t = s0 + (s1 - s0) * (k - a) / gap
                        full[k] = [t, t, 0.0]
            return [{"word": words[i], "start": full[i][0],
                     "end": full[i][1], "prob": full[i][2]} for i in range(n)]

        # Too dense or too big: split words in half and audio at a nearby
        # pause, recurse.
        mid = max(1, len(words) // 2)
        first_chars = sum(len(w) for w in words[:mid]) + mid
        total_chars = sum(len(w) for w in words) + len(words)
        cut = int(len(audio) * first_chars / max(1, total_chars))
        # snap the cut to the quietest 100 ms frame within +/- 3 s, then keep it
        # a valid interior split point
        fr = max(1, int(0.1 * SAMPLE_RATE))
        cut = _quietest_frame_near(audio, cut, 3.0)
        cut = min(max(cut, fr), len(audio) - fr)
        left = self.align_words(words[:mid], audio[:cut], t_offset, depth + 1)
        right = self.align_words(words[mid:], audio[cut:],
                                 t_offset + cut / SAMPLE_RATE, depth + 1)
        return left + right


# --------------------------------------------------------------------------- #
# Segment building
# --------------------------------------------------------------------------- #
def _split_sentences(text):
    return [m.group(0).strip() for m in _SENT_SPLIT.finditer(text) if m.group(0).strip()]


# Subtitle cue sizing: aim for short, readable phrases (a few words), not one
# word per cue and not 30-second blocks — comparable to what noScribe produces
# from Whisper's phrase-level segments.
SUB_MAX_CHARS = 45
SUB_MAX_WORDS = 12
SUB_MAX_SEC = 6.0
SUB_MIN_WORDS = 4


def _segments_from_words(word_stamps):
    """Group aligned words into subtitle-sized cues.

    A cue is ended at sentence punctuation, at a clause boundary (comma/colon)
    once it is long enough, or when it hits a length cap (characters, words or
    duration). This keeps cues to a readable phrase rather than a whole
    sentence or a fixed time window.
    """
    segments = []
    cur = []

    def flush():
        if not cur:
            return
        txt = " ".join(w["word"] for w in cur).strip()
        if txt:
            segments.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "text": " " + txt,
                "words": list(cur),
            })
        cur.clear()

    for w in word_stamps:
        cur.append(w)
        tok = w["word"]
        chars = sum(len(x["word"]) + 1 for x in cur)
        dur = cur[-1]["end"] - cur[0]["start"]
        if tok.endswith((".", "!", "?", "…")):
            flush()
        elif (len(cur) >= SUB_MIN_WORDS and tok.endswith((",", ";", ":"))
              and (chars >= SUB_MAX_CHARS * 0.55 or dur >= SUB_MAX_SEC * 0.55)):
            flush()
        elif len(cur) >= SUB_MAX_WORDS or chars >= SUB_MAX_CHARS or dur >= SUB_MAX_SEC:
            flush()
    flush()

    # Merge a tiny leftover cue (e.g. a lone "habe.") back into the previous one
    # when the result still fits, so subtitles don't get one-word fragments.
    merged = []
    for seg in segments:
        if merged and len(seg["words"]) <= 2 and (seg["end"] - seg["start"]) < 1.2:
            prev = merged[-1]
            combined = prev["words"] + seg["words"]
            if len(combined) <= SUB_MAX_WORDS + 3 and (seg["end"] - prev["start"]) <= SUB_MAX_SEC + 2:
                prev["words"] = combined
                prev["end"] = seg["end"]
                prev["text"] = " " + " ".join(w["word"] for w in combined).strip()
                continue
        merged.append(seg)
    return merged


def _segments_text_only(text, duration):
    """Short path: sentence segments with proportional (approximate) times."""
    sentences = _split_sentences(text)
    total = sum(len(s) for s in sentences) or 1
    segments, pos = [], 0
    for s in sentences:
        start = duration * pos / total
        pos += len(s)
        end = duration * pos / total
        segments.append({"start": start, "end": end, "text": " " + s, "words": None})
    return segments


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def transcribe(audio_path, language="de", need_timestamps=True,
               voxtral_repo=None, chunk_sec=None,
               corrections_path=None, speaker_names=None, ram_reserve_gb=None,
               log_cb=None, progress_cb=None, segment_cb=None):
    """
    Transcribe `audio_path` with Voxtral and return noScribe-compatible segments.

    need_timestamps=True  -> also run forced alignment for word-level times
                             (needed for VTT, speaker assignment, pauses).
    need_timestamps=False -> fast path, plain text only (approx. segment times).

    corrections_path -> optional YAML word-correction list (brand/product names)
                        applied to the transcribed text.
    chunk_sec        -> per-pass length in seconds; None/0 = pick automatically
                        from RAM (see _auto_chunk_sec). A short file that fits one
                        pass is never split regardless of this value.
    segment_cb       -> optional callable(segment_dict); called for each finished
                        segment as soon as its pass completes, so callers can
                        stream/autosave partial transcripts instead of waiting
                        for the whole file.
    """
    import soundfile as sf

    from noScribe import transcript_corrections
    corrections = transcript_corrections.load_corrections(corrections_path)
    if corrections:
        _log(log_cb, "info", f"Applying {len(corrections)} word correction(s).")
    # The speaker names the user entered are the correct spelling of words that
    # are very likely to be spoken. A speech model cannot know whether to write
    # "Markus" or "Marcus", so normalise same-sounding spellings to theirs.
    speaker_names = [n for n in (speaker_names or []) if n]
    if speaker_names:
        _log(log_cb, "info", f"Normalising spoken names to: {', '.join(speaker_names)}")

    # A missing file must be diagnosed as a missing file, not as whatever the
    # memory sizing below happens to find wrong with the machine.
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # bf16 mini as the bare-API default: always fetchable from the hub and its
    # repo name classifies correctly in MEM_MODEL.
    repo = voxtral_repo or VOXTRAL_MODELS["voxtral-mini-8bit"]
    # The unquantised source releases carry no bit width in their name, so
    # _model_kind would meter them with a quantised profile and wave a too-long
    # pass through -- and the 24B one is a 48 GB download. The GUI never gets
    # here (it only offers the published builds), but a direct caller must be
    # stopped just as early.
    if str(repo) in SOURCE_REPOS:
        raise ValueError(
            f"{repo} is an unquantised source release, not a runnable build. "
            f"Use one of the published builds ({', '.join(VOXTRAL_MODELS)}) or "
            f"convert it first with tools/quantize_voxtral.py.")

    # Size the passes BEFORE loading anything: the MemoryError for models that
    # cannot fit this machine is only worth something if it comes before 20+ GB
    # of weights have already pushed the OS into swap.
    # The config value may arrive as a string (YAML round-trip); coerce before
    # any arithmetic and fall back to auto-sizing on junk.
    try:
        chunk_sec = float(chunk_sec) if chunk_sec else None
    except (TypeError, ValueError):
        chunk_sec = None
    if not chunk_sec or chunk_sec <= 0:
        chunk_sec = _auto_chunk_sec(repo, log_cb, ram_reserve_gb)
    else:
        # A pinned voxtral_chunk_sec must not bypass the safety nets (it may
        # well predate a switch to a hungrier model). It may exceed the *auto*
        # length -- that one holds back the configurable reserve, and pinning
        # is the documented way to trade that reserve for context -- but
        # neither the hard memory ceiling past which the working set no longer
        # fits and the run stops progressing, nor the model-context cap
        # (MAX_CHUNK_SEC). A model whose shortest pass cannot fit is refused
        # outright, exactly as in the automatic path.
        ceiling = max_safe_chunk_sec(repo)  # raises MemoryError if unfit
        pinned = chunk_sec
        chunk_sec = int(min(pinned, ceiling, MAX_CHUNK_SEC))
        if chunk_sec < pinned:
            what = ("more memory than this machine has"
                    if ceiling < MAX_CHUNK_SEC else "more context than the model has")
            _log(log_cb, "warn",
                 f"voxtral_chunk_sec={pinned:.0f}s would need {what}; "
                 f"using {chunk_sec}s.")

    _log(log_cb, "info", f"Loading Voxtral model: {repo}")
    vox = _Voxtral(repo)

    aligner = None
    if need_timestamps:
        align_model = resolve_align_model(language)
        _log(log_cb, "info", f"Loading alignment model: {align_model}")
        aligner = _Aligner(align_model)

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr}")
    duration = len(audio) / SAMPLE_RATE
    # Balance the passes: take the fewest passes that keep each within the
    # RAM-safe length, then split evenly. Even passes share the memory headroom
    # and avoid a tiny leftover tail (e.g. 1143s @1006 -> 2x571s, not 1006+137),
    # which also lowers the per-pass peak.
    import math
    max_len = max(1, int(chunk_sec * SAMPLE_RATE))
    n_passes = max(1, math.ceil(len(audio) / max_len))
    chunk_len = math.ceil(len(audio) / n_passes)
    # Wide backward hunt for a real pause (a shorter pass is always memory-safe),
    # small forward reach so a cut can't grow the pass much past the RAM budget.
    back_len = min(int(SILENCE_SEARCH_SEC * SAMPLE_RATE), chunk_len // 2)
    fwd_len = int(min(SILENCE_SEARCH_SEC, 20) * SAMPLE_RATE)
    bounds = _chunk_boundaries(audio, chunk_len, back_len, fwd_len, max_len)
    n_chunks = len(bounds) - 1
    if n_chunks > 1:
        _log(log_cb, "info",
             f"Audio {duration / 60:.1f} min > one pass -> {n_chunks} passes "
             f"(pause-aligned, {OVERLAP_SEC}s overlap).")
    # Overlap gives the model lead-in context at a seam; only used on the long
    # path where the duplicate can be dropped cleanly by timestamp.
    overlap = int(OVERLAP_SEC * SAMPLE_RATE) if aligner is not None else 0
    all_segments = []
    # Progress is measured against the WHOLE audio (each pass contributes its own
    # share of the total duration, so a short final pass moves the bar only a
    # little), and never runs backward. _prog_max keeps it monotonic across the
    # intra-pass estimate, the pass-complete snap, and any split-retry re-decode.
    n_samples = max(1, len(audio))
    _prog_max = [0]

    def _emit_progress(pct):
        pct = int(pct)
        if pct > _prog_max[0]:
            _prog_max[0] = pct
        else:
            pct = _prog_max[0]
        if progress_cb:
            try:
                progress_cb(pct)
            except Exception:
                pass

    for ci in range(n_chunks):
        a0 = bounds[ci]
        a1 = bounds[ci + 1]
        a_read = max(0, a0 - overlap) if ci > 0 else a0
        chunk = audio[a_read:a1]
        t_offset = a_read / SAMPLE_RATE
        _log(log_cb, "info", f"Transcribing chunk {ci + 1}/{n_chunks} "
                             f"({a0 / SAMPLE_RATE:.0f}-{a1 / SAMPLE_RATE:.0f}s)")
        # Intra-pass liveness: Voxtral returns the whole pass at once, so estimate
        # how far the decode is from the token count and map it onto this pass's
        # slice of the overall bar (capped below the pass boundary; the real
        # position is set when the pass finishes below).
        _base = a0 / n_samples
        _span = max(0.0, (a1 - a0) / n_samples)
        _exp_tokens = max(1.0, (a1 - a0) / SAMPLE_RATE * _EST_TOKENS_PER_SEC)

        def _heartbeat(ntok, _base=_base, _span=_span, _exp=_exp_tokens):
            _emit_progress((_base + min(0.95, ntok / _exp) * _span) * 100)

        text = _transcribe_guarded(vox, chunk, language, log_cb,
                                   f"Pass {ci + 1}/{n_chunks}", token_cb=_heartbeat)
        if not text:
            continue
        if corrections:
            text = transcript_corrections.apply_corrections(text, corrections)
        if speaker_names:
            text = transcript_corrections.apply_name_corrections(
                text, speaker_names, language)
        if aligner is not None:
            words = re.findall(r"\S+", text)
            _log(log_cb, "info", f"Pass {ci + 1}/{n_chunks}: aligning word timestamps "
                                 f"({len(words)} words)...")
            stamps = aligner.align_words(words, chunk, t_offset=t_offset)
            if a_read < a0:
                # Drop the *words* already covered by the previous pass (the
                # overlap region). Filtering at word rather than cue level
                # means a cue straddling the seam can neither duplicate its
                # pre-seam words nor lose its post-seam ones.
                b = a0 / SAMPLE_RATE
                stamps = [w for w in stamps if (w["start"] + w["end"]) / 2 >= b]
            segs = _segments_from_words(stamps)
        else:
            segs = _segments_text_only(text, (a1 - a_read) / SAMPLE_RATE)
            for seg in segs:
                seg["start"] += t_offset
                seg["end"] += t_offset
        all_segments.extend(segs)
        if segment_cb:
            # Stream this pass's segments right away so the caller can show
            # and autosave a partial transcript during long files.
            for seg in segs:
                segment_cb(seg)
        # Release MLX buffers so memory stays flat across chunks of a long file.
        try:
            vox._mx.clear_cache()
        except Exception:
            pass
        # Snap the bar to this pass's true end position in the whole audio
        # (duration-weighted; the final pass lands on 100%).
        _emit_progress(a1 / n_samples * 100)

    return all_segments, {"duration": duration, "language": language}
