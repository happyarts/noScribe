"""Guard against the forced_align 32-bit index-overflow segfault.

torchaudio's CPU forced_align kernel indexes its (frames x 2*tokens+1) DP
buffer with 32-bit ints. Once the product nears 2**31 the index wraps
negative and the whole worker process dies with SIGSEGV -- observed in a
real noScribe run (2026-07-22, torchaudio 2.11; int64 targets crash the
same way, so there is no dtype workaround). ``_Aligner.align_words`` must
therefore split oversized windows *before* calling forced_align.

This test drives align_words with a fake emission (no wav2vec2 download)
over a case above FORCED_ALIGN_MAX_CELLS and asserts every actual
forced_align invocation stays under the cap while still timestamping
every word.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")
# The package __init__ loads noScribe.main lazily, but voxtral_engine still
# needs its own heavy deps at import time.
pytest.importorskip("transformers")

from noScribe.voxtral_engine import (  # noqa: E402
    _Aligner,
    FORCED_ALIGN_MAX_CELLS,
    SAMPLE_RATE,
)

VOCAB_SIZE = 30  # incl. blank


def _stub_aligner():
    al = object.__new__(_Aligner)
    al._torch = torch
    al.vocab = {ch: i + 1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}
    al.blank = 0
    al.delim = None
    gen = torch.Generator().manual_seed(0)
    # 50 fps at 16 kHz, random log-probs -- shape is all that matters here.
    al._emission = lambda audio: torch.log_softmax(
        torch.rand((len(audio) // 320, VOCAB_SIZE), generator=gen), dim=-1
    )
    return al


def test_oversized_window_splits_below_cap(monkeypatch):
    al = _stub_aligner()

    calls = []
    real_fa = torchaudio.functional.forced_align

    def checked_fa(emission, targets, blank=0):
        calls.append(emission.shape[1] * (2 * targets.shape[1] + 1))
        return real_fa(emission, targets, blank=blank)

    monkeypatch.setattr(torchaudio.functional, "forced_align", checked_fa)

    # 1100 s -> 55_000 frames; 2000 x 5-char words -> 10_000 tokens
    # -> 55_000 * 20_001 = 1.10e9 cells, just above the 2**30 cap.
    words = ["abcde"] * 2000
    audio = np.zeros(1100 * SAMPLE_RATE, dtype=np.float32)
    n_frames = len(audio) // 320
    assert n_frames * (2 * len(words) * 5 + 1) > FORCED_ALIGN_MAX_CELLS

    out = al.align_words(words, audio)

    assert calls, "forced_align was never reached"
    assert all(c <= FORCED_ALIGN_MAX_CELLS for c in calls)
    assert len(calls) >= 2  # the window was actually split
    assert len(out) == len(words)
    assert all(w["end"] >= w["start"] for w in out)
    starts = [w["start"] for w in out]
    assert starts == sorted(starts)
