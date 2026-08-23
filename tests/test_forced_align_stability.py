"""Acceptance test for the Voxtral aligner's DP against a recorded reference.

The word timestamps come from ``noScribe.ctc_align`` (``forced_align`` +
``merge_tokens``), a numpy replacement for the torchaudio kernel of the same
names. The reference in ``tests/data/`` was recorded from torchaudio 2.8 on
2026-07-22 and verified identical on 2.11: 43 cases (random emissions incl.
heavy repeats, tight T == L+R fits, single-token targets). The numpy DP must
reproduce every path and span bit-for-bit -- that is what makes it a drop-in
rather than a rewrite. ``--regenerate`` records the file from torchaudio
again, so the reference stays the *other* implementation's answer.

What the reference cannot see, tests/test_ctc_align.py covers: blank indices
other than 0, and exact ties, where the numpy DP is deliberately *better*
than torchaudio (pytorch/audio#4221).

Regenerate the reference (only after manually vetting a divergence):
    python tests/test_forced_align_stability.py --regenerate
"""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from noScribe import ctc_align  # noqa: E402

REF_PATH = Path(__file__).parent / "data" / "forced_align_ref.npz"


def _cases():
    """Deterministic case specs: (T, C, targets). Mirrors what the recorded
    reference was built from -- do not change without regenerating it."""
    rng = np.random.default_rng(42)
    specs = []
    for i in range(40):
        C = int(rng.integers(5, 60))           # vocab incl. blank
        L = int(rng.integers(1, 80))           # target length
        tg = rng.integers(1, C, size=L)
        if i % 3 == 0 and L > 3:               # force heavy repeats sometimes
            tg[1::2] = tg[0:-1:2][: len(tg[1::2])]
        R = int(np.sum(tg[1:] == tg[:-1]))
        T = int(L + R + rng.integers(0, 200))  # from tightest possible upward
        specs.append((T, C, tg))
    specs.append((1, 5, np.array([2])))        # single frame, single token
    specs.append((500, 40, np.array([7])))     # long audio, one token
    specs.append((7, 6, np.array([3, 3, 3, 3])))  # T == L+R exactly
    return specs


def _emission(T, C, tg):
    # torch's seeded RNG and log_softmax are bitwise stable across the
    # versions we care about (verified 2.8 vs 2.13).
    g = torch.Generator().manual_seed(hash((T, C, len(tg))) % (2**31))
    return torch.randn((1, T, C), generator=g).log_softmax(-1)


def _pack(paths, scores, spans):
    # the reference keeps torchaudio's batch axis on paths and scores
    return (
        np.asarray(paths).reshape(1, -1),
        np.asarray(scores).reshape(1, -1),
        np.array([[s.token, s.start, s.end] for s in spans], dtype=np.int64),
        np.array([s.score for s in spans], dtype=np.float64),
    )


def _run(T, C, tg):
    lp = _emission(T, C, tg)[0].numpy()
    paths, scores = ctc_align.forced_align(lp, tg, blank=0)
    return _pack(paths, scores, ctc_align.merge_tokens(paths, scores))


def _run_torchaudio(T, C, tg):
    F = pytest.importorskip("torchaudio.functional")
    lp = _emission(T, C, tg)
    targets = torch.tensor(tg, dtype=torch.int32).unsqueeze(0)
    paths, scores = F.forced_align(lp, targets, blank=0)
    return _pack(paths.numpy(), scores.numpy(), F.merge_tokens(paths[0], scores[0]))


@pytest.mark.parametrize("idx,spec", list(enumerate(_cases())))
def test_forced_align_matches_recorded_reference(idx, spec):
    ref = np.load(REF_PATH)
    paths, scores, spani, spanf = _run(*spec)
    # The integer outputs decide the word timestamps -- they must be
    # bit-identical everywhere (verified across torchaudio 2.8/2.11 and
    # macOS arm64 / Linux x86_64).
    assert np.array_equal(ref[f"paths_{idx}"], paths), "alignment path changed"
    assert np.array_equal(ref[f"spani_{idx}"], spani), "token spans changed"
    # Float scores depend on the platform's log_softmax reduction order
    # (last-ulp differences on other CPU architectures), so a tight
    # tolerance instead of equality.
    assert np.allclose(ref[f"scores_{idx}"], scores, rtol=0, atol=1e-5), "frame scores drifted"
    assert np.allclose(ref[f"spanf_{idx}"], spanf, rtol=0, atol=1e-5), "span scores drifted"


if __name__ == "__main__":
    import sys

    if "--regenerate" not in sys.argv:
        sys.exit("run via pytest, or pass --regenerate to rewrite the reference")
    out = {}
    for i, spec in enumerate(_cases()):
        # recorded from torchaudio on purpose: the reference is the *other*
        # implementation, not the one under test
        for key, arr in zip(("paths", "scores", "spani", "spanf"), _run_torchaudio(*spec)):
            out[f"{key}_{i}"] = arr
    REF_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(REF_PATH, **out)
    import torchaudio

    print(f"wrote {REF_PATH} from torchaudio {torchaudio.__version__}")
