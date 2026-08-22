"""Final numpy replacement for torchaudio.functional.forced_align / merge_tokens.
The same code is embedded in docs/viterbi-numpy-brief.md; this file is importable
for the benches and for the engine migration. See the brief for the measurements.
"""
import numpy as np

NEG = np.float32(-np.inf)           # same floor as torchaudio; nothing here subtracts


def forced_align(log_probs, targets, blank=0):
    """CTC Viterbi forced alignment, `torchaudio.functional.forced_align` semantics
    (unbatched): returns one label per frame and that label's log-prob per frame."""
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    N = 2 * L + 1
    repeats = int(np.count_nonzero(targets[1:] == targets[:-1]))
    if L == 0:
        raise ValueError("targets must not be empty")
    if T < L + repeats:
        raise ValueError(f"targets length is too long for CTC: {L} tokens + "
                         f"{repeats} repeats need {L + repeats} frames, got {T}")
    # Advance-by-two penalty: 0 on token states whose preceding token differs
    # (skipping the blank between them is legal), -inf everywhere else.
    pen = np.full(N, NEG, dtype=np.float32)
    pen[(np.flatnonzero(targets[1:] != targets[:-1]) + 1) * 2 + 1] = 0.0

    alpha = np.full(N, NEG, dtype=np.float32)
    nxt = np.full(N, NEG, dtype=np.float32)
    two = np.full(N, NEG, dtype=np.float32)   # two[:2] stays -inf for the whole run
    c1 = np.zeros(N, dtype=bool)              # c1[0] stays False for the whole run
    c2 = np.zeros(N, dtype=bool)
    c1v, c2v = c1.view(np.uint8), c2.view(np.uint8)
    bt = np.empty((T, N), dtype=np.uint8)     # every row t >= 1 is fully written below
    bt[0] = 0

    alpha[0] = log_probs[0, blank]
    alpha[1] = log_probs[0, targets[0]]

    for t in range(1, T):
        np.greater(alpha[:-1], alpha[1:], out=c1[1:])      # advance-one beats stay
        np.maximum(alpha[1:], alpha[:-1], out=nxt[1:])
        nxt[0] = alpha[0]
        np.add(alpha[:-2], pen[2:], out=two[2:])           # advance-two, -inf where illegal
        np.greater(two, nxt, out=c2)                       # ... beats both
        np.maximum(nxt, two, out=nxt)
        row = bt[t]                                        # 2 where the skip won, else c1
        np.add(c2v, c2v, out=row)
        np.maximum(row, c1v, out=row)
        lp = log_probs[t]
        nxt[0::2] += lp[blank]                             # blank states, one scalar
        nxt[1::2] += lp[targets]                           # token states, gather of L
        alpha, nxt = nxt, alpha

    # torchaudio ends on the final blank only if it is strictly better
    s = N - 1 if alpha[N - 1] > alpha[N - 2] else N - 2
    path = np.empty(T, dtype=np.int64)
    scores = np.empty(T, dtype=np.float32)
    ext = np.zeros(N, dtype=np.int64)
    ext[1::2] = targets
    for t in range(T - 1, -1, -1):
        tok = ext[s]
        path[t] = tok
        scores[t] = log_probs[t, tok]
        s -= int(bt[t, s])   # int(): numpy would type the in-place op as uint8
    return path, scores


def merge_tokens(path, scores, blank=0):
    """`torchaudio.functional.merge_tokens`: runs of equal non-blank labels as
    (token, start, end, score) with an exclusive end and the mean frame score."""
    path = np.asarray(path)
    cut = np.flatnonzero(path[1:] != path[:-1]) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(path)]))
    return [(int(path[s]), int(s), int(e), float(scores[s:e].mean()))
            for s, e in zip(starts, ends) if path[s] != blank]
