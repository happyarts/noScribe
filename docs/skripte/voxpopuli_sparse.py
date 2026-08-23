"""Kostet der Perzentil-Boden etwas, wenn ein Pass grossteils still ist?

`voxpopuli_floor.py` misst dichte Rede: die Stroeme bestehen fast nur aus
Sprache, und dort liegt das 99. Perzentil 19,5-26,8 dB unter dem Maximum
(Median 22; auf dem redigierten Interview nur 15,6-17,8). In einem
Pass, der ueberwiegend aus Pausen oder Vorlauf besteht -- die 60-s-Kopfprobe
auf einer Aufnahme, die mit Stille beginnt, ein Fenster ueber eine lange
Pause --, ist das oberste Prozent der Zellen immer noch Sprache, aber das
Perzentil rutscht innerhalb der Sprachzellen nach unten, und der Boden liegt
tiefer als in jedem dichten Strom. Ob das Modell dort anders hoert, misst
dieser Lauf: dieselben Goldtranskripte, dieselben Clips, aber zwischen den
Clips Raumrauschen statt 0,4 s Stille, bis der Sprachanteil bei `anteil`
liegt. Beide Boeden auf bitgleichem Audio, gepaart.

    python docs/skripte/voxpopuli_sparse.py [n_stroeme] [sekunden] [anteil] [boeden]

`boeden` in der Kurzform von `voxpopuli_floor.floor_arm` (`max,99,99c20`);
der erste Arm ist die Basis der gepaarten Auswertung.

`anteil` ist der Sprachanteil des Stroms (0,5 = halb Rede, halb Rauschen).
Das Raumrauschen ist weisses Rauschen bei -60 dBFS -- der Pegel eines
leisen Raums in einer Aufnahme, die bis 1,0 ausgesteuert ist.
"""
import pathlib
import sys

import numpy as np
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, wer                                    # noqa: E402
from fleurs_stream import paired, SR                         # noqa: E402
from noScribe.voxtral_engine import _Voxtral                 # noqa: E402
from voxpopuli_floor import floor_arm, load_clips            # noqa: E402
from mel_outlier_gap import raw_log_mel                      # noqa: E402

ROOM_DB = -60.0


def sparse_streams(clips, refs, n_streams, sec, share, seed=20260823):
    """Stroeme von `sec` Sekunden, in denen Sprache `share` der Zeit ausmacht
    und der Rest Raumrauschen ist, gleichmaessig zwischen die Clips verteilt."""
    rng = np.random.default_rng(seed)
    out, i = [], 0
    want_speech = int(sec * share * SR)
    while len(out) < n_streams and i < len(clips):
        parts, words, total = [], [], 0
        while total < want_speech and i < len(clips):
            parts.append(clips[i])
            words += norm(refs[i])
            total += len(clips[i])
            i += 1
        if total < want_speech * 0.8 or not words:
            continue
        gap = int((sec * SR - total) / (len(parts) + 1))
        room = lambda: (rng.standard_normal(gap) * 10 ** (ROOM_DB / 20)).astype(np.float32)
        audio = [room()]
        for p in parts:
            audio += [p, room()]
        out.append((np.concatenate(audio).astype(np.float32), words))
    return out


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    specs = sys.argv[4].split(",") if len(sys.argv) > 4 else ["max", "99"]

    clips, refs = load_clips(int(n_streams * sec * share / 9) + 60)
    st = sparse_streams(clips, refs, n_streams, sec, share)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} Stroeme aus VoxPopuli de (gold), je ~{sec:.0f}s, "
          f"Sprachanteil {share:.0%}, {total/60:.1f} min gesamt")

    # Die Dosis: wie weit liegt das Perzentil unter dem Maximum?
    for spec in specs:
        if spec == "max":
            continue
        p = float(spec.split("c")[0])
        gaps = [(raw_log_mel(x).max() - np.percentile(raw_log_mel(x), p)) * 10
                for x, _ in st]
        print(f"# Abstand max -> Perzentil {p:g}: median {np.median(gaps):.1f} dB "
              f"(min {min(gaps):.1f}, max {max(gaps):.1f})")

    vox = _Voxtral(str(REPO / "models" / "voxtral-mini-8bit"))
    arms = [floor_arm(s) for s in specs]

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    print(f"\n{'Boden':18s} {'WER':>7s} {'CER':>7s}")
    store = {}
    for name, ex in arms:
        vox.proc.feature_extractor = ex
        rows = []
        for x, words in st:
            hyp = norm(vox.transcribe_array(
                x, "de", max_new_tokens=int(len(x) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        print(f"{name:18s} {rate(rows,'w','ref_w'):6.2f}% {rate(rows,'c','ref_c'):6.2f}%",
              flush=True)
        store[name] = rows
        mx.clear_cache()

    a = store[arms[0][0]]  # erster Arm ist die Basis
    print(f"\nGepaart gegen {arms[0][0]} (95%; enthaelt es 0, nicht nachweisbar):")
    for name, _ in arms[1:]:
        b = store[name]
        lo_w, hi_w = paired(a, b, "w", "ref_w")
        lo_c, hi_c = paired(a, b, "c", "ref_c")
        dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
        dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
        star = " *" if (lo_w > 0) == (hi_w > 0) else ""
        print(f"  {name:18s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star}"
              f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]")
    print("  (positiv = Perzentil schlechter)")


if __name__ == "__main__":
    main()
