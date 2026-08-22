"""Runde 2: der Transienten-Schaden auf vielen Strömen statt einer Passage.

Die Dosis-Wirkung des Clamp-Bodens (`docs/voxtral-audio-vorverarbeitung.md`
§6b) stand zunächst auf einer einzigen handkorrigierten Passage. Hier läuft
sie über zehn VoxPopuli-Ströme mit Goldtranskript, jeder einmal sauber und
einmal mit einem eingesetzten Transienten, unter beiden Böden.

Die Frage ist nicht "welcher Boden ist besser" -- das misst
`voxpopuli_floor.py` auf sauberem Material -- sondern: **was kostet ein
Transient unter dem jeweiligen Boden.** Verglichen wird deshalb immer
WITHIN-Schema (sauber gegen Knall beim selben Boden), was auch jede
Verzerrung eines Referenztranskripts herausfallen lässt.

Der Knall wird relativ zur Sprachspitze des Stroms gesetzt, nicht absolut,
damit alle Ströme dieselbe Dosis sehen.

    python docs/skripte/voxpopuli_spike.py [n_stroeme] [sekunden] [dB_ueber_Sprache]
"""
import pathlib
import sys
import time

import numpy as np
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, wer                                    # noqa: E402
from fleurs_stream import streams, paired, SR                # noqa: E402
from noScribe.voxtral_engine import _Voxtral                 # noqa: E402
from voxpopuli_floor import PercentileFloor, load_clips      # noqa: E402


def with_spike(x, db_over, at=0.45, ms=100):
    """Ein `ms` langer Vollpegel-Impuls, skaliert auf `db_over` über der
    Sprachspitze dieses Stroms."""
    a = np.array(x, dtype=np.float32, copy=True)
    amp = float(np.abs(a).max()) * (10.0 ** (db_over / 20.0))
    n = int(ms * SR / 1000)
    pos = int(at * len(a))
    a[pos:pos + n] = min(amp, 1.0)
    return a


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    db_over = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0

    clips, refs = load_clips(int(n_streams * sec / 9) + 60)
    st = streams(clips, refs, n_streams, sec)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} Ströme aus VoxPopuli de (gold), je ~{sec:.0f}s, "
          f"{total/60:.1f} min gesamt")
    print(f"# Transient: {db_over:+.0f} dB über der Sprachspitze, 100 ms, bei 45 %")

    vox = _Voxtral(str(REPO / "models" / "voxtral-mini-8bit"))
    stock, pctx = vox.proc.feature_extractor, PercentileFloor(99.9)

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    def score(ex, spike):
        vox.proc.feature_extractor = ex
        rows, t0 = [], time.time()
        for x, words in st:
            a = with_spike(x, db_over) if spike else np.ascontiguousarray(x, dtype=np.float32)
            hyp = norm(vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        mx.clear_cache()
        return rows, time.time() - t0

    print(f"\n{'Boden':16s} {'Audio':10s} {'WER':>7s} {'CER':>7s}")
    store = {}
    for name, ex in (("max (ist)", stock), ("Perzentil 99,9", pctx)):
        for au, spike in (("sauber", False), ("mit Knall", True)):
            rows, el = score(ex, spike)
            store[(name, au)] = rows
            print(f"{name:16s} {au:10s} {rate(rows,'w','ref_w'):6.2f}% "
                  f"{rate(rows,'c','ref_c'):6.2f}%", flush=True)

    print("\nKosten des Transienten, gepaart je Boden "
          "(95 %; enthält es 0, nicht nachweisbar):")
    for name in ("max (ist)", "Perzentil 99,9"):
        a, b = store[(name, "sauber")], store[(name, "mit Knall")]
        lo_w, hi_w = paired(a, b, "w", "ref_w")
        lo_c, hi_c = paired(a, b, "c", "ref_c")
        dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
        dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
        star = " *" if (lo_w > 0) == (hi_w > 0) else ""
        print(f"  {name:16s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star}"
              f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]")
    print("  (positiv = der Knall schadet)")


if __name__ == "__main__":
    main()
