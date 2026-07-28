"""Kostet ein angebrochenes letztes 30-s-Fenster etwas? Mit genug Wiederholungen.

Zwei handkorrigierte Passagen haben auf diese Frage entgegengesetzt geantwortet:
auf der einen half es, das Restfragment zu beseitigen, auf der anderen schadete
es. Bei Effektgrössen um zwei CER-Punkte und Bootstrap-Intervallen von ±1,8 bis
±2,5 Punkten ist das kein Widerspruch, sondern Rauschen -- die Passagen sind zu
kurz, um es zu entscheiden.

Also viele lange Ströme statt zweier Passagen: FLEURS-Aufnahmen werden zu
Strömen von rund `SEC` Sekunden aneinandergehängt, deren Transkript bekannt ist.
Jeder Strom wird in zwei Zuständen gemessen, und weil das Paar dieselbe Sprache
enthält, ist der Vergleich gepaart:

  voll       Länge auf ein Vielfaches von 30 s gebracht -> letztes Fenster voll
  fragment   `FRAG` Sekunden darüber -> letztes Fenster enthält nur ein Fragment

Gefüllt wird jeweils vorne mit Stille, damit der gesprochene Inhalt in beiden
Zuständen identisch bleibt und dieselbe Referenz gilt.

    python docs/skripte/fleurs_stream.py <n_stroeme> <build> [sekunden] [fragment_s]
"""
import json
import pathlib
import random
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, wer

SR = 16000
WINDOW = 30 * SR
GAP = int(0.4 * SR)
RESAMPLES = 10000


def streams(clips, refs, n_streams, sec):
    """Aneinandergehängte Aufnahmen von je rund `sec` Sekunden, mit Referenz."""
    out, i = [], 0
    want = int(sec * SR)
    while len(out) < n_streams and i < len(clips):
        parts, words, total = [], [], 0
        while total < want and i < len(clips):
            parts += [clips[i], np.zeros(GAP, np.float32)]
            words += norm(refs[i])
            total += len(clips[i]) + GAP
            i += 1
        if total >= want * 0.8 and words:
            out.append((np.concatenate(parts).astype(np.float32), words))
    return out


def pad_front(x, target_mod):
    """Vorne mit Stille auffüllen, bis die Länge `target_mod` Samples über einem
    Vielfachen von 30 s liegt (0 = letztes Fenster genau voll)."""
    n = len(x)
    need = (target_mod - n) % WINDOW
    return np.concatenate([np.zeros(need, np.float32), x]).astype(np.float32)


def paired(a, b, num, den):
    rng = random.Random(20260728)
    n = len(a)
    diffs = []
    for _ in range(RESAMPLES):
        s = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i][num] for i in s) / max(1, sum(a[i][den] for i in s))
        db = sum(b[i][num] for i in s) / max(1, sum(b[i][den] for i in s))
        diffs.append((db - da) * 100)
    diffs.sort()
    return diffs[int(0.025 * RESAMPLES)], diffs[int(0.975 * RESAMPLES)]


def main():
    n_streams, build = int(sys.argv[1]), sys.argv[2]
    sec = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    frag = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

    import fleurs_gain
    fleurs_gain.N = int(n_streams * sec / 11) + 40   # ~11 s je Aufnahme
    clips, refs = fleurs_gain.load_clips()
    st = streams(clips, refs, n_streams, sec)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} Stroeme aus FLEURS de_de, je ~{sec:.0f}s, {total/60:.1f} min gesamt")
    print(f"# Build: {build}\n")
    print(f"{'Zustand':12s} {'Fragment':>9s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    import mlx.core as mx
    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(build)

    store = {}
    for name, mod in (("voll", 0), ("fragment", int(frag * SR))):
        rows = []
        t0 = time.time()
        for x, words in st:
            a = np.ascontiguousarray(pad_front(x, mod), dtype=np.float32)
            hyp = norm(vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        W = sum(r["w"] for r in rows) / max(1, sum(r["ref_w"] for r in rows)) * 100
        C = sum(r["c"] for r in rows) / max(1, sum(r["ref_c"] for r in rows)) * 100
        print(f"{name:12s} {mod/SR:8.1f}s {W:6.2f}% {C:6.2f}% {total/el:6.2f}x", flush=True)
        store[name] = rows
        json.dump(store, open("/tmp/fleurs_stream.json", "w"))
        mx.clear_cache()

    lo_w, hi_w = paired(store["voll"], store["fragment"], "w", "ref_w")
    lo_c, hi_c = paired(store["voll"], store["fragment"], "c", "ref_c")
    dW = (sum(r["w"] for r in store["fragment"]) / sum(r["ref_w"] for r in store["fragment"])
          - sum(r["w"] for r in store["voll"]) / sum(r["ref_w"] for r in store["voll"])) * 100
    dC = (sum(r["c"] for r in store["fragment"]) / sum(r["ref_c"] for r in store["fragment"])
          - sum(r["c"] for r in store["voll"]) / sum(r["ref_c"] for r in store["voll"])) * 100
    print("\nAufschlag des Fragments gegenueber dem vollen Fenster, gepaart ueber Stroeme:")
    print(f"  dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]"
          f"{'' if lo_w <= 0 <= hi_w else '  *'}   "
          f"dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]"
          f"{'' if lo_c <= 0 <= hi_c else '  *'}")
    print("\n* = Intervall schliesst 0 aus")


if __name__ == "__main__":
    main()
