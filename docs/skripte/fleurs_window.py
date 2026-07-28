"""Spielt es eine Rolle, WO im 30-s-Fenster die Sprache liegt?

Der Feature-Extractor zerlegt jede Eingabe in 30-Sekunden-Fenster und füllt das
letzte mit Nullen auf (`mlx_voxtral/audio_processing.py`, `process_audio_chunk`).
Unser Chunker schneidet an Sprechpausen, nicht am 30-s-Raster -- eine Äußerung
kann also mitten auf einer Fenstergrenze liegen und wird dann von zwei Encoder-
Fenstern gesehen, jedes mit eigenem `log_max`-Boden.

Wenn die Position im Raster etwas ausmacht, wäre das ein kostenloser Gewinn: der
Chunker müsste seine Schnitte nur zusätzlich am 30-s-Raster ausrichten. Wenn
nicht, ist die Frage erledigt.

Gemessen wird, indem jeder FLEURS-Aufnahme k Sekunden Stille vorangestellt
werden. Bei rund 12 s Länge liegt sie damit einmal am Fensteranfang, einmal in
der Mitte, einmal über der Grenze.

    python docs/skripte/fleurs_window.py <n> <build> <offset_s> [offset_s ...]
    python docs/skripte/fleurs_window.py 180 models/voxtral-mini-8bit 0 6 14 22 27
"""
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, wer
import fleurs_gain
from fleurs_gain import load_clips, paired_bootstrap

SR = 16000


def main():
    n, build = int(sys.argv[1]), sys.argv[2]
    offsets = sys.argv[3:] or ["0"]
    fleurs_gain.N = n
    clips, refs = load_clips()
    lens = np.array([len(c) / SR for c in clips])
    sec = lens.sum()
    print(f"# FLEURS de_de test, {len(clips)} Aufnahmen, {sec/60:.1f} min")
    print(f"# Laenge der Aufnahmen: Median {np.median(lens):.1f}s, "
          f"p5 {np.percentile(lens,5):.1f}s, p95 {np.percentile(lens,95):.1f}s")
    print(f"# Build: {build}\n")
    print(f"{'Offset':>8s} {'ueber Grenze':>13s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    import mlx.core as mx
    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(build)

    store = {}
    for off in offsets:
        pad = int(float(off) * SR)
        rows = []
        straddle = 0
        t0 = time.time()
        for clip, ref_text in zip(clips, refs):
            a = np.concatenate([np.zeros(pad, np.float32), clip]).astype(np.float32)
            if pad and (pad // (30 * SR)) != ((pad + len(clip)) // (30 * SR)):
                straddle += 1
            hyp = vox.transcribe_array(a, "de",
                                       max_new_tokens=int(len(a) / SR * 20) + 512)
            r, h = norm(ref_text), norm(hyp)
            rc, hc = "".join(r), "".join(h)
            rows.append(dict(w=wer(r, h)[0], ref_w=len(r),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        W = sum(r["w"] for r in rows) / max(1, sum(r["ref_w"] for r in rows)) * 100
        C = sum(r["c"] for r in rows) / max(1, sum(r["ref_c"] for r in rows)) * 100
        print(f"{off:>7s}s {straddle:12d} {W:6.2f}% {C:6.2f}% {sec/el:6.2f}x", flush=True)
        store[off] = rows
        json.dump(store, open("/tmp/fleurs_window.json", "w"))
        mx.clear_cache()

    if len(offsets) > 1:
        print("\nGegen Offset 0, gepaart ueber Aufnahmen "
              "(95%-Intervall; enthaelt es 0, ist der Unterschied nicht nachweisbar):")
        base = store[offsets[0]]
        for off in offsets[1:]:
            b = store[off]
            lo_w, hi_w = paired_bootstrap(base, b, "w")
            lo_c, hi_c = paired_bootstrap(base, b, "c")
            dW = (sum(x['w'] for x in base) - sum(x['w'] for x in b)) / sum(x['ref_w'] for x in base) * 100
            dC = (sum(x['c'] for x in base) - sum(x['c'] for x in b)) / sum(x['ref_c'] for x in base) * 100
            print(f"  0s - {off:>3s}s   dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]"
                  f"{'' if lo_w <= 0 <= hi_w else '  *'}   "
                  f"dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]"
                  f"{'' if lo_c <= 0 <= hi_c else '  *'}")
        print("\n* = Intervall schliesst 0 aus")


if __name__ == "__main__":
    main()
