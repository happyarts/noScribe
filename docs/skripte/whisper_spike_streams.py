"""Trifft der Transienten-Defekt auch die Standard-Engine? Auf vielen Strömen.

`faster_whisper/feature_extractor.py:227` trägt dieselbe Zeile wie der
Voxtral-Pfad -- `np.maximum(log_spec, log_spec.max() - 8.0)` über die ganze
übergebene Wellenform -- und `noScribe/whisper_mp_worker.py` übergibt Whisper die
ganze Datei. Wenn der Mechanismus dort greift, beträfe er jeden Nutzer der
Standardeinstellung, also weit mehr als die Voxtral-Engine.

Ein erster Versuch auf **einer** Passage blieb ergebnislos, und zwar aus einem
nachvollziehbaren Grund: Whispers eigene Streuung war größer als der gesuchte
Effekt (auf `hart_780-900` 9,00 % WER mit VAD gegen 21,09 % ohne). Der Ausweg ist
derselbe wie beim Chunker und beim Mel: viele Ströme statt einer Passage, gepaart
auf bitgleichem Audio, mit Bootstrap-Intervall.

Whisper hat hier einen zweiten Hebel, den Voxtral nicht hat: der VAD-Filter kann
den Transienten entfernen, bevor er das Mel-Maximum setzt. Deshalb laufen beide
Zustände -- ist `vad_filter=True` bereits der Schutz, ist nichts zu tun.

    python docs/skripte/whisper_spike_streams.py [n_stroeme] [sekunden] [dB] [modell]
"""
import pathlib
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, wer                                    # noqa: E402
from fleurs_stream import streams, paired, SR                # noqa: E402
from voxpopuli_floor import load_clips                       # noqa: E402
from voxpopuli_spike import pair, floor_rise_db              # noqa: E402


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    db_over = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    model_name = sys.argv[4] if len(sys.argv) > 4 else "precise"

    from faster_whisper import WhisperModel
    clips, refs = load_clips(int(n_streams * sec / 9) + 60)
    st = streams(clips, refs, n_streams, sec)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} Ströme aus VoxPopuli de (gold), je ~{sec:.0f}s, "
          f"{total/60:.1f} min gesamt")
    print(f"# Transient: {db_over:+.0f} dB über der Sprachspitze, 100 ms, bei 45 %")
    print(f"# Modell: models/{model_name}, CPU int8, beam 5\n")

    rises = [floor_rise_db(*pair(x, db_over)) for x, _ in st]
    print(f"# Kontrolle: Bodenanstieg median {np.median(rises):.1f} dB "
          f"(min {min(rises):.1f}, max {max(rises):.1f})\n")
    assert np.median(rises) > 0.7 * db_over, "Dosis kommt nicht an"

    model = WhisperModel(f"models/{model_name}", device="cpu", compute_type="int8")

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    def score(spike, vad):
        rows, t0 = [], time.time()
        for x, words in st:
            a = pair(x, db_over)[1 if spike else 0]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                sf.write(fh.name, a, SR)
                path = fh.name
            segs, _ = model.transcribe(path, language="de", beam_size=5,
                                       vad_filter=vad)
            hyp = norm(" ".join(s.text for s in segs))
            pathlib.Path(path).unlink(missing_ok=True)
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        return rows, time.time() - t0

    print(f"{'VAD':>6s} {'Audio':10s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")
    store = {}
    for vad in (True, False):
        for au, spike in (("sauber", False), ("mit Knall", True)):
            rows, el = score(spike, vad)
            store[(vad, au)] = rows
            print(f"{str(vad):>6s} {au:10s} {rate(rows,'w','ref_w'):6.2f}% "
                  f"{rate(rows,'c','ref_c'):6.2f}% {total/el:6.2f}x", flush=True)

    print("\nKosten des Transienten, gepaart je VAD-Zustand "
          "(95 %; enthält es 0, nicht nachweisbar):")
    for vad in (True, False):
        a, b = store[(vad, "sauber")], store[(vad, "mit Knall")]
        lo_w, hi_w = paired(a, b, "w", "ref_w")
        lo_c, hi_c = paired(a, b, "c", "ref_c")
        dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
        dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
        star = " *" if (lo_w > 0) == (hi_w > 0) else ""
        print(f"  vad_filter={str(vad):5s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star}"
              f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]")
    print("  (positiv = der Knall schadet)")


if __name__ == "__main__":
    main()
