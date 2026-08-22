"""Kostet die Per-Block-Normalisierung des log-Mel etwas? Auf langen Strömen.

HISTORISCH: seit mlx-voxtral 0.0.6 ist `mel-ref` der eingebaute Pfad; beide
Varianten sind identisch und die Kontroll-Assertion unten schlägt deshalb fehl.
Für eine Wiederholung müsste `stock` die 0.0.5-Fassung von
`_process_audio_array_with_chunking` sein. Ergebnis: §6 des Messdokuments.

mlx-voxtral berechnete das log-Mel bis 0.0.5 je 30-s-Block und normalisierte
jeden Block gegen sein eigenes Maximum. Voxtral spezifiziert ein Spektrogramm
über die ganze Eingabe, das erst danach geteilt wird (arXiv:2507.13264 §2.1) -- der Clamp liegt
bei `log_max - 8`, ein Block mit eigenem Maximum landet also auf einem anderen
Boden.

Zwei handkorrigierte Passagen entscheiden das nicht: auf `hart_780-900` sind
beide Fassungen wortgleich (Blockpegel-Spanne nur 0,12 log10), auf
`zoom_9890-10190` sieht der Unterschied signifikant aus -- aber jene Referenz ist
um 0,25 CER-Punkte zugunsten des Builds verzerrt, aus dessen Entwurf sie
korrigiert wurde. Dieselbe Falle wie beim Chunker (siehe
docs/voxtral-audio-vorverarbeitung.md §5).

Also viele lange Ströme: FLEURS-Aufnahmen zu je rund `SEC` Sekunden
aneinandergehängt, Transkript bekannt. Beide Fassungen sehen **bitgleiches
Audio**, der Vergleich ist also gepaart.

Aufbau und Bootstrap stammen aus fleurs_stream.py; hier werden statt zweier
Fensterzustände zwei Feature-Pfade verglichen.

    python docs/skripte/mel_stream.py [n_stroeme] [build]
"""
import json
import pathlib
import sys
import time

import numpy as np
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, wer
from fleurs_stream import streams, paired, SR
import fleurs_gain
from noScribe.voxtral_engine import _Voxtral
from mlx_voxtral.audio_processing import log_mel_spectrogram, N_SAMPLES, N_MELS, N_FRAMES

SEC = 300.0


class RefExtractor:
    """Ein Spektrogramm über das ganze gepaddete Audio, danach geteilt."""

    def __call__(self, audio, sampling_rate=SR, return_tensors="mlx", **kw):
        a = np.asarray(audio, dtype=np.float32)
        pad = (-len(a)) % N_SAMPLES
        if pad:
            a = np.pad(a, (0, pad))
        mel, _ = log_mel_spectrogram(mx.array(a))
        return {"input_features":
                mel.reshape(N_MELS, len(a) // N_SAMPLES, N_FRAMES).transpose(1, 0, 2)}


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    build = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "models" / "voxtral-mini-8bit")

    fleurs_gain.N = int(n_streams * SEC / 11) + 40   # ~11 s je Aufnahme
    st = streams(*fleurs_gain.load_clips(), n_streams, SEC)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} Stroeme aus FLEURS de_de, je ~{SEC:.0f}s, {total/60:.1f} min gesamt")
    print(f"# Build: {pathlib.Path(build).name}, Sprache 'de' gepinnt, greedy\n", flush=True)

    vox = _Voxtral(build)
    stock, ref_ex = vox.proc.feature_extractor, RefExtractor()

    # Greift der Tausch überhaupt? Sonst misst der Lauf zweimal dasselbe -- eine
    # Stunde lang, und das Ergebnis sähe aus wie ein sauberer Nulleffekt.
    probe = st[0][0]
    d = np.abs(np.array(ref_ex(probe)["input_features"])
               - np.array(stock(probe, sampling_rate=SR,
                                return_tensors="mlx")["input_features"]))
    assert d.max() > 0.1, f"Feature-Pfade unterscheiden sich kaum: {d.max()}"
    print(f"# Kontrolle: Feature max|diff| auf Strom 0 = {d.max():.4f}\n")
    print(f"{'Variante':10s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    store = {}
    for name, extractor in (("mel-ist", stock), ("mel-ref", ref_ex)):
        vox.proc.feature_extractor = extractor
        rows, t0 = [], time.time()
        for x, words in st:
            a = np.ascontiguousarray(x, dtype=np.float32)
            hyp = norm(vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        print(f"{name:10s} {rate(rows, 'w', 'ref_w'):6.2f}% "
              f"{rate(rows, 'c', 'ref_c'):6.2f}% {total/el:6.2f}x", flush=True)
        store[name] = rows
        json.dump(store, open("/tmp/mel_stream.json", "w"))
        mx.clear_cache()

    lo_w, hi_w = paired(store["mel-ist"], store["mel-ref"], "w", "ref_w")
    lo_c, hi_c = paired(store["mel-ist"], store["mel-ref"], "c", "ref_c")
    dW = rate(store["mel-ref"], "w", "ref_w") - rate(store["mel-ist"], "w", "ref_w")
    dC = rate(store["mel-ref"], "c", "ref_c") - rate(store["mel-ist"], "c", "ref_c")

    def star(lo, hi):
        return " *" if (lo > 0) == (hi > 0) else ""

    print("\nGepaarte Differenz mel-ref - mel-ist (95%; enthaelt es 0, nicht nachweisbar):")
    print(f"  dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star(lo_w, hi_w)}"
          f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]{star(lo_c, hi_c)}")
    print("  (positiv = mel-ref schlechter)")
    if len(st) < 8:
        # Der Bootstrap zieht Ströme mit Zurücklegen: bei einer Handvoll schrumpft
        # das Intervall auf die Streuung weniger Punkte und setzt einen Stern, der
        # nichts bedeutet. Bei einem einzigen Strom ist es sogar breitenlos.
        print(f"  ACHTUNG: nur {len(st)} Strom/Stroeme -- das Intervall traegt nicht. "
              f"Zum Entscheiden 18 nehmen.")


if __name__ == "__main__":
    main()
