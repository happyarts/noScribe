"""Kostet ein robuster Clamp-Boden auf sauberem Material etwas? Auf VoxPopuli.

Zwei Fragen in einem Lauf.

**Der Korpus.** FLEURS ist vorgelesen, sauber, ein Sprecher -- die Luecke, die
`docs/voxtral-audio-vorverarbeitung.md` an mehreren Stellen einraeumt.
`facebook/voxpopuli` de ist **spontane** Parlamentsrede mit Goldtranskript und
damit das naechste oeffentliche Material an unserem Anwendungsfall. Nur Zeilen
mit `is_gold_transcript` werden genommen.

**Die Frage.** Der Clamp-Boden liegt bei `log_max - 8`, und `log_max` ist das
Maximum ueber die ganze Eingabe. Ein einzelner lauter Transient hebt ihn damit
fuer den gesamten Pass: gemessen hebt ein 0,1-s-Knall den Boden um 9,6 dB und
kostet auf `hart_780-900` 1,66 WER-Punkte, waehrend ein Perzentil-Boden ihn
komplett ignoriert (dWER +0,00, Text bitgleich mit und ohne Knall). Ob der
Perzentil-Boden dafuer auf *sauberem* Material etwas kostet, entscheidet, ob er
ueberhaupt in Frage kommt -- und genau das misst dieser Lauf, gepaart auf
bitgleichem Audio.

Aufbau und Bootstrap aus fleurs_stream.py; verglichen werden zwei
Feature-Pfade auf denselben Stroemen.

    python docs/skripte/voxpopuli_floor.py [n_stroeme] [sekunden] [perzentile]

`perzentile` darf eine Liste sein (`99.0,99.9,99.99`). Die Max-Basis wird dann
nur einmal gerechnet und alle Arme gegen dieselbe Basis gepaart -- das ist der
Sweep, mit dem die Perzentilwahl entschieden wird.
"""
import io
import pathlib
import sys
import time

import numpy as np
import soundfile as sf
import mlx.core as mx

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, wer                                    # noqa: E402
from fleurs_stream import streams, paired, SR                # noqa: E402
from noScribe.voxtral_engine import _Voxtral                 # noqa: E402
from mlx_voxtral.audio_processing import (                   # noqa: E402
    log_mel_spectrogram, N_MELS, N_FRAMES)


class PercentileFloor:
    """Wie der eingebaute Pfad, aber der Boden kommt aus einem Perzentil statt
    aus dem Maximum. Das Maximum wird ausschliesslich fuer den Boden benutzt
    (danach eine feste Affinitaet), also ist das die einzige Aenderung.

    Der `global_max`-Parameter der Bibliothek ist der Hebel: ein sehr negativer
    Wert klemmt nichts, liefert also das rohe log-Mel (die Affinitaet ist
    invertierbar), daraus das Perzentil, dann der regulaere Aufruf. Kontrolle:
    mit Perzentil 100 ist die Ausgabe bitgleich zum eingebauten Pfad.
    """

    def __init__(self, pct):
        self.pct = pct

    def __call__(self, a, sampling_rate=SR, return_tensors="mlx", **kw):
        a = np.asarray(a, dtype=np.float32)
        cs = 30 * SR
        n = int(np.ceil(len(a) / cs))
        a = np.pad(a, (0, n * cs - len(a)))
        arr = mx.array(a)
        raw = np.array(log_mel_spectrogram(arr, global_max=-1e6)) * 4.0 - 4.0
        mel = log_mel_spectrogram(arr, global_max=float(np.percentile(raw, self.pct)))
        return {"input_features":
                mel.reshape(N_MELS, n, N_FRAMES).transpose(1, 0, 2)}


def load_clips(n_rows):
    """Goldtranskribierte VoxPopuli-de-Zeilen als (Audio, Referenztext)."""
    from datasets import load_dataset
    ds = load_dataset("facebook/voxpopuli", "de", split="test", streaming=True)
    clips, refs = [], []
    for row in ds:
        if not row.get("is_gold_transcript"):
            continue
        text = (row.get("raw_text") or "").strip()
        if len(text.split()) < 4:
            continue
        a = row["audio"]
        wav = np.asarray(a["array"] if isinstance(a, dict) else a.get_all_samples().data,
                         dtype=np.float32).squeeze()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        clips.append(np.ascontiguousarray(wav))
        refs.append(text)
        if len(clips) >= n_rows:
            break
    return clips, refs


def main():
    n_streams = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sec = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    pcts = ([float(x) for x in sys.argv[3].split(",")]
            if len(sys.argv) > 3 else [99.9])

    clips, refs = load_clips(int(n_streams * sec / 9) + 60)
    st = streams(clips, refs, n_streams, sec)
    total = sum(len(s[0]) for s in st) / SR
    print(f"# {len(st)} Stroeme aus VoxPopuli de (gold), je ~{sec:.0f}s, "
          f"{total/60:.1f} min gesamt")

    vox = _Voxtral(str(REPO / "models" / "voxtral-mini-8bit"))
    stock = vox.proc.feature_extractor
    arms = [(f"Perzentil {p:g}", PercentileFloor(p)) for p in pcts]

    # Greift der Tausch? Sonst misst der Lauf zweimal dasselbe.
    probe = st[0][0]
    base_feat = np.array(stock(probe, sampling_rate=SR,
                               return_tensors="mlx")["input_features"])
    for name, ex in arms:
        d = np.abs(np.array(ex(probe)["input_features"]) - base_feat).max()
        print(f"# Kontrolle: Feature max|diff| {name} gegen max = {d:.4f}")
        assert d > 1e-3, f"{name} unterscheidet sich kaum vom eingebauten Pfad"

    def rate(rows, num, den):
        return sum(r[num] for r in rows) / max(1, sum(r[den] for r in rows)) * 100

    print(f"\n{'Boden':16s} {'WER':>7s} {'CER':>7s} {'Speed':>7s}")
    store = {}
    for name, ex in [("max (ist)", stock)] + arms:
        vox.proc.feature_extractor = ex
        rows, t0 = [], time.time()
        for x, words in st:
            a = np.ascontiguousarray(x, dtype=np.float32)
            hyp = norm(vox.transcribe_array(
                a, "de", max_new_tokens=int(len(a) / SR * 20) + 512))
            rc, hc = "".join(words), "".join(hyp)
            rows.append(dict(w=wer(words, hyp)[0], ref_w=len(words),
                             c=wer(rc, hc)[0], ref_c=len(rc)))
        el = time.time() - t0
        print(f"{name:16s} {rate(rows,'w','ref_w'):6.2f}% {rate(rows,'c','ref_c'):6.2f}% "
              f"{total/el:6.2f}x", flush=True)
        store[name] = rows
        mx.clear_cache()

    a = store["max (ist)"]
    star = lambda lo, hi: " *" if (lo > 0) == (hi > 0) else ""
    print("\nGepaart gegen max (95%; enthaelt es 0, nicht nachweisbar):")
    for name, _ in arms:
        b = store[name]
        lo_w, hi_w = paired(a, b, "w", "ref_w")
        lo_c, hi_c = paired(a, b, "c", "ref_c")
        dW = rate(b, "w", "ref_w") - rate(a, "w", "ref_w")
        dC = rate(b, "c", "ref_c") - rate(a, "c", "ref_c")
        print(f"  {name:16s} dWER {dW:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}]{star(lo_w,hi_w)}"
              f"   dCER {dC:+.2f} [{lo_c:+.2f}, {hi_c:+.2f}]{star(lo_c,hi_c)}")
    print("  (positiv = Perzentil schlechter)")
    if len(st) < 8:
        print(f"  ACHTUNG: nur {len(st)} Stroeme -- das Intervall traegt nicht.")


if __name__ == "__main__":
    main()
