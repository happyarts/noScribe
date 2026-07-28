"""Finde in einer langen Aufnahme die Passage, die sich zu korrigieren lohnt.

Eine handkorrigierte Referenz kostet echte Arbeit, also sollte die Passage die
Frage beantworten können, für die sie gebaut wird. Für die Leveller-Frage aus
docs/voxtral-audio-vorverarbeitung.md heisst das: viel Sprache, und darin ein
grosser Abstand zwischen lauten und leisen Stellen. Genau danach wird hier
gesucht -- nicht nach der "schönsten" Stelle.

Bewertet werden Fenster fester Länge nach

  Sprachanteil   Anteil der Sekunden über dem geschätzten Grundrauschen
  Spreizung      p90 - p10 des Sekundenpegels über die Sprachsekunden

Die Spreizung als Perzentil-Abstand, nicht als max-min: ein einzelner Türknall
soll die Auswahl nicht bestimmen.

    python docs/skripte/find_passage.py <datei> [laenge_s] [anzahl]
"""
import pathlib
import sys

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SR = 16000
CACHE = pathlib.Path("/tmp/preproc_cache")


def production_wav(src):
    """Die Datei durch den echten Konverter, einmal, und gecacht."""
    from noScribe.audio.convert import ToWav
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / (pathlib.Path(src).stem.replace(" ", "_") + "_p0.wav")
    if not out.exists():
        with ToWav(pathlib.Path(src), out, force=True) as c:
            while c.convert():
                pass
    return out


def per_second_db(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, sr
    n = len(x) // SR
    r = x[:n * SR].reshape(n, SR)
    return 20 * np.log10(np.sqrt((r.astype(np.float64) ** 2).mean(axis=1)) + 1e-12)


def main():
    src = sys.argv[1]
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    wav = production_wav(src)
    db = per_second_db(wav)
    # Grundrauschen aus dem unteren Zehntel schätzen, Sprache 12 dB darüber.
    floor = np.percentile(db, 10)
    thr = floor + 12
    speech = db > thr
    print(f"# {pathlib.Path(src).name}: {len(db)/60:.1f} min, "
          f"Grundrauschen ~{floor:.1f} dB, Sprachschwelle {thr:.1f} dB, "
          f"Sprachanteil gesamt {speech.mean()*100:.0f}%")

    rows = []
    for t in range(0, len(db) - win, 10):
        seg, m = db[t:t + win], speech[t:t + win]
        if m.mean() < 0.7 or m.sum() < 30:
            continue
        s = seg[m]
        rows.append((float(np.percentile(s, 90) - np.percentile(s, 10)), t,
                     float(m.mean()), float(np.percentile(s, 90)),
                     float(np.percentile(s, 10))))
    rows.sort(reverse=True)

    # Kandidaten mit mindestens einer Fensterlänge Abstand, damit die Liste
    # nicht zehnmal dieselbe Stelle zeigt.
    picked = []
    for r in rows:
        if all(abs(r[1] - p[1]) >= win for p in picked):
            picked.append(r)
        if len(picked) >= topn:
            break

    print(f"\n{'von':>8s} {'bis':>8s} {'Spreizung':>10s} {'Sprache':>8s} "
          f"{'laut p90':>9s} {'leise p10':>10s}")
    for spread, t, frac, hi, lo in picked:
        print(f"{t:8d} {t+win:8d} {spread:9.1f} dB {frac*100:7.0f}% "
              f"{hi:8.1f} dB {lo:9.1f} dB")

    if rows:
        allspread = np.array([r[0] for r in rows])
        print(f"\n# Spreizung über alle {len(rows)} Kandidatenfenster: "
              f"Median {np.median(allspread):.1f} dB, p90 {np.percentile(allspread,90):.1f} dB, "
              f"max {allspread.max():.1f} dB")


if __name__ == "__main__":
    main()
