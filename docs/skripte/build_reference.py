"""Bereite eine Passage zur Handkorrektur vor.

Erzeugt drei Dateien nach der Konvention von Audiotest2/referenz/:

  <name>.wav           die Passage, durch den echten Produktionspfad
                       (noScribe/audio/convert.py), also exakt das, was die
                       Engine später zu hören bekommt
  <name>_ENTWURF.txt   Rohtranskript des aktuellen Builds, eine Zeile, als
                       Ausgangspunkt fuer die Korrektur
  <name>_PEGEL.txt     Sekundenpegel und die daraus abgeleiteten leisen
                       Strecken -- damit beim Korrigieren klar ist, wo genau
                       hinzuhören sich lohnt, und damit später getrennt
                       ausgewertet werden kann

Was das Skript NICHT kann: korrigieren. Das braucht Ohren. Der Entwurf ist der
Anfang der Arbeit, nicht das Ergebnis.

Konvention der Referenz (siehe docs/skripte/wer.py):
  //text//   von der zweiten Stimme gleichzeitig gesprochen; ein Transkript,
             das es weglässt, ist nicht falsch und wird separat gezählt

    python docs/skripte/build_reference.py <quelle> <von_s> <bis_s> <ziel-basis> [build]
"""
import pathlib
import sys

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from find_passage import production_wav, SR

QUIET_REL_DB = 10.0     # so viel unter dem lauten Pegel gilt als "leise Strecke"
MIN_QUIET_SEC = 2       # kürzere Einbrüche sind Atempausen, keine Passagen


def level_profile(seg):
    n = len(seg) // SR
    r = seg[:n * SR].reshape(n, SR).astype(np.float64)
    return 20 * np.log10(np.sqrt((r ** 2).mean(axis=1)) + 1e-12)


def quiet_runs(db, floor, loud):
    """Zusammenhängende Sekunden, die Sprache sind, aber deutlich leiser."""
    speech = db > floor + 12
    quiet = speech & (db < loud - QUIET_REL_DB)
    runs, start = [], None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start >= MIN_QUIET_SEC:
                runs.append((start, i))
            start = None
    return runs


def main():
    src, t0, t1, base = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    build = sys.argv[5] if len(sys.argv) > 5 else "models/voxtral-mini-8bit"

    wav = production_wav(src)
    x, sr = sf.read(wav, dtype="float32")
    assert sr == SR, sr
    seg = np.ascontiguousarray(x[t0 * SR:t1 * SR])

    out = pathlib.Path(base)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out.with_suffix(".wav"), seg, SR, subtype="PCM_16")

    db = level_profile(seg)
    # Das Grundrauschen muss aus der GANZEN Aufnahme kommen, nicht aus der
    # Passage: eine sprachdichte Passage hat kaum Stille, ihr eigenes 10.
    # Perzentil liegt deshalb mitten in der leisen Sprache. Mit dem
    # Passagen-Perzentil als Boden fällt genau das aus der Sprachmaske heraus,
    # was hier interessiert -- die Spreizung sähe dann viel kleiner aus, als
    # sie ist.
    floor = float(np.percentile(level_profile(x), 10))
    speech = db > floor + 12
    loud = float(np.percentile(db[speech], 90)) if speech.any() else floor
    runs = quiet_runs(db, floor, loud)
    spread = (loud - float(np.percentile(db[speech], 10))) if speech.any() else 0.0

    from noScribe.voxtral_engine import _Voxtral
    vox = _Voxtral(build)
    text = vox.transcribe_array(seg, "de",
                                max_new_tokens=int((t1 - t0) * 20) + 512).strip()
    open(f"{base}_ENTWURF.txt", "w", encoding="utf-8").write(text)

    with open(f"{base}_PEGEL.txt", "w", encoding="utf-8") as f:
        f.write(f"# {pathlib.Path(src).name} {t0}-{t1}s, {t1-t0}s, "
                f"Entwurf {len(text.split())} Woerter, Build {build}\n")
        f.write(f"# Grundrauschen {floor:.1f} dB, lauter Sprachpegel (p90) {loud:.1f} dB, "
                f"Spreizung p90-p10 {spread:.1f} dB, Sprachanteil {speech.mean()*100:.0f}%\n")
        f.write(f"# Leise Strecken: Sprache mehr als {QUIET_REL_DB:.0f} dB unter p90, "
                f"mindestens {MIN_QUIET_SEC} s am Stueck. "
                f"Sekunden relativ zum Passagenanfang.\n#\n")
        f.write(f"# {len(runs)} leise Strecken, zusammen "
                f"{sum(b-a for a, b in runs)} s "
                f"({sum(b-a for a, b in runs)/max(1,len(db))*100:.0f}% der Passage):\n")
        for a, b in runs:
            f.write(f"#   {a//60:02d}:{a%60:02d} - {b//60:02d}:{b%60:02d}  "
                    f"({b-a:3d}s, {db[a:b].mean():.1f} dB, "
                    f"{loud-db[a:b].mean():.1f} dB unter laut)\n")
        f.write("#\n# Sekunde\tdB\tSprache\tleise\n")
        for i, v in enumerate(db):
            q = any(a <= i < b for a, b in runs)
            f.write(f"{i}\t{v:.1f}\t{int(speech[i])}\t{int(q)}\n")

    print(f"{out.with_suffix('.wav')}  {t1-t0}s")
    print(f"{base}_ENTWURF.txt  {len(text.split())} Woerter")
    print(f"{base}_PEGEL.txt  Spreizung {spread:.1f} dB, "
          f"{len(runs)} leise Strecken / {sum(b-a for a, b in runs)}s")


if __name__ == "__main__":
    main()
