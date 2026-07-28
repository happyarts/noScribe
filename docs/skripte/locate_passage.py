"""Finde eine bekannte Passage in einer anderen Fassung derselben Aufnahme.

Liegt von einem Mitschnitt eine rohe und eine bearbeitete Fassung vor, zeigt eine
handkorrigierte Referenz nur auf eine davon. Die andere lässt sich nicht
ausrechnen: ein Leveller wie Auphonic schneidet Stillen, die Zeitachse ist also
nicht linear verschoben, sondern gestaucht.

Gesucht wird deshalb über die Lautstärke-Hüllkurve, und Anfang und Ende getrennt,
damit eine Stauchung innerhalb der Passage sichtbar wird. Die Hüllkurve überlebt
Levelling gut genug: das Muster von Sprache und Pause bleibt, auch wenn die
Pegel angeglichen werden.

Ausgegeben wird neben der Fundstelle die Korrelation am Optimum und die
zweitbeste Korrelation. Liegen die nah beieinander, ist der Fund nicht
vertrauenswürdig.

    python docs/skripte/locate_passage.py <passage.wav> <datei> [datei ...]
"""
import pathlib
import sys

import numpy as np
import soundfile as sf
import soxr

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from audio_audit import decode_float

SR = 16000
HOP = 800          # 50 ms Hüllkurve
EDGE_SEC = 20      # so viel vom Anfang bzw. Ende wird zum Suchen benutzt


def envelope(x):
    e = np.abs(x[:len(x) // HOP * HOP]).reshape(-1, HOP).mean(axis=1)
    return (e - e.mean()) / (e.std() + 1e-9)


def find(hay_env, needle):
    n = envelope(needle)
    c = np.correlate(hay_env, n, "valid") / len(n)
    k = int(np.argmax(c))
    # zweitbester Wert ausserhalb der Umgebung des Treffers
    mask = np.ones(len(c), bool)
    mask[max(0, k - len(n)):k + len(n)] = False
    second = float(c[mask].max()) if mask.any() else 0.0
    return k * HOP / SR, float(c[k]), second


def main():
    passage, _ = sf.read(sys.argv[1], dtype="float32")
    print(f"# Passage {len(passage)/SR:.1f}s aus {sys.argv[1]}")
    print(f"{'Datei':44s} {'Anfang':>9s} {'Ende':>9s} {'Laenge':>8s} "
          f"{'Korr':>6s} {'2.':>6s}")
    for path in sys.argv[2:]:
        x, sr = decode_float(path)
        hay = envelope(soxr.resample(x, sr, SR, quality="HQ").astype(np.float32))
        t_a, c_a, s_a = find(hay, passage[:SR * EDGE_SEC])
        t_b, c_b, s_b = find(hay, passage[-SR * EDGE_SEC:])
        end = t_b + EDGE_SEC
        print(f"{pathlib.Path(path).name[:44]:44s} {t_a:8.2f}s {end:8.2f}s "
              f"{end-t_a:7.2f}s {min(c_a,c_b):6.3f} {max(s_a,s_b):6.3f}")


if __name__ == "__main__":
    main()
