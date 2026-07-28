"""Stellen auflisten, an denen zwei Arme verschieden hören -- mit Zeitmarke.

Wozu: eine Referenz, die durch Korrektur EINES Transkripts entstanden ist, ist an
dieses Transkript angelehnt. Fehler, die beim Korrigieren übersehen wurden,
stehen danach als Wahrheit in der Referenz -- und bevorzugen genau den Arm, aus
dem der Entwurf stammte. Jeder andere Arm wird für dieselbe Stelle bestraft,
auch wenn er sie richtig hört.

Das lässt sich nicht wegrechnen, aber billig entscheiden: es sind nur eine
Handvoll Stellen, an denen die Arme überhaupt auseinandergehen. Dieses Skript
listet sie mit ungefährer Zeitmarke und dem, was die Referenz an der Stelle
sagt, sodass gezielt nachgehört werden kann statt die ganze Passage erneut.

Die Zeitmarke ist geschätzt: Wortposition mal mittlere Wortdauer. Sie zeigt die
Gegend, nicht die Sekunde.

    python docs/skripte/adjudicate.py <referenz.txt> <sekunden> <arm_a> <arm_b> [arm ...]
"""
import difflib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, OVERLAP

CONTEXT = 4


def load(label):
    return norm(open(f"/tmp/wer_{label}.txt", encoding="utf-8").read())


def stamp(pos, total, dur):
    s = int(pos / max(1, total) * dur)
    return f"{s//60:02d}:{s%60:02d}"


def main():
    ref_path, dur = sys.argv[1], float(sys.argv[2])
    a_label, others = sys.argv[3], sys.argv[4:]
    raw = open(ref_path, encoding="utf-8").read()
    ref = norm(OVERLAP.sub(" ", raw))
    a = load(a_label)

    for b_label in others:
        b = load(b_label)
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
        print(f"\n{'='*74}\n{a_label}  gegen  {b_label}: {len(ops)} Stellen\n")
        for tag, i1, i2, j1, j2 in ops:
            # Wo steht die Stelle in der Referenz? Über den gemeinsamen Kontext
            # davor gesucht, damit die Referenzworte danebenstehen können.
            pre = a[max(0, i1 - CONTEXT):i1]
            k = -1
            for s in range(len(ref) - len(pre) + 1):
                if pre and ref[s:s + len(pre)] == pre:
                    k = s + len(pre)
                    break
            ref_span = " ".join(ref[k:k + max(1, i2 - i1)]) if k >= 0 else "?"
            print(f"  ~{stamp(i1, len(a), dur)}  …{' '.join(pre)} ▸")
            print(f"      {a_label:14s} {' '.join(a[i1:i2]) or '(nichts)'!r}")
            print(f"      {b_label:14s} {' '.join(b[j1:j2]) or '(nichts)'!r}")
            print(f"      {'REFERENZ':14s} {ref_span!r}")
            print(f"      danach: {' '.join(a[i2:i2+CONTEXT])}")
    print("\nWer recht hat, entscheidet das Ohr. Wo die Referenz dem ersten Arm")
    print("folgt, obwohl der zweite richtig liegt, ist die Referenz an den ersten")
    print("angelehnt -- und der Vergleich zu seinen Gunsten verzerrt.")


if __name__ == "__main__":
    main()
