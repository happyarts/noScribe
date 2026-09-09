"""Prüft, ob der lokale Testzweig alle Änderungen der offenen PR-Branches trägt.

Hintergrund: local/main ist die Integrationslinie, auf der getestet wird. Die
PR-Branches werden nach Reviews weiterentwickelt, und ein Merge kann eine
Verbesserung stillschweigend wieder verlieren -- eine Konfliktauflösung, die
die falsche Seite nimmt, hinterlässt keine Spur, und `git merge` meldet danach
trotzdem "Already up to date". Merge-Abstammung beweist also nichts über den
Inhalt; nur ein Zeilenvergleich tut das.

Für jeden Branch: alle Zeilen, die er gegenüber upstream/main hinzufügt, müssen
in der entsprechenden Datei des Arbeitsbaums vorkommen. Fehlende Zeilen werden
gemeldet.

    python tools/check_local_current.py            # alle origin-Branches
    python tools/check_local_current.py --verbose  # jede fehlende Zeile

Fehlalarme sind möglich, wenn eine Zeile bewusst überholt wurde -- deshalb
meldet das Skript, statt zu reparieren. Exit-Code 1, wenn etwas fehlt.
"""
import argparse
import re
import subprocess
import sys

BASE = "upstream/main"
SKIP_PREFIX = ("docs/", "benchmarks-local/")
# Zeilen ohne Aussagekraft: Leerzeilen, alleinstehende Klammern, Trennlinien.
NOISE = re.compile(r"^[\s)\](}#*=/-]*$")

# Bewusste Abweichungen: local/main trägt Funktionen, die es upstream nicht
# gibt, deshalb kann eine Branch-Zeile hier zu Recht anders lauten. Jede
# Ausnahme braucht eine Begründung, sonst verdeckt sie echte Drift.
INTENTIONAL = {
    # Der Modell-Picker mit RAM-Angabe existiert nur lokal; ohne model_key()
    # würde der Anzeigetext als gemerktes Modell gespeichert.
    "config['last_whisper_model'] = self.option_menu_whisper_model.get()":
        "lokal durch model_key() ersetzt (dekorierter Picker)",
    # local/main trägt zusätzlich ctc_align (numpy-Viterbi), das es upstream
    # nicht gibt -- die Zeile ist hier also eine echte Obermenge der PR-Zeile.
    '_SUBMODULES = ("main", "audio", "exception", "transcription", "utils")':
        "lokal um ctc_align erweitert",
    # Der Voxtral-Worker ist ein drittes Spawn-Ziel und steht lokal mit in der
    # Liste; die Branch-Zeile ist eine Teilmenge der lokalen.
    'WORKER_MODULES = ["noScribe.pyannote_mp_worker", "noScribe.whisper_mp_worker"]':
        "lokal um voxtral_mp_worker erweitert",
    '# Both are ctx.Process targets, so both are re-imported in a spawn child.':
        "lokal auf drei Worker umformuliert",
    # Der Rueckgabewert der Pump-Funktion war tot und ist lokal entfernt (013a0e7);
    # der Branch traegt die Zeile noch mit der Zuweisung.
    'info = self._run_whisper_subprocess_stream(tmp_audio_file, job, on_segment_split)':
        "lokal ohne die tote info-Zuweisung",
}


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def branches():
    out = sh("git", "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")
    # origin/voxtral ist ein Spiegel von local/main, kein PR-Branch.
    return [b for b in out.split()
            if b not in ("origin/main", "origin/HEAD", "origin/local/main",
                         "origin/voxtral")]


def added_lines(branch):
    """{Datei: [hinzugefügte Zeilen]} für branch gegen upstream/main."""
    diff = sh("git", "diff", f"{BASE}...{branch}")
    out, path = {}, None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("+") and not line.startswith("+++") and path:
            body = line[1:]
            if not NOISE.match(body):
                out.setdefault(path, []).append(body.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    worst = 0
    for branch in sorted(branches()):
        missing = {}
        for path, lines in added_lines(branch).items():
            if path.startswith(SKIP_PREFIX):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    have = fh.read()
            except FileNotFoundError:
                missing[path] = ["(Datei fehlt vollständig)"]
                continue
            gone = [l for l in lines if l not in have and l not in INTENTIONAL]
            if gone:
                missing[path] = gone
        total = sum(len(v) for v in missing.values())
        worst = max(worst, total)
        mark = "OK " if not total else "!! "
        print(f"{mark}{branch:44} {total:4} Zeile(n) fehlen")
        for path, lines in missing.items():
            print(f"      {path}: {len(lines)}")
            if args.verbose:
                for l in lines[:20]:
                    print(f"        {l[:100]}")
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
