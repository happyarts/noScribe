"""Prüft, ob der lokale Testzweig alle Änderungen der offenen PR-Branches trägt.

Hintergrund: main ist die Integrationslinie, auf der getestet wird. Die
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

Ein Branch, der ganz in einem anderen steckt (gestapelt), wird über den oberen
geprüft; siehe stacked_into().

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

# Bewusste Abweichungen: main trägt Funktionen, die es upstream nicht
# gibt, deshalb kann eine Branch-Zeile hier zu Recht anders lauten. Jede
# Ausnahme braucht eine Begründung, sonst verdeckt sie echte Drift.
INTENTIONAL = {
    # feature/voice-verified-speakers fuegt voice_check an; lokal steht ctc_align
    # (Voxtral) mit in derselben Zeile, und der Voxtral-Branch ohne voice_check.
    '_SUBMODULES = ("main", "audio", "exception", "transcription", "utils", "voice_check")':
        "lokal um ctc_align erweitert",
    '_SUBMODULES = ("main", "audio", "ctc_align", "exception", "transcription", "utils")':
        "lokal zusaetzlich mit voice_check",
    # Jeder Branch traegt seinen Worker in WORKER_MODULES ein; lokal stehen alle.
    '"noScribe.voxtral_mp_worker"]':
        "lokal folgt nemotron_mp_worker in derselben Zeile",
    # Der Voxtral-Branch traegt die lokalen .gitignore-Eintraege nicht.
    '# with tools/quantize_voxtral.py).':
        "lokal geht der Kommentar mit Audiotest/ und venv/ weiter",
    # Das Test-Harness der Stimmpruefung: lokal laeuft Silero neben dem Worker
    # (fix/silero-beside-worker), das Harness reicht deshalb vad_future.
    "'speech_chunks': [], 'traceback': traceback, 'get_config': lambda key, default=None: default,":
        "lokal mit vad_future statt speech_chunks",
    # Seit dem Umbau auf die Plugins (2026-10-30): was der Voxtral- und der
    # Nemotron-Branch je fuer ihr eigenes Plugin sagen, sagt main fuer beide.
    '# The engine behind the speakers, for the header and the log':
        "lokal: die Engines hinter Sprecher:innen und Text",
    '# ("loading_nemotron").':
        "lokal mit loading_voxtral in derselben Zeile",
    '# The engine behind the text, for the log ("loading_voxtral").':
        "lokal: die Engines hinter Sprecher:innen und Text",
    '"""The engine a model runs on (\'whisper\', \'pyannote\', \'nemotron\')."""':
        "lokal mit allen vier Engines",
    '"""The engine a model runs on (\'whisper\', \'voxtral\')."""':
        "lokal mit allen vier Engines",
    "nemotron_defaults stand in for absent fields of Nemotron's settings file, as":
        "lokal ein Docstring fuer beide *_defaults",
    "voxtral_defaults stand in for absent fields of Voxtral's settings file, as":
        "lokal ein Docstring fuer beide *_defaults",
    "the desktop's older config.yml entries do for Whisper and Pyannote above.\"\"\"":
        "lokal geht der Docstring mit dem Korrekturen-Verzeichnis weiter",
    '└── local_voxtral/':
        "lokal folgt local_nemotron im Verzeichnisbaum der Plugin-README",
    # Lokal gelten die aelteren voxtral_*-Schluessel in config.yml weiter
    # (fork-only, tests/test_fork_migrations.py); VOXTRAL.md sagt es.
    'instead — that is the knob the automatic length is computed against. Raising':
        "lokal mit dem Satz zu den aelteren config.yml-Schluesseln",
    'it past 10 minutes is refused — see the cap above.':
        "lokal mit dem Satz zu den aelteren config.yml-Schluesseln",
}



def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def branches():
    out = sh("git", "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin")
    # origin/main ist die Integrationslinie, kein PR-Branch.
    return [b for b in out.split()
            if b not in ("origin", "origin/main", "origin/HEAD")]


def stacked_into(branch, others):
    """Der Branch, auf dem `branch` aufsetzt und der ihn ganz enthält, oder None.

    Ein gestapelter Branch (feature/nemotron-diarization auf
    feature/voice-verified-speakers) darf Zeilen des unteren bewusst überholen.
    Der obere wird ohnehin geprüft, und sein Diff gegen upstream/main enthält
    jede Zeile des unteren, die dort noch gilt -- den unteren eigens zu prüfen
    meldete nur die überholten."""
    tip = sh("git", "rev-parse", branch).strip()
    for other in others:
        if other != branch and sh("git", "rev-parse", other).strip() != tip and subprocess.run(
                ["git", "merge-base", "--is-ancestor", branch, other]).returncode == 0:
            return other
    return None


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
    all_branches = sorted(branches())
    for branch in all_branches:
        upper = stacked_into(branch, all_branches)
        if upper:
            print(f"   {branch:44} steckt ganz in {upper}, dort geprüft")
            continue
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
