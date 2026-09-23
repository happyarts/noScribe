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
    # feature/voice-verified-speakers fuegt voice_check an; lokal steht ctc_align
    # mit in derselben Zeile.
    '_SUBMODULES = ("main", "audio", "exception", "transcription", "utils", "voice_check")':
        "lokal um ctc_align erweitert",
    # Der Schreib-Test hebt on_segment aus main.py; lokal liest es is_voxtral,
    # also steht die Variable im Test-Scope und die Zeile endet anders.
    "'speech_chunks': []}":
        "lokal um is_voxtral ergaenzt (on_segment liest es nur hier)",
    # Wo die Nummerierung nach Auftreten (feature/speakers-in-order-of-appearance)
    # und die Stimmpruefung (warn-Parameter) zusammentreffen, lauten drei Zeilen
    # lokal anders: die Warnbedingung traegt beide Zusaetze, und das Bewegungs-
    # protokoll nennt den Sprecher unter dem Namen, unter dem er geschrieben
    # wurde -- deshalb erst nach der Transkription.
    'if names and idx == len(names):':
        "lokal zusaetzlich mit 'and warn' (Stimmpruefung schreibt still neu)",
    'if idx == len(names) and warn:':
        "lokal zusaetzlich mit 'names and' (Nummerierung nach Auftreten)",
    "f\"{before} -> {after}:{passage['text'][:60]}\", where='file')":
        "lokal unter den geschriebenen Namen (shown(...)) und nach dem Neuschreiben",
    # Der Voxtral-Worker ist ein drittes Spawn-Ziel und steht lokal mit in der
    # Liste; die Branch-Zeile ist eine Teilmenge der lokalen.
    'WORKER_MODULES = ["noScribe.pyannote_mp_worker", "noScribe.whisper_mp_worker"]':
        "lokal um voxtral_mp_worker erweitert",
    '# Both are ctx.Process targets, so both are re-imported in a spawn child.':
        "lokal auf drei Worker umformuliert",
    # feature/voxtral-engine gibt das _Info-Objekt wieder zurueck, wie upstream es
    # tut (die Entfernung wurde dort abgelehnt und bleibt lokal); lokal geben die
    # drei Funktionen nichts zurueck.
    'info = self._run_voxtral_subprocess_stream(':
        "lokal ohne Zuweisung (_Info lokal entfernt)",
    'info = self._run_whisper_subprocess_stream(tmp_audio_file, job, on_segment)':
        "lokal ohne Zuweisung (_Info lokal entfernt)",
    'reliably. Returns a simple info object (duration at least).':
        "lokal ohne Rueckgabe (_Info lokal entfernt)",
    'Returns a simple info object (duration at least).':
        "lokal ohne Rueckgabe (_Info lokal entfernt)",
    'return self._run_engine_subprocess_stream(voxtral_proc_entrypoint, args, job, on_segment)':
        "lokal ohne return (_Info lokal entfernt)",
    'return self._run_engine_subprocess_stream(whisper_proc_entrypoint, args, job, on_segment)':
        "lokal ohne return (_Info lokal entfernt)",
    # Der Branch traegt weder voice_check noch die lokalen .gitignore-Eintraege.
    '_SUBMODULES = ("main", "audio", "ctc_align", "exception", "transcription", "utils")':
        "lokal zusaetzlich mit voice_check",
    '# with tools/quantize_voxtral.py).':
        "lokal geht der Kommentar mit Audiotest/ und venv/ weiter",
}

# local/main bietet als 24B-Build voxtral-small-4bit an, der PR noch
# voxtral-small-8bit (Markus beobachtet 4 bit erst lokal, bevor der PR es
# bekommt). Diese Zeilen des PR-Branches fehlen lokal deshalb mit Absicht; kommt
# die Umstellung in den PR, faellt der Block weg.
_SMALL_4BIT_ONLY_LOCAL = [
    'The models `voxtral-mini-8bit` and `voxtral-small-8bit` then appear in the',
    '| `voxtral-small-8bit` (24B) | 25 GB | ~34 GB | quality ceiling for clean, read-aloud audio on 48 GB+; slower than realtime |',
    '**Which of the two?** It depends on the recording, not on a ranking: on clean,',
    'read-aloud speech the 24B model is clearly better (2.8 % against 4.8 % word',
    'error), on hard conversational German with crosstalk and brand names the 3B',
    'model is (4.3 % against 7.8 %). Both 24B figures are the best 24B configuration',
    'measured, which is a locally built 4-bit variant; the shipped 8-bit build scores',
    '8.3 % on the hard passage. And read the inversion with care: by *character*',
    'error the 24B model is the better half of it — it hears more and spells worse.',
    'For interviews and podcasts, pick mini — it is also the only one that runs on a',
    '32 GB or smaller machine. The measurements, including the',
    'comparison against Whisper, are in',
    'and `lm_head` to 8 bit. The encoder runs once per pass, so its precision costs no',
    'speed, but compressing it below 8 bit measurably costs accuracy on difficult',
    'audio. `lm_head` runs once per generated token and is left quantised for that',
    '[small](https://huggingface.co/MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder));',
    'GB/min, small ≈ 27 GB + ~0.8 GB/min. Rough per-pass lengths:',
    '| RAM | mini-8bit | small-8bit |',
    "| 16 GB | ~7 min | won't run |",
    "| 24 GB | 10 min | won't run |",
    "| 32 GB | 10 min | won't run (refused) |",
    '| 48 GB+ | 10 min | 10 min |',
    '# "voxtral-mini-8bit" and "voxtral-small-8bit" models in the backend:',
    '#   small: 4-bit 24B; measured 30 s = 15.6 GB and 120 s = 17.0 GB, i.e. the same',
    '#          ~linear slope as mini but a ~15 GB fixed offset from its weights, so it',
    '#          needs shorter passes / more RAM for the same length.',
    '#          only binds at 16 GB and for the small builds. small/small6/small8 below',
    '#          are still on old-path numbers (no local build to re-measure) -- safe,',
    '#          just conservative, and they target 48 GB+ machines regardless.',
    '#   small8: the shipped 24B build (8-bit LM + lm_head, bf16 encoder). Needs',
    '"small":  {"fixed": 15.2, "slope": 0.017},',
    '# cannot run at all). Each keeps the audio encoder in bf16 while quantising',
    '# the language model and lm_head to 8 bit -- see docs/voxtral-quantisation.md',
    '# for why (the encoder runs once per pass, so its precision is nearly free,',
    '# and compressing it below 8 bit measurably costs accuracy on hard audio).',
    '# mini (3B): the everyday build. Reproduces the bf16 transcript word for',
    '# word on our hard-German reference at ~4.5x the speed, runs on any Mac with',
    '# 16 GB, and beats the 24B model on difficult conversational audio.',
    '# small (24B): a quality ceiling for clean, read-aloud audio on machines',
    '# with 48 GB+. Better than mini on clean speech but slower than realtime, and',
    '# refused below ~34 GB (it would swap forever). See MEM_MODEL / min_ram_gb.',
    '"voxtral-small-8bit": "MarkusKaemmerer/Voxtral-Small-24B-2507-8bit-dense-encoder",',
    'assert v.has_local_build("voxtral-small-8bit")',
]
INTENTIONAL.update(dict.fromkeys(
    _SMALL_4BIT_ONLY_LOCAL, "lokal voxtral-small-4bit statt voxtral-small-8bit"))


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
