"""Dieselbe Sprache aus verschiedenen Quellen gegen eine Referenz messen.

Wenn von einer Aufnahme eine rohe und eine bearbeitete Fassung vorliegt, ist das
ein natürliches Experiment: gleiche Wörter, verschiedene Vorverarbeitung, eine
handkorrigierte Referenz für beide. Nur passen die Zeitachsen nicht zusammen --
ein Leveller wie Auphonic schneidet Stillen, die Passage liegt in jeder Fassung
woanders. Deshalb wird hier jeder Arm mit eigenem Fenster angegeben.

Ein Arm ist `label=datei@von:bis` und optional `+kette`, wobei die Kette ein
Name aus audio_filters.CHAINS ist:

    roh=Aufnahme.m4a@737:858.6
    roh_dyn=Aufnahme.m4a@737:858.6+dynaudnorm
    auphonic=Aufnahme-proc.m4a@725.65:845.65

Das Fenster findet man mit docs/skripte/locate_passage.py.

    python docs/skripte/pair_cer.py <referenz.txt> <build> <arm> [arm ...]
"""
import pathlib
import sys
import time

import numpy as np
import soundfile as sf
import soxr

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "docs" / "skripte"))
from wer import norm, wer, OVERLAP
from audio_filters import apply_filter, CHAINS
from audio_audit import decode_float

SR = 16000


def parse(spec):
    label, _, rest = spec.partition("=")
    rest, _, chain = rest.partition("+")
    path, _, window = rest.partition("@")
    t0, _, t1 = window.partition(":")
    return label, path, float(t0), float(t1), (chain or None)


def render(path, t0, t1, chain, cache):
    """Fenster als 16 kHz mono float32, über den echten Produktionsweg gedacht:
    (L+R)/2, dann auf 16 kHz. Der Resampler ist hier soxr statt swr -- gemessen
    macht das keinen Unterschied (Abschnitt 11), und er hält die Fenstergrenzen
    exakt."""
    if path not in cache:
        x, sr = decode_float(path)
        cache[path] = soxr.resample(x, sr, SR, quality="VHQ").astype(np.float32)
    y = cache[path][int(t0 * SR):int(t1 * SR)]
    return np.ascontiguousarray(apply_filter(y, CHAINS[chain]) if chain else y,
                                dtype=np.float32)


def main():
    ref_path, build = sys.argv[1], sys.argv[2]
    specs = [parse(s) for s in sys.argv[3:]]

    raw = open(ref_path, encoding="utf-8").read()
    overlap_words = norm(" ".join(OVERLAP.findall(raw)))
    ref = norm(OVERLAP.sub(" ", raw))
    ref_chars = "".join(ref)
    print(f"# {ref_path}: {len(ref)} Woerter / {len(ref_chars)} Zeichen")
    print(f"# Build: {build}\n")
    print(f"{'Arm':16s} {'Sek':>6s} {'peak':>6s} {'LUFS':>7s} {'WER':>7s} {'CER':>7s} "
          f"{'Sub':>4s} {'Del':>4s} {'Ins':>4s} {'W':>5s}  Overlap")

    from noScribe.voxtral_engine import _Voxtral
    from preproc_variants import loudness
    vox = _Voxtral(build)
    cache = {}
    for label, path, t0, t1, chain in specs:
        a = render(path, t0, t1, chain, cache)
        dur = len(a) / SR
        text = vox.transcribe_array(a, "de", max_new_tokens=int(dur * 20) + 512)
        hyp = norm(text)
        err, sub, dele, ins = wer(ref, hyp)
        cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
        got = sum(1 for w in overlap_words if w in hyp) if overlap_words else 0
        print(f"{label:16s} {dur:6.1f} {np.abs(a).max():6.3f} {loudness(a):7.1f} "
              f"{err/len(ref)*100:6.2f}% {cer*100:6.2f}% {sub:4d} {dele:4d} {ins:4d} "
              f"{len(hyp):5d}  {got}/{len(overlap_words)}", flush=True)
        open(f"/tmp/wer_{label}.txt", "w").write(text)


if __name__ == "__main__":
    main()
