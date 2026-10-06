#!/usr/bin/env python3
"""Nemotron diarization through noScribe's own worker, for library upgrade checks
(bench_stack.py covers Voxtral, the aligner and pyannote, not Nemotron).

Run with each interpreter:  <venv>/bin/python3 tools/upgrade-checks/bench_nemotron.py <label>
Compare:                    venv/bin/python3 tools/upgrade-checks/bench_nemotron.py --compare

Runs nemotron_proc_entrypoint on bench_4min.wav (converted to 16 kHz mono with
ToWav, as main.py does) on cpu and mps. The turns' digest goes to
nemotron_<label>_<device>.json beside this script; the per-frame probabilities,
too large for git, to nemotron-bench/<label>_<device>.npy beside the audio (a real
recording, kept in benchmarks-local/ or the directory NOSCRIBE_BENCH_AUDIO names).
--compare prints, for each pair of labels on the same device, the largest
probability difference, the frames that differ and the ones that cross 0.5.
Expected on 2026-09-30: cpu bit-identical across torch 2.13.0/2.14.1; mps
deterministic per version, but one float16 step (4.9e-4) apart in 306 of
24,001 frames between them, less than mps against cpu (823), turns identical.
"""
import hashlib
import itertools
import json
import os
import queue
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
AUDIO = Path(os.environ.get("NOSCRIBE_BENCH_AUDIO", ROOT / "benchmarks-local"))
OUT = AUDIO / "nemotron-bench"
sys.path.insert(0, str(ROOT))


def wav16k():
    from noScribe.audio.convert import ToWav
    path = OUT / "bench_4min_16k.wav"
    if not path.exists():
        with ToWav(AUDIO / "bench_4min.wav", path) as conv:
            while conv.convert():  # one frame per call
                pass
    return path


def run(label):
    import numpy as np
    import torch
    from noScribe.nemotron_mp_worker import nemotron_proc_entrypoint
    OUT.mkdir(exist_ok=True)
    src = wav16k()
    for dev in ("cpu", "mps"):
        wav = OUT / f"{label}_{dev}.wav"  # the worker writes <wav>.speakers.npy next to it
        shutil.copy(src, wav)
        q = queue.Queue()
        t0 = time.monotonic()
        nemotron_proc_entrypoint({"audio_path": str(wav), "device": dev}, q)
        sec = time.monotonic() - t0
        res = [m for m in (q.get() for _ in range(q.qsize())) if m["type"] == "result"][0]
        assert res["ok"], res
        probs = Path(res["probabilities"]["path"])
        probs.rename(OUT / f"{label}_{dev}.npy")
        wav.unlink()
        p = np.load(OUT / f"{label}_{dev}.npy")
        info = {"label": label, "device": dev, "torch": torch.__version__,
                "transformers": __import__("transformers").__version__, "sec": round(sec, 1),
                "n_turns": len(res["segments"]),
                "n_speakers": len({s["label"] for s in res["segments"]}),
                "turns_digest": hashlib.sha256(repr(res["segments"]).encode()).hexdigest()[:16],
                "probs_digest": hashlib.sha256(p.tobytes()).hexdigest()[:16]}
        (HERE / f"nemotron_{label}_{dev}.json").write_text(json.dumps(info, indent=1))
        print(json.dumps(info))


def compare():
    import numpy as np
    for dev in ("cpu", "mps"):
        labels = [json.loads(f.read_text())["label"] for f in sorted(HERE.glob(f"nemotron_*_{dev}.json"))]
        # Only runs whose probabilities are on this machine; the JSON alone holds the digests.
        runs = {lab: OUT / f"{lab}_{dev}.npy" for lab in labels if (OUT / f"{lab}_{dev}.npy").exists()}
        for a, b in itertools.combinations(runs, 2):
            x = np.load(runs[a]).astype(np.float32)
            y = np.load(runs[b]).astype(np.float32)
            d = np.abs(x - y)
            print(f"{dev} {a} vs {b}: max {d.max():.2e}, frames differing "
                  f"{int((d > 0).any(1).sum())}/{len(d)}, crossing 0.5 {int(((x > .5) != (y > .5)).sum())}")


if __name__ == "__main__":
    compare() if "--compare" in sys.argv else run(sys.argv[1] if len(sys.argv) > 1 else "unlabeled")
