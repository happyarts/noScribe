#!/usr/bin/env python3
"""Benchmark of the noScribe stack before and after a library or macOS update
(the procedure is in README.md beside this script).

Run with each interpreter:  <venv>/bin/python3 tools/upgrade-checks/bench_stack.py <label>
Compare:                    venv/bin/python3 tools/upgrade-checks/bench_stack.py --compare

Covers every layer that a macOS/Metal/driver change could shift:
  1. mlx GPU GEMM (bf16) ......... raw Metal throughput
  2. mlx quantized matmul ........ the kernel Voxtral decode lives on
  3. torch MPS GEMM .............. pyannote's device
  4. torchaudio forced_align ..... CPU DP kernel + result digest
  5. Voxtral E2E (60 s speech) ... wall time, tokens/s, mlx peak mem,
                                   greedy TEXT (must stay bit-identical)
  6. wav2vec2 emission + align ... aligner timing + timestamp digest
  7. pyannote diarization ........ wall time, segments/speakers + digest

Determinism digests catch silent numeric changes; timings catch perf
regressions. Results land in bench_<label>.json beside this script.
Functional gate on top: run the full pytest suite before and after.

The audio (speech_60s.wav, bench_4min.wav) is a real recording and stays out of
git, in benchmarks-local/ or the directory NOSCRIBE_BENCH_AUDIO names; so does
voxtral_60s_text.txt, the transcript written there for eyeball diffs.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
AUDIO = Path(os.environ.get("NOSCRIBE_BENCH_AUDIO", ROOT / "benchmarks-local"))
sys.path.insert(0, str(ROOT))


def _sysinfo():
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True).strip()
        except Exception:
            return ""
    import mlx.core as mx
    import mlx_lm, mlx_voxtral, numpy, torch, torchaudio, transformers
    import pyannote.audio
    return {
        "sw_vers": sh("sw_vers -productVersion") + " (" + sh("sw_vers -buildVersion") + ")",
        "uname": platform.platform(),
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "mlx_voxtral": getattr(mlx_voxtral, "__version__", "?"),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "pyannote.audio": pyannote.audio.__version__,
        "numpy": numpy.__version__,
    }


def _digest(x):
    return hashlib.sha256(repr(x).encode()).hexdigest()[:16]


def bench_mlx_gemm():
    import mlx.core as mx
    n = 4096
    a = mx.random.normal((n, n), dtype=mx.bfloat16, key=mx.random.key(0))
    b = mx.random.normal((n, n), dtype=mx.bfloat16, key=mx.random.key(1))
    mx.eval(a @ b)  # warmup + compile
    t0 = time.monotonic()
    iters = 20
    for _ in range(iters):
        mx.eval(a @ b)
    dt = time.monotonic() - t0
    return {"tflops": round(iters * 2 * n**3 / dt / 1e12, 2), "sec": round(dt, 3)}


def bench_mlx_quantized():
    import mlx.core as mx
    n, k = 4096, 4096
    w = mx.random.normal((n, k), key=mx.random.key(2))
    wq, scales, biases = mx.quantize(w, group_size=64, bits=8)
    x = mx.random.normal((64, k), dtype=mx.bfloat16, key=mx.random.key(3))
    mx.eval(mx.quantized_matmul(x, wq, scales, biases, group_size=64, bits=8))
    t0 = time.monotonic()
    iters = 200
    for _ in range(iters):
        mx.eval(mx.quantized_matmul(x, wq, scales, biases, group_size=64, bits=8))
    return {"ms_per_iter": round((time.monotonic() - t0) / iters * 1000, 3)}


def bench_torch_mps():
    import torch
    if not torch.backends.mps.is_available():
        return {"skipped": "no mps"}
    dev = torch.device("mps")
    g = torch.Generator().manual_seed(0)
    a = torch.randn(2048, 2048, generator=g).to(dev)
    b = torch.randn(2048, 2048, generator=g).to(dev)
    (a @ b).sum().item()  # warmup
    t0 = time.monotonic()
    iters = 50
    for _ in range(iters):
        c = a @ b
    torch.mps.synchronize()
    dt = time.monotonic() - t0
    return {"tflops": round(iters * 2 * 2048**3 / dt / 1e12, 2)}


def bench_forced_align():
    import torch
    import torchaudio.functional as F
    T, C, L = 20000, 30, 4000  # ~1.6e8 cells, well under the 2^31 cap
    g = torch.Generator().manual_seed(0)
    emission = torch.log_softmax(torch.rand((T, C), generator=g), dim=-1)
    targets = torch.randint(1, C, (1, L), generator=g, dtype=torch.int32)
    t0 = time.monotonic()
    paths, scores = F.forced_align(emission.unsqueeze(0), targets, blank=0)
    dt = time.monotonic() - t0
    return {"sec": round(dt, 3),
            "paths_digest": _digest(paths[0].tolist()),
            "score_sum": round(float(scores.sum()), 2)}


def bench_voxtral(results):
    import mlx.core as mx
    import soundfile as sf
    from noScribe.voxtral_engine import (_Voxtral, resolve_model,
                                         VOXTRAL_MODELS)
    audio, sr = sf.read(AUDIO / "speech_60s.wav", dtype="float32")
    t0 = time.monotonic()
    vox = _Voxtral(resolve_model("voxtral-mini-8bit",
                                 VOXTRAL_MODELS["voxtral-mini-8bit"]))
    load_s = time.monotonic() - t0
    mx.reset_peak_memory()
    t0 = time.monotonic()
    text = vox.transcribe_array(audio, "de", max_new_tokens=2048)
    dt = time.monotonic() - t0
    results["voxtral_e2e"] = {
        "load_sec": round(load_s, 1),
        "transcribe_sec": round(dt, 1),
        "realtime_x": round(len(audio) / sr / dt, 2),
        "words": len(text.split()),
        "text_sha": _digest(text),
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    (AUDIO / "voxtral_60s_text.txt").write_text(text)  # for eyeball diffs
    return vox, audio


def bench_aligner(results, vox, audio):
    import re
    from noScribe.voxtral_engine import _Aligner, resolve_align_model
    text = (AUDIO / "voxtral_60s_text.txt").read_text()
    words = re.findall(r"\S+", text)
    t0 = time.monotonic()
    al = _Aligner(resolve_align_model("de"))
    load_s = time.monotonic() - t0
    t0 = time.monotonic()
    stamps = al.align_words(words, audio)
    dt = time.monotonic() - t0
    results["aligner"] = {
        "load_sec": round(load_s, 1),
        "align_sec": round(dt, 1),
        "n_words": len(stamps),
        "stamps_digest": _digest([(round(w["start"], 3), round(w["end"], 3))
                                  for w in stamps]),
    }


def bench_pyannote(results):
    # Through the real worker module so device pick etc. match production.
    import importlib.util
    import queue
    spec = importlib.util.spec_from_file_location(
        "pyw", ROOT / "noScribe" / "pyannote_mp_worker.py")
    pyw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pyw)
    q = queue.Queue()
    t0 = time.monotonic()
    pyw.pyannote_proc_entrypoint({"audio_path": str(AUDIO / "bench_4min.wav")}, q)
    dt = time.monotonic() - t0
    segs = []
    while not q.empty():
        m = q.get()
        if m.get("type") == "result":
            segs = m.get("segments", [])
    results["pyannote"] = {
        "sec": round(dt, 1),
        "n_segments": len(segs),
        "n_speakers": len({s["label"] for s in segs}),
        "bounds_digest": _digest([(round(s["start"]), round(s["end"]), s["label"])
                                  for s in segs]),
    }


def compare():
    files = sorted(HERE.glob("bench_*.json"))
    runs = {f.stem.replace("bench_", ""): json.loads(f.read_text()) for f in files}
    print(f"{'metric':45s} " + " ".join(f"{k:>14s}" for k in runs))
    def walk(prefix, dicts):
        keys = sorted({k for d in dicts if isinstance(d, dict) for k in d})
        for k in keys:
            vals = [d.get(k) if isinstance(d, dict) else None for d in dicts]
            if all(isinstance(v, dict) or v is None for v in vals) and any(isinstance(v, dict) for v in vals):
                walk(f"{prefix}{k}.", vals)
            else:
                marks = ""
                svals = [str(v) for v in vals]
                if len(set(v for v in svals if v != "None")) > 1:
                    marks = "   <-- DIFFERS"
                print(f"{prefix + k:45s} " + " ".join(f"{v:>14s}" for v in svals) + marks)
    walk("", list(runs.values()))


def main():
    if "--compare" in sys.argv:
        return compare()
    label = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
    results = {"label": label, "sysinfo": _sysinfo()}
    print(json.dumps(results["sysinfo"], indent=1))
    steps = [
        ("mlx_gemm_bf16", bench_mlx_gemm),
        ("mlx_quantized_matmul", bench_mlx_quantized),
        ("torch_mps_gemm", bench_torch_mps),
        ("forced_align_cpu", bench_forced_align),
    ]
    for name, fn in steps:
        t0 = time.monotonic()
        results[name] = fn()
        print(f"{name}: {results[name]} ({time.monotonic()-t0:.1f}s)")
    vox, audio = bench_voxtral(results)
    print("voxtral_e2e:", results["voxtral_e2e"])
    bench_aligner(results, vox, audio)
    print("aligner:", results["aligner"])
    del vox
    bench_pyannote(results)
    print("pyannote:", results["pyannote"])
    out = HERE / f"bench_{label}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
