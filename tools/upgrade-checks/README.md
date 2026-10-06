# Library upgrade checks

The baseline for the next time a library in the stack gets a release: which
versions were tested, against what, and what came out. Newest first. Each row's
JSON is `bench_<label>.json` here.

## How to run a check

1. Copy the venv, so the real one stays untouched (a copied venv runs fine when
   its interpreter is called directly):
   `cp -a venv <scratch>/venvX && <scratch>/venvX/bin/python3 -m pip install "<pkg>==<ver>"`
2. Interleave the runs, so that load on the machine does not read as a regression:
   baseline → candidate → baseline again, with the same label scheme
   (`<lib><version>_<other lib><version>`):
   `<venvX>/bin/python3 tools/upgrade-checks/bench_stack.py <label>` (Voxtral, aligner,
   forced_align, pyannote, raw GEMM), then `bench_nemotron.py <label>` for the Nemotron
   worker on cpu and mps (`--compare` on either afterwards). Both write their JSON here.
   Their audio (`speech_60s.wav`, `bench_4min.wav`) is a real recording and stays out of
   git, in `benchmarks-local/` or wherever `NOSCRIBE_BENCH_AUDIO` points; so do the
   Voxtral transcript and Nemotron's per-frame probabilities they write next to it.
3. `<venvX>/bin/python3 -m pytest tests/ -q -p no:cacheprovider`.
4. Read the release notes for what the stack touches: MPS, CPU attention/matmul,
   stft/FFT, removed APIs (grep noScribe, pyannote, transformers, speechbrain for them).
5. Add a row below, delete the copied venv.

The digests must stay equal; timings only mean something against the baseline
run next to them. A digest that moved between rows with the *same* versions
came from our code (the log-Mel floor on 2026-08-23, aligner changes in
September), not from a library.

## Current digests (torch 2.13.0, torchaudio 2.11.0, transformers 5.18.0, mlx 0.32.1, pyannote.audio 4.0.7, numpy 2.2.6, macOS 27.0.1)

| Check | Digest |
|---|---|
| Voxtral text (60 s) | `d21f430b54bfa7b2` |
| Aligner word stamps | `ff4b4b2462e1f36c` |
| forced_align paths (CPU) | `a70f3c508befdd8c` |
| pyannote bounds (242 segments, 4 speakers) | `a22143a8c73f2f30` |
| Nemotron turns (145, 5 speakers), cpu and mps | `d9e331984c0a553e` |
| Nemotron probabilities cpu / mps | `6371898baa2afc11` / `4068ad505b17ad6b` |

torchcodec 0.15.0 does not load on this machine under any torch (no FFmpeg
installed); noScribe hands pyannote an in-memory waveform, so this is expected.

## Log

| Date | Candidate | Against | Result | Decision |
|---|---|---|---|---|
| 2026-09-30 | torch 2.14.1 | torch 2.13.0, tf 5.18.0 | All digests equal; Nemotron cpu bit-identical, mps one float16 step (4.9e-4) in 306 of 24,001 frames, nothing crosses 0.5, turns equal; 603 tests pass; no timing difference beyond noise (machine under load). Release notes: 2.14's MPS additions (svd, eigh, lstsq, ctc_loss, sampling) and 2.14.1's fixes to them are ops we do not use; removed `torch.qr`/`torch.cholesky` used nowhere in the stack. | Stay on 2.13: nothing gained. Safe to move whenever a dependency needs it. |
| 2026-09-30 | transformers 5.18.0 | 5.18.0.dev0, 5.16.1 | Bit-identical Voxtral text, aligner stamps, pyannote bounds (`bench_tf5180`, `bench_tf518dev`, `bench_tf516`). | Adopted (first release with Nemotron). |
| 2026-09-05 | torch 2.14.0 | torch 2.13.0, tf 5.16.1 | All digests equal (`bench_torch214` vs `bench_torch213`). | Stayed on 2.13. |
| 2026-07-23 | macOS 27 beta | macOS 26.5.2 | All digests equal (`bench_macos27` vs `bench_macos26`). | — |
| 2026-07 | torch 2.13 / torchaudio 2.11 / torchcodec 0.15 | torch 2.8 | pyannote identical output and speed (comment in `environments/requirements_macOS_arm64.txt`). | Adopted on macOS arm64; Linux/Windows stay on 2.8. |
