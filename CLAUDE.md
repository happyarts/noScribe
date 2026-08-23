# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A fork of [kaixxx/noScribe](https://github.com/kaixxx/noScribe) (`upstream`) hosted at
`happyarts/noScribe` (`origin`). noScribe is a local, offline GUI app that turns interview
audio into speaker-attributed transcripts.

The fork's substantial addition is the **Voxtral transcription engine** (`noScribe/voxtral_engine.py`,
documented in `VOXTRAL.md`) — an Apple-Silicon-only alternative to faster-whisper. Most other
fork branches are small fixes intended to go back upstream as PRs. Work destined for `upstream`
should be in English; `docs/` holds German measurement write-ups.

The repo ships **two applications**. `noScribe/` is the transcriber (customtkinter/tkinter);
`noScribeEdit/` is a separate PyQt transcript editor with its own README, requirements and
PyInstaller specs. `main.py` starts it as a detached process (`launch_editor`) so it survives
noScribe quitting — the two share no runtime state, only the transcript file. Changes on one
side rarely affect the other.

Supporting code that is neither: `tools/` (model quantisation, loop-detection calibration),
`docs/skripte/` (~30 one-off measurement scripts behind the German write-ups in `docs/`), and
`benchmarks-local/`, which is excluded through `.git/info/exclude` rather than `.gitignore` and
therefore exists only in this working copy.

## Commands

```bash
venv/bin/python3 -m pytest tests/ -q          # full suite (~356 tests)
venv/bin/python3 -m pytest tests/test_loop_breaker.py -q
venv/bin/python3 -m pytest tests/ -q -k ghost_speaker
venv/bin/python3 -m noScribe                  # launch the GUI
```

Use `venv/bin/python3` — the binary is `python3`, there is no `python`, and the system
interpreter has neither pytest nor torch. There is no pytest config file; tests are plain
pytest with no fixtures directory. The editor is a separate app with separate dependencies
(`noScribeEdit/environments/`), launched as `venv/bin/python3 noScribeEdit/noScribeEdit.py`.

**Git LFS is what ships the large model files, but nothing is LFS-tracked in this working copy**
(`git lfs ls-files` is empty). The `post-commit`, `post-checkout`, `post-merge` and `pre-push`
hooks still run on every operation and have stalled commits and cherry-picks here with no LFS
work to do. Prefix git commands with `git -c core.hooksPath=/dev/null` when one hangs.

CI (`.github/workflows/pytest.yml`) runs the suite on Linux against
`environments/requirements_linux.txt` on Python 3.10 and 3.13, so anything added must import
without the macOS-only Voxtral stack — the Voxtral tests use `pytest.importorskip`.
`.github/workflows/pyinstaller.yml` builds the frozen app on every push and pull request.

Dependencies are per-platform under `environments/`. `requirements_macOS_arm64.txt` is the
maintained one; `requirements_voxtral_macOS_arm64.txt` layers the Voxtral stack on top.

## Architecture

### Process model — the part that breaks silently

Transcription and diarization each run in a **spawned subprocess**, and three constraints
conspire here. Get any of them wrong and the test suite stays green while the frozen app dies.

1. `noScribe/__init__.py` loads submodules lazily via PEP 562. This is load-bearing: a spawn
   child re-imports the package just to reach its entrypoint, and an eager
   `from noScribe import main` would drag tkinter, customtkinter and PyAV into every worker.
2. PyInstaller cannot follow those lazy imports, so **every spec in `pyinstaller/` must list
   `noScribe.main` in `hiddenimports`**.
3. Under PyInstaller a spawn child is the frozen binary restarted from the top, so
   `mp.freeze_support()` must run in `noScribe/__main__.py` *before* anything touches
   `noScribe.main`.

`tests/test_worker_import_lightweight.py` guards 1 and 3. Claims about frozen behaviour cannot
be derived from source — prove them with a throwaway PyInstaller build.

### Worker protocol

`whisper_mp_worker.py` and `voxtral_mp_worker.py` are interchangeable behind one queue message
contract, which is why `main.py` consumes segments identically regardless of engine:

```
{"type": "log",      "level": "info|warn|error|debug", "msg": str}
{"type": "progress", "pct": int}
{"type": "segment",  "segment": {"start","end","text","words"}}
{"type": "result",   "ok": True,  "info": {...}}
{"type": "result",   "ok": False, "error": str, "trace": str}
```

A broken `segment` put is deliberately *not* swallowed — silently truncating a transcript while
reporting success is worse than failing the job. Worker modules stay stdlib-only at import time
and defer heavy imports into the entrypoint.

### Pipeline

`main.py` drives: convert to WAV (`noScribe/audio/convert.py`, ffmpeg) → diarize
(`pyannote_mp_worker`) → transcribe (whisper or voxtral worker) → merge and write the transcript.
Which engine runs is decided by the `engine` field on the `WhisperModel` dataclass in
`transcription.py` (`"whisper"` or `"voxtral"`).

`main.py` is ~4200 lines and mixes GUI, queue management and pipeline orchestration; expect to
search rather than read it.

### voxtral_engine.py

The fork's own ~2700-line module. Beyond decoding it owns: chunking with pause-aware cut points,
repetition-loop detection and a temperature ladder, per-chunk aligner-language selection, word
timestamps via CTC forced alignment (emissions from transformers, the Viterbi DP in
`noScribe/ctc_align.py` — numpy, torchaudio's kernel is not used; `docs/viterbi-numpy-brief.md`
records why), and prefix salvage when a pass has to be redone. Its tuning
constants carry comments explaining the measurement behind each value — read those before
changing a number, and check whether a test in `tests/` pins it.

## State that lives outside the repository

- **User config**: `appdirs.user_config_dir('noScribe')/config.yml`. Several behaviours are only
  reachable through it (e.g. `pyannote_xpu`, `force_whisper_cpu`, `voxtral_ram_reserve_gb`).
- **Models**: `models/` holds `fast`, `precise` and `voxtral-mini-8bit`; Voxtral repos are
  downloaded on first use.
- **UI strings**: `trans/noScribe.<lang>.yml`, one file per language. UI-text changes touch these,
  not the Python source; `de.yml` and `en.yml` are the ones kept current.

## Test suite character

Many tests are regression guards for defects found in production or in upstream libraries rather
than unit tests of new code — forced-align caps and density, loop breaking, ghost speakers, lost
head recovery, prefix-salvage alignment. Their docstrings explain the failure they prevent. When a
test looks arbitrary, read its docstring before changing it: several encode a bug that was
expensive to find, and one (`test_forced_align_stability.py`) pins the numpy forced-alignment DP
(`noScribe/ctc_align.py`) to a reference recorded from torchaudio in `tests/data/`.
