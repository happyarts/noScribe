# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A fork of [kaixxx/noScribe](https://github.com/kaixxx/noScribe) (`upstream`) hosted at
`happyarts/noScribe` (`origin`). noScribe is a local, offline GUI app that turns interview
audio into speaker-attributed transcripts.

The fork's substantial addition is the **Voxtral transcription engine** (`noScribe/voxtral_engine.py`,
documented in `VOXTRAL.md`) — an Apple-Silicon-only alternative to faster-whisper. Most other
fork branches are small fixes intended to go back upstream as PRs. Work destined for `upstream`
should be in English, and since 2026-09-01 everything in the Voxtral set is (docs, tests,
scripts). `docs/` holds the measurement write-ups.
They are sorted by topic, not by date: a finding belongs in the
write-up for its subject, or in a comment next to the constant it explains.

The repo ships **two applications**. `noScribe/` is the transcriber (customtkinter/tkinter);
`noScribeEdit/` is a separate PyQt transcript editor with its own README, requirements and
PyInstaller specs. `main.py` starts it as a detached process (`launch_editor`) so it survives
noScribe quitting — the two share no runtime state, only the transcript file. Changes on one
side rarely affect the other.

Supporting code that is neither: `tools/` (model quantisation, loop-detection calibration, and
`check_local_current.py`, which proves `local/main` still carries every line the open PR branches
add), `docs/scripts/` (~40 one-off measurement scripts behind the write-ups in `docs/`, plus
`docs/scripts/engines/` for scoring rival ASR engines), and
`benchmarks-local/`, which is excluded through `.git/info/exclude` rather than `.gitignore` and
therefore exists only in this working copy.

## Commands

```bash
venv/bin/python3 -m pytest tests/ -q          # full suite (~396 tests)
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

`tests/test_worker_import_lightweight.py` guards all three — including a walk over
`pyinstaller/*.spec` for point 2. Claims about frozen behaviour cannot
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

Not every worker sends every message: `whisper_mp_worker.py` emits no `progress`
(its own docstring advertises one, and a `segments` list on the result, that it does
not send), and `pyannote_mp_worker.py` speaks a different contract again.

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

The fork's own ~3000-line module. Beyond decoding it owns: chunking with pause-aware cut points,
repetition-loop detection and a temperature ladder, per-chunk aligner-language selection, word
timestamps via CTC forced alignment (emissions from transformers, the Viterbi DP in
`noScribe/ctc_align.py` — numpy, torchaudio's kernel is not used; `docs/viterbi-numpy-brief.md`
records why), the log-Mel clamp floor (a percentile of the spectrogram instead of its
maximum, `_PercentileFloorFeatures`, swapped onto the processor so the library stays
unpatched), and prefix salvage when a pass has to be redone. Its tuning
constants carry comments explaining the measurement behind each value — read those before
changing a number, and check whether a test in `tests/` pins it.

## State that lives outside the repository

- **User config**: `appdirs.user_config_dir('noScribe')/config.yml`. Several behaviours are only
  reachable through it (e.g. `pyannote_xpu`, `force_whisper_cpu`, `voxtral_ram_reserve_gb`).
- **Models**: `models/` holds `fast`, `precise` and `voxtral-mini-8bit`; Voxtral repos are
  downloaded on first use.
- **UI strings**: `trans/noScribe.<lang>.yml`, one file per language. UI-text changes touch these,
  not the Python source; `de.yml` and `en.yml` are the ones kept current.

## Conventions this repo keeps but does not enforce

- **`local/main` is the integration line**, not a topic branch: the PR branches are cut from
  `main`, which stays a clean mirror of upstream. `tools/check_local_current.py` exists because
  a merge can drop a branch's improvement while `git merge` still says "Already up to date";
  its `INTENTIONAL` table records the lines local/main deliberately differs on.
- **No real names.** No real file, person or brand name appears in the repo, in commit messages
  or in issues — the write-ups use a fictional set (Mona, Lena, Muster, VitaFlor, Sonvita) even
  when quoting a real measurement.
- **Commit subjects are full sentences** naming the effect, and usually the reason ("Take the
  log-Mel clamp floor from a percentile, so one loud cell no longer sets it for the whole
  pass"). No prefixes, no imperative fragments.
- **A measurement lives next to what it justifies**: a tuning constant carries it in the comment
  above it, a regression guard in its test docstring, and only what fits neither goes to `docs/`.

## The Voxtral pull request, when it happens

Voxtral goes upstream as **one** feature PR, deliberately not split, and **after** the smaller
PRs have merged. It therefore has to be built from `local/main` minus everything the other PR
branches carry. The branch that used to hold it (`feature/voxtral-engine`, tip `8f132bc`) was
deleted once it had fallen 161 commits behind: opening a PR from it would have shipped errors
that have since been corrected. Rebuild it from today's `local/main` instead.

That branch was cleanly isolated — it contained none of the other PR branches' commits — and
this was its boundary, which is what a rebuild has to re-establish:

    added     VOXTRAL.md · environments/requirements_voxtral_macOS_arm64.txt
              noScribe/{voxtral_engine,voxtral_mp_worker,transcript_corrections,ctc_align}.py
              tools/{quantize_voxtral,calibrate_loop_detection}.py
              docs/{voxtral-benchmarks,voxtral-quantisation,voxtral-audio-preprocessing,
                    voxtral-mel-clamp-floor,viterbi-numpy-brief,migration-mlx-audio,
                    other-asr-engines}.md · docs/scripts/ (incl. engines/)
              tests/data/forced_align_ref.npz
              tests/test_{voxtral_smoke,voxtral_worker,voxtral_pin,voxtral_safety_guards,
                          fast_generate,loop_breaker,forced_align_cap,forced_align_density,
                          forced_align_stability,transcript_corrections,turn_split,
                          quant_summary,quantize_group_guard,mel_floor,ctc_align,
                          align_language,cut_and_cue_quality,lost_head_recovery,
                          merged_embeddings,salvage_prefix_alignment,word_prob_format}.py
    modified  noScribe/main.py · noScribe/transcription.py · trans/*.yml (four keys:
              voxtral_path_long, voxtral_path_short, voxtral_no_disfluencies, loading_voxtral)
              noScribe/__init__.py (one word: ctc_align in _SUBMODULES)
              environments/requirements_macOS_arm64.txt (a two-line comment)
              tests/test_worker_import_lightweight.py (the voxtral worker in WORKER_MODULES)
              .gitignore (only the `models/voxtral-*` line)

Every Voxtral hunk in `noScribe/main.py` is interleaved with small-PR hunks, so build the
branch as upstream/main + the still-open small branches merged (tag that as the base), take
`main.py` wholesale from `local/main` with the fork-only items below removed, and rebase onto
upstream/main as the small PRs land. `tools/check_local_current.py` says which branch owns
which line. `fix/lazy-main-import` is the one hard dependency (`_SUBMODULES` does not exist
upstream); the three Voxtral depended on only textually — speaker-names, header-labels,
torch-2.13-stack — merged upstream on 2026-09-08 and are now part of the base.

**On `local/main` but neither Voxtral nor one of the open small PRs** — keep them out of the
Voxtral PR:

- `find_ghost_speakers` and `split_at_speaker_change` in `main.py` (with
  `tests/test_ghost_speaker.py`, `tests/test_segment_speaker_split.py` and the
  `warn_ghost_speaker` key in `de`/`en`) are engine-agnostic and since 2026-09-01 have their own
  branch, `feature/ghost-speaker-turn-split`, cut from `main` and merged into `local/main`:
  upstream PR #341. The wrapper `on_segment_split` sits after `on_segment` rather than renaming
  it, so the branch merges cleanly with the other open PRs. `docs/diarization.md` stays local: it also documents the
  fast-embeddings path below, and upstream has no `docs/` directory.
- `noScribe/pyannote_fast_embeddings.py` and its hook in `pyannote_mp_worker.py`: the
  diarization speed-up that upstream pyannote-audio#2048 supersedes. Fork-only until then.
- `Romy` → `Mona` in `tests/test_utils.py` and `tests/test_apostrophe_fix.py`: an edit to
  upstream's own test data. Stays local.
- The removal of the `_Info` object that `_run_engine_subprocess_stream` used to return:
  upstream carries the same dead value (`info_obj` in `_run_whisper_subprocess_stream`), so
  this is a candidate for its own small PR, written against upstream's shape rather than
  cherry-picked — the fork's shared pump does not exist there. Not part of Voxtral.

**These are fork-only and must never appear in that pull request**, however convenient the diff
makes it look:

- **`README.md`** — it carries the fork banner (Voxtral, no packaged download, pointers back to
  kaixxx/noScribe and noscribe.de). It exists for the fork alone. This one is not a judgement
  call; leaving it in would push the fork's advertising into the upstream project.
- **`CLAUDE.md`** — upstream has no such file, and this one describes the fork's own workflow.
- **`tools/check_local_current.py`** — it checks `local/main` against the *fork's* PR branches;
  upstream has nothing for it to do.
- **The deletion of `environments/marked_for_deletion/`** — that directory is the upstream
  maintainers' own staging area (created 2025-05-29, and the name says they mean to remove it
  themselves). Removing it for them is not this fork's call, so the deletion stays local.
- **`.github/workflows/pyinstaller.yml`** — that change belongs to the CI pull request, not to
  Voxtral.
- **`noScribeEdit/`** — the editor source is tracked in this fork for local work only
  (since 2026-09-01; `.gitignore` no longer excludes it). Upstream keeps the editor in
  its own repository, kaixxx/noScribeEditor, and editor changes go there as PRs from
  happyarts/noScribeEditor. `noScribeEdit/Test/` stays ignored: it holds a real recording.
- `environments/requirements_win_cpu.txt` and `requirements_win_cuda.txt` — they belong to
  `fix/pyannote-soundfile-load`, not to Voxtral; Windows is the upstream maintainers' platform.
- **`local_build_stamp()` in `noScribe/main.py`** and its call in the transcript header, plus
  `tests/test_local_build_stamp.py` — stamps transcripts with the git commit and the author's
  initials; local provenance only. The PR keeps `t('doc_header', version=app_version)`.
- The `.gitignore` lines for `noScribeEdit/`, `Audiotest/`, `Audiotest2/` and `venv/` — local
  environment; only `models/voxtral-*` is Voxtral's.

## Test suite character

Many tests are regression guards for defects found in production or in upstream libraries rather
than unit tests of new code — forced-align caps and density, loop breaking, ghost speakers, lost
head recovery, prefix-salvage alignment. Their docstrings explain the failure they prevent. When a
test looks arbitrary, read its docstring before changing it: several encode a bug that was
expensive to find, and one (`test_forced_align_stability.py`) pins the numpy forced-alignment DP
(`noScribe/ctc_align.py`) to a reference recorded from torchaudio in `tests/data/`.
