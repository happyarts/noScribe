# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A fork of [kaixxx/noScribe](https://github.com/kaixxx/noScribe) (`upstream`) hosted at
`happyarts/noScribe` (`origin`). noScribe is a local, offline GUI app that turns interview
audio into speaker-attributed transcripts.

The fork's substantial addition is the **Voxtral transcription engine** (`noScribe/voxtral_engine.py`,
documented in `VOXTRAL.md`) — an Apple-Silicon-only alternative to faster-whisper. Most other
fork branches are small fixes intended to go back upstream as PRs. Work destined for `upstream`
should be in English, and everything in the Voxtral set is (docs, tests, scripts). `docs/` holds the measurement write-ups.
They are sorted by topic, not by date: a finding belongs in the
write-up for its subject, or in a comment next to the constant it explains.

The repo ships **two applications**. `noScribe/` is the transcriber (customtkinter/tkinter);
`noScribeEdit/` is a separate PyQt transcript editor with its own README, requirements and
PyInstaller specs. `main.py` starts it as a detached process (`launch_editor`) so it survives
noScribe quitting — the two share no runtime state, only the transcript file. Changes on one
side rarely affect the other.

Supporting code that is neither: `tools/` (model quantisation, loop-detection calibration, and
`check_local_current.py`, which proves `main` still carries every line the open PR branches
add; `upgrade-checks/` logs every library version tried), `docs/scripts/` (one-off measurement scripts behind the write-ups in `docs/`, plus
`docs/scripts/engines/` for scoring rival ASR engines), and
`benchmarks-local/`, which is excluded through `.git/info/exclude` rather than `.gitignore` and
therefore exists only in this working copy.

## Commands

```bash
venv/bin/python3 -m pytest tests/ -q          # full suite
venv/bin/python3 -m pytest tests/test_loop_breaker.py -q
venv/bin/python3 -m pytest tests/ -q -k voice_check
venv/bin/python3 -m noScribe                  # launch the GUI
```

Use `venv/bin/python3` — the binary is `python3`, there is no `python`, and the system
interpreter has neither pytest nor torch. There is no pytest config file; tests are plain
pytest with no fixtures directory. The editor is a separate app with separate dependencies
(`noScribeEdit/environments/`), launched as `venv/bin/python3 noScribeEdit/noScribeEdit.py`.

**Git LFS is what ships the large model files, but nothing is LFS-tracked in this working copy**
(`git lfs ls-files` is empty). The `post-commit`, `post-checkout`, `post-merge` and `pre-push`
hooks still run on every operation and have stalled commits and cherry-picks here with no LFS
work to do. Prefix git commands with `git -c core.hooksPath=/dev/null` when one hangs;
`--no-verify` does not help, it skips only the pre-commit and commit-msg hooks.

CI (`.github/workflows/pytest.yml`) runs the suite on Linux against
`environments/requirements_linux.txt` on Python 3.11 and 3.13, so anything added must import
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
be derived from source — prove them with a throwaway PyInstaller build (a minimal package that
reproduces the structure freezes in about a minute and needs no models).

### Plugins and the worker protocol

Since upstream's api-server merge (#377, merged into `main` 2026-10-30) every engine is a
**backend plugin** behind `noScribe/plugins/registry.py`; `main.py` never starts a worker
itself. Models are `ModelRef`s (`local-whisper:precise`, `local-voxtral:voxtral-mini-8bit`),
the GUI's model menu comes from the registry, and `noScribe/plugins/README.md` (upstream's)
is the contract. Upstream ships `local_whisper`, `local_pyannote` and `remote_http`; the
fork adds `local_voxtral` and `local_nemotron` (registered in `plugins/factory.py`). All
local plugins share one `LocalInferenceBackend` (`inference.py`), whose
`run_transcription_worker`/`run_diarization_worker` (a fork refactor of upstream's two
loops) run any worker that speaks the queue contract:

```
{"type": "log",      "level": "info|warn|error|debug", "msg": str}
{"type": "status",   "message_id": str, "params": {...}, "level": str}  # translated in main.py
{"type": "state",    "device": str|None, "stage": str}                   # for the CPU fallback
{"type": "progress", "pct": int}                    # diarization adds "step"
{"type": "segment",  "segment": {"start","end","text","words"}}
{"type": "result",   "ok": True,  "info": {...}}    # diarization: "segments", "extensions"
{"type": "result",   "ok": False, ...}              # worker_errors.error_result()
```

Not every worker sends every message: Whisper emits no `progress` (main.py then derives it
from the segments; an engine that sends its own, Voxtral, keeps the bar). Plugin-specific
data travels in `extensions`, keyed by backend ID and JSON-compatible so a server can
carry it: pyannote's `centroids` (and, on a second call with `embed_spans`, `embeddings`)
under `local-pyannote`, Nemotron's per-frame `probabilities` as base64 float16 under
`local-nemotron` (`nemotron_mp_worker.encode_probabilities`, ~7.7 MB per audio hour).
main.py adds `extensions["noscribe"]` (`needs_timing`, `speaker_names`) to local
transcription requests; Voxtral reads it. The diarization reaches the transcription as
`TranscriptionRequest.diarization`: Voxtral cuts loops at its turns, and Whisper's speech
map (`whisper_mp_worker.speech_turns_from`) uses pyannote's turns only.
Per-plugin settings live in `plugin-settings/<backend-id>.yml` in the user config dir;
the older config.yml keys (`force_nemotron_cpu`, `voxtral_chunk_sec`,
`voxtral_ram_reserve_gb`) still fill fields a plugin file leaves out.

A broken `segment` put is deliberately *not* swallowed — silently truncating a transcript while
reporting success is worse than failing the job. Worker modules stay stdlib-only at import time
and defer heavy imports into the entrypoint.

### Pipeline

`main.py` drives: convert to WAV (`noScribe/audio/convert.py`, PyAV) → diarize
(`pyannote_mp_worker` or `nemotron_mp_worker`) → transcribe (whisper or voxtral worker) → voice
check → merge and write the transcript. `select_diarization_model` picks the diarization: a
remote server's own first, else what `diarization_engine` in config.yml says:
`auto` (default) takes Nemotron whenever the installed transformers knows
`nemotron3_diarization`, librosa is installed (transformers builds the processor's mel filters
with it) and its weights are on disk (shipped or in the HF cache), `pyannote`/`nemotron`
force one (`nemotron` downloads the weights); a job with a fixed speaker count always stays with
pyannote. The transcript header names the engine ("auto (Nemotron)").
Which transcription engine runs is decided by the model's backend (`local-voxtral` or
`local-whisper`); a bare model name saved before the plugins is migrated by `_model_ref`.

`main.py` is large and mixes GUI, queue management and pipeline orchestration; expect to
search rather than read it.

### voxtral_engine.py

The fork's own module, and a large one. Beyond decoding it owns: chunking with pause-aware cut points,
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
  reachable through it (e.g. `pyannote_xpu`, `force_whisper_cpu`, `diarization_engine`), and the
  plugins read `plugin-settings/<backend-id>.yml` beside it.
- **Models**: `models/` holds `fast`, `precise` and `voxtral-mini-8bit`; Voxtral repos are
  downloaded on first use, and so is `nvidia/Nemotron-3-Diarization` (HF cache) unless a copy
  sits in `models/nemotron-diarization`.
- **The venv follows the requirements, not a frozen set**: on 2026-10-20 it was upgraded to what a
  fresh install gets (transformers 5.20.2, huggingface_hub 2.2, librosa 1.0) plus the bumped
  Voxtral pins (mlx 0.32.4, mlx-lm 0.32.2), with torch still 2.13; every result stayed
  bit-identical. A machine on an older venv fails `tests/test_voxtral_pin.py` until it runs
  `pip install --upgrade --upgrade-strategy eager -r` on both requirements files, with `datasets`
  named in the same call: the measurement scripts need it, no requirements file lists it, and
  left out it is ignored while fsspec moves past what it allows. 5.18.0 is the
  first transformers with Nemotron's model, so `>=5.18` stays the floor. Every library upgrade check
  (versions tried, digests, decision, how to run one) is logged in
  `tools/upgrade-checks/README.md`, with each run's `bench_*.json` beside it; start a new
  check from there, with the scripts beside it (`bench_stack.py`, `bench_nemotron.py`); only
  their audio, a real recording, stays in `benchmarks-local/`. `~/Documents/transformers-src`
  is a clone of transformers for upstream work there; `~/Documents/transformers-src-analysis/README.md`
  says how to work in it (dev venv, checks, scripts, PR texts).
- **UI strings**: `trans/noScribe.<lang>.yml`, one file per language. UI-text changes touch these,
  not the Python source; `de.yml` and `en.yml` are the ones kept current.

## Conventions this repo keeps but does not enforce

- **`main` is the integration line**, not a topic branch: the PR branches are cut from
  `upstream-main`, the local branch that stays a clean mirror of `upstream/main` (`feature/nemotron-diarization` is the one branch
  stacked on another; the check follows it through the upper branch). `tools/check_local_current.py` exists because
  a merge can drop a branch's improvement while `git merge` still says "Already up to date";
  its `INTENTIONAL` table records the lines `main` deliberately differs on.
- **Stage explicit paths on branches cut from `upstream-main`.** They carry upstream's `.gitignore`, which
  does not exclude `Audiotest/`, `Audiotest2/` (private recordings), `models/voxtral-*` or
  `venv/`, so `git add -A` there would stage them.
- **`main` is the fork's default branch**, tracking `origin/main` — it is *not* upstream's
  `main`, which lives locally as `upstream-main`.
- **Fix on the topic branch first, then merge it into `main`** — not the other way round, or
  the change has to be carried into the PR branch by hand later.
  On GitHub, never use "Contribute → Open pull request" from it (it would offer the whole fork
  to kaixxx) or "Sync fork → Discard commits" (it would reset it to upstream).
- **Never write `kaixxx/noScribe#N` in a pull request or issue on the fork.** GitHub turns it into
  a permanent cross-reference on the upstream PR that cannot be edited away.
- **No real names.** No real file, person or brand name appears in the repo, in commit messages
  or in issues — the write-ups use a fictional set (Mona, Lena, Muster, VitaFlor, Sonvita) even
  when quoting a real measurement.
- **Commit subjects are full sentences** naming the effect, and usually the reason ("Take the
  log-Mel clamp floor from a percentile, so one loud cell no longer sets it for the whole
  pass"). No prefixes, no imperative fragments.
- **A measurement lives next to what it justifies**: a tuning constant carries it in the comment
  above it, a regression guard in its test docstring, and only what fits neither goes to `docs/`.
- **Pull requests and issues to other projects follow that project's rules.** Read its
  CONTRIBUTING and PR checks first and run its own check commands (for transformers `make
  fix-repo`, `make typing`, `make check-repo`); keep the diff minimal and comments short;
  disclose AI help; where agent-written PRs are unwelcome, ask the maintainer first; tick a
  checklist box only once it is true.

## The Voxtral pull request

**State on 2026-10-30:** `main` carries upstream's plugin architecture (#377) with Voxtral and
Nemotron as bundled plugins. All five PR branches were rebuilt on it, reviewed, merged into
`main` and force-pushed (the pre-plugin tips are tagged `backup/*-pre-plugins`); Nemotron is
upstream PR #379, stacked on #351, with pyannote kept beside it for now. Fork-only migrations
(the renamed `voxtral-small-8bit`, the `voxtral_*` keys in config.yml) live on `main` alone,
with `tests/test_fork_migrations.py`. Upstream's own CI is red on
`test_empty_cli_result_returns_failure` (no Whisper models on the runner), on its `main` too.

Two worktrees under `benchmarks-local/` are frozen code for measurement scripts, not work in
progress: `src-local-main` (2026-09-17, loaded by `whisper-split-ab/` and `ami/`) and `wt-nemo`
(the pre-plugin Nemotron branch, loaded by `nemotron/vc_voxnew.py`). Both are detached; keep
them while those scripts are meant to reproduce their numbers.

Voxtral goes upstream as **one** feature PR, deliberately not split. Its branch is
`feature/voxtral-engine`, rebuilt on 2026-09-20 (and again on 2026-09-23, after the Opus 5.5 review; rebased with the other open branches onto upstream 0.8 on 2026-10-05, each now carrying its own adaptations of upstream's tests, and again on 2026-10-06 onto upstream's `av<19` pin) from `upstream/main` plus `main`'s
Voxtral set (the earlier branch of that name, tip `8f132bc`, was deleted once it had fallen far
behind). It is cut from `upstream-main` and carries **neither of the still-open small PRs**, so its diff
is Voxtral alone; it overlaps with both textually in `noScribe/main.py`, and whichever merges
first, the rest is rebased. It is five commits (engine, integration, tests, tools, docs); keep
that shape when rebuilding. `docs/scripts/` needs `git add -f` there -- upstream's `.gitignore`
has a `scripts/` entry. The PR text lives in `benchmarks-local/voxtral-pr-description.md`.

To refresh the branch after `main` moved: check the Voxtral-only files out of
`main` wholesale, and carry `main.py` changes over by hand -- its Voxtral hunks (the job
extension, `_model_ref`, the menu hint, `_engine_progress`, `end_open_line`) sit between the
other branches' hunks. `tests/test_voxtral_plugin.py` and `tests/test_nemotron_plugin.py` are
one file per branch on `main` too.
`tools/check_local_current.py` must stay green for the branch; its `INTENTIONAL` table lists the
lines the branch deliberately differs on. In the branch's copy of `docs/migration-mlx-audio.md`
three references to files outside the PR (`docs/diarization.md`, `benchmarks-local/`) are removed.

This is its boundary:

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
                          merged_embeddings,salvage_prefix_alignment,word_prob_format,
                          voxtral_transcribe_guards,align_input_level,voxtral_plugin}.py
              noScribe/plugins/local_voxtral/
    modified  noScribe/main.py · noScribe/transcription.py · trans/*.yml (four keys:
              voxtral_path_long, voxtral_path_short, voxtral_no_disfluencies, loading_voxtral)
              noScribe/__init__.py (one word: ctc_align in _SUBMODULES)
              environments/requirements_macOS_arm64.txt (a two-line comment)
              tests/test_worker_import_lightweight.py (the voxtral worker in WORKER_MODULES and the
                                                      comment above it)
              pyinstaller/noScribe_macOS.spec (the MLX stack in `excludes`)
              .gitignore (only the `models/voxtral-*` line)
              noScribe/plugins/factory.py (registers local-voxtral)
              noScribe/inference.py (run_transcription_worker split out of transcribe(),
                                     and no device assumed)
              tests/test_plugins.py (the local-voxtral manifest)

Every branch Voxtral depended on is part of upstream: the three it touched only textually
(speaker-names, header-labels, torch-2.13-stack) and the one hard dependency,
`fix/lazy-main-import` (`_SUBMODULES`).

**On `main` but neither Voxtral nor one of the open small PRs** — keep them out of the
Voxtral PR:

- `noScribe/voice_check.py`, its second call to the diarization worker
  (`_run_voice_embeddings`, the `embed_spans` mode and `_centroids` in `pyannote_mp_worker.py`),
  `check_voices` in `main.py`, `tests/test_voice_{check,rewrite,embeddings}.py` and the
  `voice_check_*` keys in all nine languages are engine-agnostic and have their own branch,
  `feature/voice-verified-speakers`, cut from `upstream-main` and merged into `main`: upstream
  PR #351. It checks each passage's speaker against the voice and replaced the sentence
  split (`split_at_speaker_change`), which moved 83 of 100 words rightly on Whisper segments
  overall but did net harm on some corpora. Its margins are chosen for the fewest wrong words
  afterwards, whichever side caused them -- not for rare broken ones. The module docstring carries the measurement and
  everything that was tried and left out; the scripts and data behind it are in
  `benchmarks-local/whisper-split-ab/`, `benchmarks-local/ami/` and `benchmarks-local/callhome/`.
  `NOSCRIBE_VOICE_CHECK=0` switches it off. Voxtral depends on it in practice: the check
  repairs a misattribution that Voxtral's own short-cue merge causes (`voxtral_engine.py`), so
  the Voxtral PR should follow this one. Its second opinion is pluggable: `relabel()` takes an
  evidence object, `Voice` (pyannote's embeddings against its centroids, what was measured) or
  `Probabilities` (on the Nemotron branch below).
- **Nemotron 3 Diarization** (`noScribe/nemotron_mp_worker.py`, `voice_check.Probabilities`,
  `select_diarization_model` in `main.py`, `noScribe/plugins/local_nemotron/`,
  `loading_nemotron` in all nine languages,
  `tests/test_nemotron_{worker,plugin}.py`) has its own branch, `feature/nemotron-diarization`, stacked on
  `feature/voice-verified-speakers` rather than cut from `upstream-main`, and merged into `main`.
  Proposed upstream in issue #360 (Kai agreed on transformers, no pyannote fallback, 8 speakers
  as the limit); measurements in `benchmarks-local/nemotron/README.md`: fewer wrong speakers
  than pyannote in English and German, ~7x faster on CPU, the voice check on its probabilities
  roughly halves wrong words against today's. It chunks the feature extraction itself because
  transformers' one-pass spectrogram needs ~4 GB per hour of audio; transformers keeps that by
  design (huggingface/transformers#49090), so the worker uses the processor's streaming calls
  for the features, bit-identical, and runs the model offline.
  The worker takes the recording's end from an offline call, because transformers' streaming
  call for a last chunk raises on short ends in a mode without look-ahead, which the worker's
  `(step, 0)` is (numbers in the docstring of `features`). Our fix is
  huggingface/transformers#49213; once a release carries it, the offline call can go.
  The branch is five commits (plugin and worker, integration, requirements, the remote
  plugin's line reader, the server variant) ready for review; its PR text is in
  `benchmarks-local/nemotron/pr-description.md`, and pyannote's removal comes as a separate PR.
  A throwaway frozen build ran it: the specs collect `transformers.models.nemotron3_diarization`
  and `nemotron_asr_streaming` plus torchcodec's metadata. PyInstaller, bisected on macOS: before
  6.14 a frozen transformers fails its own import-structure scan of `__init__.pyc`, before 6.21 a
  frozen scipy 1.18 misses `scipy._cyutility` (6.14) or array_api_compat's `numpy.fft` (6.15-6.20);
  so macOS and Linux ask for >=6.21, and Windows keeps 6.14.1 (its spec collects all of scipy).
  The requirements ask for `transformers>=5.18` since its release (2026-09-30). Not done until
  upstream agrees: removing pyannote (worker, models, `pyannote_fast_embeddings`, requirements,
  specs), dropping the speaker-count setting, and shipping bf16 weights with the OpenMDW licence.
  On a server it runs as on the desktop (the server builds its registry from the same factory):
  `select_diarization_model` lets `diarization_engine` choose among a server's engines, the
  voice check reads the probabilities a server sends, and the server retries Nemotron on its
  CPU. That needed the remote plugin to stop reading events with `iter_lines(chunk_size=1)`,
  quadratic in the line length (200 kB: 27 s; Nemotron's ~7.7 MB per audio hour never
  arrived); `_iter_lines` reads with `read1`. The Docker lock files still lack transformers
  and librosa: no Docker here, so `docker/update-lock.sh` is Kai's to run.
- `noScribe/pyannote_fast_embeddings.py` and its hook in `pyannote_mp_worker.py` (with
  `tests/test_pyannote_fast_embeddings.py`): the
  diarization speed-up that upstream pyannote-audio#2048 supersedes. Fork-only until then.
- `Romy` → `Mona` in `tests/test_utils.py` and `tests/test_apostrophe_fix.py`: an edit to
  upstream's own test data. Stays local.
- (The `_Info` question is settled: since #377 every engine returns `TranscriptionInfo`
  through the plugin, and Voxtral's carries duration and language like Whisper's.)

**These are fork-only and must never appear in that pull request**, however convenient the diff
makes it look:

- **`README.md`** — it carries the fork banner (Voxtral, no packaged download, pointers back to
  kaixxx/noScribe and noscribe.de). It exists for the fork alone. This one is not a judgement
  call; leaving it in would push the fork's advertising into the upstream project.
- **`CLAUDE.md`** — upstream has no such file, and this one describes the fork's own workflow.
- **`tools/check_local_current.py`** — it checks `main` against the *fork's* PR branches;
  upstream has nothing for it to do. Nor for **`tools/upgrade-checks/`**, the fork's own log.
- **The deletion of `environments/marked_for_deletion/`** — that directory is the upstream
  maintainers' own staging area (created 2025-05-29, and the name says they mean to remove it
  themselves). Removing it for them is not this fork's call, so the deletion stays local.
- **`noScribeEdit/`** — the editor source is tracked in this fork for local work only
  (`.gitignore` no longer excludes it). Upstream keeps the editor in
  its own repository, kaixxx/noScribeEditor, and editor changes go there as PRs from
  happyarts/noScribeEditor. `noScribeEdit/Test/` stays ignored: it holds a real recording.
- **`local_build_stamp()` in `noScribe/main.py`** and its call in the transcript header, plus
  `tests/test_local_build_stamp.py` — stamps transcripts with the git commit and the author's
  initials; local provenance only. The PR keeps `t('doc_header', version=app_version)`.
- The `.gitignore` changes for `noScribeEdit/` (upstream ignores the whole directory, the fork
  only its `Test/`, `build/` and `dist/`), `Audiotest/`, `Audiotest2/` and `venv/` — local
  environment; only `models/voxtral-*` is Voxtral's.

## Test suite character

Many tests are regression guards for defects found in production or in upstream libraries rather
than unit tests of new code — forced-align caps and density, loop breaking, lost head
recovery, prefix-salvage alignment. Their docstrings explain the failure they prevent. When a
test looks arbitrary, read its docstring before changing it: several encode a bug that was
expensive to find, and one (`test_forced_align_stability.py`) pins the numpy forced-alignment DP
(`noScribe/ctc_align.py`) to a reference recorded from torchaudio in `tests/data/`.
