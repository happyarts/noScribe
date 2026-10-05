"""noScribe.nemotron_mp_worker without the model: the turns it hands to main.py,
the features it builds, and when `auto` may choose it."""
from pathlib import Path

import numpy as np
import pytest

from noScribe import nemotron_mp_worker as nw


def test_turns_are_noscribe_turns_sorted_by_start():
    """main.py's overlap assignment stops scanning at the first turn that starts
    after a segment, so the turns must come sorted -- and a turn of no length
    would only be noise in the log."""
    got = nw.turns([{'Start': 3.2, 'End': 4.0, 'Speaker': 1}, {'Start': 0.0, 'End': 2.5, 'Speaker': 0},
                    {'Start': 5.0, 'End': 5.0, 'Speaker': 1}])
    assert got == [{'start': 0, 'end': 2500, 'label': 'SPEAKER_00'},
                   {'start': 3200, 'end': 4000, 'label': 'SPEAKER_01'}]


def test_features_in_chunks_equal_one_pass():
    """The worker computes the spectrogram in chunks through the processor's streaming
    calls, which must not change a single value, nor the mask that leaves the last
    frame out, whatever the chunk size and length -- including the last valid frame,
    which the streamed chunks leave out and a separate call has to supply."""
    module = pytest.importorskip(
        'transformers.models.nemotron3_diarization.processing_nemotron3_diarization')
    from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
        NemotronAsrStreamingFeatureExtractor)
    fe = NemotronAsrStreamingFeatureExtractor(
        feature_size=128, hop_length=160, n_fft=512, win_length=400, preemphasis=0.97, sampling_rate=16000)
    processor = module.Nemotron3DiarizationProcessor(fe)
    rng = np.random.default_rng(0)
    for length in (16000 * 3, 16000 * 3 + 77, 16000 * 3 + 160, 16000 * 3 + 350):  # the last: under one window left
        audio = (rng.standard_normal(length) * 0.05).astype(np.float32)
        whole = processor(audio, sampling_rate=16000)
        for chunk_frames in (8, 56, 800, 80_000):
            got = nw.features(processor, audio, 16000, chunk_frames=chunk_frames)
            assert got.input_features.shape == whole.input_features.shape
            assert (got.input_features == whole.input_features).all(), (length, chunk_frames)
            assert (got.attention_mask == whole.attention_mask.long()).all()


def test_auto_needs_the_model_in_transformers_and_its_weights_on_disk(monkeypatch, tmp_path):
    """main.py picks the engine in the GUI process, which must not import transformers
    for it, and `auto` must never start a download."""
    import importlib.util
    import subprocess
    import sys
    import types
    import huggingface_hub
    code = ('import sys; from noScribe.nemotron_mp_worker import available; available(); '
            'assert "transformers" not in sys.modules')
    subprocess.run([sys.executable, '-c', code], check=True, cwd=Path(__file__).resolve().parent.parent)

    (tmp_path / 'models' / 'nemotron3_diarization').mkdir(parents=True)
    knows = types.SimpleNamespace(origin=str(tmp_path / '__init__.py'))
    older = types.SimpleNamespace(origin=str(tmp_path / 'models' / '__init__.py'))
    monkeypatch.setattr(nw, 'model_source', lambda: nw.MODEL_REPO)
    whole = {'config.json', 'processor_config.json', 'model.safetensors'}
    for spec, cached, expected in ((knows, whole, True), (knows, set(), False), (knows, {'model.safetensors'}, False),
                                   (older, whole, False), (None, whole, False)):
        monkeypatch.setattr(importlib.util, 'find_spec', lambda name, spec=spec: spec)
        monkeypatch.setattr(huggingface_hub, 'try_to_load_from_cache',
                            lambda repo, name, cached=cached: f'/cache/{name}' if name in cached else None)
        assert nw.available() is expected, (spec, cached)


@pytest.mark.parametrize("inherited", [None, "0", "false"])
def test_hub_telemetry_is_off_before_transformers_is_imported(tmp_path, inherited):
    """huggingface_hub reads HF_HUB_DISABLE_TELEMETRY once, at import, so the
    worker has to set it before transformers pulls the hub in. A stand-in
    transformers reports the value it was imported under through the worker's
    own error path (the pattern of test_pyannote_font_cache)."""
    import os
    import subprocess
    import sys
    import textwrap
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text("def set_num_threads(n): pass\n")
    (tmp_path / "transformers").mkdir()
    (tmp_path / "transformers" / "__init__.py").write_text(
        'import os\nraise RuntimeError("telemetry=%r" % os.environ.get("HF_HUB_DISABLE_TELEMETRY"))\n')
    probe = textwrap.dedent("""
        import queue
        from noScribe.nemotron_mp_worker import nemotron_proc_entrypoint
        q = queue.Queue()
        nemotron_proc_entrypoint({"audio_path": "unused.wav"}, q)
        while True:
            message = q.get_nowait()
            if message["type"] == "result":
                print(message["error"])
                break
    """)
    repo = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), repo])}
    env.pop("HF_HUB_DISABLE_TELEMETRY", None)
    if inherited is not None:
        env["HF_HUB_DISABLE_TELEMETRY"] = inherited
    proc = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "telemetry='1'" in proc.stdout, proc.stdout
