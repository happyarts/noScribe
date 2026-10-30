"""The bundled local-voxtral plugin without its model: what it hands its worker,
what it refuses, and how main.py migrates saved choices, asks for job data and
leaves Voxtral out of the CPU fallback."""
from types import SimpleNamespace

import pytest

from noScribe.inference import DiarizationResult, DiarizationSegment, TranscriptionRequest
from noScribe.models import ModelRef
from noScribe.plugins.local_voxtral import LocalVoxtralPlugin
from noScribe.plugins.local_voxtral.settings import VoxtralWorkerSettings
from noScribe.worker_errors import WorkerError, error_result


class Host:
    """Records what the plugin asks the shared local backend to run."""

    def __init__(self):
        self.runs = []

    def run_transcription_worker(self, entrypoint, args, on_segment, **callbacks):
        self.runs.append((entrypoint.__name__, args, callbacks))
        return 'info'


def voxtral(tmp_path, monkeypatch, settings=None):
    from noScribe import voxtral_engine
    monkeypatch.setattr(voxtral_engine, 'resolve_model', lambda name, repo: f'repo/{name}')
    host = Host()
    return LocalVoxtralPlugin(host, corrections_dir=tmp_path, settings=settings), host


def test_voxtral_hands_its_worker_the_job(tmp_path, monkeypatch):
    """Language, the job data main.py adds (whether times are needed, the names),
    the diarization's turns in seconds, and its own settings -- 0 meaning "the
    engine decides", which the worker expects as None."""
    plugin, host = voxtral(tmp_path, monkeypatch, VoxtralWorkerSettings(chunk_sec=120, ram_reserve_gb=0))
    status = []
    diarization = DiarizationResult((DiarizationSegment(1500, 4200, 'SPEAKER_01'),))
    request = TranscriptionRequest(
        'a.wav', ModelRef('local-voxtral', 'voxtral-mini-8bit'), language='de', include_disfluencies=False,
        diarization=diarization,
        extensions={'noscribe': {'needs_timing': False, 'speaker_names': ['Mona', 'Lena']}})
    assert plugin.transcribe(request, on_segment=print,
                             on_status=lambda *message: status.append(message)) == 'info'
    (name, args, callbacks), = host.runs
    assert name == 'voxtral_proc_entrypoint'
    assert args == {
        'audio_path': 'a.wav', 'language_code': 'de', 'need_timestamps': False,
        'voxtral_repo': 'repo/voxtral-mini-8bit', 'chunk_sec': 120,
        'corrections_path': str(tmp_path / 'voxtral_corrections.yml'), 'speaker_names': ['Mona', 'Lena'],
        'ram_reserve_gb': None, 'speaker_turns': [[1.5, 4.2, 'SPEAKER_01']]}
    assert (tmp_path / 'voxtral_corrections.yml').exists()
    assert status == [('voxtral_path_short', {}, 'debug')]
    assert 'on_status' in callbacks


def test_voxtral_without_job_data_takes_the_safe_path(tmp_path, monkeypatch):
    """A request without main.py's job data (a server, a script) gets times, and
    disfluencies it cannot honour are said, not silently dropped."""
    plugin, host = voxtral(tmp_path, monkeypatch)
    status = []
    plugin.transcribe(TranscriptionRequest('a.wav', ModelRef('local-voxtral', 'voxtral-mini-8bit')),
                      on_segment=print, on_status=lambda *message: status.append(message[0]))
    (_, args, _), = host.runs
    assert args['need_timestamps'] is True and args['speaker_turns'] is None and args['speaker_names'] is None
    assert args['language_code'] is None
    assert status == ['voxtral_path_long', 'voxtral_no_disfluencies']


def test_voxtral_is_offered_on_apple_silicon_only(tmp_path, monkeypatch):
    from noScribe.plugins.local_voxtral import plugin as module
    plugin, _ = voxtral(tmp_path, monkeypatch)
    monkeypatch.setattr(module, '_apple_silicon', lambda: False)
    assert plugin.list_models() == []


def test_voxtral_settings_refuse_what_the_engine_cannot_use():
    assert VoxtralWorkerSettings.from_mapping({'ram_reserve_gb': 4}, VoxtralWorkerSettings(chunk_sec=90)) == \
        VoxtralWorkerSettings(chunk_sec=90, ram_reserve_gb=4)
    for bad in ({'chunk_sec': -1}, {'ram_reserve_gb': 'high'}, {'device': 'cuda'}):
        with pytest.raises(ValueError):
            VoxtralWorkerSettings.from_mapping(bad)


def test_the_registry_reads_voxtrals_settings_file(tmp_path):
    from noScribe.inference import LocalWorkerSettings
    from noScribe.plugins.factory import create_builtin_registry
    (tmp_path / 'plugin-settings').mkdir()
    (tmp_path / 'plugin-settings' / 'local-voxtral.yml').write_text('chunk_sec: 60\n')
    registry = create_builtin_registry({}, LocalWorkerSettings(), tmp_path / 'plugin-settings',
                                       voxtral_defaults=VoxtralWorkerSettings(ram_reserve_gb=6))
    assert registry.get('local-voxtral').settings == VoxtralWorkerSettings(chunk_sec=60, ram_reserve_gb=6)


def test_voxtral_does_not_retry_on_the_cpu(monkeypatch):
    """Voxtral runs on MLX; a "retry Whisper on the CPU" would only repeat the failure."""
    m = pytest.importorskip('noScribe.main')
    logged = []
    registry = SimpleNamespace(get=lambda backend_id: SimpleNamespace(settings=SimpleNamespace(device='auto')))
    app = SimpleNamespace(cancel=False, inference_backend=registry,
                          logn=lambda text='', *args, **kwargs: logged.append(text))
    job = m.TranscriptionJob()
    job.transcription_model = ModelRef('local-voxtral', 'voxtral-mini-8bit')
    error = WorkerError(error_result(RuntimeError('CUDA out of memory'), 'cuda', 'transcription'))
    assert m.App._handle_cuda_fallback(app, 'whisper', error, job) is False
    assert not job.whisper_cpu_fallback and logged == []


def test_job_data_tells_whether_the_output_needs_times():
    m = pytest.importorskip('noScribe.main')

    def needs(**options):
        options = {'speaker_detection': 'none', 'pause': 'none', 'timestamps': False, **options}
        job = m.create_transcription_job(transcript_file='x.txt', **options)
        return m._job_extension(job)['noscribe']['needs_timing']

    assert needs() is False
    assert needs(timestamps=True) is True
    assert needs(speaker_detection='auto') is True
    assert needs(pause='1sec+') is True
    job = m.create_transcription_job(transcript_file='x.html', speaker_names='Mona, Lena')
    assert m._job_extension(job) == {'noscribe': {'needs_timing': True, 'speaker_names': ['Mona', 'Lena']}}


def test_the_shared_worker_loop_runs_voxtrals_arguments(tmp_path, monkeypatch):
    """The plugins run their workers through LocalInferenceBackend's own loop,
    which was written for Whisper's arguments: Voxtral has no device setting, and
    a loop that read worker_args['device'] failed every Voxtral job before the
    worker started. Here the real loop runs the real arguments, with a worker
    that answers in a thread instead of a spawned process."""
    import queue
    import threading
    from noScribe import inference, voxtral_mp_worker
    from noScribe.inference import LocalInferenceBackend

    class Process:
        def __init__(self, target, args):
            self.thread = threading.Thread(target=target, args=args, daemon=True)
            self.exitcode = 0

        def start(self):
            self.thread.start()

        def is_alive(self):
            return self.thread.is_alive()

        def join(self, timeout=None):
            self.thread.join(timeout)

        def terminate(self):
            pass

        def close(self):
            pass

    class Queue(queue.Queue):
        def close(self):
            pass

        def join_thread(self):
            pass

    seen = []

    def worker(args, q):
        seen.append(args)
        q.put({"type": "progress", "pct": 50})
        q.put({"type": "segment", "segment": {"start": 0.0, "end": 1.0, "text": " Hallo", "words": []}})
        q.put({"type": "result", "ok": True, "info": {"duration": 1.0, "language": "de"}})

    monkeypatch.setattr(inference.mp, "get_context", lambda method: SimpleNamespace(Queue=Queue, Process=Process))
    monkeypatch.setattr(voxtral_mp_worker, "voxtral_proc_entrypoint", worker)
    from noScribe import voxtral_engine
    monkeypatch.setattr(voxtral_engine, "resolve_model", lambda name, repo: f"repo/{name}")
    plugin = LocalVoxtralPlugin(LocalInferenceBackend(), corrections_dir=tmp_path)
    segments, progress = [], []
    info = plugin.transcribe(TranscriptionRequest("a.wav", ModelRef("local-voxtral", "voxtral-mini-8bit")),
                             on_segment=segments.append, on_progress=lambda pct, detail: progress.append(pct))
    assert "device" not in seen[0]
    assert [segment.text for segment in segments] == [" Hallo"] and progress == [50.0]
    assert info.language == "de" and info.duration == 1.0
