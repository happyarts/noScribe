"""With speaker detection, Whisper takes its speech map from the diarization.

Silero dropped a quiet voice before Whisper could hear it; the diarization's
turns keep it (the measurement is above SPEECH_MAP_MIN_GAP_S in
noScribe/whisper_mp_worker.py). The swap works through the one name
faster-whisper imports into its transcribe module, so the guard that matters
most is that it still looks the map up there: a faster-whisper release that
stops doing so would leave the feature silently dead.
"""
import numpy as np
import pytest

from noScribe import whisper_mp_worker as w

SR = 16000


def test_turns_become_padded_chunks_with_short_gaps_closed():
    turns = [[1.0, 2.0], [2.3, 3.0], [5.0, 6.0]]   # 0.3 s gap closed, 2 s gap kept
    got = w.speech_map(turns, 10 * SR)
    pad = int(w.SPEECH_MAP_PAD_S * SR)
    assert got == [{'start': SR - pad, 'end': 3 * SR + pad}, {'start': 5 * SR - pad, 'end': 6 * SR + pad}]


def test_chunks_stay_inside_the_audio_and_empty_turns_vanish():
    got = w.speech_map([[-1.0, 0.5], [3.0, 3.0], [9.8, 12.0]], 10 * SR)
    assert got[0]['start'] == 0 and got[-1]['end'] == 10 * SR
    assert len(got) == 2


def test_the_gap_to_close_is_the_constant():
    turns = [[1.0, 2.0], [2.7, 3.0]]   # 0.6 s gap, 0.5 s after padding
    assert len(w.speech_map(turns, 10 * SR)) == 2
    pytest.MonkeyPatch.context  # noqa -- the monkeypatch fixture below varies it


def test_a_longer_min_gap_merges_what_the_shorter_keeps(monkeypatch):
    turns = [[1.0, 2.0], [2.7, 3.0]]
    monkeypatch.setattr(w, "SPEECH_MAP_MIN_GAP_S", 1.0)
    assert len(w.speech_map(turns, 10 * SR)) == 1


ft = pytest.importorskip("faster_whisper.transcribe")


def test_faster_whisper_still_asks_its_module_for_the_speech_map():
    """transcribe() and detect_language() must look get_speech_timestamps up as a
    module global -- the name _speech_map_from replaces."""
    assert hasattr(ft, "get_speech_timestamps")
    for fn in (ft.WhisperModel.transcribe, ft.WhisperModel.detect_language):
        assert "get_speech_timestamps" in fn.__code__.co_names, fn.__qualname__


def test_the_swap_answers_with_the_turns_and_is_undone():
    silero = ft.get_speech_timestamps
    audio = np.zeros(10 * SR, np.float32)
    with w._speech_map_from([[1.0, 2.0]]) as mapped:
        assert mapped
        assert ft.get_speech_timestamps(audio, None) == w.speech_map([[1.0, 2.0]], len(audio))
    assert ft.get_speech_timestamps is silero


def test_the_swap_is_undone_when_the_call_inside_fails():
    silero = ft.get_speech_timestamps
    with pytest.raises(RuntimeError):
        with w._speech_map_from([[1.0, 2.0]]):
            raise RuntimeError("transcription failed")
    assert ft.get_speech_timestamps is silero


@pytest.mark.parametrize("turns", [None, []])
def test_without_turns_silero_stays(turns):
    silero = ft.get_speech_timestamps
    with w._speech_map_from(turns) as mapped:
        assert not mapped and ft.get_speech_timestamps is silero


def test_main_hands_the_diarization_to_the_whisper_worker(tmp_path, monkeypatch):
    pytest.importorskip("tkinter")
    import noScribe.main as m
    monkeypatch.setattr(m, "config", {})
    app = m.HeadlessApp()
    seen = {}
    monkeypatch.setattr(app, "_run_engine_subprocess_stream", lambda entry, args, *a, **k: seen.update(args))
    job = m.create_transcription_job(audio_file=str(tmp_path / "a.wav"), transcript_file=str(tmp_path / "a.html"),
                                     speaker_detection="auto", cli_mode=True)
    app._run_whisper_subprocess_stream("x.wav", job, None,
                                       [{"start": 1500, "end": 4200, "label": "SPEAKER_00"}])
    assert seen["speech_turns"] == [[1.5, 4.2]]
    app._run_whisper_subprocess_stream("x.wav", job, None)
    assert seen["speech_turns"] is None


def test_the_worker_transcribes_inside_the_swap(tmp_path, monkeypatch):
    """The wiring itself: whisper_proc_entrypoint must hand faster-whisper the turns'
    map while transcribe() runs, and while detect_language() runs for 'Auto' --
    a stand-in model asks for it in both."""
    import queue, types
    import faster_whisper
    import soundfile as sf
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(10 * SR, np.int16), SR)
    asked = []

    class FakeModel:
        def __init__(self, *a, **k):
            self.model = types.SimpleNamespace(is_multilingual=True)
            self.feature_extractor = types.SimpleNamespace(sampling_rate=SR)

        def detect_language(self, audio, **k):
            asked.append(("detect", ft.get_speech_timestamps(np.zeros(10 * SR, np.float32), k.get("vad_parameters"))))
            return "de", 1.0, []

        def transcribe(self, audio, **k):
            asked.append(ft.get_speech_timestamps(np.zeros(10 * SR, np.float32), k.get("vad_parameters")))
            return iter([]), types.SimpleNamespace(duration=10.0, language="de", language_probability=1.0)

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    want = w.speech_map([[1.0, 2.0]], 10 * SR)
    for language, expected in (("German", [want]), ("Auto", [("detect", want), want])):
        asked.clear()
        q = queue.Queue()
        w.whisper_proc_entrypoint({
            "whisper_model": types.SimpleNamespace(path=tmp_path), "device": "cpu", "audio_path": str(wav),
            "language_name": language, "language_code": "de", "locale": "en",
            "speech_turns": [[1.0, 2.0]]}, q)
        results = []
        while not q.empty():
            m = q.get_nowait()
            if m["type"] == "result":
                results.append(m)
        assert results and results[-1]["ok"], results
        assert asked == expected, language


@pytest.mark.parametrize("engine, handed", [("pyannote", True), ("nemotron", False)])
def test_only_pyannotes_turns_reach_whisper(tmp_path, monkeypatch, engine, handed):
    """Nemotron's many short turns cost words as Whisper's speech map on German
    (see main.py where Whisper is started), so its diarization is not handed on."""
    pytest.importorskip("tkinter")
    import soundfile as sf
    import noScribe.main as m
    monkeypatch.setattr(m, "config_dir", str(tmp_path))
    monkeypatch.setattr(m, "config", {})
    audio = tmp_path / "a.wav"
    sf.write(audio, np.zeros(SR * 3, dtype=np.int16), SR)
    app = m.HeadlessApp()
    turns = [{"start": 0, "end": 2000, "label": "SPEAKER_00"}]
    monkeypatch.setattr(app, "_diarization_engine", lambda job: engine)
    monkeypatch.setattr(app, "_run_diarize_subprocess", lambda *a, **k: turns)
    seen = []
    monkeypatch.setattr(app, "_run_whisper_subprocess_stream", lambda *args: seen.append(args))
    job = m.create_transcription_job(audio_file=str(audio), transcript_file=str(tmp_path / "a.html"),
                                     speaker_detection="auto", cli_mode=True)
    app._process_single_job(job)
    assert seen, "Whisper was not started"
    assert (seen[0][3] == turns) == handed
