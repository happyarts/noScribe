"""Prove that loading the diarization waveform via soundfile is a drop-in
replacement for the previous ``torchaudio.load`` call.

The pyannote worker only ever loads noScribe's own converted audio
(16 kHz mono pcm_s16le WAV, see noScribe/audio/convert.py) and passes it to
the pipeline as an in-memory ``{"waveform": tensor, "sample_rate": int}``
dict.  So the loader just has to produce the exact same tensor -- which this
test asserts bit-for-bit against torchaudio while both libraries are
installed side by side.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

from noScribe.pyannote_mp_worker import load_waveform


@pytest.fixture()
def converted_wav(tmp_path):
    """A WAV exactly like noScribe's conversion step writes: 16 kHz mono PCM16."""
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 3) / 16000.0
    signal = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    path = tmp_path / "converted.wav"
    sf.write(path, np.clip(signal, -1.0, 1.0), 16000, subtype="PCM_16")
    return path


def test_load_waveform_shape_dtype_rate(converted_wav):
    waveform, sample_rate = load_waveform(str(converted_wav))
    assert sample_rate == 16000
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 2 and waveform.shape[0] == 1  # (channels, frames)
    assert waveform.shape[1] == 16000 * 3
    assert waveform.is_contiguous()


def test_load_waveform_bit_identical_to_torchaudio(converted_wav):
    torchaudio = pytest.importorskip("torchaudio")
    expected, expected_rate = torchaudio.load(str(converted_wav))
    actual, actual_rate = load_waveform(str(converted_wav))
    assert actual_rate == expected_rate
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)  # bit-for-bit, not just allclose
