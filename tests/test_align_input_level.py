"""The aligner sees each window at zero mean and unit variance, whatever the level.

Found on a voice 17 dB below its neighbour: fed the raw signal, the aligner
stamped its words 10-20 s early, into the louder speaker's time. Mechanism and
measurement are in the comment in _Aligner._forward_windows.

The model here is a stand-in whose logits are a plain function of its input,
so any level that reaches it shows up in the emissions.
"""
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from noScribe.voxtral_engine import EMISSION_WINDOW_SEC, SAMPLE_RATE, _Aligner  # noqa: E402

FRAME = 320  # samples per emission frame, as wav2vec2's 20 ms stride
WIN, MIN_WIN = int(EMISSION_WINDOW_SEC * SAMPLE_RATE), int(0.025 * SAMPLE_RATE)  # as _emission passes them


class _LevelSensitiveModel:
    """logits[t] = [mean, rms, max] of frame t -- no normalisation of its own."""

    def __init__(self):
        self.seen = []

    def __call__(self, batch):
        x = batch[0]
        self.seen.append(x.clone())
        frames = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
        logits = torch.stack([frames.mean(1), frames.pow(2).mean(1).sqrt(), frames.max(1).values], 1)
        return types.SimpleNamespace(logits=logits.unsqueeze(0))


def _aligner():
    al = object.__new__(_Aligner)
    al._torch = torch
    al.device = "cpu"
    al.model = _LevelSensitiveModel()
    return al


def _speech_like(seconds, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    envelope = 0.5 + 0.5 * np.sin(np.linspace(0, 40 * np.pi, n)) ** 2
    return (rng.standard_normal(n) * envelope * 0.1).astype(np.float32)


@pytest.mark.parametrize("gain", [0.14, 0.3, 3.0])  # 0.14 = the 17 dB of the finding
def test_emissions_do_not_depend_on_the_recording_level(gain):
    """Equal up to the extractor's 1e-7 epsilon, which at 17 dB down is ~0.1 % of
    the variance (~2e-3 in these emissions); without the normalisation the gap is
    ~0.4, so the tolerance still tells the two apart."""
    audio = _speech_like(45)  # three windows at 20 s, the last one short
    ref = _aligner()._forward_windows(torch.from_numpy(audio), WIN, MIN_WIN)
    got = _aligner()._forward_windows(torch.from_numpy(audio * gain), WIN, MIN_WIN)
    assert len(ref) == len(got) == 3
    for a, b in zip(ref, got):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-2)


def _extractor_norm(x):
    """Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm without padding, in numpy."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / np.sqrt(x.var() + 1e-7)


def test_every_window_reaches_the_model_normalised_on_its_own():
    """Per window, not per call: a loud window must not set the scale of a quiet one,
    a DC offset is removed, and a very quiet window keeps the extractor's epsilon."""
    loud, quiet = _speech_like(20, seed=1), _speech_like(20, seed=2) * 0.005 + 0.3
    al = _aligner()
    al._forward_windows(torch.from_numpy(np.concatenate([loud, quiet])), WIN, MIN_WIN)
    assert len(al.model.seen) == 2
    for x, src in zip(al.model.seen, (loud, quiet)):
        np.testing.assert_allclose(x.numpy(), _extractor_norm(src), atol=2e-4)


def test_matches_the_feature_extractor_itself():
    transformers = pytest.importorskip("transformers")
    x = _speech_like(7, seed=3) * 0.02 + 0.01
    al = _aligner()
    al._forward_windows(torch.from_numpy(x), WIN, MIN_WIN)
    want = transformers.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm([x], None)[0]
    np.testing.assert_allclose(al.model.seen[0].numpy(), want, atol=2e-4)


def test_a_short_rest_takes_the_scale_of_the_window_that_ends_with_it():
    """The rest after the last full window must not be scaled on its own: 0.3 s of
    room tone after speech would otherwise reach the model at unit variance."""
    speech = _speech_like(20, seed=4)
    tone = (np.random.default_rng(5).standard_normal(int(0.3 * SAMPLE_RATE)) * 1e-3).astype(np.float32)
    x = np.concatenate([speech, tone])
    al = _aligner()
    al._forward_windows(torch.from_numpy(x), WIN, MIN_WIN)
    assert len(al.model.seen) == 2
    ref = x[len(x) - WIN:]
    want = (tone - ref.astype(np.float64).mean()) / np.sqrt(ref.astype(np.float64).var() + 1e-7)
    np.testing.assert_allclose(al.model.seen[1].numpy(), want, atol=2e-4)
    assert float(al.model.seen[1].std()) < 0.05  # stays room tone, not speech


def test_silence_does_not_blow_up():
    """An all-zero window has no variance; the epsilon keeps it finite."""
    al = _aligner()
    out = al._forward_windows(torch.zeros(5 * SAMPLE_RATE), WIN, MIN_WIN)
    assert np.isfinite(out[0].numpy()).all()
