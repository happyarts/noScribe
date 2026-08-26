"""Guards for the per-chunk embedding fast path (pyannote_fast_embeddings).

pyannote's stock ``get_embeddings`` runs the full embedding model once per
(chunk x speaker slot); the fast path runs it once per chunk with the
speaker masks stacked. That is only valid because the WeSpeaker mask enters
at the pooling stage -- these tests pin the equivalence against the stock
implementation on a randomly initialised model, so a pyannote upgrade that
moves the mask earlier (as the SpeechBrain/NeMo/ONNX backends do) or
reorders the mask-selection logic fails loudly here instead of silently
degrading diarization quality. The install gate is additionally exercised
against the real bundled pipeline: its failure mode is the silent kind (the
stock path quietly returns, 2.2x slower), which no runtime error would ever
surface.
"""
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("pyannote.audio")

from pyannote.audio.core.io import Audio
from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.pipelines.speaker_verification import (
    PyannoteAudioPretrainedSpeakerEmbedding,
)
from pyannote.core import SlidingWindow, SlidingWindowFeature

from noScribe.pyannote_fast_embeddings import (
    fast_get_embeddings,
    install_fast_embeddings,
)

REPO = Path(__file__).resolve().parent.parent


def _pipeline_stub(embedding):
    """Just the attributes get_embeddings actually touches."""
    return types.SimpleNamespace(
        training=False,
        _embedding=embedding,
        _audio=Audio(sample_rate=16000, mono="downmix"),
        embedding_batch_size=4,
    )


@pytest.fixture(scope="module")
def wespeaker_wrapper():
    """A randomly initialised WeSpeaker model behind the real pipeline
    wrapper -- the equivalence claim is about arithmetic, not about trained
    weights, so no model download is needed and this runs in CI on CPU."""
    torch.manual_seed(0)
    model = WeSpeakerResNet34()
    model.eval()
    return PyannoteAudioPretrainedSpeakerEmbedding(
        model, device=torch.device("cpu")
    )


@pytest.fixture(scope="module")
def test_inputs():
    """5 chunks x 20 mask frames x 3 speakers over 6 s of noise, with the
    awkward cases the stock code handles: a fully inactive speaker slot,
    NaN frames from partial stitching, and (via exclude_overlap) chunks
    whose overlap-free speech is too short so the full mask must be used."""
    file = {"waveform": torch.randn(1, 16000 * 6), "sample_rate": 16000}
    rng = np.random.default_rng(1)
    data = (rng.random((5, 20, 3)) > 0.5).astype(float)
    data[0, :, 2] = 0.0
    data[1, 3:5, 0] = np.nan
    binseg = SlidingWindowFeature(
        data, SlidingWindow(start=0.0, duration=2.0, step=1.0)
    )
    return file, binseg


@pytest.mark.parametrize("exclude_overlap", [False, True])
def test_fast_path_matches_stock(wespeaker_wrapper, test_inputs,
                                 exclude_overlap):
    file, binseg = test_inputs
    stub = _pipeline_stub(wespeaker_wrapper)
    expected = SpeakerDiarization.get_embeddings(
        stub, file, binseg, exclude_overlap=exclude_overlap
    )
    actual = fast_get_embeddings(
        stub, file, binseg, exclude_overlap=exclude_overlap
    )
    assert actual.shape == expected.shape == (5, 3, 256)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_install_engages_on_the_bundled_pipeline():
    """The gate's failure mode is silent (install returns False, stock path
    runs, diarization is quietly 2.2x slower), so it must be exercised
    against the real Pipeline class -- including pyannote's __setattr__
    interception and the .to(device) the worker calls after install."""
    from pyannote.audio import Pipeline

    config = REPO / "pyannote" / "config.yaml"
    if not config.exists():
        pytest.skip("bundled pyannote pipeline not present")
    pipeline = Pipeline.from_pretrained(REPO / "pyannote")
    assert install_fast_embeddings(pipeline) is True
    assert pipeline.get_embeddings.__func__ is fast_get_embeddings
    pipeline.to(torch.device("cpu"))
    assert pipeline.get_embeddings.__func__ is fast_get_embeddings
    # installing twice rebinds the same plain function, no wrapping
    assert install_fast_embeddings(pipeline) is True
    assert pipeline.get_embeddings.__func__ is fast_get_embeddings


def test_install_refuses_unsafe_pipelines(wespeaker_wrapper):
    """Every gate has a silent-decline reason; each must actually decline.
    SpeechBrain/NeMo/ONNX backends apply masks before the trunk, a
    two_emb_layer head crashes BatchNorm1d on (batch, speakers, dim), and
    the training path carries an embedding cache the fast path does not."""
    # not a SpeakerDiarization at all (stubs included)
    stub = _pipeline_stub(wespeaker_wrapper)
    assert install_fast_embeddings(stub) is False
    assert "get_embeddings" not in vars(stub)

    from pyannote.audio import Pipeline

    config = REPO / "pyannote" / "config.yaml"
    if not config.exists():
        pytest.skip("bundled pyannote pipeline not present")

    # wrong backend type
    pipeline = Pipeline.from_pretrained(REPO / "pyannote")
    real_embedding = pipeline._embedding
    object.__setattr__(pipeline, "_embedding", object())
    assert install_fast_embeddings(pipeline) is False
    assert "get_embeddings" not in pipeline.__dict__
    object.__setattr__(pipeline, "_embedding", real_embedding)

    # WeSpeaker variant whose head is not pooling-transparent
    resnet = pipeline._embedding.model_.resnet
    assert resnet.two_emb_layer is False, "bundled model changed shape"
    resnet.two_emb_layer = True
    try:
        assert install_fast_embeddings(pipeline) is False
    finally:
        resnet.two_emb_layer = False

    # training pipeline (embedding cache not carried by the fast path)
    pipeline.training = True
    try:
        assert install_fast_embeddings(pipeline) is False
    finally:
        pipeline.training = False


def test_progress_hook_still_reaches_100_percent():
    """main.py drives its progress bar from completed/total; the fast path
    reports fewer, larger batches, which is fine -- but it must still end at
    completed == total or the bar sticks below 100 %. A stub backend keeps
    this a millisecond test."""

    class FakeBackend:
        def __call__(self, waveforms, masks=None):
            return np.zeros(
                (waveforms.shape[0], masks.shape[1], 7), dtype=np.float32
            )

    file = {"waveform": torch.randn(1, 16000 * 6), "sample_rate": 16000}
    binseg = SlidingWindowFeature(
        np.ones((5, 20, 3)), SlidingWindow(start=0.0, duration=2.0, step=1.0)
    )
    calls = []

    def hook(step_name, artifact, file=None, total=None, completed=None):
        calls.append((step_name, total, completed))

    out = fast_get_embeddings(
        _pipeline_stub(FakeBackend()), file, binseg, hook=hook
    )
    assert out.shape == (5, 3, 7)
    assert calls and all(name == "embeddings" for name, _, _ in calls)
    _, total, completed = calls[-1]
    assert completed == total


def test_runtime_failure_falls_back_to_stock(wespeaker_wrapper, test_inputs):
    """The fast path promises 'at worst slow, never wrong': if the grouped
    call breaks mid-run (an unpinned pyannote upgrade), the stock result
    must come back instead of an aborted diarization."""

    class BrokenBackend:
        calls = 0

        def __call__(self, waveforms, masks=None):
            if masks is not None and masks.ndim == 3:
                raise RuntimeError("grouped call broken")
            # stock's per-pair path: 1-D-per-item masks stacked to 2-D
            return wespeaker_wrapper(waveforms, masks=masks)

    file, binseg = test_inputs
    stub = _pipeline_stub(BrokenBackend())
    with pytest.warns(RuntimeWarning, match="falling back"):
        out = fast_get_embeddings(stub, file, binseg)
    expected = SpeakerDiarization.get_embeddings(
        _pipeline_stub(wespeaker_wrapper), file, binseg
    )
    np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-4)
