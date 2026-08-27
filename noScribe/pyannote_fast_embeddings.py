"""Compute the speaker-embedding trunk once per chunk instead of once per
(chunk, speaker) pair.

pyannote's ``SpeakerDiarization.get_embeddings`` iterates (chunk x speaker
slot) and runs the full embedding model for every pair, so with the
community-1 segmentation model (3 local speaker slots) the same 10-second
waveform passes through the WeSpeaker ResNet three times. With the
pyannote-native WeSpeaker backend the speaker mask only enters at the
statistics-pooling stage (``ResNet.forward_embedding``); the convolutional
trunk -- which is nearly all of the compute -- sees identical input each
time. The fast path below stacks each chunk's speaker masks into one
``(speakers, frames)`` weight tensor and makes a single model call per
chunk, which is the same arithmetic in a different grouping: ``StatsPool``
executes the identical ``_pool`` per speaker whether the weights arrive
flat or stacked, and the wrapper's ``__call__`` passes 3-D weights through
untouched. Two deliberate contract changes against stock: the progress hook
sees ~3x fewer, larger batches (its artifact is now (batch, speakers,
dimension)), and the training-time embedding cache is not carried --
``install_fast_embeddings`` declines training pipelines instead.

Measured 2026-08-26/27 on the M1 Max: embeddings step 9.0 s -> 4.0 s on
bench_4min.wav (max abs embedding difference 2.3e-6), and on the VoxConverse
dev set (216 files, 20.3 h) diarization end to end 47.1 -> 21.4 min at a
DER identical on every single file (7.18 %), with both automatic and fixed
speaker counts. The win is platform-neutral -- it removes work, so CUDA and
CPU-only machines (where embeddings are ~20x slower than MPS) benefit at
least as much.

The redundancy is architectural upstream, not an oversight that generalises:
the generic pipeline also serves embedding backends that apply masks to the
waveform *before* the trunk (SpeechBrain, NeMo, ONNX -- see
``pipelines/speaker_verification.py``), where per-speaker trunk inputs
really differ. ``install_fast_embeddings`` therefore binds the fast path
onto the one pipeline *instance* the worker builds, and only after checking
every assumption the grouping rests on -- any other configuration keeps the
stock method, and a failure at run time falls back to it, so this can at
worst be slow, never wrong. Patching the instance rather than the class
follows the house precedent of ``_PercentileFloorFeatures`` in
``voxtral_engine`` (swap a component on the object we own, leave the
library untouched).
"""
import inspect
import math
import types
import warnings


def install_fast_embeddings(pipeline) -> bool:
    """Bind the per-chunk fast path onto this pipeline instance.

    Returns True when the fast path will actually run. Returns False -- and
    leaves the stock, slow-but-correct ``get_embeddings`` in place -- when
    any assumption the grouping rests on does not hold on this pipeline:
    a backend that does not demonstrably compute per-speaker embeddings
    from stacked masks (probed, see ``_accepts_stacked_masks``), a training
    pipeline (whose embedding cache the fast path does not carry), a
    changed method signature, or an installation that pyannote's attribute
    machinery did not accept.
    """
    try:
        from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
        from pyannote.audio.pipelines.speaker_verification import (
            PyannoteAudioPretrainedSpeakerEmbedding,
        )
    except ImportError:
        return False

    if not isinstance(pipeline, SpeakerDiarization):
        return False
    if getattr(pipeline, "training", False):
        return False

    embedding = getattr(pipeline, "_embedding", None)
    if not isinstance(embedding, PyannoteAudioPretrainedSpeakerEmbedding):
        return False
    if not _accepts_stacked_masks(embedding):
        return False

    # If upstream adds or renames a parameter, apply() will call with it;
    # decline rather than crash at that call site. Compare names only --
    # the stock method annotates its parameters, ours does not.
    stock = SpeakerDiarization.get_embeddings
    if list(inspect.signature(stock).parameters) != list(
        inspect.signature(fast_get_embeddings).parameters
    ):
        return False

    # pyannote's Pipeline.__setattr__ intercepts Parameter, sub-Pipeline,
    # nn.Module and BaseInference values; a bound method is none of those
    # and falls through to object.__setattr__, where the instance attribute
    # wins the self.get_embeddings lookup in apply(). Verify rather than
    # assume -- a future interception branch must yield False here, not a
    # silent revert to the slow path.
    pipeline.get_embeddings = types.MethodType(fast_get_embeddings, pipeline)
    if getattr(pipeline.get_embeddings, "__func__", None) is not fast_get_embeddings:
        pipeline.__dict__.pop("get_embeddings", None)
        return False
    return True


def _accepts_stacked_masks(embedding) -> bool:
    """Probe whether the backend computes per-speaker embeddings from one
    call with (batch, speakers, frames) masks.

    Probing the actual call (the approach of pyannote-audio#2048) is more
    robust than deriving the capability from model classes and constructor
    flags: any head that cannot take the extra speaker dimension simply
    fails or returns the wrong shape here. On top of the shape check, the
    stacked result must match per-speaker calls on the same input -- that
    pins the semantics, not just the container. Costs two small forwards
    over one second of noise, once per worker process.
    """
    import numpy as np
    import torch

    try:
        rng = torch.Generator().manual_seed(0)
        waveform = torch.randn(1, 1, embedding.sample_rate, generator=rng)
        masks = (torch.rand(2, 16, generator=rng) > 0.5).float()
        masks[:, :4] = 1.0  # no empty mask: empty ones pool to zeros anyway

        stacked = embedding(waveform, masks=masks[None])
        per_speaker = embedding(
            torch.vstack([waveform, waveform]), masks=masks
        )
    except Exception:
        return False

    if getattr(stacked, "shape", None) != (1, 2, embedding.dimension):
        return False
    return bool(np.allclose(stacked[0], per_speaker, atol=1e-4, rtol=1e-3))


def fast_get_embeddings(
    self, file, binary_segmentations, exclude_overlap=False, hook=None
):
    """Drop-in for ``SpeakerDiarization.get_embeddings`` on inference
    pipelines, one model call per chunk. Mirrors the stock mask selection,
    NaN handling and padding -- the equivalence is pinned by
    tests/test_pyannote_fast_embeddings.py -- and falls back to the stock
    implementation if anything raises mid-run.
    """
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

    try:
        return _fast_get_embeddings(
            self, file, binary_segmentations, exclude_overlap=exclude_overlap,
            hook=hook,
        )
    except Exception as e:
        # A slow result beats a dead pipeline: an upstream change that
        # breaks the grouping mid-run (unpinned pyannote) surfaces as this
        # warning on the worker's stderr, not as a failed transcription.
        warnings.warn(
            f"noScribe fast embedding path failed ({type(e).__name__}: {e}); "
            f"falling back to the stock implementation",
            RuntimeWarning,
        )
        return SpeakerDiarization.get_embeddings(
            self, file, binary_segmentations, exclude_overlap=exclude_overlap,
            hook=hook,
        )


def _fast_get_embeddings(self, file, binary_segmentations, exclude_overlap, hook):
    import numpy as np
    import torch
    from pyannote.audio.pipelines.speaker_diarization import batchify

    # From here on this mirrors the stock implementation -- only the
    # grouping differs: one iteration per chunk, all of the chunk's speaker
    # masks stacked into a single (speakers, frames) weight tensor.
    duration = binary_segmentations.sliding_window.duration
    num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

    if exclude_overlap:
        min_num_samples = self._embedding.min_num_samples
        num_samples = duration * self._embedding.sample_rate
        min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
        clean_frames = 1.0 * (
            np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
        )
        clean_data = binary_segmentations.data * clean_frames
    else:
        min_num_frames = -1
        clean_data = binary_segmentations.data

    def iter_waveform_and_masks():
        for c, (chunk, masks) in enumerate(binary_segmentations):
            waveform, _ = self._audio.crop(file, chunk, mode="pad")
            masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
            clean_masks = np.nan_to_num(clean_data[c], nan=0.0).astype(np.float32)
            per_speaker = [
                clean_masks[:, s]
                if np.sum(clean_masks[:, s]) > min_num_frames
                else masks[:, s]
                for s in range(num_speakers)
            ]
            yield waveform[None], torch.from_numpy(np.stack(per_speaker))[None]

    batches = batchify(
        iter_waveform_and_masks(),
        batch_size=self.embedding_batch_size,
        fillvalue=(None, None),
    )
    batch_count = math.ceil(num_chunks / self.embedding_batch_size)

    embedding_batches = []

    if hook is not None:
        hook("embeddings", None, total=batch_count, completed=0)

    for i, batch in enumerate(batches, 1):
        waveforms, masks = zip(*(b for b in batch if b[0] is not None))

        waveform_batch = torch.vstack(waveforms)
        # (batch_size, 1, num_samples)
        mask_batch = torch.vstack(masks)
        # (batch_size, num_speakers, num_frames)

        embedding_batch = self._embedding(waveform_batch, masks=mask_batch)
        # (batch_size, num_speakers, dimension) -- the wrapper moves both
        # tensors to its device and hands the 3-D masks to the pooling stage

        embedding_batches.append(embedding_batch)

        if hook is not None:
            hook("embeddings", embedding_batch, total=batch_count, completed=i)

    embeddings = np.vstack(embedding_batches)
    # (num_chunks, num_speakers, dimension), same as the stock rearrange --
    # which also validated the row count, so keep that guarantee: a chunk
    # iteration that stopped early must not flow into clustering.
    if embeddings.shape[0] != num_chunks:
        raise RuntimeError(
            f"embedding rows ({embeddings.shape[0]}) != chunks ({num_chunks})"
        )
    return embeddings
