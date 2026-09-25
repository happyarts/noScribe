"""Speaker diarization with NVIDIA's Nemotron 3 Diarization, in a child process.

An alternative to pyannote_mp_worker, chosen with `diarization_engine: nemotron`
in config.yml (which downloads the model on first use), and by default (`auto`)
whenever the installed transformers knows the model and its weights are here. It answers with the same result message -- turns as
{"start": ms, "end": ms, "label": "SPEAKER_nn"} -- and instead of centroids it
leaves the model's own speaker probabilities (one row per FRAME_S, one column
per speaker, float16) in a .npy file next to the audio, which is what the voice
check compares against (noScribe.voice_check.Probabilities).

Measured against pyannote in the fork's benchmarks-local/nemotron/README.md: on
the 15 AMI test meetings (not in Nemotron's training data; VoxConverse and AMI
train/dev are) 11.6 % of words end up under the wrong speaker against 15.7 %,
at about a seventh of the time; on 40 German CallHome calls 6.8 % of speech time
against 11.8 %; a 4.8 h video call runs through its speaker cache in 50 s.

What it cannot do: more than eight speakers, and a fixed number of speakers
(main.py keeps such a job with pyannote). It needs transformers with `nemotron3_diarization`, which is
on transformers' main branch only so far, and neither the requirements nor the
PyInstaller specs carry that yet.

Messages:
  {"type":"log","level":"info|warn|error|debug","msg":str}
  {"type":"progress","pct":int}
  {"type":"result","ok":True,"segments":[{"start":ms,"end":ms,"label":str}],
                             "probabilities":{"path":str,"columns":[label],"frame_s":float}}
  {"type":"result","ok":False,"error":str,"trace":str}

Stdlib-only at import time, like the other workers: the heavy imports happen in
the entrypoint.
"""
import os
import sys
import traceback

if sys.version_info >= (3, 12):
    import importlib.resources as impres
else:
    import importlib_resources as impres

# The Hugging Face repository, used when no copy ships in models/.
MODEL_REPO = "nvidia/Nemotron-3-Diarization"
MODEL_DIR = "nemotron-diarization"  # under models/

# The model's output frame: transformers upsamples its 80 ms encoder frames to
# the 10 ms of the spectrogram.
FRAME_S = 0.01

# A speaker counts as talking where the model's probability exceeds this: the
# model card's and transformers' default, and what the measurement used.
THRESHOLD = 0.5


def model_source():
    """The shipped copy under models/ if there is one, else the Hub repository."""
    try:
        path = impres.files("models") / MODEL_DIR
        if (path / "config.json").is_file():
            return str(path)
    except Exception:
        pass
    return MODEL_REPO


def available():
    """Whether the installed transformers knows the model and its weights are here,
    shipped or in the Hugging Face cache, so `auto` never starts a download or fails
    offline. Found without importing transformers: main.py asks this in the GUI
    process, which never loads it."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("transformers")
        if spec is None or not spec.origin:
            return False
        if not os.path.isdir(os.path.join(os.path.dirname(spec.origin), "models", "nemotron3_diarization")):
            return False
        if model_source() != MODEL_REPO:
            return True
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(MODEL_REPO, "model.safetensors"), str)
    except Exception:
        return False


def label(column):
    """The diarization label of the model's speaker column `column`."""
    return f"SPEAKER_{int(column):02d}"


def features(processor, audio, sample_rate, chunk_frames=60000, device="cpu"):
    """The model's input for a whole recording, as processor(audio) gives it, but
    computed ten minutes at a time: in one pass transformers' spectrogram took
    12.3 GB for a 4.8 h recording whose features are 0.9 GB. transformers keeps the
    one pass by design (huggingface/transformers#49090) and points to the
    processor's streaming calls, whose chunks reproduce it frame for frame; only
    the model's forward stays offline, since its streaming mode caches speakers
    differently. What is left after the last whole chunk comes from an offline call
    on the recording's end: the streaming call for a last chunk left out the last
    valid frame, differed from the one pass on a few short ends and refused one
    under a window long. Bit-identical to the one pass."""
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    hop = processor.feature_extractor.hop_length
    valid = len(audio) // hop  # frames one pass counts as valid; it adds one more, masked
    step = chunk_frames // processor.subsampling_factor
    processor.streaming_modes = dict(processor.streaming_modes, noscribe_offline=(step, 0))
    processor.set_streaming_mode("noscribe_offline")
    if len(audio) <= processor.num_samples_first_audio_chunk:
        return processor(audio, sampling_rate=sample_rate).to(device)

    feats, done = None, 0
    start, end, first = 0, processor.num_samples_first_audio_chunk, True
    while end <= len(audio):
        part = processor(audio[start:end], sampling_rate=sample_rate, is_streaming=True,
                         is_first_audio_chunk=first).input_features
        if feats is None:
            feats = torch.empty((1, valid + 1, part.shape[2]), dtype=part.dtype, device=device)
        feats[:, done:done + part.shape[1]] = part
        done += part.shape[1]
        start = processor.audio_chunk_start(done)
        end, first = start + processor.num_samples_per_audio_chunk, False
    tail = max(0, done - 16) * hop  # on the frame grid, and far enough back for the window and pre-emphasis
    feats[:, done:valid] = processor(audio[tail:], sampling_rate=sample_rate).input_features[:, done - tail // hop:valid - tail // hop]
    feats[:, valid:] = processor.feature_extractor.padding_value
    mask = torch.zeros(feats.shape[:2], dtype=torch.long, device=device)
    mask[:, :valid] = 1
    return BatchFeature({"input_features": feats, "attention_mask": mask})


def turns(speaker_dicts):
    """transformers' [{"Start": s, "End": s, "Speaker": k}] as noScribe's turns,
    sorted by start, which the overlap assignment in main.py relies on."""
    out = [{"start": round(d["Start"] * 1000), "end": round(d["End"] * 1000),
            "label": label(d["Speaker"])} for d in speaker_dicts]
    return sorted((t for t in out if t["end"] > t["start"]), key=lambda t: (t["start"], t["end"]))


def nemotron_proc_entrypoint(args: dict, q):
    def plog(level, msg):
        try:
            q.put({"type": "log", "level": level, "msg": str(msg)})
        except Exception:
            pass

    device = ''
    try:
        # First: pyannote_mp_worker sets the OpenMP variables torch has to find at import.
        from .pyannote_mp_worker import load_waveform, select_device
        import numpy as np
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        try:
            from transformers.models import nemotron3_diarization  # noqa: F401
        except ImportError:
            import transformers
            raise RuntimeError(f"Nemotron diarization needs a transformers that knows the model "
                               f"(5.18 or newer); this one is {transformers.__version__}") from None

        audio_file = args["audio_path"]
        if not os.path.exists(audio_file):
            raise FileNotFoundError(audio_file)
        device = select_device(args.get("device", ""))

        source = model_source()
        plog("debug", f"Loading Nemotron diarization from {source} on {device}")
        q.put({"type": "progress", "pct": 0})
        processor = AutoProcessor.from_pretrained(source)

        waveform, sample_rate = load_waveform(audio_file)
        audio = waveform[0].numpy()  # ToWav writes mono; a view, not a copy
        del waveform
        if sample_rate != processor.feature_extractor.sampling_rate:
            raise RuntimeError(f"audio at {sample_rate} Hz, the model needs "
                               f"{processor.feature_extractor.sampling_rate} Hz")
        inputs = features(processor, audio, sample_rate, device=device)
        del audio
        # Loaded after the features, so it is not held through their peak.
        model = AutoModelForAudioFrameClassification.from_pretrained(source).to(device).eval()
        inputs = inputs.to(device, dtype=model.dtype)
        # Offline mode: the whole recording in one call, cut into chunks with a
        # speaker cache inside the model, so there is no length limit.
        with torch.inference_mode():
            logits = model(**inputs).logits
        mask = inputs.attention_mask
        del inputs  # the whole recording's features, on the device
        speaker_dicts = processor.extract_speaker_dict(logits, mask, threshold=THRESHOLD)[0]
        probabilities = torch.sigmoid(logits[0].float()).cpu().numpy().astype(np.float16)
        probabilities_path = os.path.splitext(audio_file)[0] + ".speakers.npy"
        np.save(probabilities_path, probabilities)
        q.put({"type": "progress", "pct": 100})
        q.put({"type": "result", "ok": True, "segments": turns(speaker_dicts),
               "probabilities": {"path": probabilities_path, "frame_s": FRAME_S,
                                 "columns": [label(k) for k in range(probabilities.shape[1])]}})
    except Exception as e:
        try:
            q.put({"type": "result", "ok": False,
                   "error": f"{type(e).__name__}: {e} (device_{device[:3]})",  # main.py's CUDA fallback reads this
                   "trace": traceback.format_exc()})
        except Exception:
            pass
