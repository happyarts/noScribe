"""Speaker diarization with NVIDIA's Nemotron 3 Diarization, in a child process.

An alternative to pyannote_mp_worker, chosen with `diarization_engine: nemotron`
in config.yml. It answers with the same result message -- turns as
{"start": ms, "end": ms, "label": "SPEAKER_nn"} -- and instead of centroids it
leaves the model's own speaker probabilities (one row per FRAME_S, one column
per speaker, float16) in a .npy file next to the audio, which is what the voice
check compares against (noScribe.voice_check.Probabilities).

Measured against pyannote in the fork's benchmarks-local/nemotron/README.md: on
the 15 AMI test meetings (not in Nemotron's training data; VoxConverse and AMI
train/dev are) 11.6 % of words end up under the wrong speaker against 15.7 %,
at about a seventh of the time; on 40 German CallHome calls 6.8 % of speech time
against 11.8 %; a 4.8 h Zoom call runs through its speaker cache in 50 s.

What it cannot do: more than eight speakers, and a fixed number of speakers
(main.py keeps such a job with pyannote; here `num_speakers` would only be
noted and ignored). It needs transformers with `nemotron3_diarization`, which is
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
MODEL_DIR = "nemotron-diarization"  # under transcription.DIR_PACKAGE_MODELS

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
        if path.is_dir():
            return str(path)
    except Exception:
        pass
    return MODEL_REPO


def label(column):
    """The diarization label of the model's speaker column `column`."""
    return f"SPEAKER_{int(column):02d}"


def features(processor, audio, sample_rate, chunk_frames=60000):
    """The model's input for a whole recording, as processor(audio) gives it, but
    computed ten minutes at a time: in one pass transformers' spectrogram took
    12.3 GB for a 4.8 h recording whose features are 0.9 GB (5.4 GB peak this way).
    Bit-identical to the one-pass features: each chunk's STFT windows are cut from
    the padded recording as its feature extractor's docstring describes
    (center=False), and the pre-emphasis runs once over the whole recording so no
    chunk starts without its previous sample."""
    import numpy as np
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    fe = processor.feature_extractor
    hop, n_fft, pre = fe.hop_length, fe.n_fft, fe.preemphasis
    n = len(audio) // hop + 1  # frames of one centered pass (torch.stft)
    x = np.pad(audio, (n_fft // 2, n_fft // 2))  # center=True's constant padding
    if pre:
        x[n_fft // 2 + 1:n_fft // 2 + len(audio)] -= pre * audio[:-1]
    fe.preemphasis = None
    try:
        parts = [fe(x[hop * k:hop * (min(n, k + chunk_frames) - 1) + n_fft], sampling_rate=sample_rate,
                    center=False, return_tensors="pt").input_features
                 for k in range(0, n, chunk_frames)]
    finally:
        fe.preemphasis = pre
    feats = torch.cat(parts, dim=1)
    # One pass counts only floor(L / hop) frames as valid: the last is masked and padded.
    valid = len(audio) // hop
    feats[:, valid:] = fe.padding_value
    mask = torch.zeros(feats.shape[:2], dtype=torch.long)
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
        import numpy as np
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        from .pyannote_mp_worker import load_waveform, select_device
        try:
            from transformers.models import nemotron3_diarization  # noqa: F401
        except ImportError:
            import transformers
            raise RuntimeError(f"Nemotron diarization needs a transformers that knows the model "
                               f"(5.18 or newer); this one is {transformers.__version__}") from None

        audio_file = args["audio_path"]
        if not os.path.exists(audio_file):
            raise FileNotFoundError(audio_file)
        if args.get("num_speakers"):
            plog("warn", "Nemotron diarization finds the number of speakers itself; "
                         f"the requested {args['num_speakers']} is ignored")

        device = select_device(args.get("device", ""))

        source = model_source()
        plog("debug", f"Loading Nemotron diarization from {source} on {device}")
        q.put({"type": "progress", "pct": 0})
        processor = AutoProcessor.from_pretrained(source)
        model = AutoModelForAudioFrameClassification.from_pretrained(source).to(device).eval()

        waveform, sample_rate = load_waveform(audio_file)
        audio = waveform[0].numpy()  # ToWav writes mono; a view, not a copy
        del waveform
        if sample_rate != processor.feature_extractor.sampling_rate:
            raise RuntimeError(f"audio at {sample_rate} Hz, the model needs "
                               f"{processor.feature_extractor.sampling_rate} Hz")
        # Offline mode: the whole recording in one call, cut into chunks with a
        # speaker cache inside the model, so there is no length limit.
        inputs = features(processor, audio, sample_rate).to(device, dtype=model.dtype)
        del audio
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
