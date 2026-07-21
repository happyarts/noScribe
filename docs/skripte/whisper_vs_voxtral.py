"""Whisper and Voxtral on the same audio, scored with the same metric.

noScribe's own settings are used for Whisper (the models shipped under
models/, beam search, German) so the comparison reflects what a noScribe user
actually gets, not a tuned-for-the-benchmark configuration.

    python docs/skripte/whisper_vs_voxtral.py <reference.txt> <audio.wav>
"""
import sys, time
import soundfile as sf

import pathlib
REPO = pathlib.Path(__file__).resolve().parents[2]  # docs/skripte/x.py -> repo root
sys.path.insert(0, str(REPO))
_wer = open(REPO / 'docs/skripte/wer.py').read().split('raw = open')[0]
_ns = {'__file__': str(REPO / 'docs/skripte/wer.py')}
exec(_wer, _ns)
norm, wer, OVERLAP = _ns['norm'], _ns['wer'], _ns['OVERLAP']

REF_PATH, WAV = sys.argv[1], sys.argv[2]
raw = open(REF_PATH, encoding='utf-8').read()
ref = norm(OVERLAP.sub(' ', raw))
ref_chars = "".join(ref)
audio, sr = sf.read(WAV, dtype='float32')
dur = len(audio) / sr

print(f"# {WAV}  {dur:.0f}s | Referenz {len(ref)} Woerter\n")
print(f"{'Modell':28s} {'WER':>7s} {'CER':>7s} {'Sub':>5s} {'Del':>5s} {'Ins':>5s} {'Speed':>7s}")


def score(name, text, elapsed):
    hyp = norm(text)
    err, sub, dele, ins = wer(ref, hyp)
    cer = wer(ref_chars, "".join(hyp))[0] / max(1, len(ref_chars))
    print(f"{name:28s} {err/len(ref)*100:6.2f}% {cer*100:6.2f}% {sub:5d} {dele:5d} {ins:5d} "
          f"{dur/elapsed:6.2f}x", flush=True)
    open(f"/tmp/cmp_{name}.txt", 'w').write(text)


# --- Whisper (faster-whisper), as noScribe runs it -------------------------
from faster_whisper import WhisperModel
for wname in ("precise", "fast"):
    model = WhisperModel(f"models/{wname}", device="cpu", compute_type="int8")
    t0 = time.time()
    segments, _ = model.transcribe(WAV, language="de", beam_size=5)
    text = " ".join(s.text.strip() for s in segments)
    score(f"whisper-{wname}", text, time.time() - t0)
    del model

# --- Voxtral ---------------------------------------------------------------
import mlx.core as mx
from noScribe.voxtral_engine import _Voxtral

for path in ("models/voxtral-mini-8bit", "models/voxtral-small-4bit-lh4"):
    vox = _Voxtral(path)
    t0 = time.time()
    text = vox.transcribe_array(audio, 'de', max_new_tokens=int(dur * 20) + 512)
    score(path.split('/')[-1], text, time.time() - t0)
    del vox
    mx.clear_cache()
