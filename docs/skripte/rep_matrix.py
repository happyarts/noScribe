"""Punctuation vs loop resistance for repetition_penalty candidates."""
import sys, re, zlib, soundfile as sf
sys.path.insert(0,'/Users/markus/Documents/noScribe')
from noScribe.voxtral_engine import _Voxtral, _looks_degenerate

def stats(t):
    w=t.split(); best=run=1
    for a,b in zip(w,w[1:]): run=run+1 if a==b else 1; best=max(best,run)
    return len(w), t.count(',')/max(len(w),1)*100, best, _looks_degenerate(t)

# 1) Satzzeichen-Qualität: mini auf 600s (dort haben wir Whisper=10.62 als Referenz)
audio600,_ = sf.read('/tmp/first600.wav', dtype='float32')
mini = _Voxtral('models/voxtral-mini')
print("=== mini 600s: Satzzeichen (Whisper-Referenz: 10.62 Kommas/100W) ===", flush=True)
for rep in (1.0, 1.1, 1.15, 1.2):
    t = mini.transcribe_array(audio600, 'de', max_new_tokens=12512, repetition_penalty=rep)
    w,c,run,deg = stats(t)
    print(f"  rep={rep:<5}  {w:5d} W | Kommas {c:5.2f}/100W | maxRun {run:3d} | {'ENTARTET' if deg else 'ok'}", flush=True)
del mini
import mlx.core as mx; mx.clear_cache()

# 2) Schleifenresistenz: small auf der kritischen 410s-Stelle
audio410,_ = sf.read('/tmp/loop_410.wav', dtype='float32')
small = _Voxtral('models/voxtral-small')
print("\n=== small 410s: Schleifenresistenz an der kritischen Stelle ===", flush=True)
for rep in (1.0, 1.1, 1.15):
    t = small.transcribe_array(audio410, None, max_new_tokens=8704, repetition_penalty=rep)
    w,c,run,deg = stats(t)
    print(f"  rep={rep:<5}  {w:5d} W | Kommas {c:5.2f}/100W | maxRun {run:3d} | {'ENTARTET' if deg else 'ok'}", flush=True)
