"""Better retry: split the failed pass and re-run at rep=1.0 (no penalty),
so legitimate repetitions like "nicht nicht" survive."""
import sys, soundfile as sf
sys.path.insert(0,'/Users/markus/Documents/noScribe')
from noScribe.voxtral_engine import _Voxtral, _looks_degenerate
a,_ = sf.read('/tmp/loop_410.wav', dtype='float32')
vox=_Voxtral('models/voxtral-small')

def stats(t):
    w=t.split(); best=run=1
    for x,y in zip(w,w[1:]): run=run+1 if x==y else 1; best=max(best,run)
    return len(w), best, _looks_degenerate(t)

half=len(a)//2
parts=[]
for i,seg in enumerate((a[:half], a[half:])):
    t = vox.transcribe_array(seg, None, max_new_tokens=4608, repetition_penalty=1.0)
    w,r,d = stats(t)
    print(f"### Hälfte {i+1} (205s, rep=1.0): {w} W | maxRun {r} | {'ENTARTET' if d else 'ok'}", flush=True)
    parts.append(t)
joined=" ".join(parts)
w,r,d = stats(joined)
print(f"### zusammen: {w} W | maxRun {r} | {'ENTARTET' if d else 'ok'}", flush=True)
import re
for m in re.finditer(r'.{70}nicht nicht.{70}', joined): print("  ✓ DOPPELTES NICHT:", m.group(0).strip(), flush=True)
if 'nicht nicht' not in joined:
    for m in re.finditer(r'.{60}können es dir.{60}', joined): print("  Stelle:", m.group(0).strip(), flush=True)
