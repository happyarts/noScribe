import sys, soundfile as sf
sys.path.insert(0,'/Users/markus/Documents/noScribe')
from noScribe.voxtral_engine import _Voxtral, _looks_degenerate, RETRY_REPETITION_PENALTY
a,_ = sf.read('/tmp/loop_410.wav', dtype='float32')
vox=_Voxtral('models/voxtral-small')
for rep in (1.0, RETRY_REPETITION_PENALTY):
    t = vox.transcribe_array(a, None, max_new_tokens=8704, repetition_penalty=rep)
    w=t.split(); best=run=1
    for x,y in zip(w,w[1:]): run=run+1 if x==y else 1; best=max(best,run)
    print(f"\n=== small 410s, repetition_penalty={rep} ===", flush=True)
    print(f"  {len(w)} Wörter | längste Kette: {best} | Detektor: {'ENTARTET' if _looks_degenerate(t) else 'ok'}", flush=True)
    print(f"  Ende: ...{t[-150:]}", flush=True)
