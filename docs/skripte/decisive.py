"""Isolate repetition_penalty and pass length at realistic length."""
import sys, re, soundfile as sf
sys.path.insert(0,'/Users/markus/Documents/noScribe')
from noScribe.voxtral_engine import _Voxtral, SAMPLE_RATE
audio,_ = sf.read('/tmp/first600.wav', dtype='float32')
vox = _Voxtral('models/voxtral-mini')

def one(a, lang, rep, maxnew):
    inp = vox.proc.apply_transcrition_request(audio=a, language=lang, sampling_rate=SAMPLE_RATE)
    mi = {"input_ids": inp.input_ids, "input_features": inp.input_features}
    if getattr(inp,"attention_mask",None) is not None: mi["attention_mask"]=inp.attention_mask
    out = vox.model.generate(**mi, max_new_tokens=maxnew, temperature=0.0, repetition_penalty=rep)
    return vox.proc.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def stats(t):
    w=len(t.split()); return w, t.count(',')/w*100, len(re.findall(r'[.!?]',t))/w*100

runs = {}
runs['a 600s rep=1.2 (aktuell)'] = one(audio,'de',1.2,4096)
print('a done', flush=True)
runs['b 600s rep=1.0']          = one(audio,'de',1.0,4096)
print('b done', flush=True)
half = len(audio)//2
runs['c 2x300s rep=1.0']        = one(audio[:half],'de',1.0,4096)+' '+one(audio[half:],'de',1.0,4096)
print('c done', flush=True)

print("\n=== WHISPER Referenz: Kommas 10.62 | Satzenden 7.10 (pro 100 W) ===")
for k,t in runs.items():
    w,c,p = stats(t)
    print(f"{k:26s} {w:5d} W | Kommas {c:5.2f} | Satzenden {p:5.2f}")
import json; json.dump(runs, open('/tmp/decisive.json','w'), ensure_ascii=False, indent=1)
