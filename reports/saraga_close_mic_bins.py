#!/usr/bin/env python3
"""Saraga close-mic accuracy, binned by how far each window sits above its concert bleed floor.

If close-mic errors are caused by bleed, accuracy climbs as the margin grows;
if they come from a recording-domain shift, it stays flat.
Run from the repo root: python reports/saraga_close_mic_bins.py (about 10 min).
"""
import numpy as np, collections, json, sys
import librosa
from indian_mss.data.saraga import Saraga
from indian_mss.classify.live import LiveWindowConfig, mic_ratios
from indian_mss.classify.data import mfcc_windows
from indian_mss.classify.train import load_classifier, predict_proba
from indian_mss.cli.evaluate import _crop_centre
dev="cuda"; clf,m=load_classifier("checkpoints/classifier/classifier.pt",dev)
L=m['classes']; mc,wc=m['mfcc_config'],m['window_config']; sr=mc.sample_rate
ds=Saraga("data/saraga",sr,mixture="stems")
by=collections.defaultdict(list)
for t in ds.track_ids(): by[t.split('/')[1]].append(t)
live=LiveWindowConfig()
rows=collections.defaultdict(list)  # stem -> (excess_db, correct, pred)
for conc in sorted(by):
    tr=[]
    for t in by[conc]:
        _,stems,_=_crop_centre(ds.load(t),60.0); tr.append({k:v.copy() for k,v in stems.items()})
    rat=[mic_ratios(s,sr,wc) for s in tr]
    floors={x:np.percentile(np.concatenate([r[x] for _,r,_ in rat if x in r]),live.floor_percentile) for x in ("vocal","violin","mridangam","ghatam") if any(x in r for _,r,_ in rat)}
    for stems,(starts,r,loud) in zip(tr,rat):
        for x in r:
            sel=loud[x]&(r[x]>=floors[x]+live.margin_db)
            if not sel.any(): continue
            p=predict_proba(clf,mfcc_windows(stems[x],starts[sel],mc,wc.window_seconds),dev)
            pred=p.argmax(1)
            for e,pr in zip(r[x][sel]-floors[x],pred): rows[x].append((e,pr==L.index(x),L[pr]))
    print(conc[:30],flush=True,file=sys.stderr)
bins=[10,15,20,25,30,100]
print("close-mic accuracy by ratio above the concert's bleed floor (dB)")
for x,v in rows.items():
    e=np.array([a for a,_,_ in v]); c=np.array([b for _,b,_ in v]); pr=np.array([p for *_,p in v])
    out=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        s=(e>=lo)&(e<hi)
        if s.sum()<20: out.append(f"[{lo},{hi}) n={s.sum():4d}     -"); continue
        vals,cnt=np.unique(pr[s&~c],return_counts=True); top=vals[cnt.argmax()] if len(cnt) else '-'
        out.append(f"[{lo},{hi}) n={s.sum():4d} {c[s].mean():5.1%} (top err {top})")
    print(f"{x:10s} "+" | ".join(out))
