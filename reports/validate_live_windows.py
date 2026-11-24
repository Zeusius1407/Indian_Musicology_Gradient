#!/usr/bin/env python3
"""Validate the live window rule (classify/live.py) on Sanidha with simulated bleed.

Each clean stem gets every other stem leaked in through its own random room
response, at two bleed levels. For each margin it prints, per instrument, how
many windows the rule keeps, their precision against the clean-stem rule, and
the share of kept windows where the instrument is under half its own mic energy.
Run from the repo root: python reports/validate_live_windows.py (about 15 min).
"""
import numpy as np, collections
from scipy.signal import fftconvolve
from indian_mss.data.sanidha import Sanidha
from indian_mss.classify.data import WindowConfig, window_rms_db, active_starts
from indian_mss.train.augment import synthetic_rir
sr=22050; cfg=WindowConfig(); rng=np.random.default_rng(0)
ALL=('vocal','violin','mridangam','ghatam')
ds=Sanidha("data/sanidha_flat", sr, stems=ALL)
by=collections.defaultdict(list)
for t in ds.track_ids(): by[t.split('_')[0]].append(t)
for GAIN in ((-20.0,-8.0),(-14.0,-2.0)):
  data=collections.defaultdict(list)
  for conc, tids in sorted(by.items()):
    per=collections.defaultdict(lambda: collections.defaultdict(list))
    for t in tids:
        it=ds.load(t); n=len(it.mixture); k=min(n,120*sr); s=(n-k)//2
        present=[a for a in ALL if a in it.stems]; refs={a:it.stems[a][s:s+k] for a in present}
        mics={}
        for x in present:
            m=refs[x].copy()
            for j in present:
                if j!=x: m+=10**(rng.uniform(*GAIN)/20)*fftconvolve(refs[j], synthetic_rir(rng,sr,80.0))[:k]
            mics[x]=m
        E={x:10**(window_rms_db(mics[x],sr,cfg)[1]/10) for x in present}
        starts,_=window_rms_db(mics[present[0]],sr,cfg)
        for x in present:
            own=10**(window_rms_db(refs[x],sr,cfg)[1]/10)
            oth=sum(E[j] for j in present if j!=x)
            micdb=10*np.log10(E[x]+1e-20)
            per[x]['r'].append(10*np.log10(E[x]/np.maximum(oth,1e-20)))
            per[x]['ok'].append(micdb>=max(cfg.silence_db,np.percentile(micdb,95)+cfg.silence_rel_db))
            per[x]['tr'].append(np.isin(starts, active_starts(refs[x],sr,cfg)))
            per[x]['share'].append(own/np.maximum(E[x],1e-20))
    for x,d in per.items(): data[x].append({k:np.concatenate(v) for k,v in d.items()})
  print("bleed gain per pair", GAIN)
  for margin in (3,6,10):
    line=[]
    for x in ALL:
        sh=[];tr=[];n_all=0
        for d in data[x]:
            sel=d['ok']&(d['r']>=np.percentile(d['r'],5)+margin)
            sh.append(d['share'][sel]); tr.append(d['tr'][sel]); n_all+=d['tr'].sum()
        sh=np.concatenate(sh); tr=np.concatenate(tr)
        line.append(f"{x}: n={len(sh):5d} P={tr.mean():.3f} own<50%={np.mean(sh<0.5):.3f} own-share p5={np.percentile(sh,5):.2f}")
    print(f" margin {margin:2d} dB | "+" | ".join(line))
  line=[]
  for x in ALL:
     sh=np.concatenate([d['share'][d['ok']] for d in data[x]]); line.append(f"{x}: own<50%={np.mean(sh<0.5):.3f}")
  print(" naive floors only | "+" | ".join(line))
