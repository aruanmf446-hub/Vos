#!/usr/bin/env python3
"""Render only approved review windows, or refresh Doctor Strange gaze windows."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT=Path(__file__).resolve().parent
RANGES=[(0,39),(39,56),(198,239),(359,400),(444,575),(624,665),(706,747)]
GAZE=[(0,39),(359,400),(706,747)]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gaze-only',action='store_true')
    p.add_argument('--force',action='store_true')
    a=p.parse_args()
    windows=GAZE if a.gaze_only else RANGES
    indices=[n for s,e in windows for n in range(2*s,2*e)]
    boundaries=sorted(set([39,76,173,241,272,349,404,666,749]+[q for r in RANGES for q in r if q>0]))
    command=['python',str(ROOT/'render60.py'),'--flow-method','rife',
             '--indices',','.join(map(str,indices)),'--cuts',','.join(map(str,boundaries)),
             '--output',str(ROOT/'work/review_rife')]
    if a.force or a.gaze_only:command.append('--force')
    print('Refreshing' if a.gaze_only else 'Rendering',len(indices),'approved frames',flush=True)
    subprocess.run(command,check=True)
