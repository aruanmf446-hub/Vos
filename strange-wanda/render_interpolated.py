#!/usr/bin/env python3
"""Interpolate edited anchors with edited-image DIS flow and source priors.

This aligns independently generated facial details before blending. Original
source flow initializes DIS and optionally supplies a small nonlinear motion
correction; shot-edge extrapolation continues to use only original motion.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from pathlib import Path
from collections import defaultdict
import time

import cv2
import numpy as np

from render import ROOT, SourceMotion, numbers, remap


def inverse_at_fraction(flow: np.ndarray, fraction: float, grid: np.ndarray):
    """Invert x_intermediate = x_anchor + fraction*flow(x_anchor)."""
    mapping = grid - fraction * flow
    for _ in range(10):
        next_map = grid - fraction * remap(flow, mapping)
        mapping = 0.35 * mapping + 0.65 * next_map
    error = np.linalg.norm(mapping + fraction * remap(flow, mapping) - grid, axis=2)
    return mapping, np.exp(-error / 1.5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'work/source')
    p.add_argument('--anchors', type=Path, default=ROOT/'work/anchors')
    p.add_argument('--output', type=Path, default=ROOT/'work/rendered_interpolated')
    p.add_argument('--cache', type=Path, default=ROOT/'work/flow')
    p.add_argument('--indices', type=numbers)
    p.add_argument('--cuts', type=numbers, default=[143,294])
    p.add_argument('--source-motion-strength', type=float, default=0.25)
    p.add_argument('--force', action='store_true')
    args=p.parse_args()
    cv2.setNumThreads(2)
    paths={int(q.stem):q for q in args.anchors.glob('[0-9][0-9][0-9][0-9].png')}
    # Source may also contain an untouched outro beyond the edited anchor range.
    count=max(paths)+1
    targets=args.indices if args.indices is not None else list(range(count))
    args.output.mkdir(parents=True,exist_ok=True)
    targets=[t for t in targets if args.force or not (args.output/f'{t:04d}.png').exists()]
    motion=SourceMotion(args.source,args.cache,288,'medium')
    groups=defaultdict(list)
    boundaries=[0]+args.cuts+[count]
    for start,end in zip(boundaries,boundaries[1:]):
        aa=sorted(i for i in paths if start<=i<end)
        if not aa: raise ValueError(f'No anchors for shot {start}:{end}')
        for t in targets:
            if not start<=t<end: continue
            i=bisect_left(aa,t)
            if i<len(aa) and aa[i]==t: pair=(t,t)
            elif i==0: pair=(aa[0],aa[0])
            elif i==len(aa): pair=(aa[-1],aa[-1])
            else: pair=(aa[i-1],aa[i])
            groups[pair].append(t)
    flow_dir=args.cache/'edited-dis-medium-288-v1'
    flow_dir.mkdir(exist_ok=True,parents=True)
    started=time.monotonic();done=0
    def read(i):
        im=cv2.imread(str(paths[i]));return cv2.resize(im,motion.full_size,interpolation=cv2.INTER_LANCZOS4)
    def gray(im):
        return cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),motion.small_size,interpolation=cv2.INTER_AREA)
    for (a,b),tt in groups.items():
        print(f'Edited anchors {a:04d}/{b:04d}: {len(tt)} frames',flush=True)
        ia=read(a);ib=read(b) if b!=a else ia
        if a==b:
            mm=motion.anchor_maps(a,tt)
            for t in tt:
                result=ia if t==a else motion.warp(ia,mm[t][0])
                cv2.imwrite(str(args.output/f'{t:04d}.png'),result,[cv2.IMWRITE_PNG_COMPRESSION,2]);done+=1
            continue
        # Source priors also ensure the edited flow follows the shot's camera.
        original_ab=motion.anchor_maps(b,[a])[a][0]
        original_ba=motion.anchor_maps(a,[b])[b][0]
        cache_path=flow_dir/f'{a:04d}-{b:04d}.npz'
        if cache_path.exists():
            with np.load(cache_path) as saved:
                fab=saved['forward'].astype(np.float32);fba=saved['backward'].astype(np.float32)
        else:
            dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setUseSpatialPropagation(True);dis.setUseMeanNormalization(True)
            dis.setVariationalRefinementIterations(8)
            fab=dis.calc(gray(ia),gray(ib),original_ab.copy())
            fba=dis.calc(gray(ib),gray(ia),original_ba.copy())
            np.savez_compressed(cache_path,forward=fab.astype(np.float16),backward=fba.astype(np.float16))
        cfa=motion.consistent(fab,fba);cfb=motion.consistent(fba,fab)
        actual_a=motion.anchor_maps(a,tt) if args.source_motion_strength else None
        actual_b=motion.anchor_maps(b,tt) if args.source_motion_strength else None
        for t in tt:
            s=(t-a)/(b-a)
            ma,qa=inverse_at_fraction(fab,s,motion.grid)
            mb,qb=inverse_at_fraction(fba,1-s,motion.grid)
            if args.source_motion_strength:
                la,_=inverse_at_fraction(original_ab,s,motion.grid)
                lb,_=inverse_at_fraction(original_ba,1-s,motion.grid)
                ma+=args.source_motion_strength*(actual_a[t][0]-(la-motion.grid))
                mb+=args.source_motion_strength*(actual_b[t][0]-(lb-motion.grid))
            ca=remap(cfa,ma)*qa*motion.validity(ma)
            cb=remap(cfb,mb)*qb*motion.validity(mb)
            # A soft floor limits per-pixel seams in textureless face/cloak areas.
            ca=(.15+.85*ca)*motion.validity(ma);cb=(.15+.85*cb)*motion.validity(mb)
            wa=motion.warp(ia,ma-motion.grid);wb=motion.warp(ib,mb-motion.grid)
            phase=s*s*(3-2*s)
            denominator=(1-phase)*ca+phase*cb
            alpha=np.divide(phase*cb,denominator,out=np.full_like(cb,phase),where=denominator>1e-6)
            alpha=cv2.GaussianBlur(alpha,(5,5),0)
            alpha=cv2.resize(alpha,motion.full_size,interpolation=cv2.INTER_LINEAR)[...,None]
            result=np.clip(wa.astype(np.float32)*(1-alpha)+wb.astype(np.float32)*alpha+.5,0,255).astype(np.uint8)
            cv2.imwrite(str(args.output/f'{t:04d}.png'),result,[cv2.IMWRITE_PNG_COMPRESSION,2]);done+=1
        print(f'Rendered {done}/{len(targets)} in {time.monotonic()-started:.1f}s',flush=True)
    print(f'Finished: {args.output}',flush=True)

if __name__=='__main__':main()
