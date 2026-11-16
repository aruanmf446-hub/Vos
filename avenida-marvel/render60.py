#!/usr/bin/env python3
"""Render edited keyframes at 60 fps, following original motion within each shot.

Inputs are action crops (576x364) at 30 fps in work/action, and edited anchors
whose filenames retain their original frame numbers in work/anchors. Output
NNNN.png names are indices at the requested output fps, not source indices.
Hard cuts are never interpolated. Example short preview:
  python render60.py --start 0 --end 39 --allow-partial --output work/preview
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import math
import time

import cv2
import numpy as np

from render_source import SourceMotion, numbers, remap

ROOT = Path(__file__).resolve().parent
CUTS = [39, 76, 173, 241, 272, 349, 404, 666, 749]


def inverse_map(flow: np.ndarray, fraction: float, grid: np.ndarray):
    """Invert a forward field at fractional time by damped fixed-point updates."""
    if fraction == 0:
        return grid.copy(), np.ones(grid.shape[:2], np.float32)
    mapping = grid - fraction * flow
    for _ in range(8):
        candidate = grid - fraction * remap(flow, mapping)
        mapping = .30 * mapping + .70 * candidate
    error = np.linalg.norm(mapping + fraction * remap(flow, mapping) - grid, axis=2)
    return mapping, np.exp(-error / 1.25)


def fractional_map(maps: dict, position: float, first: int, last: int):
    lo = max(first, min(last, int(math.floor(position))))
    hi = max(first, min(last, int(math.ceil(position))))
    if lo == hi:
        return maps[lo][0], maps[lo][1]
    weight = position - lo
    return (maps[lo][0] * (1-weight) + maps[hi][0] * weight,
            maps[lo][1] * (1-weight) + maps[hi][1] * weight)


def forward_splat(image, flow, reliability, fraction, motion):
    """Forward-project visible pixels instead of inverting a folding flow map.

    Inverse iteration can retain a second silhouette in disoccluded regions.
    Bilinear splatting records actual coverage, so the other anchor fills holes.
    A gentle motion priority favors moving foreground at overlaps.
    """
    full_flow = cv2.resize(flow,motion.full_size,interpolation=cv2.INTER_LINEAR)
    full_flow[...,0] *= motion.width/motion.fw
    full_flow[...,1] *= motion.height/motion.fh
    target = motion.full_grid+fraction*full_flow
    confidence = cv2.resize(reliability,motion.full_size,interpolation=cv2.INTER_LINEAR)
    priority = np.exp(np.minimum(np.linalg.norm(full_flow,axis=2),80)/35)
    importance = (.01+.99*confidence)*priority
    x,y = target[...,0].ravel(),target[...,1].ravel()
    x0,y0 = np.floor(x).astype(np.int32),np.floor(y).astype(np.int32)
    fx,fy = x-x0,y-y0
    pixels = image.reshape(-1,3).astype(np.float32)
    importance = importance.ravel()
    count = motion.height*motion.width
    denominator = np.zeros(count,np.float64)
    rgb = np.zeros((count,3),np.float64)
    for dx,dy,w in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        xx,yy=x0+dx,y0+dy
        valid=(xx>=0)&(xx<motion.width)&(yy>=0)&(yy<motion.height)
        bins=yy[valid]*motion.width+xx[valid]
        weights=w[valid]*importance[valid]
        denominator += np.bincount(bins,weights=weights,minlength=count)
        for channel in range(3):
            rgb[:,channel] += np.bincount(bins,weights=weights*pixels[valid,channel],minlength=count)
    normalized=np.divide(rgb,denominator[:,None],out=np.zeros_like(rgb),where=denominator[:,None]>1e-8)
    return normalized.reshape(motion.height,motion.width,3).astype(np.float32),np.minimum(denominator.reshape(motion.height,motion.width),1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=ROOT/'work/action')
    parser.add_argument('--anchors', type=Path, default=ROOT/'work/anchors')
    parser.add_argument('--output', type=Path, default=ROOT/'work/rendered')
    parser.add_argument('--cache', type=Path, default=ROOT/'work/flow60')
    parser.add_argument('--source-fps', type=float, default=30)
    parser.add_argument('--fps', type=float, default=60)
    parser.add_argument('--cuts', type=numbers, default=CUTS)
    parser.add_argument('--start', type=int, default=0, help='First source frame, inclusive')
    parser.add_argument('--end', type=int, help='Last source frame, exclusive')
    parser.add_argument('--indices', type=numbers, help='Only these output frame indices at --fps')
    parser.add_argument('--flow-width', type=int, default=384)
    parser.add_argument('--source-motion-strength', type=float, default=.12)
    parser.add_argument('--warp', choices=['inverse','splat'], default='inverse')
    parser.add_argument('--flow-method', choices=['dis','raft','raft-large','rife'], default='rife')
    parser.add_argument('--raft-updates', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--allow-partial', action='store_true', help='Skip shots without anchors while sheets are still being generated')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    files = sorted(args.source.glob('[0-9][0-9][0-9][0-9].png'))
    anchors = {int(p.stem): p for p in args.anchors.glob('[0-9][0-9][0-9][0-9].png')}
    if not files or not anchors:
        parser.error('Source frames and edited anchors must both exist')
    frame_count = int(files[-1].stem) + 1
    end = args.end if args.end is not None else frame_count
    if not 0 <= args.start < end <= frame_count:
        parser.error('--start/--end must be within source frame range')
    ratio = args.fps / args.source_fps
    selected = args.indices if args.indices is not None else list(range(math.ceil(args.start*ratio), math.ceil(end*ratio)))
    if any(i < 0 or i >= math.ceil(frame_count*ratio) for i in selected):
        parser.error('Output frame index outside full video')
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [i for i in selected if args.force or not (args.output/f'{i:04d}.png').exists()]
    if not selected:
        print('All requested frames already exist.', flush=True)
        return
    motion = SourceMotion(args.source, args.cache, min(args.flow_width, 576), 'medium')
    bounds = [0] + [c for c in args.cuts if 0 < c < frame_count] + [frame_count]
    groups = defaultdict(list)
    skipped = 0
    for shot_start, shot_end in zip(bounds, bounds[1:]):
        targets = [i for i in selected if shot_start <= i/ratio < shot_end]
        if not targets:
            continue
        keys = sorted(k for k in anchors if shot_start <= k < shot_end)
        if not keys:
            if args.allow_partial:
                skipped += len(targets)
                continue
            parser.error(f'No edited anchors in shot [{shot_start}, {shot_end})')
        if args.allow_partial:
            covered = [i for i in targets if keys[0] <= min(i/ratio,shot_end-1) <= keys[-1]]
            skipped += len(targets)-len(covered)
            targets = covered
        for i in targets:
            t = min(i/ratio, shot_end-1)
            pos = bisect_left(keys, t)
            if pos < len(keys) and keys[pos] == t:
                a = b = keys[pos]
            elif pos == 0:
                a = b = keys[0]
            elif pos == len(keys):
                a = b = keys[-1]
            else:
                a, b = keys[pos-1:pos+1]
            groups[(a, b, shot_start, shot_end)].append((i,t))
    edited = {}
    def read(index):
        if index not in edited:
            im = cv2.imread(str(anchors[index]), cv2.IMREAD_COLOR)
            if im is None: raise FileNotFoundError(anchors[index])
            edited[index] = cv2.resize(im, motion.full_size, interpolation=cv2.INTER_LANCZOS4)
        return edited[index]
    def gray(im):
        return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), motion.small_size, interpolation=cv2.INTER_AREA)
    flow_dir = args.cache/f'edited-{args.flow_method}-{motion.fw}-v3'
    flow_dir.mkdir(parents=True, exist_ok=True)
    neural = None
    started = time.monotonic()
    done = 0
    for (a,b,shot_start,shot_end), targets in groups.items():
        print(f'Anchors {a:04d}/{b:04d}: {len(targets)} frames', flush=True)
        ia, ib = read(a), read(b)
        needed = sorted(set(max(shot_start,min(shot_end-1,n)) for _,t in targets for n in (math.floor(t),math.ceil(t))))
        if a == b:
            maps = motion.anchor_maps(a, needed)
            for i,t in targets:
                if t == a: result = ia
                else:
                    flow, _ = fractional_map(maps,t,shot_start,shot_end-1)
                    result = motion.warp(ia,flow)
                cv2.imwrite(str(args.output/f'{i:04d}.png'),result,[cv2.IMWRITE_PNG_COMPRESSION,2])
                done += 1
            continue
        if args.flow_method == 'rife':
            if neural is None:
                from render_rife_backend import Rife
                neural=Rife(threads=args.threads)
            for i,t in targets:
                result=neural.between(ia,ib,(t-a)/(b-a),(a,b))
                cv2.imwrite(str(args.output/f'{i:04d}.png'),result,[cv2.IMWRITE_PNG_COMPRESSION,2])
                done += 1
            print(f'Rendered {done}/{len(selected)-skipped} in {time.monotonic()-started:.1f}s',flush=True)
            continue
        source_ab = motion.anchor_maps(b,[a])[a][0]
        source_ba = motion.anchor_maps(a,[b])[b][0]
        fingerprints = [f'{anchors[k].stat().st_mtime_ns}:{anchors[k].stat().st_size}' for k in (a,b)]
        signature = hashlib.sha1('|'.join(fingerprints).encode()).hexdigest()[:12]
        path = flow_dir/f'{a:04d}-{b:04d}-{signature}.npz'
        if path.exists():
            with np.load(path) as saved:
                forward, backward = saved['forward'].astype(np.float32), saved['backward'].astype(np.float32)
        else:
            if args.flow_method.startswith('raft'):
                if neural is None:
                    from render_neural_flow import RaftFlow
                    neural = RaftFlow(threads=args.threads,updates=args.raft_updates,large=args.flow_method=='raft-large')
                forward,backward = neural.both(ia,ib,motion.small_size)
                np.savez_compressed(path,forward=forward.astype(np.float16),backward=backward.astype(np.float16))
            else:
                dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                dis.setUseMeanNormalization(True)
                dis.setUseSpatialPropagation(True)
                dis.setVariationalRefinementIterations(12)
                ga, gb = gray(ia), gray(ib)
                def best_flow(left, right, prior):
                    candidates = [dis.calc(left,right,prior.copy()), dis.calc(left,right,None)]
                    def score(field):
                        mapping = motion.grid+field
                        valid = motion.validity(mapping)
                        delta = np.abs(remap(right,mapping).astype(np.float32)-left.astype(np.float32))
                        return float(np.sum(np.minimum(delta,32)*valid)/max(1,float(np.sum(valid))) + 32*(1-float(np.mean(valid))))
                    return min(candidates,key=score)
                forward = best_flow(ga,gb,source_ab)
                backward = best_flow(gb,ga,source_ba)
                np.savez_compressed(path,forward=forward.astype(np.float16),backward=backward.astype(np.float16))
        reliability_a = motion.consistent(forward,backward)
        reliability_b = motion.consistent(backward,forward)
        original_a = motion.anchor_maps(a,needed) if args.source_motion_strength else None
        original_b = motion.anchor_maps(b,needed) if args.source_motion_strength else None
        for i,t in targets:
            f = (t-a)/(b-a)
            if args.warp == 'splat':
                wa,ca=forward_splat(ia,forward,reliability_a,f,motion)
                wb,cb=forward_splat(ib,backward,reliability_b,1-f,motion)
                denominator=(1-f)*ca+f*cb
                alpha=np.divide(f*cb,denominator,out=np.full_like(cb,f),where=denominator>1e-7)[...,None]
                result=np.clip(wa*(1-alpha)+wb*alpha+.5,0,255).astype(np.uint8)
                cv2.imwrite(str(args.output/f'{i:04d}.png'),result,[cv2.IMWRITE_PNG_COMPRESSION,2])
                done += 1
                continue
            ma,qa = inverse_map(forward,f,motion.grid)
            mb,qb = inverse_map(backward,1-f,motion.grid)
            if args.source_motion_strength:
                base_a,_ = inverse_map(source_ab,f,motion.grid)
                base_b,_ = inverse_map(source_ba,1-f,motion.grid)
                actual_a,_ = fractional_map(original_a,t,shot_start,shot_end-1)
                actual_b,_ = fractional_map(original_b,t,shot_start,shot_end-1)
                taper = 4*f*(1-f)
                ma += args.source_motion_strength * taper * (actual_a-(base_a-motion.grid))
                mb += args.source_motion_strength * taper * (actual_b-(base_b-motion.grid))
            valid_a, valid_b = motion.validity(ma), motion.validity(mb)
            ca = remap(reliability_a,ma)*qa
            cb = remap(reliability_b,mb)*qb
            ca = (.025+.975*ca)*valid_a
            cb = (.025+.975*cb)*valid_b
            # Linear temporal weights avoid introducing easing at every anchor.
            denominator = (1-f)*ca+f*cb
            alpha = np.divide(f*cb,denominator,out=np.full_like(cb,f),where=denominator>1e-7)
            wa, wb = motion.warp(ia,ma-motion.grid), motion.warp(ib,mb-motion.grid)
            # When details cannot be aligned, avoid averaging two eyes/mouths.
            # Matching regions retain a linear dissolve; conflicts prefer the
            # reliable/nearer anchor through a sharper, still continuous weight.
            low_a = cv2.resize(wa,motion.small_size,interpolation=cv2.INTER_AREA).astype(np.float32)
            low_b = cv2.resize(wb,motion.small_size,interpolation=cv2.INTER_AREA).astype(np.float32)
            disagreement = cv2.GaussianBlur(np.mean(np.abs(low_a-low_b),axis=2),(3,3),0)
            power = 1+2*np.clip((disagreement-5)/22,0,1)
            left, right = np.power(1-alpha,power), np.power(alpha,power)
            alpha = np.divide(right,left+right,out=alpha,where=left+right>1e-8)
            alpha = cv2.GaussianBlur(alpha,(3,3),0)
            alpha = cv2.resize(alpha,motion.full_size,interpolation=cv2.INTER_LINEAR)[...,None]
            result = np.clip(wa.astype(np.float32)*(1-alpha)+wb.astype(np.float32)*alpha+.5,0,255).astype(np.uint8)
            cv2.imwrite(str(args.output/f'{i:04d}.png'),result,[cv2.IMWRITE_PNG_COMPRESSION,2])
            done += 1
        print(f'Rendered {done}/{len(selected)-skipped} in {time.monotonic()-started:.1f}s',flush=True)
    manifest = dict(source_fps=args.source_fps, fps=args.fps, source_frames=frame_count,
                    full_output_frames=math.ceil(frame_count*ratio), rendered_this_run=done,
                    skipped_without_anchors=skipped, cuts=args.cuts, width=motion.width,height=motion.height,
                    method=args.flow_method)
    (args.output/'render-info.json').write_text(json.dumps(manifest,indent=2))
    print(f'Finished {done} frames: {args.output}',flush=True)


if __name__ == '__main__':
    main()
