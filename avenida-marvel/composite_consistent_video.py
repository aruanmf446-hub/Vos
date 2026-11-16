#!/usr/bin/env python3
"""Preserve source video background around existing generated character video.

This does no identity generation: it composites existing edited frames through
temporally stabilized source/generated person mattes onto original video pixels.
"""
from pathlib import Path
import argparse, json, subprocess, time
import cv2
import numpy as np

from composite_consistent_test import ROOT, WORK, PersonMatte, soft_mask, mix


def remap_mask(mask, flow):
    h,w=mask.shape
    y,x=np.mgrid[:h,:w].astype(np.float32)
    return cv2.remap(mask,x+flow[...,0],y+flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',type=int,default=359)
    p.add_argument('--end',type=int,default=400)
    p.add_argument('--input',type=Path,default=WORK/'review_rife')
    p.add_argument('--out',type=Path,default=WORK/'consistency_test'/'strange_frames')
    p.add_argument('--temporal-weight',type=float,default=.2)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    model=PersonMatte();cv2.setNumThreads(2)
    flow_model=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    mask_cache={};original_cache={};previous_alpha=None;previous_gray=None
    record=[];started=time.monotonic()
    comparison=a.out.parent/(a.out.name+'-comparison.mp4')
    fixed=a.out.parent/(a.out.name+'-fixed.mp4')
    def encoder(path,size):
        return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size',size,'-framerate','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    ec=encoder(comparison,'1152x364');ef=encoder(fixed,'576x364')
    for index in range(a.start*2,a.end*2):
        si=index//2
        if si not in original_cache:
            original_cache={si:cv2.imread(str(WORK/'action'/f'{si:04d}.png'))}
        src=original_cache[si]
        gen=cv2.imread(str(a.input/f'{index:04d}.png'))
        if src is None or gen is None:raise FileNotFoundError(index)
        if si not in mask_cache:
            _,mo=model(src);mask_cache={si:mo}
        else:mo=mask_cache[si]
        _,mn=model(gen)
        hard=np.maximum(mo,mn)
        alpha=soft_mask(hard,3,3)
        gray=cv2.cvtColor(gen,cv2.COLOR_BGR2GRAY)
        if previous_alpha is not None and a.temporal_weight:
            # Backward current->previous optical flow transports the previous
            # matte without holding its silhouette at obsolete screen positions.
            flow=flow_model.calc(gray,previous_gray,None)
            transported=remap_mask(previous_alpha,flow)
            stable=(1-a.temporal_weight)*alpha+a.temporal_weight*transported
            # Current hard coverage wins; temporal averaging only softens edges.
            alpha=np.maximum(alpha*.85,stable)
            alpha[alpha<.015]=0
        previous_alpha=alpha;previous_gray=gray
        comp=mix(src,gen,alpha)
        outside=alpha==0
        changed_bg=(np.abs(gen.astype(np.float32)-src.astype(np.float32)).mean(2))[outside]
        delta=np.abs(comp.astype(np.float32)-src.astype(np.float32)).mean(2)[outside]
        record.append({'index':index,'coverage':float((alpha>0).mean()),'old_background_mae':float(changed_bg.mean()),'fixed_background_mae':float(delta.mean())})
        cv2.imwrite(str(a.out/f'{index:04d}.png'),comp,[cv2.IMWRITE_PNG_COMPRESSION,2])
        pair=np.concatenate([gen,comp],axis=1)
        cv2.putText(pair,'Previous generated background',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1,cv2.LINE_AA)
        cv2.putText(pair,'Original video background',(584,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,220,255),1,cv2.LINE_AA)
        ec.stdin.write(pair.tobytes());ef.stdin.write(comp.tobytes())
        if index%20==0:print(f'{index}: {time.monotonic()-started:.1f}s',flush=True)
    for proc in [ec,ef]:
        proc.stdin.close()
        if proc.wait():raise RuntimeError('ffmpeg failed')
    summary={'frames':len(record),'duration':len(record)/60,'runtime':time.monotonic()-started,'mean_original_background_error':float(np.mean([r['old_background_mae'] for r in record])),'mean_fixed_background_error':float(np.mean([r['fixed_background_mae'] for r in record])),'comparison':str(comparison),'fixed':str(fixed),'frames_metrics':record}
    (a.out.parent/(a.out.name+'-metrics.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='frames_metrics'},indent=2))


if __name__=='__main__':main()
