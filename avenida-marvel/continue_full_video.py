#!/usr/bin/env python3
"""Assemble every source frame, keeping unavailable replacement shots original."""
from pathlib import Path
from bisect import bisect_left
import json, subprocess, time
import cv2
import numpy as np

from render_rife_backend import Rife
from render_source import SourceMotion
from composite_consistent import OriginalBackgroundCompositor

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'
CUTS=[0,39,76,173,241,272,349,404,666,749,778]
NATIVE=WORK/'continuacao_30fps.mp4'
FINAL=Path('/workspace/shared/downloads/video-continuo-em-revisao-60fps.mp4')


def main():
    cv2.setNumThreads(2)
    anchors={int(p.stem):p for p in (WORK/'anchors').glob('[0-9][0-9][0-9][0-9].png')}
    image_cache={}
    def read(index):
        if index not in image_cache:
            image=cv2.imread(str(anchors[index]))
            if image is None:raise RuntimeError(f'Unreadable anchor {index}')
            image_cache[index]=cv2.resize(image,(576,364),interpolation=cv2.INTER_LANCZOS4)
        return image_cache[index]
    motion=SourceMotion(WORK/'action',WORK/'flow_full',384,'medium')
    rife=Rife(threads=2)
    compositor=OriginalBackgroundCompositor(segmentation_stride=2)
    shot_data=[]
    for start,end in zip(CUTS,CUTS[1:]):
        keys=sorted(k for k in anchors if start<=k<end)
        data={'start':start,'end':end,'keys':keys,'edge_maps':{},'edge_count':0,'replaced_count':0}
        if keys:
            for a,targets in [(keys[0],list(range(start,keys[0]))),(keys[-1],list(range(keys[-1]+1,end)))]:
                if targets:
                    data['edge_maps'].update({i:(a,flow) for i,(flow,_) in motion.anchor_maps(a,targets).items()})
            data['edge_count']=len(data['edge_maps'])
        shot_data.append(data)
    command=['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size','576x854','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','medium','-crf','17','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(NATIVE)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    qa=WORK/'full_qa';qa.mkdir(exist_ok=True)
    inspect={0,38,39,55,57,75,76,100,172,173,240,241,271,272,348,349,403,404,443,444,574,575,600,623,624,665,666,705,706,748,749,777}
    started=time.monotonic();shot_id=0
    try:
        for n in range(778):
            if n>=CUTS[shot_id+1]:shot_id+=1
            shot=shot_data[shot_id];keys=shot['keys']
            source=cv2.imread(str(WORK/'source'/f'{n:04d}.png'))
            if source is None:raise RuntimeError(f'Missing original frame {n}')
            full=source.copy()
            if keys:
                cached=WORK/'review_rife'/f'{n*2:04d}.png'
                if n in anchors:
                    generated=read(n)
                elif cached.exists():
                    generated=cv2.imread(str(cached))
                elif n in shot['edge_maps']:
                    a,flow=shot['edge_maps'][n]
                    generated=motion.warp(read(a),flow)
                else:
                    pos=bisect_left(keys,n);a,b=keys[pos-1],keys[pos]
                    fraction=(n-a)/(b-a)
                    generated=rife.between(read(a),read(b),fraction,('edited',a,b))
                    # For long gaps, gently follow the original performance
                    # instead of imposing only linear anchor-to-anchor timing.
                    if b-a>20:
                        old_a=motion.read(a);old_b=motion.read(b)
                        old_intermediate=rife.between(old_a,old_b,fraction,('original',a,b))
                        small_a=cv2.resize(cv2.cvtColor(source[230:594],cv2.COLOR_BGR2GRAY),(384,243))
                        small_b=cv2.resize(cv2.cvtColor(old_intermediate,cv2.COLOR_BGR2GRAY),(384,243))
                        dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                        residual=dis.calc(small_a,small_b,None)
                        full_flow=cv2.resize(residual,(576,364));full_flow[...,0]*=576/384;full_flow[...,1]*=364/243
                        full_flow=np.clip(full_flow,-12,12)*.30
                        yy,xx=np.mgrid[:364,:576].astype(np.float32)
                        generated=cv2.remap(generated,xx+full_flow[...,0],yy+full_flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                composed,_=compositor.process(source[230:594],generated,n,shot_id)
                full[230:594]=composed
                shot['replaced_count']+=1
            if n in inspect:cv2.imwrite(str(qa/f'{n:04d}.jpg'),full,[cv2.IMWRITE_JPEG_QUALITY,94])
            process.stdin.write(full.tobytes())
            if n%30==0:print(f'Frame {n}/778; shot {shot_id}; {time.monotonic()-started:.1f}s',flush=True)
        process.stdin.close()
        if process.wait():raise RuntimeError('Native encoder failed')
    except BaseException:
        if process.poll() is None:process.kill();process.wait()
        raise
    report={'source_frames':778,'duration':778/30,'no_cuts_removed':True,'silent':True,'output_fps':60,'shots':[{k:v for k,v in s.items() if k!='edge_maps'} for s in shot_data]}
    (WORK/'full_timeline_status.json').write_text(json.dumps(report,indent=2))
    print('Interpolating final full timeline to 60 FPS',flush=True)
    temporary=FINAL.with_name(FINAL.stem+'.encoding.mp4')
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(NATIVE),'-vf','tpad=stop_mode=clone:stop_duration=0.15,minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=8,scale=720:1068:flags=lanczos,setsar=1','-frames:v','1556','-an','-c:v','libx264','-preset','medium','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],check=True)
    temporary.replace(FINAL)
    print(f'Finished full-duration draft: {FINAL}',flush=True)


if __name__=='__main__':main()
