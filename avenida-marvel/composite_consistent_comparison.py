#!/usr/bin/env python3
"""Export the approved continuous before/after background-compositing test."""
from pathlib import Path
import json,time,subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from composite_consistent import OriginalBackgroundCompositor

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'
DEST=Path('/workspace/shared/downloads')
COMPARISON=DEST/'comparacao-fundo-fluidez-60fps.mp4'
FIXED=DEST/'teste-fundo-original-60fps.mp4'


def encoder(path,size,filters=None):
    args=['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size',size,'-framerate','60','-i','pipe:0']
    if filters:args+=['-vf',filters]
    args+=['-an','-c:v','libx264','-preset','medium','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    return subprocess.Popen(args,stdin=subprocess.PIPE)


def main():
    DEST.mkdir(exist_ok=True,parents=True)
    out=WORK/'consistency_test'/'wanda_comparison'
    out.mkdir(exist_ok=True,parents=True)
    ids=list(range(888,1150))
    missing=[i for i in ids if not (WORK/'review_rife'/f'{i:04d}.png').exists()]
    if missing:raise RuntimeError(f'Missing rendered frames {missing}')
    header=Image.new('RGB',(1152,36),(18,20,25));draw=ImageDraw.Draw(header)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    draw.text((14,5),'Antes',font=font,fill=(245,245,245))
    draw.text((590,5),'Depois — fundo original',font=font,fill=(245,245,245))
    header=np.array(header)[...,::-1].copy()
    comparison_tmp=COMPARISON.with_name(COMPARISON.stem+'.encoding.mp4')
    fixed_tmp=FIXED.with_name(FIXED.stem+'.encoding.mp4')
    ec=encoder(comparison_tmp,'1152x400')
    ef=encoder(fixed_tmp,'576x854','scale=720:1068:flags=lanczos,setsar=1')
    c=OriginalBackgroundCompositor()
    records=[];started=time.monotonic();last_source=None;full=None
    for n,i in enumerate(ids):
        si=i//2
        if si!=last_source:
            full=cv2.imread(str(WORK/'source'/f'{si:04d}.png'));last_source=si
        src=full[230:594]
        gen=cv2.imread(str(WORK/'review_rife'/f'{i:04d}.png'))
        fixed,metrics=c.process(src,gen,si,'Wanda_dialogue')
        records.append({'index':i,**metrics})
        composite=np.empty((400,1152,3),dtype=np.uint8);composite[:36]=header
        composite[36:,:576]=gen;composite[36:,576:]=fixed
        ec.stdin.write(composite.tobytes())
        complete=full.copy();complete[230:594]=fixed
        ef.stdin.write(complete.tobytes())
        if i in [888,977,978,1000,1067,1068,1100,1149]:
            cv2.imwrite(str(out/f'{i:04d}.png'),composite)
        if n%60==0:print(f'{n}/{len(ids)} frames, {time.monotonic()-started:.1f}s',flush=True)
    for proc in [ec,ef]:
        proc.stdin.close()
        if proc.wait():raise RuntimeError('Encoding failed')
    comparison_tmp.replace(COMPARISON);fixed_tmp.replace(FIXED)
    result={'source_range':[444,575],'global_output_range':[888,1150],'source_frames':131,'output_frames':262,'fps':60,'duration':262/60,'runtime':time.monotonic()-started,'same_continuous_shot':True,'source_outside_matte_mae_before':float(np.mean([r['before_background_mae'] for r in records])),'source_outside_matte_mae_after':float(np.mean([r['after_background_mae'] for r in records])),'mean_mask_coverage':float(np.mean([r['coverage'] for r in records])),'comparison':str(COMPARISON),'fixed':str(FIXED),'frames':records}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2),flush=True)


if __name__=='__main__':main()
