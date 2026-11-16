#!/usr/bin/env python3
"""Read-only visual/temporal audit of the latest draft and original performance."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'
OUT=WORK/'motion_review'
ORIGINAL=Path('/tmp/codex-remote-attachments/6afa7d92-0128-8191-adc5-72526702d031/1E20634F-D279-4FB4-8BDA-96DB412A04A7/1-ssstik.io_1794801823730-2.mp4')
CUTS=[0,39,76,173,241,272,349,404,666,749,778]
WINDOWS=[(50,76),(76,113),(472,543),(566,633),(698,715)]
SEAMS=[489,534,575,624]


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True,parents=True)
    detector=cv2.FaceDetectorYN.create(str(ROOT/'models/yunet.onnx'),'',(576,364),.5,.3,5000)
    source=cv2.VideoCapture(str(ORIGINAL))
    edited=cv2.VideoCapture(str(WORK/'continuacao_corrigida_30fps.mp4'))
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    hh,ww=182,288
    yy,xx=np.mgrid[:hh,:ww].astype(np.float32)
    keep={55,60,65,489,534,624}
    contact_ranges={'stand':list(range(54,66)), 'seam489':list(range(484,496)), 'seam534':list(range(529,541)), 'gap575':list(range(570,582))}
    wanted=keep|{i for indices in contact_ranges.values() for i in indices}
    saved={}
    records=[]
    previous=None
    def face(im):
        _,faces=detector.detect(im)
        if faces is None:return None
        # The principal actor is normally the largest foreground face.
        best=max(faces,key=lambda r:r[2]*r[3])
        x,y,w,h=map(float,best[:4])
        return [x+w/2,y+h/2,w,h,float(best[-1])]
    def motion(a,b):
        ga=cv2.resize(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),(ww,hh))
        gb=cv2.resize(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),(ww,hh))
        backward=dis.calc(gb,ga,None)
        warped=cv2.remap(ga,xx+backward[...,0],yy+backward[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        residual=np.abs(gb.astype(np.float32)-warped.astype(np.float32))
        roi=np.s_[0:155,70:275]
        return {'motion_compensated_roi_mae':float(residual[roi].mean()),'mean_flow_xy':backward[roi].mean(axis=(0,1)).tolist(),'mean_abs_flow':float(np.linalg.norm(backward[roi],axis=2).mean())}
    n=0
    while True:
        ok1,a=source.read();ok2,b=edited.read()
        if not(ok1 and ok2):break
        a=a[230:594];b=b[230:594]
        relevant=any(s<=n<e for s,e in WINDOWS)
        if n in wanted:saved[n]=(a.copy(),b.copy())
        if relevant:
            fa,fb=face(a),face(b)
            record={'frame':n,'time':n/30,'source_face':fa,'edited_face':fb}
            if fa and fb:
                record['center_offset_pixels']=[fb[0]-fa[0],fb[1]-fa[1]]
                record['width_ratio']=fb[2]/fa[2]
            if previous and n not in CUTS:
                record['source_motion']=motion(previous[0],a)
                record['edited_motion']=motion(previous[1],b)
            records.append(record)
        previous=(a,b);n+=1
    source.release();edited.release()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    def tile(im,label):
        p=Image.new('RGB',(384,267),(22,22,25))
        p.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((384,243)),(0,24))
        ImageDraw.Draw(p).text((7,3),label,font=font,fill='white')
        return p
    for name,indices in contact_ranges.items():
        sheet=Image.new('RGB',(1536,801))
        for j,i in enumerate(indices):
            if i in saved:sheet.paste(tile(saved[i][1],f'{i} / {i/30:.3f}s'),((j%4)*384,(j//4)*267))
        sheet.save(OUT/f'{name}-consecutive.jpg',quality=93)
    paired=Image.new('RGB',(768,267*len(keep)))
    for row,i in enumerate(sorted(keep)):
        if i in saved:
            paired.paste(tile(saved[i][0],f'{i} original'),(0,row*267))
            paired.paste(tile(saved[i][1],f'{i} draft'),(384,row*267))
    paired.save(OUT/'source-draft-framing.jpg',quality=93)
    by={r['frame']:r for r in records}
    summary=[]
    for start,end in WINDOWS:
        rows=[r for r in records if start<=r['frame']<end]
        delta=[r['center_offset_pixels'] for r in rows if 'center_offset_pixels' in r]
        motion_rows=[r for r in rows if 'edited_motion' in r]
        summary.append({'range':[start,end],'face_pairs':len(delta),'median_xy_offset':np.median(delta,axis=0).tolist() if delta else None,'median_offset_magnitude':float(np.median(np.linalg.norm(delta,axis=1))) if delta else None,'source_motion_compensated_roi_mae':float(np.mean([r['source_motion']['motion_compensated_roi_mae'] for r in motion_rows])) if motion_rows else None,'edited_motion_compensated_roi_mae':float(np.mean([r['edited_motion']['motion_compensated_roi_mae'] for r in motion_rows])) if motion_rows else None})
    result={'frames_decoded':n,'source_fps':30,'note':'Face selection is largest-face heuristic and may select the seated actor while the standing face is partially cropped; do not use early stand offsets as ground truth. Optical-flow residual cannot prove visually correct anatomy.','summary':summary,'records':records,'compositor_issue':'Registration changes only every8frames; current .75/.25 update is applied as a step then held. Use a target plus every-frame smoothing or precomputed smooth transform curve.'}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'frames':n,'summary':summary,'contacts':str(OUT)},indent=2),flush=True)


if __name__=='__main__':main()
