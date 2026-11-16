"""Replace only Doom's motion with stable source-face tracking, preserving shape."""
from pathlib import Path
import sys,json,subprocess,time
import cv2,numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
helper=(R/'deliver_full_video.py').read_text().split("log('Preparing Doom motion")[0]
exec(compile(helper,str(R/'deliver_full_video.py'),'exec'),globals())
detector=cv2.FaceDetectorYN.create(str(R/'models/yunet.onnx'),'',(576,364),.30,.3,5000)
positions=[]
for n in range(272,349):
    _,faces=detector.detect(src(n))
    candidates=[] if faces is None else [p for p in faces if p[2]*p[3]>2200 and p[1]<150 and p[-1]>.6]
    if not candidates:positions.append([np.nan,np.nan]);continue
    p=max(candidates,key=lambda p:p[2]*p[3])
    positions.append([(p[4]+p[6])/2,(p[5]+p[7])/2])
positions=smooth(positions,1.2);reference=positions[302-272].copy()
base=clean_frame(cv2.resize(cv2.imread('/workspace/generated_images/exec-bf3d6e55-f887-4ac8-a0c8-5b855eca5571.png'),(576,364)))
alpha=np.load(D/'doom-full-alpha.npy')
premul=cv2.GaussianBlur(base,(0,0),.6).astype(np.float32)*alpha[...,None]
pad=96
premul=cv2.copyMakeBorder(premul,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
alpha=cv2.copyMakeBorder(alpha,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
camera=np.load(D/'camera/0272.npy');plate=plates[302]
native=D/'complete-native-30fps.mp4';patched=D/'complete-doom-stable-30fps.mp4'
cap=cv2.VideoCapture(str(native));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==778
encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','576x854','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','16','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(patched)],stdin=subprocess.PIPE)
for n in range(778):
    ok,full=cap.read()
    if not ok:raise RuntimeError(f'Missing native frame {n}')
    if 272<=n<349:
        dx,dy=positions[n-272]-reference
        move=np.array([[1,0,dx-pad],[0,1,dy-pad]],np.float32)
        fg=cv2.warpAffine(premul,move,(576,364));a=cv2.warpAffine(alpha,move,(576,364))
        source=src(n);removal=original_mask(n)
        back=cv2.warpAffine(plate,camera[n-272][:2],(576,364),borderMode=cv2.BORDER_REFLECT_101)
        bg=source*(1-removal[...,None])+back*removal[...,None]
        full[230:594]=np.clip(fg+bg*(1-a[...,None])+.5,0,255).astype(np.uint8)
        if n in [272,282,302,322,348]:cv2.imwrite(str(D/'qa'/f'{n:04d}.jpg'),full)
    encoder.stdin.write(full.tobytes())
encoder.stdin.close();cap.release()
if encoder.wait():raise RuntimeError('Patch encoding failed')
check=cv2.VideoCapture(str(patched));assert int(check.get(cv2.CAP_PROP_FRAME_COUNT))==778;check.release()
native.replace(D/'complete-native-before-pose-correction-30fps.mp4');patched.replace(native)
report=json.loads((D/'delivery-report.json').read_text())
report['Doom_tracking']='Source face positions, smoothed at native FPS; rigid costume layer; no deformation of helmet or armor.'
report['Doom_motion_limits']='Single approved pose; hand gestures and body acting remain approximate.'
(D/'delivery-report.json').write_text(json.dumps(report,indent=2))
np.save(D/'doom-face-track.npy',positions)
print('Stable Doom patch complete; exporting complete MP4.',flush=True)
subprocess.run(['python',str(R/'export_adjacent60.py'),str(native),str(FINAL)],check=True)
print('Completed full MP4 with stable Doom tracking.',flush=True)
