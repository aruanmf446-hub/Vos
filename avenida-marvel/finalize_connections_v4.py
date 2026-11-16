"""Decode and validate the v4 timeline; compare matched v3 and v4 frames."""
from pathlib import Path
import subprocess,json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;V=R/'work/continuity-v4';O=Path('/workspace/shared/downloads')
NEW=O/'video-marvel-conexoes-v4-60fps.mp4';OLD=O/'video-marvel-continuidade-v3-60fps.mp4'
CUTS=[39,76,173,241,272,349,404,666,749]
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(NEW)]))
vs=[s for s in probe['streams'] if s['codec_type']=='video'];assert len(vs)==1 and not any(s['codec_type']=='audio' for s in probe['streams'])
s=vs[0];assert (s['width'],s['height'],s['avg_frame_rate'],int(s['nb_frames']))==(720,1068,'60/1',1556)
cap=cv2.VideoCapture(str(NEW));count=0;kept={};needed={j for c in CUTS for j in [2*c-2,2*c-1,2*c]}
while True:
 ok,frame=cap.read()
 if not ok:break
 if count in needed:kept[count]=frame[288:743].copy()
 count+=1
cap.release();assert count==1556
boundaries=[]
for c in CUTS:
 hold=float(np.abs(kept[2*c-1].astype(float)-kept[2*c-2]).mean());change=float(np.abs(kept[2*c].astype(float)-kept[2*c-1]).mean())
 assert hold<3,(c,hold)
 assert change>8,(c,change)
 boundaries.append({'source_cut':c,'output_cut':2*c,'pre_cut_hold_mae':hold,'cut_mae':change})
windows=[('Estranho',0,39),('Wanda / Hulk',173,241),('Doom',272,349),('Wanda',520,595)]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
old=cv2.VideoCapture(str(OLD));new=cv2.VideoCapture(str(NEW))
compare=O/'comparacao-conexoes-v3-v4.mp4'
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','720x976','-r','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(compare)],stdin=subprocess.PIPE)
def label(text):
 im=Image.new('RGB',(720,32),(18,20,24));ImageDraw.Draw(im).text((12,1),text,font=font,fill='white');return np.asarray(im)[...,::-1]
comparison_count=0
for name,start,end in windows:
 old.set(cv2.CAP_PROP_POS_FRAMES,2*start);new.set(cv2.CAP_PROP_POS_FRAMES,2*start)
 for i in range(2*(end-start)):
  ok,a=old.read();ok2,b=new.read();assert ok and ok2
  pa=cv2.resize(a[288:743],(720,456),interpolation=cv2.INTER_AREA);pb=cv2.resize(b[288:743],(720,456),interpolation=cv2.INTER_AREA)
  out=np.vstack([label('Anterior · '+name),pa,label('Revisão · '+name),pb]);enc.stdin.write(out.tobytes());comparison_count+=1
enc.stdin.close();assert enc.wait()==0
old.release();new.release()
subprocess.run(['ffmpeg','-v','error','-i',str(NEW),'-f','null','-'],check=True)
report={'file':str(NEW),'bytes':NEW.stat().st_size,'sha256':hashlib.sha256(NEW.read_bytes()).hexdigest(),'decoded_frames':count,'fps':60,'duration_seconds':count/60,'resolution':[720,1068],'audio':False,'original_cuts_checked':boundaries,'comparison_file':str(compare),'comparison_frames':comparison_count,'source_motion_measurements_per_second':30,'output_midpoints':'Scene-space trajectories and separate camera interpolation; integrated inverse velocity field sampled once; shared RGB/alpha RIFE only for opening pose changes.','new_image_generations':0,'reworked_shots':[0,2,3,4,5,6,7,8],'retained_shots':[1,9],'retained_flash_frames':[747,748],'limitations':['2D motion transfer; no new hand poses, hidden surfaces or physically simulated cloth.','Opening uses nine approved views; standing and final portrait retain v2.','Lip interiors and reactions are approximate; no claim of complete reconstruction of original acting.']}
(V/'export-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'frames':count,'duration':count/60,'cuts':len(boundaries),'audio':False,'comparison_frames':comparison_count,'bytes':NEW.stat().st_size},indent=2))
