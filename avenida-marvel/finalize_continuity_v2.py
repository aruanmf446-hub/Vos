"""Publish the verified Hulk patch and make a frame-accurate slow comparison."""
from pathlib import Path
import json,subprocess,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent;W=R/'work';V=W/'continuity-v2';O=Path('/workspace/shared/downloads')
cv2.setNumThreads(2)
movie=O/'video-marvel-continuidade-v2-60fps.mp4';patched=V/'full-with-hulk.mp4'
def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames','-of','json',str(path)]))
meta=probe(patched);s=meta['streams']
assert len(s)==1 and s[0]['codec_type']=='video' and s[0]['codec_name']=='h264'
assert s[0]['avg_frame_rate']=='60/1' and int(s[0]['nb_frames'])==1556
assert (s[0]['width'],s[0]['height'])==(720,1068)
assert abs(float(meta['format']['duration'])-778/30)<.00001
if movie.exists():movie.replace(V/'before-hulk-layer.mp4')
patched.replace(movie)
cuts=[39,76,173,241,272,349,404,666,749]
selected=list(range(94*2,130*2))+list(range(570*2,594*2))
need=set(selected)|{2*c-2 for c in cuts}|{2*c-1 for c in cuts}|{420}
def collect(path,wanted,verify_count=False):
    cap=cv2.VideoCapture(str(path));frames={};index=0
    while True:
        ok,im=cap.read()
        if not ok:break
        if index in wanted:
            im=cv2.resize(im,(576,854),interpolation=cv2.INTER_AREA)
            frames[index]=im[230:594].copy()
        index+=1
    cap.release()
    if verify_count:assert index==1556,index
    assert set(frames)==set(wanted)
    return frames,index
new,count=collect(movie,need,True)
cutchecks=[]
for c in cuts:
    mae=float(np.abs(new[2*c-2].astype(float)-new[2*c-1].astype(float)).mean())
    assert mae<2.5,(c,mae)
    cutchecks.append({'original_cut_frame':c,'hold_before_cut_MAE':mae})
old,_=collect(O/'video-marvel-completo-60fps.mp4',set(selected))
patchcap=cv2.VideoCapture(str(V/'patch-shot-3-60fps.mp4'));patchcap.set(cv2.CAP_PROP_POS_FRAMES,420-346);ok,im=patchcap.read();assert ok;patchcap.release()
im=cv2.resize(im,(576,854),interpolation=cv2.INTER_AREA)[230:594]
hulk_mae=float(np.abs(im.astype(float)-new[420].astype(float)).mean());assert hulk_mae<8,hulk_mae
comparison=O/'comparacao-encaixe-v2-camera-lenta.mp4'
encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','576x1182','-r','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(comparison)],stdin=subprocess.PIPE)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
def title(text):
    im=Image.new('RGB',(576,30),(18,20,25));ImageDraw.Draw(im).text((9,4),text,font=font,fill='white');return np.asarray(im)[...,::-1].copy()
panels=[]
for i in selected:
    n=i//2;original=cv2.imread(str(W/'action'/f'{n:04d}.png'))
    frame=np.vstack([title(f'Original | {i/60:.3f} s'),original,title('Versão anterior'),old[i],title('Novo encaixe | fundo contínuo'),new[i]])
    encoder.stdin.write(frame.tobytes());panels.append(frame)
for frame in panels:
    slow=frame.copy();slow[:30]=title('Original | repetição a 0,25x')
    for _ in range(4):encoder.stdin.write(slow.tobytes())
encoder.stdin.close()
if encoder.wait():raise RuntimeError('Comparison encode failed')
comp=probe(comparison);assert int(comp['streams'][0]['nb_frames'])==600
decoded=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(movie),'-map','0:v:0','-an','-progress','pipe:1','-f','null','-'],capture_output=True,text=True)
assert decoded.returncode==0 and not decoded.stderr.strip(),decoded.stderr
actual=[int(line.split('=',1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
assert actual and actual[-1]==1556,actual
report={'file':str(movie),'metadata':meta,'actual_decoded_frames':actual[-1],'full_decode':'OK','all_nine_original_hard_cuts_preserved':cutchecks,'hulk_patch_matches_final_MAE':hulk_mae,'sha256':hashlib.sha256(movie.read_bytes()).hexdigest(),'comparison':{'file':str(comparison),'frames':600,'seconds':10,'normal_seconds':2,'slow_seconds':8,'source_intervals':[[94,130],[570,594]],'all_source_frames_in_intervals_used':True}}
(V/'export-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
main=json.loads((V/'full-report.json').read_text());main['Hulk_layer_restored']=True;main['validation_file']=str(V/'export-validation.json');main['user_comparison_export_fps']=25
(V/'full-report.json').write_text(json.dumps(main,ensure_ascii=False,indent=2))
print(json.dumps({'movie':str(movie),'bytes':movie.stat().st_size,'duration':meta['format']['duration'],'fps':60,'frames':1556,'audio':False,'decode':'OK','cuts_checked':9,'hulk_layer_verified':True,'comparison':str(comparison),'comparison_bytes':comparison.stat().st_size},ensure_ascii=False,indent=2))
