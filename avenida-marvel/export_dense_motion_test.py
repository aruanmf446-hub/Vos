from pathlib import Path
import subprocess, cv2, numpy as np
from PIL import Image,ImageDraw,ImageFont
from composite_continuity_v2 import ContinuousBackgroundCompositor
from texture_continuity_test import TextureMatch

ROOT=Path(__file__).resolve().parent;W=ROOT/'work';D=W/'dense_motion_test'
OUT=Path('/workspace/shared/downloads')

def encoder(path,size):
    return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',size,'-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

c=ContinuousBackgroundCompositor();texture=TextureMatch()
old=cv2.VideoCapture(str(W/'continuacao_corrigida_30fps.mp4'));old.set(cv2.CAP_PROP_POS_FRAMES,39)
header=Image.new('RGB',(1728,36),(18,20,25));draw=ImageDraw.Draw(header);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
for x,text in [(12,'Original'),(588,'Versão anterior'),(1164,'Novo teste — poses mais próximas')]:draw.text((x,5),text,font=font,fill='white')
head=np.asarray(header)[...,::-1].copy()
e=encoder(OUT/'comparacao-reacao-wanda-original.mp4','1728x400')
f=encoder(OUT/'teste-reacao-wanda-mais-denso.mp4','576x854')
(D/'composed').mkdir(exist_ok=True)
for i in range(39,76):
    source=cv2.imread(str(W/'source'/f'{i:04d}.png'));gen=cv2.imread(str(D/'rendered'/f'{i:04d}.png'))
    frame,diag=c.process(source[230:594],gen,i,'stand')
    frame,_=texture.process(frame)
    full=source.copy();full[230:594]=frame
    ok,previous=old.read()
    if not ok:raise RuntimeError('Missing comparison frame')
    panel=np.empty((400,1728,3),np.uint8);panel[:36]=head;panel[36:,:576]=source[230:594];panel[36:,576:1152]=previous[230:594];panel[36:,1152:]=frame
    e.stdin.write(panel.tobytes());f.stdin.write(full.tobytes())
    cv2.imwrite(str(D/'composed'/f'{i:04d}.jpg'),panel)
for proc in [e,f]:
    proc.stdin.close()
    if proc.wait():raise RuntimeError('Encoding failed')
old.release()
print('Created37-frame original-time motion comparison and standalone test.',flush=True)
