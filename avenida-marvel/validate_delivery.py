"""Check the delivered media, duration, silence, decode, and hard-cut timing."""
from pathlib import Path
import hashlib,json,subprocess,cv2,numpy as np
R=Path(__file__).resolve().parent;D=R/'work/delivery'
final=Path('/workspace/shared/downloads/video-marvel-completo-60fps.mp4')
native=D/'complete-native-30fps.mp4';cuts=[39,76,173,241,272,349,404,666,749]
meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration','-of','json',str(final)]))
streams=meta['streams'];assert len(streams)==1 and streams[0]['codec_type']=='video',meta
v=streams[0]
assert v['codec_name']=='h264' and v['avg_frame_rate']=='60/1'
assert int(v['nb_frames'])==1556 and int(v['nb_read_frames'])==1556
assert (v['width'],v['height'])==(720,1068)
assert abs(float(v['duration'])-778/30)<.00001
decoded=subprocess.run(['ffmpeg','-v','error','-i',str(final),'-f','null','-'],capture_output=True,text=True)
assert decoded.returncode==0 and not decoded.stderr.strip(),decoded.stderr
a=cv2.VideoCapture(str(native));b=cv2.VideoCapture(str(final))
assert int(a.get(cv2.CAP_PROP_FRAME_COUNT))==778
def read(cap,n):
    cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,im=cap.read()
    assert ok,f'Missing frame {n}'
    if im.shape[:2]!=(854,576):im=cv2.resize(im,(576,854),interpolation=cv2.INTER_AREA)
    return im[230:594].astype(np.float32)
cut_checks=[]
for n in cuts:
    old=read(b,2*n-2);half=read(b,2*n-1)
    hold_error=float(np.abs(old-half).mean());assert hold_error<2.5,(n,hold_error)
    native_error=float(np.abs(read(a,n)-read(b,2*n)).mean());assert native_error<8,(n,native_error)
    cut_checks.append({'source_frame':n,'hold_before_cut_MAE':hold_error,'native_frame_MAE':native_error})
a.release();b.release()
report={'path':str(final),'bytes':final.stat().st_size,'sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'metadata':meta,'full_decode':'OK','source_frames':778,'output_frames':1556,'audio':False,'all_nine_original_hard_cuts_preserved':cut_checks}
(D/'export-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'file':str(final),'seconds':v['duration'],'frames':1556,'fps':60,'audio':False,'decode':'OK','hard_cuts_checked':len(cut_checks),'bytes':final.stat().st_size},indent=2))
