"""Join the refined opening at the existing cut, with exact frame counts."""
from pathlib import Path
import subprocess,json,cv2,hashlib
R=Path(__file__).resolve().parent;V=R/'work/continuity-v4';O=Path('/workspace/shared/downloads')
opening=V/'opening-final-60fps.mp4';full=O/'video-marvel-conexoes-v4-60fps.mp4';temporary=V/'full-finalizing.mp4'
def count(p):
 return int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','csv=p=0',str(p)]))
assert count(opening)==78 and count(full)==1556
command=['ffmpeg','-y','-v','error','-i',str(opening),'-i',str(full),'-filter_complex','[0:v]trim=start_frame=0:end_frame=78,setpts=PTS-STARTPTS[a];[1:v]trim=start_frame=78:end_frame=1556,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0[v]','-map','[v]','-frames:v','1556','-r','60','-an','-c:v','libx264','-preset','fast','-crf','17','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
subprocess.run(command,check=True);assert count(temporary)==1556
full.replace(V/'before-opening-cleanup-60fps.mp4');temporary.replace(full)
report=json.loads((V/'full-report.json').read_text());report['opening_refinement']={'output_interval':[0,78],'preserved_cut':78,'background':'One clean approved whole-frame plate for the whole opening. The mixed source-recovery atlas with visible scraps is no longer used.','silhouette':'Signed-distance interpolation of warped mattes near the middle of each pose transition; exact RGBA endpoints preserved.'};(V/'full-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(V/'opening-finalization.json').write_text(json.dumps({'command':command,'output_frames':1556,'opening_frames':78,'cut_frame':78,'sha256':hashlib.sha256(full.read_bytes()).hexdigest()},ensure_ascii=False,indent=2));print('Abertura refinada unida no corte original; 1556 quadros.')
