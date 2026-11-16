from pathlib import Path
import json,sys,numpy as np,cv2
import render_continuity_v3 as m
V=m.W/'continuity-v4';V.mkdir(exist_ok=True)
def stats(p,q):
 t=p.fixed_mesh.triangles;rest=p.registered[t];dst=q[t]
 area=lambda a: np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
 ar=area(rest);ad=area(dst);xy=p.points[t].astype(int);x=np.clip(xy[...,0],0,575);y=np.clip(xy[...,1],0,363);opaque=p.alpha[y,x].mean(1)>.5
 live=opaque&(np.abs(ar)>.05);ratio=ad[live]/ar[live]
 return {'visible_triangles':int(live.sum()),'inverted':int((ratio<=0).sum()),'nearly_collapsed':int((ratio<.025).sum()),'max_area_ratio':float(ratio.max()),'min_area_ratio':float(ratio.min())}
records=[]
for ref,ns in [(15,[10,14,15,16,19,20]),(302,[272,284,302,320,330,348]),(474,[404,490,529,534,535,536,580,624,665])]:
 p=m.Puppet(ref)
 for n in ns:records.append({'ref':ref,'frame':n,**stats(p,p.positions(n))})
print(json.dumps(records,indent=2));(V/'audit-v3.json').write_text(json.dumps(records,indent=2))
