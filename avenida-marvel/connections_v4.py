"""Repair folded opaque triangles and regularize trajectories in scene space."""
import numpy as np,cv2
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.spatial import cKDTree

def cross2(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
def project_no_folds(rest,target,triangles,active):
 q=np.asarray(target,np.float64).copy();p=np.asarray(rest,np.float64);tri=triangles[active]
 ar=cross2(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
 sign=np.sign(ar);minimum=np.maximum(np.abs(ar)*.005,.0005)
 def badness(x):
  a,b,c=x[tri[:,0]],x[tri[:,1]],x[tri[:,2]]
  return cross2(b-a,c-a)*sign
 before=int((badness(q)<minimum).sum())
 for iteration in range(180):
  area=badness(q);bad=area<minimum
  if not np.any(bad):break
  t=tri[bad];a,b,c=q[t[:,0]],q[t[:,1]],q[t[:,2]];sg=sign[bad,None]
  ga=np.stack([b[:,1]-c[:,1],c[:,0]-b[:,0]],1)*sg
  gb=np.stack([c[:,1]-a[:,1],a[:,0]-c[:,0]],1)*sg
  gc=np.stack([a[:,1]-b[:,1],b[:,0]-a[:,0]],1)*sg
  grad=np.stack([ga,gb,gc],1);step=(minimum[bad]*1.05-area[bad])/np.maximum((grad*grad).sum((1,2)),1e-12)
  delta=grad*step[:,None,None]*.95
  correction=np.zeros_like(q);count=np.zeros(len(q))
  np.add.at(correction,t.ravel(),delta.reshape(-1,2));np.add.at(count,t.ravel(),1)
  q+=correction/np.maximum(count[:,None],1)
 retained=1.
 if np.any(badness(q)<minimum*.99):
  # A rigid least-squares pose is the valid endpoint of the line search.
  pc=p.mean(0);qc=q.mean(0);u,_,vt=np.linalg.svd((p-pc).T@(q-qc));rot=u@vt
  if np.linalg.det(rot)<0:u[:,-1]*=-1;rot=u@vt
  scale=np.clip(np.sum(((p-pc)@rot)*(q-qc))/np.maximum(((p-pc)**2).sum(),1e-9),.6,1.7)
  rigid=(p-pc)@rot*scale+qc
  lo,hi=0.,1.
  for _ in range(22):
   t=(lo+hi)/2;candidate=rigid+(q-rigid)*t
   if np.all(badness(candidate)>=minimum):lo=t
   else:hi=t
  retained=lo;q=rigid+(q-rigid)*lo
 after=int((badness(q)<minimum*.99).sum())
 return q.astype(np.float32),{'folds_before':before,'folds_after':after,'deformation_retained':retained,'mean_vertex_adjustment':float(np.linalg.norm(q-target,axis=1).mean())}

class SceneTrajectories:
 def __init__(self,puppet,raw,cameras,reference_camera):
  self.raw=np.asarray(raw,np.float32);self.cameras=np.asarray(cameras,np.float32)
  inv=np.linalg.inv(self.cameras);world=np.einsum('nij,nkj->nki',inv[:,:2,:2],self.raw)+inv[:,None,:2,2]
  clean=gaussian_filter1d(median_filter(world,size=(3,1,1),mode='nearest'),.8,axis=0,mode='nearest')
  if puppet.hasface:
   _,face_ids=cKDTree(puppet.points).query(puppet.face[:468,:2]);clean[:,face_ids]=gaussian_filter1d(world[:,face_ids],.35,axis=0,mode='nearest')
  ir=np.linalg.inv(reference_camera);rest=puppet.registered@ir[:2,:2].T+ir[:2,2]
  tri=puppet.fixed_mesh.triangles;xy=puppet.points[tri].astype(int);coverage=puppet.alpha[np.clip(xy[...,1],0,363),np.clip(xy[...,0],0,575)].mean(1)
  areas=np.abs(cross2(rest[tri[:,1]]-rest[tri[:,0]],rest[tri[:,2]]-rest[tri[:,0]]));active=(coverage>.5)&(areas>.05)
  solved=[];self.records=[]
  for target in clean:
   q,record=project_no_folds(rest,target,tri,active);solved.append(q);self.records.append(record)
  solved=np.asarray(solved);self.output=np.einsum('nij,nkj->nki',self.cameras[:,:2,:2],solved)+self.cameras[:,None,:2,2]
  self.summary={'visible_triangles':int(active.sum()),'inverted_before_filter':int(sum(r['folds_before'] for r in self.records)),'inverted_after_repair':int(sum(r['folds_after'] for r in self.records)),'mean_adjustment':float(np.mean([r['mean_vertex_adjustment'] for r in self.records])),'minimum_deformation_retained':float(min(r['deformation_retained'] for r in self.records))}
