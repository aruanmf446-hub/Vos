"""Render a constant triangular topology to avoid per-frame remeshing changes."""
import cv2,numpy as np
from scipy.spatial import Delaunay
class FixedMesh:
 def __init__(self,points):
  self.points=np.asarray(points,np.float32)
  self.triangles=Delaunay(self.points).simplices
  self.source=self.points[self.triangles].astype(np.float64)
 def mapping(self,destination,size=(576,364)):
  w,h=size;dst=np.asarray(destination,np.float64)[self.triangles]
  homogeneous=np.concatenate([dst,np.ones((*dst.shape[:2],1))],axis=2)
  det=np.linalg.det(homogeneous);good=np.abs(det)>1e-5
  coeff=np.zeros((len(dst),3,2),np.float64);coeff[good]=np.linalg.solve(homogeneous[good],self.source[good])
  result=np.full((h,w,2),-1000,np.float32)
  for triangle,fit,valid in zip(dst,coeff,good):
   if not valid:continue
   lo=np.maximum(np.floor(triangle.min(0)).astype(int),[0,0]);hi=np.minimum(np.ceil(triangle.max(0)).astype(int)+1,[w,h])
   bw,bh=hi-lo
   if bw<=0 or bh<=0:continue
   mask=np.zeros((bh,bw),np.uint8);cv2.fillConvexPoly(mask,np.rint((triangle-lo)*256).astype(np.int32),1,shift=8)
   ys,xs=np.nonzero(mask)
   if len(xs)==0:continue
   xx=xs+lo[0];yy=ys+lo[1]
   values=xx[:,None]*fit[0]+yy[:,None]*fit[1]+fit[2]
   result[yy,xx]=values
  return result
