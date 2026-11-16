"""Integrate a smooth velocity field instead of drawing overlapping triangles.

The inverse deformation is composed as coordinate fields; RGB and alpha are
sampled together once. The global camera/rigid pose remains a separate affine.
"""
import cv2,numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
class ContinuousWarp:
 def __init__(self,points,pad=128):
  self.pad=pad;self.h=364+2*pad;self.w=576+2*pad
  yy,xx=np.mgrid[:self.h,:self.w].astype(np.float32);self.grid=np.dstack([xx,yy])
  xy=self.grid.reshape(-1,2)-pad;tri=Delaunay(points);ids=tri.find_simplex(xy);valid=np.flatnonzero(ids>=0);s=ids[valid]
  bary=np.einsum('nij,nj->ni',tri.transform[s,:2],xy[valid]-tri.transform[s,2]);weights=np.c_[bary,1-bary.sum(1)].astype(np.float32)
  self.interpolation=csr_matrix((weights.ravel(),(np.repeat(valid,3),tri.simplices[s].ravel())),shape=(len(xy),len(points)),dtype=np.float32)
 def inverse(self,velocity):
  field=(self.interpolation@velocity).reshape(self.h,self.w,2).astype(np.float32)
  field=cv2.GaussianBlur(field,(0,0),1.15)
  # Eight squaring steps keep the initial map close to the identity, even at
  # the lips/eyelids where the original triangular warp could turn inside out.
  d=-field/256
  for _ in range(8):
   coords=self.grid+d;d=d+cv2.remap(d,coords[...,0],coords[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  return d
