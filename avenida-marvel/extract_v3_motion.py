"""Measure source acting and approved-keyframe geometry; no image generation."""
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks.python import vision,BaseOptions
R=Path(__file__).resolve().parent;W=R/'work';V=W/'continuity-v3';V.mkdir(exist_ok=True)
face=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(R/'models/mediapipe/face_landmarker.task')),num_faces=2,min_face_detection_confidence=.35,min_face_presence_confidence=.35))
pose=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(R/'models/mediapipe/pose_landmarker_lite.task')),num_poses=2,min_pose_detection_confidence=.3,min_pose_presence_confidence=.3))
def extract(im,n):
 image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(im,cv2.COLOR_BGR2RGB));f=face.detect(image);p=pose.detect(image)
 fs=[]
 for entry in f.face_landmarks:
  a=np.array([[l.x*576,l.y*364,l.z*576] for l in entry],np.float32);fs.append(a)
 if fs:
  if 39<=n<76:
   f=min(fs,key=lambda a:a[:,1].mean())
  else:f=max(fs,key=lambda a:np.ptp(a[:,0])*np.ptp(a[:,1]))
 else:f=np.full((478,3),np.nan,np.float32)
 ps=[np.array([[l.x*576,l.y*364,l.z*576,l.visibility] for l in entry],np.float32) for entry in p.pose_landmarks]
 if ps:
  if np.isfinite(f).all():
   ctr=f[:,:2].mean(0);p=min(ps,key=lambda a:np.linalg.norm(a[0,:2]-ctr))
  else:p=max(ps,key=lambda a:np.linalg.norm(a[11,:2]-a[23,:2]))
 else:p=np.full((33,4),np.nan,np.float32)
 # The standing shot has two overlapping subjects; its selector is not
 # validated for the rising actor. Never present the seated face as Wanda.
 if 39<=n<76:f[:]=np.nan;p[:]=np.nan
 return f,p
try:
 path=V/'source-geometry.npz'
 if not path.exists():
  fs=[];ps=[]
  for n in range(778):
   im=cv2.imread(str(W/'action'/f'{n:04d}.png'));f,p=extract(im,n);fs.append(f);ps.append(p)
   if n%60==0:print('Source geometry',n,flush=True)
  np.savez_compressed(path,face=np.asarray(fs),pose=np.asarray(ps))
 path=V/'anchor-geometry.npz'
 if not path.exists():
  fs=[];ps=[];ns=[]
  inputs={int(p.stem):p for p in (W/'anchors').glob('*.png')}
  for n in [210,302,404,474,489,500,624]:
   if n in inputs:continue
   for d in [W/'delivery/raw',W/'generated_native_v3']:
    p=d/f'{n:04d}.png'
    if p.exists():inputs[n]=p;break
  inputs[302]=R.parent/'generated_images/exec-bf3d6e55-f887-4ac8-a0c8-5b855eca5571.png'
  for n,path_in in sorted(inputs.items()):
   im=cv2.resize(cv2.imread(str(path_in)),(576,364));f,p=extract(im,n);fs.append(f);ps.append(p);ns.append(n)
  np.savez_compressed(path,frames=np.array(ns),face=np.asarray(fs),pose=np.asarray(ps))
 print('Geometry saved',flush=True)
finally:
 face.close();pose.close()
