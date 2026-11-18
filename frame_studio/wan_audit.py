"""Audit the pinned upstream CPU functions directly, without Torch/models/inference."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
from types import SimpleNamespace

WAN_REV='1ea34ff48f87168174e12956e200b1d908b1c5ff'
SAM_REV='0e78a118995e66bb27d78518c4bd9a3e95b4e266'


def load_file(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def extract_method(path,cls,name,namespace):
    tree=ast.parse(Path(path).read_text());node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def audit(wan):
    for binary in ('ffmpeg','ffprobe'):
        if not shutil.which(binary):raise ValueError('Dependencia ausente: '+binary)
    import numpy as np
    import cv2
    wan=Path(wan).resolve()
    revision=subprocess.check_output(['git','-C',str(wan),'rev-parse','HEAD'],text=True).strip() if shutil.which('git') else (wan/'.git/HEAD').read_text().strip()
    if revision!=WAN_REV:raise ValueError('Revisao Wan diferente da auditada')
    pre=wan/'wan/modules/animate/preprocess'
    geometry=load_file(pre/'pose2d_utils.py','vos_official_pose_geometry')
    util=load_file(pre/'utils.py','vos_official_animate_utils')
    namespace={'np':np,'cv2':cv2,'bbox_from_detector':geometry.bbox_from_detector,'crop':geometry.crop}
    preprocess=extract_method(pre/'pose2d.py','ViTPose','preprocess',namespace)
    images=extract_method(pre/'pose2d.py','Pose2d','load_images',namespace)
    # Noncentral actor, distinct red/blue values: detect bbox and channel errors.
    rgb=np.zeros((160,256,3),dtype=np.uint8);rgb[40:130,150:220]=(230,40,10)
    box=np.array([150,40,220,130],dtype=np.float32)
    official_input=images(None,[rgb])[0]
    actual,center,scale=preprocess(official_input,box)
    if not np.array_equal(official_input,cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise ValueError('Contrato de canais mudou')
    if not np.allclose(center,[185,85]) or actual.shape!=(3,256,192) or not np.isfinite(actual).all():raise ValueError('Contrato de recorte ViTPose mudou')
    try:
        from .wan_target import prepare_pose_tensor
    except ImportError:
        from wan_target import prepare_pose_tensor
    adapted,adapted_center,adapted_scale=prepare_pose_tensor(SimpleNamespace(preprocess=preprocess),rgb,box)
    if not np.array_equal(adapted,actual) or not np.array_equal(adapted_center,center) or not np.array_equal(adapted_scale,scale):raise ValueError('Adaptador diverge do preprocessamento oficial')
    # Exercise actual face conversion and pose normalization.
    kps=np.zeros((1,133,3),dtype=np.float32);kps[0,:,0]=180;kps[0,:,1]=80;kps[0,:,2]=1
    kps[0,23:91,0]=np.linspace(170,195,68);kps[0,23:91,1]=np.linspace(65,95,68)
    meta=geometry.load_pose_metas_from_kp2ds_seq(kps,256,160)[0]
    face=util.get_face_bboxes(meta['keypoints_face'][:,:2],1.3,(160,256))
    if not (0<=face[0]<face[1]<=256 and 0<=face[2]<face[3]<=160):raise ValueError('Contrato de rosto invalido')
    requirements=(wan/'requirements_animate.txt').read_text()
    if SAM_REV not in requirements:raise ValueError('Revisao SAM2 mudou')
    return {'wan_revision':revision,'sam_revision':SAM_REV,'test':'real upstream CPU geometry, normalization and face functions; no inference',
        'pose_shape':list(actual.shape),'pose_center':center.tolist(),'face_bbox':list(face),
        'source_sha256':{name:hashlib.sha256((pre/name).read_bytes()).hexdigest() for name in ('pose2d.py','pose2d_utils.py','utils.py','sam_utils.py','video_predictor.py')}}


def dependencies():
    import importlib
    from importlib.metadata import version
    names=('torch','torchvision','torchaudio','numpy','cv2','decord','peft','onnxruntime','pandas','matplotlib','loguru','sentencepiece','moviepy','flash_attn','sam2')
    result={}
    for name in names:
        module=importlib.import_module(name);result[name]=getattr(module,'__version__','import OK')
    from diffusers import FluxKontextPipeline
    result['diffusers']=version('diffusers')
    result['onnx_providers']=importlib.import_module('onnxruntime').get_available_providers()
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wan',required=True);parser.add_argument('--dependencies',action='store_true');args=parser.parse_args()
    report=audit(args.wan)
    if args.dependencies:report['dependencies']=dependencies()
    print(json.dumps(report,indent=2))
