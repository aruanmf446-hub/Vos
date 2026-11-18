"""Experimental, opt-in SAM2 target tracking and selected-person Wan conditioning.
Uses the pinned Wan preprocessing modules and local checkpoints, no Hub/API calls.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from PIL import Image
try:
    from . import target_masks as target
except ImportError:
    import target_masks as target


def upstream(wan):
    path=Path(wan).resolve()/'wan/modules/animate/preprocess'
    if not (path/'sam_utils.py').is_file():raise ValueError('Preprocessador Wan oficial ausente')
    sys.path.insert(0,str(path))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')


def sam_points(points,size):
    """Return absolute XY pixels and labels separately for SAM2 normalize_coords=True."""
    import numpy as np
    pixels=np.array([[x*size[0],y*size[1]] for x,y,_ in points],dtype=np.float32)
    pixels=np.minimum(pixels,np.array([size[0]-1,size[1]-1],dtype=np.float32))
    return pixels,np.array([p[2] for p in points],dtype=np.int32)


def track(work,wan,checkpoint,precision="bf16"):
    work=Path(work).resolve();target.invalidate(work);cfg=target.config(work)
    _,frames,size=target.context(work)
    if len(frames)>600:raise ValueError('Rastreamento experimental limitado a 600 quadros nativos')
    upstream(wan)
    import numpy as np
    import torch
    from sam_utils import build_sam2_video_predictor
    folder=work/'target_masks';shutil.rmtree(folder,ignore_errors=True);folder.mkdir()
    with tempfile.TemporaryDirectory(prefix='sam2-frames-',dir=work) as temporary:
        for n,f in enumerate(frames):
            with Image.open(work/f['file']) as im:im.convert('RGB').save(Path(temporary)/f'{n:06d}.jpg',quality=95)
        if precision=="fp16":
            import sam2.modeling.sam.transformer as sam_attention
            sam_attention.USE_FLASH_ATTN=False;sam_attention.MATH_KERNEL_ON=True;sam_attention.OLD_GPU=True
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16 if precision=="fp16" else torch.bfloat16):
            predictor=build_sam2_video_predictor('sam2_hiera_l.yaml',str(Path(checkpoint)/'process_checkpoint/sam2/sam2_hiera_large.pt'))
            state=predictor.init_state(video_path=temporary,offload_video_to_cpu=True,offload_state_to_cpu=True)
            for key,entry in sorted(cfg['keyframes'].items(),key=lambda kv:int(kv[0])):
                pts,labels=sam_points(entry['points'],size)
                predictor.add_new_points(inference_state=state,frame_idx=int(key),obj_id=1,points=pts,labels=labels,normalize_coords=True)
            seeds=work/'target_seeds'
            if seeds.is_dir():
                for path in sorted(seeds.glob('frame_*.png')):
                    n=int(path.stem.split('_')[1])
                    if not 0<=n<len(frames):raise ValueError('Quadro-chave corrigido fora da selecao')
                    mask=np.array(target.load_mask(path,size))>0
                    predictor.add_new_mask(inference_state=state,frame_idx=n,obj_id=1,mask=mask)
            seen=set()
            for n,ids,logits in predictor.propagate_in_video(state):
                if list(ids)!=[1] or n in seen or not 0<=n<len(frames):raise ValueError('SAM2 retornou objeto/quadros inesperados')
                pixels=(logits[0,0]>0).cpu().numpy().astype('uint8')*255
                if pixels.shape!=(size[1],size[0]):raise ValueError('Dimensoes SAM2 divergentes')
                Image.fromarray(pixels).save(target.mask_path(work,n));seen.add(n)
            if seen!=set(range(len(frames))):raise ValueError('Rastreamento incompleto: nenhuma aprovacao liberada')
    report=target.audit(work)
    print('SAM2 terminou; revise todos os quadros nativos. Alertas:',len(report['issues']))


def bbox_xyxy_from_mask(mask):
    """BBox [left, top, right, bottom] exigido pelo ViTPose oficial.
    Nao usar [x, y, width, height]: bbox_from_detector trata os ultimos
    campos como coordenadas absolutas e deslocaria a pose para outra pessoa.
    """
    import numpy as np
    yy, xx = np.where(mask)
    if len(xx) == 0:
        raise ValueError("Mascara vazia para estimativa de pose")
    return np.array([xx.min(), yy.min(), xx.max()+1, yy.max()+1], dtype=np.float32)


def prepare_pose_tensor(model,rgb,bbox):
    """Match the actual official Pose2d.load_images -> ViTPose.preprocess handoff."""
    import cv2
    return model.preprocess(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),bbox)


def prepare_selected(work,wan,checkpoint):
    """Pose and face come only from pixels inside the reviewed target mask.
    Original, unmasked scene is used for src_bg. No automatic person detector.
    """
    work=Path(work).resolve();target.require_review(work);upstream(wan)
    import cv2
    import numpy as np
    from pose2d import Pose2d
    from pose2d_utils import AAPoseMeta,load_pose_metas_from_kp2ds_seq
    from human_visualization import draw_aapose_by_meta_new
    from utils import resize_by_area,get_face_bboxes
    try:
        from .wan_bridge import cmd
    except ImportError:
        from wan_bridge import cmd
    info=json.loads((work/'bridge.json').read_text());_,frames,size=target.context(work)
    temporal=json.loads((work/'temporal_map.json').read_text())
    stage=work/'wan_processed';stage.mkdir(exist_ok=True)
    pose=Pose2d(checkpoint=str(Path(checkpoint)/'process_checkpoint/pose2d/vitpose_h_wholebody.onnx'),detector_checkpoint=None)
    with tempfile.TemporaryDirectory(prefix='target-conditioning-',dir=work) as temporary:
        root=Path(temporary)
        for name in ('pose','face','bg','mask'):(root/name).mkdir()
        for n,source_index in enumerate(temporal['wan_to_native']):
            with Image.open(work/frames[source_index]['file']) as im:original=np.array(im.convert('RGB'))
            original=resize_by_area(original,1280*720,divisor=16);h,w=original.shape[:2]
            native=target.load_mask(target.mask_path(work,source_index),size)
            mask=np.array(native.resize((w,h),Image.Resampling.NEAREST))>0
            isolated=original*mask[:,:,None]
            bbox=bbox_xyxy_from_mask(mask)
            # ViTPose.preprocess expects XYXY, not XYWH; use upstream
            # crop + normalization so pose follows the selected person.
            # Match official Pose2d.load_images before ViTPose normalization.
            tensor,center,scale=prepare_pose_tensor(pose.model,isolated,bbox)
            kps=pose.model(tensor[None],center[None],scale[None])
            meta=load_pose_metas_from_kp2ds_seq(kps,width=w,height=h)[0]
            x1,x2,y1,y2=get_face_bboxes(meta['keypoints_face'][:,:2],scale=1.3,image_shape=(h,w))
            if x2<=x1 or y2<=y1:raise ValueError(f'Rosto do alvo invalido no quadro Wan {n}')
            face=cv2.resize(isolated[y1:y2,x1:x2],(512,512))
            skeleton=draw_aapose_by_meta_new(np.zeros_like(original),AAPoseMeta.from_humanapi_meta(meta))
            outputs={'pose':skeleton,'face':face,'bg':original*(~mask)[:,:,None],'mask':np.repeat((mask.astype('uint8')*255)[:,:,None],3,axis=2)}
            for name,array in outputs.items():Image.fromarray(array.astype('uint8')).save(root/name/f'frame_{n:06d}.png')
        for name in ('pose','face','bg','mask'):
            # Lossless RGB avoids mask compression expanding editable pixels.
            cmd(['ffmpeg','-v','error','-y','-framerate','30','-i',root/name/'frame_%06d.png','-c:v','libx264rgb','-crf','0','-pix_fmt','rgb24',stage/f'src_{name}.mp4'])
    shutil.copyfile(work/'reference.png',stage/'src_ref.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--precision',choices=['bf16','fp16'],default='bf16');parser.add_argument('action',choices=['track','preprocess'])
    for name in ('work','wan','checkpoint'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.action=='track':track(args.work,args.wan,args.checkpoint,args.precision)
    else:prepare_selected(args.work,args.wan,args.checkpoint)
