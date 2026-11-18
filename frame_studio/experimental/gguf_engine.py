"""Local ComfyUI library execution. No server, tunnel or external inference API.
Each process holds only one model group; no weights enter diagnostic archives.
"""
import argparse
import gc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def package(name,path):
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def boot(root,text=False,efficient_only=False):
    root=Path(root).resolve();comfy=root/'ComfyUI';sys.path.insert(0,str(comfy))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DO_NOT_TRACK='1')
    sys.argv=['vos-gguf-local-library','--disable-dynamic-vram','--disable-async-offload','--disable-smart-memory',
        '--preview-method','auto','--use-pytorch-cross-attention','--disable-xformers','--disable-all-custom-nodes','--force-fp16',
        '--fp32-text-enc' if text else '--fp16-text-enc']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    import nodes
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(not efficient_only)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    wrapper=package('vos_wan_wrapper',comfy/'custom_nodes/ComfyUI-WanVideoWrapper')
    gguf=package('vos_city_gguf',comfy/'custom_nodes/ComfyUI-GGUF')
    registry=dict(nodes.NODE_CLASS_MAPPINGS);registry.update(wrapper.NODE_CLASS_MAPPINGS);registry.update(gguf.NODE_CLASS_MAPPINGS)
    return torch,registry


def invoke(registry,name,**kwargs):
    node=registry[name]();method=getattr(node,node.FUNCTION)
    inspect.signature(method).bind(**kwargs) # refuse drift, no fabricated fallback nodes
    return method(**kwargs)


def warm(root):
    torch,nodes=boot(root)
    for name in ('WanVideoModelLoader','WanVideoBlockSwap','WanVideoVAELoader','WanVideoLoraSelect','WanVideoAnimateEmbeds','WanVideoSampler','WanVideoDecode','WanVideoClipVisionEncode','WanVideoTextEmbedBridge','CLIPLoaderGGUF','CLIPVisionLoader','CLIPTextEncode'):
        if name not in nodes:raise ValueError('Node real ausente: '+name)
        nodes[name].INPUT_TYPES()
    from vos_wan_wrapper.gguf.gguf_utils import dequantize_blocks_Q3_K
    import gguf
    block,size=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q3_K]
    with torch.inference_mode():
        output=dequantize_blocks_Q3_K(torch.zeros((1,size),device='cuda',dtype=torch.uint8),block,size,dtype=torch.float16)
        if output.dtype!=torch.float16 or not torch.isfinite(output).all():raise ValueError('Kernel GGUF Q3_K invalido')
    print(json.dumps({'runtime_imports_verified':True,'q3_kernel_verified':True,'nodes_verified':True,'real_generation':False,'server_started':False}))


def run(stage,root,work):
    torch,nodes=boot(root,text=stage=='text',efficient_only=stage=='sample')
    from PIL import Image
    import numpy as np
    from frame_studio.experimental.gguf_t4 import fit
    from frame_studio.wan_bridge import extract_video_png,file_hash
    root=Path(root);work=Path(work);plan=json.loads((work/'gguf_plan.json').read_text())
    with torch.inference_mode():
        if stage=='text':
            clip=invoke(nodes,'CLIPLoaderGGUF',clip_name='umt5-xxl-encoder-Q3_K_M.gguf',type='wan')[0]
            positive=invoke(nodes,'CLIPTextEncode',clip=clip,text='视频中的人在做动作')[0][0][0].detach().cpu()
            negative=invoke(nodes,'CLIPTextEncode',clip=clip,text='')[0][0][0].detach().cpu()
            if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():raise ValueError('T5 produziu nao-finitos')
            torch.save({'positive':positive,'negative':negative},work/'gguf_text.pt');return
        if stage in ('conditioning','decode'):
            vae=invoke(nodes,'WanVideoVAELoader',model_name='Wan2_1_VAE_bf16.safetensors',precision='fp32',use_cpu_cache=True,verbose=True)[0]
        if stage=='conditioning':
            def tensor_image(path,mask=False,face=False):
                with Image.open(path) as im:
                    im=im.convert('L' if mask else 'RGB')
                    if not face:im=fit(im,plan,mask)
                    return torch.from_numpy(np.array(im,dtype=np.float32)/255.)
            def video(name,mask=False,face=False):
                with tempfile.TemporaryDirectory(prefix='gguf-input-',dir=work) as temporary:
                    files=extract_video_png(work/'wan_processed'/name,Path(temporary)/'frames')
                    if len(files)!=plan['count']:raise ValueError('Condicionamento com frames extras/ausentes')
                    items=[tensor_image(p,mask,face) for p in files]
                    items += [items[-1]]*(plan['padded_count']-len(items))
                    return torch.stack(items)
            from PIL import ImageOps
            with Image.open(work/'wan_processed/src_ref.png') as ref:
                ref=ref.convert('RGB')
                clip_reference=torch.from_numpy(np.array(ref,dtype=np.float32)/255.).unsqueeze(0)
                reference=torch.from_numpy(np.array(ImageOps.pad(ref,tuple(plan['internal_size']),method=Image.Resampling.LANCZOS),dtype=np.float32)/255.).unsqueeze(0)
            clip=invoke(nodes,'CLIPVisionLoader',clip_name='clip_vision_h.safetensors')[0]
            clip_embeds=invoke(nodes,'WanVideoClipVisionEncode',clip_vision=clip,image_1=clip_reference,
                strength_1=1.,strength_2=1.,force_offload=True,crop='center',combine_embeds='average')[0]
            # Explicit mask mapping: white in VOS = replaced region. Upstream node does 1-mask for its BG latent mask.
            embeds=invoke(nodes,'WanVideoAnimateEmbeds',vae=vae,width=plan['internal_size'][0],height=plan['internal_size'][1],
                num_frames=33,force_offload=True,frame_window_size=33,colormatch='disabled',pose_strength=1.,face_strength=1.,
                ref_images=reference,clip_embeds=clip_embeds,pose_images=video('src_pose.mp4'),face_images=video('src_face.mp4',face=True),
                bg_images=video('src_bg.mp4'),mask=video('src_mask.mp4',mask=True),tiled_vae=True)[0]
            if not embeds.get('is_masked') or embeds.get('looping'):raise ValueError('Replacement/window mapping divergente')
            embeds['vae']=None
            def cpu(value):
                if isinstance(value,torch.Tensor):return value.detach().cpu()
                if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
                if isinstance(value,list):return [cpu(v) for v in value]
                return value
            torch.save(cpu(embeds),work/'gguf_conditioning.pt');return
        if stage=='sample':
            text=torch.load(work/'gguf_text.pt',map_location='cpu',weights_only=True)
            text_embeds=invoke(nodes,'WanVideoTextEmbedBridge',positive=[[text['positive'],{}]],negative=[[text['negative'],{}]])[0]
            del text
            embeds=torch.load(work/'gguf_conditioning.pt',map_location='cpu',weights_only=True)
            swap=invoke(nodes,'WanVideoBlockSwap',blocks_to_swap=30,offload_img_emb=True,offload_txt_emb=True,
                use_non_blocking=False,prefetch_blocks=0)[0]
            lora=invoke(nodes,'WanVideoLoraSelect',lora='WanAnimate_relight_lora_fp16.safetensors',strength=1.,unique_id=None,merge_loras=False)[0]
            model=invoke(nodes,'WanVideoModelLoader',model='Wan2.2-Animate-14B-Q3_K_M.gguf',base_precision='fp16',quantization='disabled',
                load_device='offload_device',attention_mode='sdpa',block_swap_args=swap,lora=lora)[0]
            if model.model.diffusion_model.pose_patch_embedding is None or model.model.diffusion_model.face_encoder is None:raise ValueError('Loader nao detectou Wan Animate')
            samples=invoke(nodes,'WanVideoSampler',model=model,image_embeds=embeds,steps=20,cfg=1.,shift=5.,seed=plan['seed'],
                force_offload=False,scheduler='unipc',riflex_freq_index=0,text_embeds=text_embeds,rope_function='comfy')[0]
            # Process exits before decoding: avoid shifting the entire DiT into scarce host RAM.
            torch.save(samples,work/'gguf_samples.pt');return
        if stage=='decode':
            samples=torch.load(work/'gguf_samples.pt',map_location='cpu',weights_only=True)
            images=invoke(nodes,'WanVideoDecode',vae=vae,samples=samples,enable_vae_tiling=True,tile_x=256,tile_y=256,tile_stride_x=128,tile_stride_y=128)[0]
            if images.shape[0]!=33 or tuple(images.shape[1:3])!=(plan['internal_size'][1],plan['internal_size'][0]):raise ValueError('Saida nao corresponde a 33 frames/sem referencia do contrato WanAnimate')
            if not torch.isfinite(images).all():raise ValueError('Saida nao-finita, nao publicar video')
            with tempfile.TemporaryDirectory(prefix='gguf-output-',dir=work) as temporary:
                directory=Path(temporary)
                for n in range(plan['count']):
                    array=(images[n].clamp(0,1).cpu().numpy()*255).round().astype('uint8')
                    image=Image.fromarray(array).crop(tuple(plan['content_box'])).resize(tuple(plan['stage_size']),Image.Resampling.LANCZOS)
                    image.save(directory/f'frame_{n:06d}.png')
                import subprocess
                subprocess.run(['ffmpeg','-v','error','-y','-framerate','30','-i',str(directory/'frame_%06d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(work/'wan_generated.mp4')],check=True)
            receipt={'schema':'vos-gguf-engine/1','model':'Wan2.2-Animate-14B-Q3_K_M','real_generation':True,
                'plan_sha256':file_hash(work/'gguf_plan.json'),'video_sha256':file_hash(work/'wan_generated.mp4'),
                'internal_size':plan['internal_size'],'padded_frames':33,'published_frames':plan['count'],'server_started':False}
            (work/'gguf_engine_receipt.json').write_text(json.dumps(receipt,indent=2));return
    raise ValueError('Etapa desconhecida')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['warm','text','conditioning','sample','decode'],required=True)
    parser.add_argument('--root',required=True);parser.add_argument('--work')
    args=parser.parse_args()
    if args.stage=='warm':warm(args.root)
    else:run(args.stage,args.root,args.work)
