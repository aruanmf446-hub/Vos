"""ModelScope-only Q3_K_M orchestration. Baseline notebook is never modified."""
import argparse
import ast
from fractions import Fraction
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from PIL import Image
from .gguf_profile import require_attempt
from ..wan_bridge import file_hash,stage_hashes,video_info,generation_hashes
from .. import target_masks as target


def manifest(path):
    data=json.loads(Path(path).read_text())
    if data.get('schema')!='vos-t4-gguf/1' or data['model']!='Wan2.2-Animate-14B' or data['quantization']!='Q3_K_M':raise ValueError('Manifesto experimental desconhecido')
    targets=set()
    for f in data['files']:
        if f['target'] in targets or Path(f['target']).is_absolute() or '..' in Path(f['target']).parts:raise ValueError('Destino de peso inseguro/duplicado')
        targets.add(f['target'])
        if len(f['sha256'])!=64 or f['size']<=0 or '/' not in f['repo']:raise ValueError('Peso sem metadados suficientes')
    if sum(f['size'] for f in data['files'])!=data['total_bytes']:raise ValueError('Orcamento de disco divergente')
    return data


def verify_weights(root,data):
    root=Path(root).resolve()
    for f in data['files']:
        path=(root/f['target']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size!=f['size'] or file_hash(path)!=f['sha256']:
            raise ValueError('Peso incompleto/truncado/SHA incorreto: '+f['path'])


def download_weights(root,data,report,enabled=False):
    require_attempt(report,enabled)
    import shutil
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free<data['total_bytes']+16*2**30:raise ValueError('Sem disco para pesos, instalacao e intermediarios')
    from modelscope.hub.file_download import model_file_download
    for f in data['files']:
        dest=root/f['target'];dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.is_file() and dest.stat().st_size==f['size'] and file_hash(dest)==f['sha256']:continue
        # Download one exact file/revision. No snapshot of unused BF16 weights.
        result=Path(model_file_download(f['repo'],file_path=f['path'],revision=f['revision'],local_dir=str(root/'download-staging')))
        if result.stat().st_size!=f['size'] or file_hash(result)!=f['sha256']:raise ValueError('Download ModelScope inconsistente: '+f['path'])
        result.replace(dest)
    verify_weights(root,data)


def layout(root):
    root=Path(root)
    return {'comfy':root/'ComfyUI','wrapper':root/'ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper',
        'gguf':root/'ComfyUI/custom_nodes/ComfyUI-GGUF','wan':root/'Wan2.2'}


def audit_sources(root,data):
    paths=layout(root)
    for repo in data['repositories']:
        try:actual=subprocess.check_output(['git','-C',str(paths[repo['name']]),'rev-parse','HEAD'],text=True).strip()
        except FileNotFoundError:
            # Read-only CPU Docker audit has no Git; CI checks out detached pins.
            actual=(paths[repo['name']]/'.git/HEAD').read_text().strip()
        if actual!=repo['commit']:raise ValueError('Revisao de motor mudou: '+repo['name'])
    # Parse real method signatures/required inputs without importing CUDA-dependent nodes.
    schemas={}
    classes={'nodes_model_loading.py':['WanVideoModelLoader','WanVideoBlockSwap','WanVideoVAELoader','WanVideoLoraSelect'],
        'nodes.py':['WanVideoAnimateEmbeds','WanVideoDecode','WanVideoClipVisionEncode'],
        'nodes_sampler.py':['WanVideoSampler']}
    namespace={'folder_paths':type('Paths',(),{'get_filename_list':lambda *args:['fixture']})(),
        'attention_modes':['sdpa'],'scheduler_list':['unipc'],'rope_functions':['comfy']}
    for filename,names in classes.items():
        tree=ast.parse((paths['wrapper']/filename).read_text())
        for name in names:
            cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==name)
            node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='INPUT_TYPES');node.decorator_list=[]
            scope=dict(namespace);exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),scope)
            schemas[name]=scope['INPUT_TYPES'](None)
    model=schemas['WanVideoModelLoader']
    if 'fp16' not in model['required']['base_precision'][0] or 'sdpa' not in model['optional']['attention_mode'][0]:raise ValueError('Motor sem FP16/SDPA')
    for field in ('pose_images','face_images','bg_images','mask','ref_images','tiled_vae'):
        if field not in schemas['WanVideoAnimateEmbeds']['optional']:raise ValueError('Node Animate nao atende Replacement: '+field)
    source=(paths['wrapper']/'gguf/gguf_utils.py').read_text()
    if 'def dequantize_blocks_Q3_K' not in source or 'def dequantize_blocks_BF16' not in source:raise ValueError('Dequantizador sem Q3/BF16-storage')
    engine=ast.parse(Path(__file__).with_name('gguf_engine.py').read_text())
    definitions={}
    sources=[paths['wrapper']/f for f in classes]+[paths['gguf']/'nodes.py',paths['comfy']/'nodes.py']
    for source in sources:
        for cls in ast.parse(source.read_text()).body:
            if isinstance(cls,ast.ClassDef):definitions[cls.name]=cls
    for call in ast.walk(engine):
        if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='invoke' or len(call.args)<2 or not isinstance(call.args[1],ast.Constant):continue
        name=call.args[1].value;cls=definitions[name]
        function=next(ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FUNCTION' for t in n.targets))
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==function)
        positional=[p.arg for p in method.args.args][1:]
        required=set(positional[:len(positional)-len(method.args.defaults)] if method.args.defaults else positional)
        supplied={k.arg for k in call.keywords};allowed=set(positional)|{k.arg for k in method.args.kwonlyargs}
        if required-supplied or (not method.args.kwarg and supplied-allowed):raise ValueError('Assinatura real de node divergiu: '+name)
    return schemas


def geometry(width,height,edge=320):
    scale=min(edge/width,edge/height,1)
    w=max(1,round(width*scale));h=max(1,round(height*scale))
    cw=max(64,math.ceil(w/16)*16);ch=max(64,math.ceil(h/16)*16)
    x=(cw-w)//2;y=(ch-h)//2
    return {'stage_size':[width,height],'internal_size':[cw,ch],'content_box':[x,y,x+w,y+h]}


def fit(image,plan,mask=False):
    box=plan['content_box'];image=image.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    canvas=Image.new(image.mode,tuple(plan['internal_size']));canvas.paste(image,(box[0],box[1]));return canvas


def make_plan(work):
    work=Path(work);info=json.loads((work/'bridge.json').read_text());target.require_review(work)
    if Fraction(json.loads((work/'manifest.json').read_text())['selection']['duration'])>1:raise ValueError('Primeiro experimento limitado a UMA cena de ate 1 segundo')
    count=info['frame_count']
    if not 1<=count<=30:raise ValueError('Contagem experimental fora de 1..30')
    for name in ('src_pose.mp4','src_face.mp4','src_bg.mp4','src_mask.mp4'):video_info(work/'wan_processed'/name,count)
    if json.loads((work/'mask_review.json').read_text())!=stage_hashes(work):raise ValueError('Mascara/pose atual nao revisada')
    with Image.open(work/'mask_preflight/frame_000000.png') as im:plan=geometry(*im.size)
    plan.update({'schema':'vos-gguf-plan/1','real_generation':False,'count':count,'padded_count':33,'padding':'repeat last condition, trim tail only',
        'stage_hashes':stage_hashes(work),'native_target':target.signature(work),'manifest_sha256':file_hash(work/'manifest.json'),
        'steps':20,'seed':42,'cfg':1.0,'shift':5.0,'block_swap':30})
    (work/'gguf_plan.json').write_text(json.dumps(plan,indent=2));return plan


def accept_result(work,engine_receipt):
    work=Path(work);target.require_review(work);plan=json.loads((work/'gguf_plan.json').read_text())
    if plan['stage_hashes']!=stage_hashes(work) or plan['native_target']!=target.signature(work):raise ValueError('Condicionamento/alvo alterado')
    receipt=json.loads(Path(engine_receipt).read_text())
    if receipt.get('real_generation') is not True or receipt.get('model')!='Wan2.2-Animate-14B-Q3_K_M' or receipt.get('plan_sha256')!=file_hash(work/'gguf_plan.json') or receipt.get('video_sha256')!=file_hash(work/'wan_generated.mp4'):
        raise ValueError('Recibo de geracao GGUF ausente/inconsistente')
    if video_info(work/'wan_generated.mp4',plan['count'])!=tuple(plan['stage_size']):raise ValueError('Dimensoes GGUF incompatíveis com o stage VOS')
    (work/'generation.json').write_text(json.dumps(generation_hashes(work),indent=2))


def run_generation(work,root,data,report,session,enabled=False):
    work=Path(work)
    for name in ('generation.json','wan_generated.mp4','gguf_engine_receipt.json','gguf_text.pt','gguf_conditioning.pt','gguf_samples.pt'):(work/name).unlink(missing_ok=True)
    require_attempt(report,enabled);verify_weights(root,data);audit_sources(root,data);make_plan(work)
    script=Path(__file__).with_name('gguf_engine.py')
    for stage in ('text','conditioning','sample','decode'):
        session.run([sys.executable,str(script),'--stage',stage,'--root',str(root),'--work',str(work)],label='GGUF '+stage+' (experimental real)')
    accept_result(work,work/'gguf_engine_receipt.json')


def install(root,data,report,session,enabled=False):
    require_attempt(report,enabled)
    import importlib.metadata
    root=Path(root);root.mkdir(parents=True,exist_ok=True);paths=layout(root)
    for repo in data['repositories']:
        path=paths[repo['name']]
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            session.run(['git','clone','--filter=blob:none','--no-checkout',repo['url'],str(path)],label='Codigo '+repo['name'])
        session.run(['git','-C',str(path),'checkout','--detach',repo['commit']],label='Fixar '+repo['name'])
    audit_sources(root,data)
    import os
    os.environ.setdefault('CUDA_HOME','/usr/local/cuda')
    capability=report.get('compute_capability',[7,5])
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{capability[0]}.{capability[1]}'
    # Preserve the Colab Torch/CUDA wheel (in particular its sm_75 kernels).
    constraints=['numpy==1.26.4','transformers==4.57.6','gguf==0.17.1']
    for name in ('torch','torchvision','torchaudio'):
        try:constraints.append(name+'=='+importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            if name=='torch':raise ValueError('PyTorch CUDA do Colab ausente')
    lock=root/'constraints.txt';lock.write_text('\n'.join(constraints)+'\n')
    session.run([sys.executable,'-m','pip','install','--no-build-isolation','-c',str(lock),'-r',str(paths['comfy']/'requirements.txt'),
        '-r',str(paths['wrapper']/'requirements.txt'),'-r',str(paths['gguf']/'requirements.txt'),
        '-r',str(paths['wan']/'requirements_animate.txt'),'moviepy==1.0.3'],label='Dependencias locais sem FlashAttention2')
    session.run([sys.executable,'-m','pip','install','modelscope'],label='SDK de arquivos ModelScope')
    # Imports, actual schemas and synthetic Q3 CUDA kernel must work BEFORE huge downloads.
    session.run([sys.executable,str(Path(__file__).with_name('gguf_engine.py')),'--stage','warm','--root',str(root)],label='Probe real do motor GGUF sem pesos')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit-root',required=True);parser.add_argument('--manifest',required=True)
    args=parser.parse_args();audit_sources(args.audit_root,manifest(args.manifest))
    print(json.dumps({'source_contracts_verified':True,'real_generation':False,'gpu_tested':False}))
