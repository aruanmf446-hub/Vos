"""Optional explicit target selection. Pillow only; SAM2 lives in wan_target.py.
Native PNG masks are authoritative; never interpolate masks between keyframes.
"""
import base64
import hashlib
import io
import json
import math
from pathlib import Path
from PIL import Image, ImageChops


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def atomic_json(path, data):
    path=Path(path);temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(path)


def invalidate(work):
    for name in ('target_review.json','preprocess.json','mask_review.json','generation.json'):
        (Path(work)/name).unlink(missing_ok=True)


def context(work):
    work=Path(work);manifest=json.loads((work/'manifest.json').read_text())
    return manifest,manifest['frames'],(manifest['source']['width'],manifest['source']['height'])


def save_selection(work, data):
    work=Path(work);manifest,frames,size=context(work)
    if not isinstance(data,dict) or not isinstance(data.get('keyframes'),dict) or not data['keyframes']:
        raise ValueError('Selecione o alvo no primeiro quadro; adicione pontos positivos/negativos')
    clean={}
    for key,entry in data['keyframes'].items():
        n=int(key)
        if str(n)!=key or not 0<=n<len(frames):raise ValueError('Quadro-chave invalido')
        points=entry.get('points',[])
        if not isinstance(points,list) or len(points)>200:raise ValueError('Pontos invalidos')
        for point in points:
            if len(point)!=3 or point[2] not in (0,1) or any(not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in point[:2]):
                raise ValueError('Ponto fora do quadro')
        if points:clean[key]={'points':points}
    if '0' not in clean or not any(p[2]==1 for p in clean['0']['points']):
        raise ValueError('Marque pelo menos um ponto positivo no alvo no quadro zero')
    old=work/'target.json'
    if old.is_file():
        previous=json.loads(old.read_text()).get('keyframes',{})
        positive=lambda entry:[p for p in entry.get('points',[]) if p[2]==1]
        if positive(previous.get('0',{}))!=positive(clean['0']):
            import shutil
            shutil.rmtree(work/'target_seeds',ignore_errors=True)
        else:
            for key in set(previous)|set(clean):
                if previous.get(key)!=clean.get(key):
                    (work/'target_seeds'/f'frame_{int(key):06d}.png').unlink(missing_ok=True)
    invalidate(work)
    atomic_json(work/'target.json',{'schema':'vos-target/1','name':str(data.get('name','Personagem selecionado'))[:100],
        'manifest_sha256':digest(work/'manifest.json'),'frame_count':len(frames),'size':size,'keyframes':clean})


def config(work):
    work=Path(work);data=json.loads((work/'target.json').read_text());_,frames,size=context(work)
    if data.get('schema')!='vos-target/1' or data['manifest_sha256']!=digest(work/'manifest.json') or data['frame_count']!=len(frames) or data['size']!=list(size):
        raise ValueError('Selecao pertence a outro manifesto/resolucao/tempo')
    return data


def mask_path(work,n):return Path(work)/'target_masks'/f'frame_{n:06d}.png'


def load_mask(path,size):
    with Image.open(path) as im:
        if im.format!='PNG' or im.size!=size or getattr(im,'n_frames',1)!=1:raise ValueError('Mascara PNG com dimensoes incompatíveis')
        if 'transparency' in im.info and im.convert('RGBA').getchannel('A').getextrema()!=(255,255):raise ValueError('Mascara com transparencia')
        if 'A' in im.getbands() and im.getchannel('A').getextrema()!=(255,255):raise ValueError('Mascara com transparencia')
        mask=im.convert('L');mask.load()
        if any(n and value not in (0,255) for value,n in enumerate(mask.histogram())):
            raise ValueError('Mascara deve ser binaria (0/255), sem transparencia ou suavizacao')
        return mask


def signature(work):
    work=Path(work);cfg=config(work)
    expected={f'frame_{n:06d}.png' for n in range(cfg['frame_count'])}
    folder=work/'target_masks'
    if not folder.is_dir() or {p.name for p in folder.iterdir()}!=expected:raise ValueError('Mascaras nativas faltantes ou extras')
    return {'selection_sha256':digest(work/'target.json'),'masks':[digest(mask_path(work,n)) for n in range(cfg['frame_count'])]}


def audit(work):
    cfg=config(work);size=tuple(cfg['size']);previous=None;rows=[];issues=[]
    for n in range(cfg['frame_count']):
        mask=load_mask(mask_path(work,n),size);area=mask.histogram()[255];box=mask.getbbox();reasons=[]
        if not box or area/(size[0]*size[1])>0.65:reasons.append('area vazia/excessiva')
        if previous and box:
            prev,prev_area,prev_box=previous
            intersection=ImageChops.multiply(mask,prev).histogram()[255]
            union=area+prev_area-intersection;iou=intersection/max(union,1)
            center=lambda b:((b[0]+b[2])/2/size[0],(b[1]+b[3])/2/size[1])
            if iou<0.05 or math.dist(center(box),center(prev_box))>0.25 or max(area,prev_area)/max(min(area,prev_area),1)>3:
                reasons.append('possivel troca de alvo/oclusao/salto; corrigir e rastrear novamente')
        for x,y,label in cfg['keyframes'].get(str(n),{}).get('points',[]):
            actual=mask.getpixel((min(size[0]-1,int(x*size[0])),min(size[1]-1,int(y*size[1]))))>0
            if actual!=bool(label):reasons.append('mascara contradiz ponto positivo/negativo')
        if reasons:issues.append({'frame':n,'reasons':reasons})
        rows.append({'frame':n,'coverage':area/(size[0]*size[1]),'box':box,'issues':reasons})
        previous=(mask,area,box) if box else None
    signature(work)
    result={'schema':'vos-target-audit/1','issues':issues,'frames':rows,
        'limitation':'Heuristicas nao identificam pessoas. Revise TODOS os quadros; SAM2 nao foi validado nesta maquina.'}
    atomic_json(Path(work)/'target_audit.json',result)
    return result


def apply_corrections(work, corrections):
    """Exact-frame brush masks; these can also seed a subsequent SAM2 propagation."""
    work=Path(work);cfg=config(work);invalidate(work)
    for key,encoded in corrections.items():
        n=int(key)
        if str(n)!=key or not 0<=n<cfg['frame_count']:raise ValueError('Correcao fora da sequencia')
        raw=base64.b64decode(encoded.split(',')[-1],validate=True)
        if len(raw)>cfg['size'][0]*cfg['size'][1]*4+1048576:raise ValueError('Correcao grande demais')
        temp=work/f'correction_{n}.png'
        try:
            temp.write_bytes(raw);mask=load_mask(temp,tuple(cfg['size']))
            mask.save(mask_path(work,n))
            seeds=work/'target_seeds';seeds.mkdir(exist_ok=True);mask.save(seeds/f'frame_{n:06d}.png')
        finally:temp.unlink(missing_ok=True)


def approve(work, reviewed):
    work=Path(work);invalidate(work);cfg=config(work)
    if set(reviewed)!=set(range(cfg['frame_count'])):raise ValueError('Revise e marque cada quadro nativo antes de aprovar')
    report=audit(work)
    if report['issues']:raise ValueError('Rastreamento suspeito: corrija os quadros sinalizados; geracao bloqueada')
    atomic_json(work/'target_review.json',signature(work))


def require_review(work):
    work=Path(work);review=work/'target_review.json'
    if not review.is_file() or json.loads(review.read_text())!=signature(work):
        raise ValueError('Alvo/máscaras não revisados ou alterados; geracao bloqueada')


def editor_payload(work, mode):
    work=Path(work);manifest,frames,size=context(work)
    if len(frames)>600:raise ValueError('Editor experimental limitado a 600 quadros nativos; exporte cena menor')
    cfg=config(work) if (work/'target.json').exists() else {'keyframes':{},'name':''}
    images=[];masks=[]
    for n,f in enumerate(frames):
        with Image.open(work/f['file']) as im:
            im=im.convert('RGB');im.thumbnail((640,640));buf=io.BytesIO();im.save(buf,'JPEG',quality=80)
            images.append('data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode())
        if mode=='review':
            mask=load_mask(mask_path(work,n),size);buf=io.BytesIO();mask.save(buf,'PNG')
            masks.append('data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())
    return {'mode':mode,'name':cfg['name'],'keyframes':cfg['keyframes'],'images':images,'masks':masks,'size':size,
            'timestamps':[f['timestamp'] for f in frames], 'issues':audit(work)['issues'] if mode=='review' else []}


def show_editor(work, mode='prompts'):
    """Local Colab iframe only. No upload to another site or inference service."""
    from google.colab.output import eval_js
    script=(Path(__file__).parent/'static/target_editor.js').read_text()
    invalidate(work)
    result=eval_js(script+'\nvosTargetEditor('+json.dumps(editor_payload(work,mode))+')',timeout_sec=3600)
    if result.get('cancelled'):raise ValueError('Editor cancelado; nenhuma aprovacao registrada')
    if mode=='prompts':save_selection(work,result)
    else:
        if result.get('needs_tracking'):
            save_selection(work,result)
            apply_corrections(work,result.get('corrections',{}))
            raise ValueError('Pontos atualizados. Execute SAM2 novamente e reabra a revisao.')
        apply_corrections(work,result.get('corrections',{}))
        approve(work,result.get('reviewed',[]))
    return result


def export_review(work,output):
    """Backup for mask review, intentionally not a VOS edited-frame import ZIP."""
    import zipfile
    work=Path(work);require_review(work);output=Path(output)
    if output.resolve()==(work/'original_vos.zip').resolve():raise ValueError('Nao sobrescreva o ZIP original')
    temporary=output.with_suffix('.partial')
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('LEIA.txt','Backup de selecao/mascaras, NAO importar este ZIP no VOS. Use VOS_EDITADO.zip para importar imagens completas.\n')
            for name in ('manifest.json','target.json','target_audit.json','target_review.json'):z.write(work/name,name)
            for path in sorted((work/'target_masks').glob('*.png')):z.write(path,'target_masks/'+path.name)
            for path in sorted((work/'target_seeds').glob('*.png')):z.write(path,'target_seeds/'+path.name)
        temporary.replace(output)
    finally:temporary.unlink(missing_ok=True)
    return output
