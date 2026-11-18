"""Selection/mask progress backup, separate from diagnostic ZIP and VOS edits.
Restores only against the exact original export. Never resumes model inference.
"""
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile
try:
    from . import target_masks as target
except ImportError:
    import target_masks as target


METADATA={'manifest.json','bridge.json','temporal_map.json','target.json','target_review.json','target_audit.json','progress.json'}


def export_progress(work,output):
    work=Path(work);output=Path(output);manifest,frames,size=target.context(work)
    if output.resolve()==(work/'original_vos.zip').resolve():raise ValueError('Nao sobrescreva o ZIP original')
    status='no_selection'
    if (work/'target.json').exists():
        target.config(work);status='selection_unreviewed'
        try:target.require_review(work);status='masks_reviewed'
        except (ValueError,OSError,KeyError):pass
    info=json.loads((work/'bridge.json').read_text())
    progress={'schema':'vos-progress/1','manifest_sha256':target.digest(work/'manifest.json'),
              'original_zip_sha256':info['original_zip_sha256'],'status':status,
              'limitation':'Nao retoma inferencia; requer mesmo ZIP original e referencia guardados pelo usuario. Nao importar no VOS.'}
    temp=output.with_suffix('.partial')
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('progress.json',json.dumps(progress,indent=2))
            for name in METADATA-{'progress.json'}:
                if (work/name).is_file():z.write(work/name,name)
            for folder in ('target_masks','target_seeds'):
                for path in sorted((work/folder).glob('frame_*.png')):z.write(path,folder+'/'+path.name)
        temp.replace(output)
    finally:temp.unlink(missing_ok=True)
    return output


def restore_progress(work,archive):
    work=Path(work);manifest,frames,size=target.context(work)
    with tempfile.TemporaryDirectory(prefix='restore-',dir=work) as temporary,zipfile.ZipFile(archive) as z:
        stage=Path(temporary);infos=z.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>2*len(frames)+20 or sum(i.file_size for i in infos)>256*1024**2:raise ValueError('Backup duplicado/grande demais')
        for i in infos:
            match=re.fullmatch(r'(target_masks|target_seeds)/frame_(\d{6})\.png',i.filename)
            if i.filename not in METADATA and not match:raise ValueError('Arquivo/caminho nao permitido no backup')
            if match and int(match[2])>=len(frames):raise ValueError('Mascara fora da sequencia')
            if i.flag_bits&1 or ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('Links/senha nao permitidos')
            limit=size[0]*size[1]*4+1048576 if match else 8*1024**2
            if i.file_size>limit:raise ValueError('Arquivo de backup grande demais')
            path=stage/i.filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(i))
        if (stage/'manifest.json').read_bytes()!=(work/'manifest.json').read_bytes():raise ValueError('Backup pertence a outro manifesto. Use exatamente o ZIP original; nunca altere o manifesto.')
        info=json.loads((work/'bridge.json').read_text());progress=json.loads((stage/'progress.json').read_text())
        if progress.get('schema')!='vos-progress/1' or progress['original_zip_sha256']!=info['original_zip_sha256'] or progress['manifest_sha256']!=target.digest(work/'manifest.json'):raise ValueError('Backup nao corresponde ao ZIP original')
        if not (stage/'target.json').exists():
            if (work/'target.json').exists():raise ValueError('Backup sem selecao nao substitui um alvo existente; prepare um trabalho novo.')
            return 'no_selection'
        cfg=target.config(stage)
        # Revalidate prompt coordinates, while preserving corrected seeds.
        validation=stage/'validation';validation.mkdir();shutil.copyfile(stage/'manifest.json',validation/'manifest.json')
        target.save_selection(validation,{'name':cfg['name'],'keyframes':cfg['keyframes']})
        for folder in ('target_masks','target_seeds'):
            for path in (stage/folder).glob('*.png'):target.load_mask(path,size)
        reviewed=progress['status']=='masks_reviewed'
        if reviewed:
            target.require_review(stage)
            if target.audit(stage)['issues']:raise ValueError('Backup de mascaras contem alertas de rastreamento')
        target.invalidate(work)
        for folder in ('target_masks','target_seeds'):
            shutil.rmtree(work/folder,ignore_errors=True)
            if (stage/folder).exists():shutil.move(str(stage/folder),str(work/folder))
        shutil.copyfile(stage/'target.json',work/'target.json')
        if reviewed:
            shutil.copyfile(stage/'target_audit.json',work/'target_audit.json')
            target.atomic_json(work/'target_review.json',target.signature(work))
        return 'masks_reviewed' if reviewed else 'selection_unreviewed'
