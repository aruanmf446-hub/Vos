from pathlib import Path, PurePosixPath
from fractions import Fraction
from bisect import bisect_right
import json, os, shutil, stat, zipfile
from PIL import Image, UnidentifiedImageError
from .errors import StudioError
from .jobs import run_process
from .probe import input_args
from .storage import atomic_json, read_json, sha256, identifier
from .timeline import SCHEMA, rational, native_timeline, select_bounds, resample, sampling_report, replace_interval

INSTRUCTIONS = '''# Editar um pacote Vos Frame Studio

Este ZIP contém quadros completos, não apenas personagens recortados.

1. Leia manifest.json e preserve-o integralmente. Ele identifica o projeto, a seleção, cada cena e os timestamps. Não troque manifestos entre projetos.
2. Preserve caminhos e nomes de todos os PNGs. Não renumere, exclua ou acrescente quadros.
3. Preserve resolução, enquadramento, cenário, perspectiva e quantidade de imagens. Edite apenas o conteúdo solicitado.
4. Mantenha o movimento e a composição coerentes entre imagens consecutivas. Não trate cada quadro como um retrato independente.
5. Devolva quadros completos em PNG, sem transparência e sem rotação EXIF. Não substitua uma imagem completa por um recorte de personagem.
6. Deixe os quadros não editados no ZIP. Para editar só uma cena, prefira exportar essa cena pelo aplicativo.
7. O manifesto determina a ordem e o tempo; a ordem física dos arquivos dentro do ZIP não importa.
8. ChatGPT/Codex pode ter limites de quantidade e geração. Não presuma que centenas de imagens serão geradas em uma só operação. Trabalhe em lotes compatíveis e indique claramente o que foi efetivamente editado.
9. Não use fluxo óptico, deformação ou interpolação para afirmar que novos quadros foram gerados. O aplicativo fará montagem exata; alterações feitas nas próprias imagens pelo editor externo não são corrigidas automaticamente.
10. Não altere áudio ou duração pelo manifesto. O aplicativo sincroniza o áudio da fonte, se ele ainda estiver disponível.

Compacte manifest.json, este arquivo e scenes/ na raiz de um ZIP. O aplicativo verifica os arquivos antes da montagem e identifica faltas, duplicações, extras e resoluções incompatíveis.
'''


def ensure_originals(store, project, first, end, job):
    root=store.path(project['id']);existing=root/'originals'
    if all((existing/f'frame_{n:06d}.png').is_file() for n in range(first,end)):return
    stage=root/'staging'/('extract-'+identifier());stage.mkdir()
    try:
        def progress(line):
            if line.startswith('frame='):
                try:job.update('Extraindo todos os quadros selecionados',5+35*int(line.split('=')[1])/(end-first))
                except ValueError:pass
        command=[store.settings.ffmpeg,'-v','error','-nostdin',*input_args(root/'source.media'),'-map','0:v:0','-an','-sn','-dn',
                 '-vf',f'select=between(n\\,{first}\\,{end-1})','-fps_mode','passthrough','-threads','1',
                 '-compression_level','3','-start_number',str(first),'-progress','pipe:1',str(stage/'frame_%06d.png')]
        run_process(command,job,on_line=progress)
        expected={f'frame_{n:06d}.png' for n in range(first,end)}
        actual={p.name for p in stage.glob('*.png')}
        if actual!=expected:
            raise StudioError('A extração não produziu a sequência esperada. Nenhum pacote foi liberado.',code='extraction_count',details=[{'missing':sorted(expected-actual),'extra':sorted(actual-expected)}])
        size=(project['source']['width'],project['source']['height'])
        for n,name in enumerate(sorted(expected)):
            job.check()
            with Image.open(stage/name) as image:
                if image.size!=size:raise StudioError('A resolução decodificada difere do manifesto. A extração foi bloqueada.',code='resolution_mismatch')
            if not (existing/name).exists():os.replace(stage/name,existing/name)
    finally:
        shutil.rmtree(stage,ignore_errors=True)


def package_plan(project, selection, fps='native'):
    a,b,begin,end=select_bounds(project,selection)
    original=native_timeline(project)
    selected=[f for f in original if begin<=rational(f['start'])<end]
    frames=resample(original,project,begin,end,fps)
    return a,b,begin,end,frames,sampling_report(selected,frames,fps)


def export_package(store, project, selection, fps, confirmed, job):
    a,b,begin,end,frames,sampling=package_plan(project,selection,fps)
    if len(frames)>store.settings.max_frames*2:
        raise StudioError('O pacote excede o limite de quadros. Selecione um trecho menor.',code='frame_limit')
    if fps!='native' and not confirmed:
        raise StudioError('Confirme a amostragem: esse modo pode descartar ou duplicar imagens.',code='sampling_confirmation',details=[sampling])
    ensure_originals(store,project,a,b,job)
    root=store.path(project['id']);package_id=identifier();folder=root/'packages'/package_id;folder.mkdir()
    temp=root/'staging'/(package_id+'.zip');output=root/'outputs'/(package_id+'.zip')
    try:
        entries=[]
        for order,f in enumerate(frames):
            job.check();idx=f['source_index'];serial=idx if fps=='native' else order
            path=f"scenes/{f['scene_id']}/frame_{serial:06d}.png"
            image=root/'originals'/f'frame_{idx:06d}.png'
            entries.append({'id':f'{package_id}:{order}','order':order,'global_index':idx,'source_index':idx,
                            'original_pts':project['frames'][idx]['original_pts'],'scene_id':f['scene_id'],'file':path,
                            'source_start':f['start'],'timestamp':str(rational(f['start'])-begin),'duration':f['duration'],
                            'width':project['source']['width'],'height':project['source']['height'],'original_sha256':sha256(image)})
        scenes=[]
        for s in project['scenes']:
            sf=[f for f in entries if f['scene_id']==s['id']]
            if sf:scenes.append({**s,'selected_start':sf[0]['source_start'],'selected_end':str(rational(sf[-1]['source_start'])+rational(sf[-1]['duration'])),'expected_frames':len(sf)})
        manifest={'schema':SCHEMA,'project_id':project['id'],'package_id':package_id,'project_revision':project['revision'],
                  'source':project['source'],'selection':{'scope':selection.get('scope','full'),'first_frame':a,'end_frame':b,'source_start':str(begin),'source_end':str(end),'duration':str(end-begin)},
                  'sampling':sampling,'expected_frame_count':len(entries),'scenes':scenes,'frames':entries,
                  'audio_sync':{'source_video_origin':project['source']['first_timestamp'],'selected_absolute_start':str(rational(project['source']['first_timestamp'])+begin),'selected_absolute_end':str(rational(project['source']['first_timestamp'])+end)},
                  'export_defaults':{'exact_assembly':True,'fps':'native','preserve_audio':True,'codec':'h264','width':project['source']['width'],'height':project['source']['height']}}
        atomic_json(folder/'manifest.json',manifest)
        with zipfile.ZipFile(temp,'w',allowZip64=True) as archive:
            archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2),compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr('INSTRUCOES_EDICAO.md',INSTRUCTIONS,compress_type=zipfile.ZIP_DEFLATED)
            for i,f in enumerate(entries):
                job.check();archive.write(root/'originals'/f"frame_{f['source_index']:06d}.png",f['file'],compress_type=zipfile.ZIP_STORED)
                job.update('Preparando ZIP e manifesto',45+50*(i+1)/len(entries))
        job.check(force=True);os.replace(temp,output)
        return {'package_id':package_id,'artifact':package_id,'filename':'vos-projeto-'+package_id[:8]+'.zip','frames':len(entries),'sampling':sampling}
    except BaseException:
        shutil.rmtree(folder,ignore_errors=True);temp.unlink(missing_ok=True);raise


def safe_member(name):
    raw=name.rstrip('/')
    path=PurePosixPath(raw)
    if not raw or '\\' in raw or ':' in raw or '\x00' in raw or raw.startswith('/') or '//' in raw or any(p in ['.','..'] for p in raw.split('/')):
        raise StudioError('O ZIP contém um caminho inválido.',code='unsafe_zip',details=[name])
    if len(path.parts)>6:raise StudioError('Caminho excessivamente profundo no ZIP.',code='unsafe_zip',details=[name])
    return path.as_posix()


def validate_import(store, project, import_id, job):
    root=store.path(project['id']);folder=root/'imports'/import_id;archive_path=folder/'upload.zip';images=folder/'frames';images.mkdir()
    issues=[];changed=[];manifest=None;normalized_prefix=''
    def issue(code,message,**details):issues.append({'code':code,'message':message,**details})
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos=archive.infolist()
            if len(infos)>store.settings.max_frames*2+100:raise StudioError('O ZIP contém arquivos demais.',code='unsafe_zip')
            for info in infos:
                safe_member(info.filename)
                mode=(info.external_attr>>16)&0o170000
                if stat.S_ISLNK(mode):raise StudioError('Links simbólicos não são permitidos no ZIP.',code='unsafe_zip',details=[info.filename])
                if info.flag_bits&1:raise StudioError('ZIP com senha não é suportado.',code='unsafe_zip')
            files=[i for i in infos if not i.is_dir()]
            names=[i.filename for i in files]
            if 'manifest.json' not in names:
                candidates=[n for n in names if n.count('/')==1 and n.endswith('/manifest.json')]
                if len(candidates)==1:
                    prefix=candidates[0].split('/')[0]+'/'
                    if all(n.startswith(prefix) for n in names):normalized_prefix=prefix
            members={};seen=set()
            for info in files:
                name=info.filename[len(normalized_prefix):]
                if name.lower() in seen:issue('duplicate_file','Nome duplicado no ZIP.',file=name)
                seen.add(name.lower());members[name]=info
            if issues:raise StudioError('Há arquivos duplicados no ZIP.',code='invalid_zip',details=issues)
            mi=members.get('manifest.json')
            if mi is None or mi.file_size>8*1024**2:raise StudioError('manifest.json ausente ou grande demais.',code='invalid_manifest')
            try:manifest=json.loads(archive.read(mi))
            except (ValueError,RecursionError,UnicodeDecodeError):raise StudioError('manifest.json inválido.',code='invalid_manifest')
            if not isinstance(manifest,dict) or manifest.get('project_id')!=project['id']:
                raise StudioError('O manifesto pertence a outro projeto ou é inválido.',code='wrong_project')
            package_id=manifest.get('package_id','')
            from .storage import ID
            if not isinstance(package_id,str) or not ID.fullmatch(package_id):raise StudioError('Identificador de pacote inválido.',code='invalid_manifest')
            canonical_path=root/'packages'/package_id/'manifest.json'
            if not canonical_path.exists():raise StudioError('O pacote original não está mais disponível nesta sessão. Reexporte o ZIP.',code='package_expired')
            canonical=read_json(canonical_path)
            if manifest!=canonical:raise StudioError('O manifesto foi alterado. Devolva o manifesto original sem modificações.',code='manifest_changed')
            if manifest['project_revision']!=project['revision']:raise StudioError('Os cortes foram alterados após a exportação. Exporte um novo pacote.',code='revision_changed')
            expected={f['file']:f for f in manifest['frames']}
            for name in sorted(expected.keys()-members.keys()):issue('missing_frame','Quadro ausente.',file=name,global_index=expected[name]['global_index'])
            for name in sorted(members.keys()-expected.keys()-{'manifest.json','INSTRUCOES_EDICAO.md'}):issue('extra_file','Arquivo extra não previsto no manifesto.',file=name)
            max_frame=project['source']['width']*project['source']['height']*8+1024**2
            total=0
            for order,(name,f) in enumerate(expected.items()):
                job.check();info=members.get(name)
                if info is None:continue
                total+=info.file_size
                if total>store.settings.project_bytes or info.file_size>max_frame or info.file_size/max(info.compress_size,1)>200:
                    issue('zip_limit','Tamanho ou razão de compressão incompatível com um quadro.',file=name);continue
                target=images/name;target.parent.mkdir(parents=True,exist_ok=True)
                try:
                    with archive.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,64*1024)
                    with Image.open(target) as image:
                        if image.format!='PNG':raise StudioError('O quadro deve ser um PNG.',code='invalid_image')
                        if image.size!=(f['width'],f['height']):raise StudioError('Resolução incorreta.',code='wrong_resolution',details=[{'expected':[f['width'],f['height']],'actual':list(image.size)}])
                        if getattr(image,'n_frames',1)!=1:raise StudioError('PNG animado não é um quadro único.',code='invalid_image')
                        if image.getexif().get(274,1)!=1:raise StudioError('Remova a rotação EXIF sem mudar o enquadramento.',code='image_orientation')
                        image.load()
                        if 'A' in image.getbands() and image.getchannel('A').getextrema()!=(255,255):raise StudioError('O quadro contém transparência. Devolva a imagem completa, incluindo o cenário.',code='transparent_image')
                        if image.mode=='P' and 'transparency' in image.info:
                            if image.convert('RGBA').getchannel('A').getextrema()!=(255,255):raise StudioError('O quadro contém transparência.',code='transparent_image')
                    if sha256(target)!=f['original_sha256']:changed.append(f['order'])
                except StudioError as e:issue(e.code,e.message,file=name,global_index=f['global_index'],details=e.details)
                except (OSError,ValueError,zipfile.BadZipFile,Image.DecompressionBombError):issue('invalid_image','Imagem inválida ou ZIP corrompido.',file=name,global_index=f['global_index'])
                job.update('Validando todos os quadros do ZIP',5+90*(order+1)/len(expected))
    except StudioError as e:
        if e.code in ['timeout','cancelled','disk_limit']:raise
        if not issues:issue(e.code,e.message,details=e.details)
    except (zipfile.BadZipFile,OSError,RuntimeError,ValueError) as e:
        issue('invalid_zip','Não foi possível ler o ZIP. Envie um arquivo ZIP válido.')
    finally:
        archive_path.unlink(missing_ok=True)
    valid=not issues
    report={'id':import_id,'valid':valid,'status':'validated' if valid else 'invalid','issues':issues,'changed_orders':changed,
            'changed_files':len(changed),'expected_frames':manifest.get('expected_frame_count') if isinstance(manifest,dict) else None,
            'normalized_outer_folder':normalized_prefix,'manifest':manifest if valid else None}
    if not valid:shutil.rmtree(images,ignore_errors=True)
    job.check(force=True);atomic_json(folder/'import.json',report)
    project=store.get(project['id'],project['owner'])
    if import_id not in project['imports']:project['imports'].append(import_id);store.save(project)
    return {k:report[k] for k in ['id','valid','issues','changed_files','expected_frames','normalized_outer_folder']}


def apply_import(store, project, import_id):
    root=store.path(project['id']);path=root/'imports'/import_id/'import.json'
    if not path.exists():raise StudioError('Importação não encontrada.',status=404)
    report=read_json(path)
    if not report['valid']:raise StudioError('Corrija os erros do ZIP antes de aplicar a importação.',code='invalid_import')
    manifest=report['manifest']
    if manifest['project_id']!=project['id'] or manifest['project_revision']!=project['revision']:raise StudioError('O pacote não corresponde à revisão atual deste projeto.',code='wrong_project')
    frames=[{'start':f['source_start'],'duration':f['duration'],'source_index':f['source_index'],'scene_id':f['scene_id'],
             'asset':{'kind':'import','import_id':import_id,'path':f['file']}} for f in manifest['frames']]
    begin=rational(manifest['selection']['source_start']);end=rational(manifest['selection']['source_end'])
    project['timeline']=replace_interval(project['timeline'],frames,begin,end,rational(project['source']['duration']))
    store.save(project);return {'applied':True,'frames':len(frames),'source_start':str(begin),'source_end':str(end)}
