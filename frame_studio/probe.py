from fractions import Fraction
from pathlib import Path
import json, math
from .errors import StudioError
from .jobs import run_process
from .timeline import rational, scenes_for, native_timeline

SAFE_FORMATS = 'mov,matroska,webm,avi,mpeg,mpegts,asf,flv,ogg,nut,ivf'


def input_args(path):
    return ['-protocol_whitelist', 'file,pipe', '-format_whitelist', SAFE_FORMATS, '-threads', '1', '-i', str(path)]


def json_probe(path, job, extra=()):
    data=[];size=0
    def append(line):
        nonlocal size
        size+=len(line)
        if size>8*1024*1024:
            raise StudioError('Os metadados do arquivo excedem o limite permitido.',code='metadata_limit')
        data.append(line)
    run_process([job.store.settings.ffprobe, '-v', 'error', '-protocol_whitelist', 'file,pipe', '-format_whitelist', SAFE_FORMATS,
                 *extra, '-of', 'json', str(path)], job, on_line=append)
    try:
        return json.loads('\n'.join(data))
    except ValueError:
        raise StudioError('Não foi possível ler as informações do vídeo.', code='invalid_video')


def probe_project(store, project, job):
    path = store.path(project['id'])/'source.media'
    metadata=json_probe(path, job, ['-show_streams', '-show_format'])
    videos=[s for s in metadata.get('streams', []) if s['codec_type']=='video' and not s.get('disposition', {}).get('attached_pic')]
    if not videos:
        raise StudioError('O arquivo não contém uma faixa de vídeo válida.', code='invalid_video')
    video=videos[0];audio=next((s for s in metadata['streams'] if s['codec_type']=='audio'), None)
    declared=video.get('duration', metadata.get('format',{}).get('duration','0'))
    if declared not in [None,'N/A'] and float(declared)>store.settings.max_seconds+.05:
        raise StudioError(f'O vídeo excede {store.settings.max_seconds} segundos. Envie um trecho menor.', code='duration_limit')
    w,h=int(video['width']),int(video['height'])
    if w*h>store.settings.max_pixels:
        raise StudioError(f'Resolução {w}×{h} excede o limite de {store.settings.max_pixels:,} pixels.', code='resolution_limit')
    rotation=int(round(float(next((s['rotation'] for s in video.get('side_data_list',[]) if 'rotation' in s),0))))%360
    if rotation not in [0,90,180,270]:
        raise StudioError('Rotação não ortogonal não é suportada no modo exato. Normalize o vídeo externamente.', code='unsupported_rotation')
    sar=rational(video.get('sample_aspect_ratio','1:1').replace(':','/')) if video.get('sample_aspect_ratio') not in [None,'N/A','0:1'] else Fraction(1)
    dw,dh=(h,w) if rotation in [90,270] else (w,h)
    if rotation in [90,270]:sar=1/sar
    tb=rational(video['time_base']);raw=[]
    def frame_line(line):
        values=dict(s.split('=',1) for s in line.split('|') if '=' in s)
        if 'best_effort_timestamp' not in values:return
        try:
            pts=int(values['best_effort_timestamp']);duration=int(values.get('duration',values.get('pkt_duration','0')))
        except ValueError:
            raise StudioError('Há quadros sem timestamp utilizável. A montagem exata requer timestamps válidos.', code='invalid_timestamps')
        if int(values.get('width',w))!=w or int(values.get('height',h))!=h:
            raise StudioError('A resolução muda durante o vídeo. Divida o arquivo antes de usar o modo exato.', code='changing_resolution')
        if raw and pts<=raw[-1][0]:
            raise StudioError('Timestamps repetidos ou fora de ordem na origem. Nenhum quadro foi corrigido silenciosamente.', code='invalid_timestamps')
        raw.append((pts,duration))
        if len(raw)>store.settings.max_frames:
            raise StudioError(f'O vídeo excede {store.settings.max_frames} quadros. Envie um trecho menor.', code='frame_limit')
        if (pts-raw[0][0])*tb>store.settings.max_seconds:
            raise StudioError('O vídeo excede o limite de duração.', code='duration_limit')
        if len(raw)%60==0:job.update('Lendo os timestamps originais',min(90,5+len(raw)*80/store.settings.max_frames))
    run_process([store.settings.ffprobe,'-v','error','-protocol_whitelist','file,pipe','-format_whitelist',SAFE_FORMATS,
                 '-threads','1','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp,pkt_duration,duration,width,height',
                 '-of','compact=p=0:nk=0',str(path)],job,on_line=frame_line)
    if not raw:raise StudioError('Nenhum quadro foi decodificado.',code='invalid_video')
    first=raw[0][0];last_duration=raw[-1][1]
    end_pts=None
    try:
        end_pts=int(video.get('start_pts',first))+int(video['duration_ts'])
    except (KeyError,ValueError,TypeError):pass
    if end_pts is not None and end_pts>raw[-1][0]:last_duration=end_pts-raw[-1][0]
    if last_duration<=0:
        raise StudioError('O último quadro não tem duração definida. A montagem exata exige uma duração verificável.',code='invalid_timestamps')
    total=(raw[-1][0]+last_duration-first)*tb
    if total>store.settings.max_seconds+tb:raise StudioError('Duração acima do limite.',code='duration_limit')
    frames=[]
    for i,(pts,_) in enumerate(raw):
        ticks=raw[i+1][0]-pts if i+1<len(raw) else last_duration
        frames.append({'index':i,'original_pts':pts,'start':str((pts-first)*tb),'duration':str(ticks*tb)})
    source={'sha256':project['source_sha256'],'original_name':project['name'],'coded_width':w,'coded_height':h,'width':dw,'height':dh,
            'rotation':rotation,'sample_aspect_ratio':str(sar),'time_base':str(tb),'first_pts':first,'first_timestamp':str(first*tb),
            'duration':str(total),'duration_ticks':int(total/tb),'decoded_frames':len(frames),'avg_frame_rate':video.get('avg_frame_rate','0/1'),
            'r_frame_rate':video.get('r_frame_rate','0/1'),'is_vfr':len({f['duration'] for f in frames})>1,
            'audio':None if audio is None else {'codec':audio.get('codec_name'),'start_time':audio.get('start_time','0'),'sample_rate':int(audio.get('sample_rate',48000)),'channels':audio.get('channels')}}
    project.update(source=source,frames=frames,scenes=scenes_for(frames,total),status='ready',revision=1)
    project['timeline']=native_timeline(project)
    job.check(force=True);store.save(project)
    return {'project_id':project['id'],'frames':len(frames)}
