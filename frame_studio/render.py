from pathlib import Path
from fractions import Fraction
from collections import Counter
from io import BytesIO
import json, os, shutil
import av
from PIL import Image
from .errors import StudioError
from .jobs import managed_process, run_process
from .packages import ensure_originals
from .probe import json_probe, input_args
from .storage import identifier, atomic_json, sha256
from .timeline import rational, select_bounds, slice_timeline, resample, sampling_report, validate_coverage, clock_for


def resolve_asset(store, project, f):
    root=store.path(project['id']);a=f['asset']
    if a['kind']=='original':return root/'originals'/f"frame_{a['index']:06d}.png"
    if a['kind']=='import':return root/'imports'/a['import_id']/'frames'/a['path']
    raise StudioError('Origem de quadro desconhecida.',code='wrong_project')


def render_plan(project, selection, fps='native', allow_drop=False):
    a,b,begin,end=select_bounds(project,selection)
    timeline=project['timeline'];validate_coverage(timeline,Fraction(0),rational(project['source']['duration']))
    original=slice_timeline(timeline,begin,end)
    frames=resample(timeline,project,begin,end,fps)
    sampling=sampling_report(original,frames,fps)
    if sampling['discarded_frames'] and not allow_drop:
        raise StudioError('Esse FPS descarta quadros. Confirme o descarte ou preserve a cadência original.',code='drop_confirmation',details=[sampling])
    return a,b,begin,end,frames,sampling


def rgb_frame(image):
    """No numpy dependency, no resizing. Copy one RGB image into a padded plane."""
    w,h=image.size;frame=av.VideoFrame(w,h,'rgb24');plane=frame.planes[0];data=image.tobytes()
    if plane.line_size==w*3:plane.update(data)
    else:
        padded=bytearray(plane.buffer_size)
        for y in range(h):padded[y*plane.line_size:y*plane.line_size+w*3]=data[y*w*3:(y+1)*w*3]
        plane.update(padded)
    return frame


def encode_images(store,project,frames,begin,output,fingerprints,crf,job):
    clock=clock_for(frames,begin);tb=Fraction(1,clock);source=project['source'];w,h=source['width'],source['height']
    duration=sum((rational(f['duration']) for f in frames),Fraction(0));rate=(Fraction(len(frames),1)/duration).limit_denominator(100000)
    pixfmt='yuv420p' if w%2==0 and h%2==0 else 'yuv444p'
    durations={int((rational(f['start'])-begin)/tb):int(rational(f['duration'])/tb) for f in frames}
    container=av.open(str(output),'w',options={'movflags':'+faststart','video_track_timescale':str(clock)})
    stream=container.add_stream('libx264',rate=rate);stream.width=w;stream.height=h;stream.pix_fmt=pixfmt
    stream.time_base=tb;ctx=stream.codec_context;ctx.time_base=tb;ctx.thread_count=1
    ctx.sample_aspect_ratio=rational(source['sample_aspect_ratio']);ctx.colorspace=1;ctx.color_primaries=1;ctx.color_trc=1
    ctx.options={'crf':str(crf),'preset':'veryfast','tune':'zerolatency','bf':'0','x264-params':'ref=1:bframes=0:scenecut=0'}
    def mux(packet):
        at=Fraction(packet.pts)*packet.time_base
        key=at/tb
        if key.denominator!=1 or int(key) not in durations:raise StudioError('O codificador alterou um timestamp de quadro.',code='encoding_timestamps')
        delta=Fraction(durations[int(key)])*tb/packet.time_base
        if delta.denominator!=1:raise StudioError('Duração de quadro não representável pelo codificador.',code='encoding_timestamps')
        packet.duration=int(delta);container.mux(packet)
    previous_scene=None
    try:
        with fingerprints.open('wb') as sig:
            for i,f in enumerate(frames):
                job.check();path=resolve_asset(store,project,f)
                if not path.is_file():raise StudioError('Um quadro necessário não está disponível. Reexporte a cena ou envie o vídeo novamente.',code='missing_frame',details=[f])
                with Image.open(path) as original:
                    if original.size!=(w,h):raise StudioError('Resolução incorreta na montagem.',code='wrong_resolution',details=[f])
                    image=original.convert('RGB');sig.write(image.resize((32,32),Image.Resampling.BOX).tobytes())
                    frame=rgb_frame(image).reformat(format=pixfmt,src_colorspace='ITU709',dst_colorspace='ITU709')
                    frame.pts=int((rational(f['start'])-begin)/tb);frame.time_base=tb
                    if f['scene_id']!=previous_scene:frame.pict_type=av.video.frame.PictureType.I
                    previous_scene=f['scene_id']
                    for packet in stream.encode(frame):mux(packet)
                    image.close()
                job.update('Montando as imagens na ordem do manifesto',20+45*(i+1)/len(frames))
            for packet in stream.encode():mux(packet)
    finally:
        container.close()
    return pixfmt


def mux_audio(store,project,video,output,begin,end,job):
    origin=rational(project['source']['first_timestamp']);a=origin+begin;b=origin+end;duration=end-begin
    def decimal(v):return f'{float(v):.12f}'
    af=f'atrim=start={decimal(a)}:end={decimal(b)},asetpts=PTS-({decimal(a)})/TB,aresample=async=1:first_pts=0,apad'
    run_process([store.settings.ffmpeg,'-v','error','-nostdin','-y','-copyts','-i',str(video),*input_args(store.path(project['id'])/'source.media'),
                 '-map','0:v:0','-map','1:a:0','-c:v','copy','-af',af,'-c:a','aac','-b:a','192k','-threads','1',
                 '-t',decimal(duration),'-movflags','+faststart',str(output)],job)


def read_exact(pipe,size):
    data=bytearray()
    while len(data)<size:
        block=pipe.read(size-len(data))
        if not block:break
        data.extend(block)
    return bytes(data)


def validate_video(store,project,output,frames,begin,end,fingerprints,audio_expected,job):
    data=json_probe(output,job,['-show_streams','-show_format']);video=next(s for s in data['streams'] if s['codec_type']=='video')
    actual_audio=any(s['codec_type']=='audio' for s in data['streams'])
    expected_size=(project['source']['width'],project['source']['height'])
    if (video['width'],video['height'])!=expected_size or actual_audio!=audio_expected:raise StudioError('O arquivo final falhou na validação de resolução ou áudio.',code='output_validation')
    actual_sar=rational(video.get('sample_aspect_ratio','1:1').replace(':','/'))
    if actual_sar!=rational(project['source']['sample_aspect_ratio']):
        raise StudioError('O MP4 alterou a proporção de exibição dos pixels.',code='output_validation')
    tb=rational(video['time_base']);packets=[]
    def packet_line(line):
        values=dict(s.split('=',1) for s in line.split('|') if '=' in s)
        if 'pts' in values:
            try:packets.append((int(values['pts'])*tb,int(values['duration'])*tb))
            except (ValueError,KeyError):raise StudioError('Timestamp ou duração ausente no MP4 final.',code='output_validation')
    run_process([store.settings.ffprobe,'-v','error','-select_streams','v:0','-show_packets','-show_entries','packet=pts,duration','-of','compact=p=0:nk=0',str(output)],job,on_line=packet_line)
    expected=[(rational(f['start'])-begin,rational(f['duration'])) for f in frames]
    if packets!=expected:raise StudioError('O MP4 alterou a quantidade, a ordem ou os timestamps dos quadros. Download bloqueado.',code='output_validation',details=[{'expected_frames':len(expected),'actual_frames':len(packets)}])
    if packets[-1][0]+packets[-1][1]!=end-begin:raise StudioError('A duração final difere da seleção.',code='output_validation')
    count=0;worst=0;first_error=None;last_error=None
    command=[store.settings.ffmpeg,'-v','error','-nostdin','-threads','1','-i',str(output),'-map','0:v:0','-an','-vf','format=rgb24,scale=32:32:flags=area','-fps_mode','passthrough','-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    with fingerprints.open('rb') as expected_sigs,managed_process(command,job) as proc:
        while True:
            actual=read_exact(proc.stdout,32*32*3)
            if not actual:break
            expected_sig=expected_sigs.read(32*32*3)
            if len(actual)!=3072 or len(expected_sig)!=3072:raise StudioError('A decodificação final produziu quadros a mais ou dados incompletos.',code='output_validation')
            error=sum(abs(a-b) for a,b in zip(actual,expected_sig))/3072
            if error>20:raise StudioError('O conteúdo decodificado difere do quadro esperado. A sequência não foi liberada.',code='content_validation',details=[{'frame':count,'mean_thumbnail_error':error}])
            worst=max(worst,error);first_error=error if count==0 else first_error;last_error=error;count+=1
            job.update('Validando todos os timestamps, imagens e cortes',75+23*count/len(frames))
    if count!=len(frames):raise StudioError('A decodificação não encontrou todos os quadros.',code='output_validation')
    cuts=[rational(s['start'])-begin for s in project['scenes'] if begin<rational(s['start'])<end]
    points={p[0] for p in packets}
    if any(c not in points for c in cuts):raise StudioError('Um corte não está presente no MP4 final.',code='output_validation')
    counts=Counter(f['scene_id'] for f in frames)
    return {'passed':True,'decoded_frames':count,'expected_frames':len(frames),'timestamps_exact':True,'duration_exact':True,
            'duration':str(end-begin),'duration_seconds':float(end-begin),'resolution':list(expected_size),'sample_aspect_ratio':project['source']['sample_aspect_ratio'],
            'audio':actual_audio,'complete_decode':True,'first_frame_verified':True,'last_frame_verified':True,'all_frame_fingerprints_verified':True,
            'fingerprint_max_mean_error':worst,'first_frame_mean_error':first_error,'last_frame_mean_error':last_error,
            'cuts':[str(c) for c in cuts],'scene_counts':dict(counts),'codec':video.get('codec_name'),'pixel_format':video.get('pix_fmt'),
            'note':'H.264 altera pixels por compressão. As comparações visuais são assinaturas reduzidas; a ordem, a geometria e os tempos vêm do plano validado. Não são avaliação da qualidade da edição externa.'}


def build_video(store,project,selection,fps,allow_drop,preserve_audio,crf,job):
    if crf not in [16,18,20]:raise StudioError('Qualidade permitida: CRF 16, 18 ou 20.')
    a,b,begin,end,frames,sampling=render_plan(project,selection,fps,allow_drop)
    if len(frames)>store.settings.max_frames*2:raise StudioError('A exportação excede o limite de quadros. Use um trecho menor.',code='frame_limit')
    ensure_originals(store,project,a,b,job)
    root=store.path(project['id']);output_id=identifier();stage=root/'staging'/('build-'+output_id);stage.mkdir()
    try:
        video=stage/'video.mp4';final=stage/'final.mp4';signatures=stage/'fingerprints.bin'
        pixfmt=encode_images(store,project,frames,begin,video,signatures,crf,job)
        audio_expected=bool(preserve_audio and project['source']['audio'])
        if audio_expected:job.update('Sincronizando o áudio original',68);mux_audio(store,project,video,final,begin,end,job)
        else:os.replace(video,final)
        validation=validate_video(store,project,final,frames,begin,end,signatures,audio_expected,job)
        validation.update(sampling=sampling,scope=selection.get('scope','full'),source_start=str(begin),source_end=str(end),sha256=sha256(final),bytes=final.stat().st_size)
        job.check(force=True);atomic_json(root/'outputs'/(output_id+'.json'),validation);os.replace(final,root/'outputs'/(output_id+'.mp4'))
        project=store.get(project['id'],project['owner']);project['outputs'].append({'id':output_id,'scope':selection.get('scope','full'),'validation':validation});store.save(project)
        return {'artifact':output_id,'validation':validation,'filename':('vos-completo' if selection.get('scope','full') in ['full','all_scenes'] else 'vos-'+selection['scope'])+'.mp4'}
    finally:shutil.rmtree(stage,ignore_errors=True)
