from pathlib import Path
from fractions import Fraction
import io,subprocess,time,zipfile,json
from PIL import Image,ImageDraw
from frame_studio.config import Settings
from frame_studio.storage import Store,sha256,read_json,identifier
from frame_studio.jobs import Job
from frame_studio.probe import probe_project
from frame_studio.timeline import scenes_for,native_timeline
from frame_studio.packages import export_package,validate_import


def make_video(path,frames=18,rate='30',audio=True,size=(160,96),offset=0,sar=None):
    directory=path.parent/('fixture-'+identifier());directory.mkdir()
    for n in range(frames):
        color=[(180+n%6*8,30,40),(25,160+n%6*9,35),(30,40,180+n%6*8)][min(n//6,2)]
        image=Image.new('RGB',size,color);draw=ImageDraw.Draw(image)
        for bit in range(6):draw.rectangle((5+bit*15,5,15+bit*15,28),fill='white' if n&(1<<bit) else 'black')
        draw.rectangle((size[0]-26, size[1]-24, size[0]-7,size[1]-8),fill=(20+n*8,230-n*5,90))
        image.save(directory/f'frame_{n:06d}.png')
    command=['ffmpeg','-v','error','-y','-framerate',rate,'-i',str(directory/'frame_%06d.png')]
    if audio:command+=['-f','lavfi','-i',f'sine=frequency=700:sample_rate=48000:duration={float(Fraction(frames)/Fraction(rate))}']
    if sar:command+=['-vf','setsar='+sar]
    command+=['-c:v','libx264','-bf','0','-threads','1','-pix_fmt','yuv420p' if size[0]%2==0 and size[1]%2==0 else 'yuv444p']
    if audio:command+=['-c:a','aac','-shortest']
    if offset:command+=['-output_ts_offset',str(offset)]
    command+=[str(path)];subprocess.run(command,check=True)
    return path


def job_for(store,project):
    job=Job(store,project,'test');job.deadline=time.monotonic()+120;return job


def create_project(root,rate='30',audio=True,size=(160,96),offset=0,sar=None):
    settings=Settings(data_dir=root/'data',reserve_bytes=1024**2,max_seconds=10,max_frames=500,project_bytes=64*1024**2,total_bytes=128*1024**2)
    store=Store(settings);project=store.create('owner','teste.mp4');p=store.path(project['id'])/'source.media'
    source=make_video(root/'source.mp4',rate=rate,audio=audio,size=size,offset=offset,sar=sar)
    p.write_bytes(source.read_bytes());project['source_sha256']=sha256(p);store.save(project)
    probe_project(store,project,job_for(store,project));project=store.get(project['id'],'owner')
    project['scenes']=scenes_for(project['frames'],Fraction(project['source']['duration']),[6,12]);project['timeline']=native_timeline(project);store.save(project)
    return store,project


def package(store,project,selection=None,fps='native'):
    result=export_package(store,project,selection or {'scope':'full'},fps,fps!='native',job_for(store,project))
    return store.path(project['id'])/'outputs'/(result['artifact']+'.zip')


def rewrite_zip(source,dest,*,missing=None,modify=None,extra=None,manifest_change=None,reverse=False,duplicate=None,wrapper=''):
    with zipfile.ZipFile(source) as archive:
        entries=[(i.filename,archive.read(i)) for i in archive.infolist()]
    if reverse:entries.reverse()
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_STORED) as out:
        for name,data in entries:
            if name==missing:continue
            if name=='manifest.json' and manifest_change:
                m=json.loads(data);manifest_change(m);data=json.dumps(m).encode()
            if modify and name in modify:data=modify[name]
            out.writestr(wrapper+name,data)
            if name==duplicate:out.writestr(wrapper+name,data)
        for name,data in extra or []:out.writestr(wrapper+name,data)
    return dest


def import_zip(store,project,path):
    iid=identifier();folder=store.path(project['id'])/'imports'/iid;folder.mkdir();(folder/'upload.zip').write_bytes(path.read_bytes())
    validate_import(store,project,iid,job_for(store,project));return iid,read_json(folder/'import.json')


def png(color,size=(160,96),mode='RGB'):
    stream=io.BytesIO();Image.new(mode,size,color).save(stream,format='PNG');return stream.getvalue()
