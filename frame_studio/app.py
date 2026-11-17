from contextlib import asynccontextmanager
from pathlib import Path
from bisect import bisect_left
import asyncio, hashlib, hmac, json, logging, os, re, secrets, shutil, time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from starlette.requests import ClientDisconnect
from .config import Settings
from .errors import StudioError
from .storage import Store, ID, identifier, atomic_json, read_json
from .jobs import Queue, run_process
from .probe import probe_project, input_args
from .timeline import rational, scenes_for, native_timeline, select_bounds
from .packages import export_package, validate_import, apply_import, package_plan
from .render import build_video, render_plan

STATIC=Path(__file__).parent/'static'


class Selection(BaseModel):
    scope: Literal['full','all_scenes','scene','range']='full'
    scene_id: str | None=None
    first_frame: int | None=None
    end_frame: int | None=None


class PackageOptions(BaseModel):
    selection: Selection=Field(default_factory=Selection)
    fps: Literal['native','15','24','25','30','60']='native'
    confirm_sampling: bool=False


class BuildOptions(BaseModel):
    selection: Selection=Field(default_factory=Selection)
    fps: Literal['native','30','60']='native'
    allow_drop: bool=False
    preserve_audio: bool=True
    crf: Literal[16,18,20]=16


class Cuts(BaseModel):
    cuts: list[int]=Field(default_factory=list,max_length=500)


class Detect(BaseModel):
    threshold: float=Field(default=.35,ge=.05,le=.9)


class Login(BaseModel):
    password: str=Field(max_length=512)


class BodyLimit(Exception):pass


class BodyLimitMiddleware:
    def __init__(self,app,settings):self.app,self.settings=app,settings
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        path=scope.get('path','');limit=65536
        if path=='/api/projects' and scope.get('method')=='POST':limit=self.settings.video_bytes
        elif path.endswith('/imports') and scope.get('method')=='POST':limit=self.settings.zip_bytes
        count=0
        async def limited():
            nonlocal count
            message=await receive()
            if message['type']=='http.request':
                count+=len(message.get('body',b''))
                if count>limit:raise BodyLimit()
            return message
        length=dict(scope.get('headers',[])).get(b'content-length',b'0')
        try:length=int(length)
        except ValueError:length=limit+1
        try:
            if length>limit:raise BodyLimit()
            await self.app(scope,limited,send)
        except BodyLimit:
            response=JSONResponse({'error':{'code':'upload_limit','message':'Arquivo ou requisição acima do limite permitido. Envie um trecho menor.','details':[]}},status_code=413)
            await response(scope,receive,send)


def create_app(settings=None):
    settings=settings or Settings();store=Store(settings);queue=Queue(store)
    async def cleanup_loop():
        while True:
            await asyncio.sleep(60)
            await asyncio.to_thread(store.cleanup)
    @asynccontextmanager
    async def lifespan(app):
        store.cleanup();task=asyncio.create_task(cleanup_loop())
        yield
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
        await asyncio.to_thread(queue.close)
    app=FastAPI(title='Vos Frame Studio',version='1.0.0',lifespan=lifespan)
    app.state.store,app.state.queue,app.state.settings=store,queue,settings
    app.add_middleware(BodyLimitMiddleware,settings=settings)
    attempts={}
    def signature(kind,token):
        text=kind+':'+token+(':'+hashlib.sha256(settings.password.encode()).hexdigest() if kind=='auth' else '')
        return hmac.new(settings.secret.encode(),text.encode(),hashlib.sha256).hexdigest()
    @app.exception_handler(StudioError)
    async def studio_error(request,error):
        return JSONResponse({'error':{'code':error.code,'message':error.message,'details':error.details}},status_code=error.status)
    @app.exception_handler(BodyLimit)
    async def body_error(request,error):
        return JSONResponse({'error':{'code':'upload_limit','message':'Arquivo acima do limite configurado.','details':[]}},status_code=413)
    @app.middleware('http')
    async def session(request,call_next):
        token=request.cookies.get('vos_session','')
        fresh=not re.fullmatch(r'[a-f0-9]{64}',token)
        if fresh:token=secrets.token_hex(32)
        request.state.token,request.state.owner=token,store.owner(token)
        request.state.authenticated=not settings.password or hmac.compare_digest(request.cookies.get('vos_auth',''),signature('auth',token))
        path=request.url.path
        if path.startswith('/api/'):
            if path not in ['/api/session','/api/login'] and not request.state.authenticated:
                return JSONResponse({'error':{'code':'login_required','message':'Informe a senha de acesso.','details':[]}},status_code=401)
            if request.method not in ['GET','HEAD','OPTIONS']:
                if not hmac.compare_digest(request.headers.get('x-csrf-token',''),signature('csrf',token)):
                    return JSONResponse({'error':{'code':'csrf','message':'Sessão inválida. Recarregue a página.','details':[]}},status_code=403)
        response=await call_next(request)
        if fresh:response.set_cookie('vos_session',token,httponly=True,secure=settings.secure_cookies,samesite='lax',max_age=604800)
        if path.startswith('/api/'):
            response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    def project_for(request,pid,ready=True):
        project=store.get(pid,request.state.owner)
        if ready and project['status']!='ready':raise StudioError('Aguarde a leitura do vídeo ou envie um arquivo válido.',status=409)
        return project
    def idle(pid):
        if pid in store.active:raise StudioError('Aguarde ou cancele o processamento atual antes de modificar o projeto.',status=409)
    def valid_id(value):
        if not ID.fullmatch(value):raise StudioError('Arquivo ou importação não encontrado.',status=404)
    def public_project(project):
        return {k:project[k] for k in ['id','name','status','revision','source','frames','scenes','imports','outputs'] if k in project}
    async def upload(request,path,limit,pid):
        size=0;digest=hashlib.sha256()
        try:
            with path.open('wb') as out:
                async for block in request.stream():
                    size+=len(block)
                    if size>limit:raise StudioError('O upload excede o limite configurado.',code='upload_limit',status=413)
                    out.write(block);digest.update(block)
                    if size%(4*1024**2)<len(block):store.budget(pid)
            if not size:raise StudioError('O arquivo enviado está vazio.')
            store.budget(pid);return digest.hexdigest()
        except BaseException:
            path.unlink(missing_ok=True);raise

    @app.get('/healthz')
    def health():return {'status':'ok','application':'Vos Frame Studio','mode':'exact','workers':1}
    @app.get('/')
    def index():return FileResponse(STATIC/'index.html')
    app.mount('/static',StaticFiles(directory=STATIC),name='static')
    @app.get('/api/session')
    def bootstrap(request:Request):
        return {'csrf':signature('csrf',request.state.token),'authenticated':request.state.authenticated,'password_required':bool(settings.password),
                'limits':{'video_mb':settings.video_bytes//1024**2,'zip_mb':settings.zip_bytes//1024**2,'duration_seconds':settings.max_seconds,'frames':settings.max_frames,'pixels':settings.max_pixels,'ttl_seconds':settings.ttl},
                'storage_notice':'Os projetos são temporários. No Render gratuito, reinicialização ou hibernação pode apagar os arquivos. Baixe o ZIP e os resultados; não há recuperação garantida.'}
    @app.post('/api/login')
    def login(options:Login,request:Request):
        if len(attempts)>1000:
            cutoff=time.monotonic()-60
            for host,values in list(attempts.items()):
                if not values or values[-1]<cutoff:attempts.pop(host,None)
            if len(attempts)>1000:raise StudioError('Muitas tentativas. Aguarde um minuto.',status=429)
        key=request.client.host if request.client else 'unknown';now=time.monotonic();recent=[t for t in attempts.get(key,[]) if t>now-60]
        if len(recent)>=12:raise StudioError('Muitas tentativas. Aguarde um minuto.',status=429)
        if settings.password and not hmac.compare_digest(options.password,settings.password):
            attempts[key]=recent+[now];raise StudioError('Senha incorreta.',status=401)
        attempts.pop(key,None);response=JSONResponse({'authenticated':True});response.set_cookie('vos_auth',signature('auth',request.state.token),httponly=True,secure=settings.secure_cookies,samesite='strict',max_age=604800);return response
    @app.post('/api/logout')
    def logout():
        response=JSONResponse({'ok':True});response.delete_cookie('vos_auth');response.delete_cookie('vos_session');return response
    @app.get('/api/projects')
    def projects(request:Request):return store.list(request.state.owner)
    @app.post('/api/projects')
    async def new_project(request:Request,filename:str='video.mp4'):
        name=''.join(c for c in filename.replace('\\','/').split('/')[-1] if ord(c)>=32)[:180]
        project=store.create(request.state.owner,name or 'video');root=store.path(project['id'])
        try:
            project['source_sha256']=await upload(request,root/'source.media',settings.video_bytes,project['id']);project['status']='probing';store.save(project)
            def work(job):
                try:return probe_project(store,project,job)
                except BaseException:
                    project['status']='failed';store.save(project);raise
            job=queue.submit(project,'probe',work);return {'project_id':project['id'],'job':job.id}
        except BaseException:shutil.rmtree(root,ignore_errors=True);raise
    @app.get('/api/projects/{pid}')
    def get_project(pid:str,request:Request):return public_project(project_for(request,pid,False))
    @app.delete('/api/projects/{pid}')
    def delete_project(pid:str,request:Request):
        project_for(request,pid,False);idle(pid);shutil.rmtree(store.path(pid));return {'deleted':True}
    @app.get('/api/projects/{pid}/source')
    def source(pid:str,request:Request):
        project=project_for(request,pid,False)
        import mimetypes
        media=mimetypes.guess_type(project['name'])[0] or 'application/octet-stream'
        return FileResponse(store.path(pid)/'source.media',media_type=media)
    @app.post('/api/projects/{pid}/scenes')
    def update_scenes(pid:str,options:Cuts,request:Request):
        project=project_for(request,pid);idle(pid)
        if any(f['asset']['kind']=='import' for f in project['timeline']):raise StudioError('Remova as edições aplicadas antes de alterar os cortes.',status=409)
        project['scenes']=scenes_for(project['frames'],rational(project['source']['duration']),options.cuts);project['revision']+=1;project['timeline']=native_timeline(project);store.save(project);return public_project(project)
    @app.post('/api/projects/{pid}/reset-edits')
    def reset_edits(pid:str,request:Request):
        project=project_for(request,pid);idle(pid);project['timeline']=native_timeline(project);store.save(project);return {'reset':True}
    @app.post('/api/projects/{pid}/detect-scenes')
    def detect(pid:str,options:Detect,request:Request):
        project=project_for(request,pid)
        def work(job):
            timestamps=[]
            def line(value):
                if 'pts_time:' in value:
                    try:timestamps.append(rational(value.split('pts_time:',1)[1].split()[0]))
                    except (ValueError,IndexError):pass
            run_process([settings.ffmpeg,'-v','error','-nostdin',*input_args(store.path(pid)/'source.media'),'-map','0:v:0','-an',
                         '-vf',f'select=gt(scene\\,{options.threshold}),metadata=print:file=-','-fps_mode','passthrough','-threads','1','-f','null','-'],job,on_line=line)
            origin=rational(project['source']['first_timestamp']);times=[rational(f['start']) for f in project['frames']];cuts=[]
            for at in timestamps:
                t=at-origin;n=bisect_left(times,t)
                if 0<n<len(times):
                    if n>0 and abs(times[n-1]-t)<abs(times[n]-t):n-=1
                    if n>0:cuts.append(n)
            return {'suggested_cuts':sorted(set(cuts)),'note':'Revise os cortes sugeridos e aplique-os manualmente.'}
        return {'job':queue.submit(project,'detect',work).id}
    @app.post('/api/projects/{pid}/package-plan')
    def package_preview(pid:str,options:PackageOptions,request:Request):
        project=project_for(request,pid);a,b,start,end,frames,sampling=package_plan(project,options.selection.model_dump(),options.fps)
        return {'source_start':str(start),'source_end':str(end),'duration':str(end-start),'expected_frames':len(frames),'sampling':sampling}
    @app.post('/api/projects/{pid}/packages')
    def extract(pid:str,options:PackageOptions,request:Request):
        project=project_for(request,pid)
        return {'job':queue.submit(project,'extract',lambda job:export_package(store,project,options.selection.model_dump(),options.fps,options.confirm_sampling,job)).id}
    @app.post('/api/projects/{pid}/imports')
    async def receive_import(pid:str,request:Request):
        project=project_for(request,pid);idle(pid);iid=identifier();folder=store.path(pid)/'imports'/iid;folder.mkdir()
        try:
            await upload(request,folder/'upload.zip',settings.zip_bytes,pid)
            job=queue.submit(project,'validate_import',lambda job:validate_import(store,project,iid,job));return {'import_id':iid,'job':job.id}
        except BaseException:shutil.rmtree(folder,ignore_errors=True);raise
    @app.get('/api/projects/{pid}/imports/{iid}')
    def import_details(pid:str,iid:str,request:Request):
        project_for(request,pid);valid_id(iid);p=store.path(pid)/'imports'/iid/'import.json'
        if not p.exists():raise StudioError('A validação ainda não terminou ou a importação expirou.',status=404)
        return read_json(p)
    @app.post('/api/projects/{pid}/imports/{iid}/apply')
    def apply(pid:str,iid:str,request:Request):
        project=project_for(request,pid);valid_id(iid);idle(pid);return apply_import(store,project,iid)
    @app.get('/api/projects/{pid}/frames/source/{index}')
    def source_frame(pid:str,index:int,request:Request):
        project=project_for(request,pid)
        if not 0<=index<len(project['frames']):raise StudioError('Quadro não encontrado.',status=404)
        path=store.path(pid)/'originals'/f'frame_{index:06d}.png'
        if not path.is_file():raise StudioError('Exporte os quadros desta cena antes de compará-los.',status=404)
        return FileResponse(path,media_type='image/png')
    @app.get('/api/projects/{pid}/frames/import/{iid}/{order}')
    def edited_frame(pid:str,iid:str,order:int,request:Request):
        project_for(request,pid);valid_id(iid);root=store.path(pid)/'imports'/iid;p=root/'import.json'
        if not p.exists():raise StudioError('Importação não encontrada.',status=404)
        report=read_json(p)
        if not report['valid'] or not 0<=order<len(report['manifest']['frames']):raise StudioError('Quadro não disponível.',status=404)
        f=report['manifest']['frames'][order];return FileResponse(root/'frames'/f['file'],media_type='image/png')
    @app.post('/api/projects/{pid}/render-plan')
    def render_preview(pid:str,options:BuildOptions,request:Request):
        project=project_for(request,pid);a,b,start,end,frames,sampling=render_plan(project,options.selection.model_dump(),options.fps,True)
        return {'frames':len(frames),'duration':str(end-start),'source_start':str(start),'source_end':str(end),'sampling':sampling,'odd_dimensions':bool(project['source']['width']%2 or project['source']['height']%2)}
    @app.post('/api/projects/{pid}/renders')
    def render(pid:str,options:BuildOptions,request:Request):
        project=project_for(request,pid)
        return {'job':queue.submit(project,'render',lambda job:build_video(store,project,options.selection.model_dump(),options.fps,options.allow_drop,options.preserve_audio,options.crf,job)).id}
    @app.get('/api/projects/{pid}/artifacts/{aid}.{extension}')
    def artifact(pid:str,aid:str,extension:str,request:Request,download:bool=True):
        project=project_for(request,pid);valid_id(aid)
        if extension not in ['zip','mp4','json']:raise StudioError('Arquivo não encontrado.',status=404)
        path=store.path(pid)/'outputs'/(aid+'.'+extension)
        if not path.is_file():raise StudioError('Resultado não encontrado ou expirado.',status=404)
        if extension in ['mp4','json'] and not any(o['id']==aid for o in project['outputs']):raise StudioError('Resultado não validado.',status=404)
        if extension=='zip' and not (store.path(pid)/'packages'/aid/'manifest.json').exists():raise StudioError('Pacote indisponível.',status=404)
        mime={'zip':'application/zip','mp4':'video/mp4','json':'application/json'}[extension]
        return FileResponse(path,media_type=mime,filename=('vos-'+aid[:8]+'.'+extension) if download else None,content_disposition_type='attachment' if download else 'inline')
    @app.get('/api/jobs/{jid}')
    def get_job(jid:str,request:Request):return queue.get(jid,request.state.owner).public()
    @app.post('/api/jobs/{jid}/cancel')
    def cancel(jid:str,request:Request):
        job=queue.get(jid,request.state.owner);job.cancelled.set();return {'cancel_requested':True}
    return app


app=create_app()
