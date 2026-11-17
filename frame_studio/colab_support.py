"""Local diagnostic collection and interruptible subprocesses for Colab (stdlib)."""
import datetime
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback
import time
import zipfile
import re
import shutil
import importlib.metadata



def redact(text):
    for key,value in os.environ.items():
        if re.search(r'token|password|secret|credential|api_key',key,re.I) and len(value)>=8:
            text=text.replace(value,'[REDACTED]')
    text=re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._-]+',r'\1[REDACTED]',text)
    text=re.sub(r'(?i)((?:token|password|secret|api[_-]?key|authorization)["\']?\s*[:=]\s*["\']?)[^\s"\',}]+',r'\1[REDACTED]',text)
    text=re.sub(r'(https?://)[^/\s:@]+:[^/\s@]+@',r'\1[REDACTED]@',text)
    return re.sub(r'\b(?:hf_|gh[pousr]_|sk-)[A-Za-z0-9_-]{8,}', '[REDACTED]',text)


class Session:
    def __init__(self,folder,work=None,download=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.work=Path(work) if work else None;self.download=download
        self.current_stage=None;self.failure_stage=None;self.last_error=None
        self.counter=max([int(p.stem) for p in self.folder.glob("*.json") if p.stem.isdecimal()]+[0])

    def bundle(self):
        # Diagnostics contain sanitized bounded text only, never masks/videos/weights.
        output=self.folder/'VOS_LOGS.zip';temp=output.with_suffix('.partial');budget=2*1024**2-1024
        candidates=[p for p in sorted(self.folder.iterdir()) if p.is_file() and (p.suffix=='.log' or (p.suffix=='.json' and (p.stem.isdecimal() or p.name in {'resources.json','versions.json','VOS_GPU_DIAGNOSTICO.json','synthetic_smoke.json'})))]
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in candidates[-50:]:
                with p.open('rb') as stream:
                    size=p.stat().st_size;stream.seek(max(0,size-32768));raw=stream.read(32768)
                text=redact(raw.decode('utf-8',errors='replace'))
                if size>32768:text='[Truncado: ultimos 32 KiB]\n'+text
                payload=text.encode()
                if len(payload)>budget:break
                z.writestr(p.name,payload);budget-=len(payload)
            z.writestr('LEIA.txt','Diagnostico pequeno e sanitizado. Selecao/mascaras ficam no backup de progresso separado. Nenhuma retomada de inferencia.\n')
        temp.replace(output)
        return output

    def snapshot(self):
        data={'disk_free_bytes':shutil.disk_usage(self.folder).free,'python':sys.version.split()[0]}
        try:
            values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            data['available_ram_bytes']=int(values['MemAvailable'].split()[0])*1024
        except (OSError,KeyError,ValueError):data['available_ram_bytes']=None
        if shutil.which('nvidia-smi'):
            try:data['gpu_after_stage']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],text=True,timeout=3).strip()[:1000]
            except (OSError,subprocess.SubprocessError):data['gpu_after_stage']='indisponivel'
        versions={}
        for name in ('torch','torchvision','torchaudio','numpy','Pillow','opencv-python','diffusers','transformers','tokenizers','accelerate','modelscope','sam2','moviepy','onnxruntime','flash-attn'):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]='nao instalado'
        (self.folder/'versions.json').write_text(json.dumps(versions,indent=2))
        (self.folder/'resources.json').write_text(json.dumps(data,indent=2))

    def safe_bundle(self):
        try:return self.bundle()
        except OSError as error:
            print('Nao foi possivel compactar logs (disco/permissao):',error,flush=True)
            print('Guarde os .log existentes e a saida do notebook; nao reinicie antes de baixar.',flush=True)
            return None

    def run(self,args,label='etapa',cwd=None):
        self.snapshot()
        self.counter+=1;log=self.folder/f'{self.counter:02d}.log'
        record={'stage':self.current_stage,'label':label,'argv':list(map(str,args)),'cwd':str(cwd) if cwd else None,
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'python':sys.version,'platform':platform.system()+' '+platform.release(),'status':'running'}
        meta=log.with_suffix('.json')
        try:meta.write_text(json.dumps(record,indent=2))
        except OSError as error:raise RuntimeError('Disco/permissao insuficiente para logs; etapa nao iniciada') from error
        process=None
        try:
            with log.open('w') as stream:
                process=subprocess.Popen(list(map(str,args)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,errors='replace',bufsize=1,start_new_session=True)
                for line in process.stdout:
                    stream.write(line);stream.flush();print(line,end='',flush=True)
                code=process.wait()
                if code:raise subprocess.CalledProcessError(code,list(map(str,args)))
            record['status']='success'
        except BaseException as error:
            self.last_error=error;self.failure_stage=self.current_stage
            if process:
                # Kill the group even if its leader has exited, so GPU grandchildren stop.
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            record['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            record['error']=str(error)
            with log.open('a') as stream:stream.write('\n'+traceback.format_exc())
            try:meta.write_text(json.dumps(record,indent=2))
            except OSError:pass
            artifact=self.safe_bundle()
            print('Falha/interrupcao. Logs locais:',artifact,flush=True)
            print('OOM: nao pule o diagnostico; reduza a cena e guarde o traceback. Sem recursos gratuitos suficientes, interrompa.',flush=True)
            if self.download and artifact:
                try:self.download(str(artifact))
                except Exception:print('Download automatico indisponivel; baixe VOS_LOGS.zip pelo painel Arquivos.')
            raise
        finally:
            if process and process.stdout:process.stdout.close()
            record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:meta.write_text(json.dumps(record,indent=2))
            except OSError:print('Falha ao gravar metadados; preserve a saida do notebook.')
            try:self.snapshot()
            except OSError:pass
            self.safe_bundle()


def install_error_hook(session, ipython):
    """Catch notebook-level SDK/editor failures too. No secrets/environment dump."""
    def collect(result):
        error=result.error_in_exec or result.error_before_exec
        if error is None or error is session.last_error:return
        if not session.failure_stage:session.failure_stage=session.current_stage
        if hasattr(session,'pipeline'):session.pipeline.failure()
        path=session.folder/f'notebook_failure-{time.time_ns()}.log'
        try:path.write_text(''.join(traceback.format_exception(type(error),error,error.__traceback__)))
        except OSError:print('Nao ha espaco para registrar traceback; preserve a saida do notebook.')
        output=session.safe_bundle();print('Diagnosticos de falha:',output)
        if session.download and output:
            try:session.download(str(output))
            except Exception:print('Baixe VOS_LOGS.zip no painel Arquivos.')
    previous=getattr(ipython,'_vos_failure_hook',None)
    if previous:ipython.events.unregister('post_run_cell',previous)
    ipython.events.register('post_run_cell',collect);ipython._vos_failure_hook=collect
    return collect


class Pipeline:
    stages=('install','prepare','selection','track','preprocess','review','infer','finish')
    def __init__(self,session):
        self.session=session;self.ready={s:False for s in self.stages};self.current=None;self.failed=None
    def begin(self,stage,preflight_ok=True):
        if not preflight_ok:raise RuntimeError('Diagnostico bloqueado. Nao iniciar uploads/downloads/inferencia.')
        index=self.stages.index(stage)
        if self.failed and index>self.stages.index(self.failed):raise RuntimeError('Etapa anterior falhou: '+self.failed+'. Corrija e execute novamente essa etapa.')
        if index and not self.ready[self.stages[index-1]]:raise RuntimeError('Execute a etapa anterior: '+self.stages[index-1])
        for name in self.stages[index:]:self.ready[name]=False
        self.current=stage;self.failed=None
    def complete(self,stage):
        if self.current!=stage:raise RuntimeError('Etapa fora de ordem')
        self.ready[stage]=True
    def failure(self):
        if self.current:self.ready[self.current]=False;self.failed=self.current


class Pipeline:
    stages=('install','prepare','selection','track','preprocess','review','infer','finish')
    def __init__(self,session):self.session=session;self.ready={stage:False for stage in self.stages}
    def begin(self,stage,preflight=True,requires=()):
        if not preflight:raise RuntimeError('Diagnostico bloqueado: nenhuma acao seguinte foi autorizada pelo ambiente.')
        failure=self.session.failure_stage
        if failure and self.stages.index(stage)>self.stages.index(failure):
            raise RuntimeError('Etapa anterior falhou: '+failure+'. Corrija e reexecute essa etapa ou uma anterior.')
        if any(not self.ready[name] for name in requires):raise RuntimeError('Etapas anteriores incompletas: '+', '.join(requires))
        index=self.stages.index(stage)
        for name in self.stages[index:]:self.ready[name]=False
        self.session.current_stage=stage;self.session.failure_stage=None;self.session.last_error=None
    def complete(self,stage):
        if self.session.failure_stage or self.session.current_stage!=stage:raise RuntimeError('Etapa nao concluiu: '+stage)
        self.ready[stage]=True
    def failure(self):
        stage=self.session.current_stage
        if stage:self.ready[stage]=False;self.session.failure_stage=stage
