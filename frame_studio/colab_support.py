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


class Session:
    def __init__(self,folder,work=None,download=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.work=Path(work) if work else None;self.download=download
        self.counter=max([int(p.stem) for p in self.folder.glob("*.json") if p.stem.isdecimal()]+[0])

    def bundle(self):
        output=self.folder/'VOS_LOGS.zip';temp=output.with_suffix('.partial')
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.folder.iterdir()):
                if p.is_file() and p.suffix in ('.log','.json','.txt'):z.write(p,p.name)
            if self.work and self.work.is_dir():
                for name in ('bridge.json','temporal_map.json','target.json','target_audit.json','target_review.json','preprocess.json','mask_report.json','mask_review.json','generation.json','memory_profile.json'):
                    p=self.work/name
                    if p.is_file():z.write(p,'job/'+name)
        temp.replace(output)
        return output

    def safe_bundle(self):
        try:return self.bundle()
        except OSError as error:
            print('Nao foi possivel compactar logs (disco/permissao):',error,flush=True)
            print('Guarde os .log existentes e a saida do notebook; nao reinicie antes de baixar.',flush=True)
            return None

    def run(self,args,label='etapa',cwd=None):
        self.counter+=1;log=self.folder/f'{self.counter:02d}.log'
        record={'label':label,'argv':list(map(str,args)),'cwd':str(cwd) if cwd else None,
                'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'python':sys.version,'platform':platform.system()+' '+platform.release(),'status':'running'}
        meta=log.with_suffix('.json')
        try:meta.write_text(json.dumps(record,indent=2))
        except OSError as error:raise RuntimeError('Disco/permissao insuficiente para logs; etapa nao iniciada') from error
        process=None
        try:
            with log.open('w') as stream:
                process=subprocess.Popen(list(map(str,args)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,bufsize=1,start_new_session=True)
                for line in process.stdout:
                    stream.write(line);stream.flush();print(line,end='',flush=True)
                code=process.wait()
                if code:raise subprocess.CalledProcessError(code,list(map(str,args)))
            record['status']='success'
        except BaseException as error:
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
            self.safe_bundle()


def install_error_hook(session, ipython):
    """Catch notebook-level SDK/editor failures too. No secrets/environment dump."""
    def collect(result):
        error=result.error_in_exec or result.error_before_exec
        if error is None:return
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
