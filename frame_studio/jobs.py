from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os, signal, subprocess, tempfile, threading, time
from .errors import StudioError
from .storage import identifier


class Job:
    def __init__(self, store, project, kind):
        self.store, self.project, self.owner, self.kind = store, project['id'], project['owner'], kind
        self.id, self.status, self.stage = identifier(), 'queued', 'Aguardando processamento'
        self.progress, self.result, self.error = 0, None, None
        self.cancelled = threading.Event()
        self.deadline, self.last_budget = None, 0
        self._lock = threading.Lock()

    def update(self, stage, progress):
        with self._lock:
            self.stage, self.progress = stage, min(99, max(self.progress, int(progress)))

    def check(self, *, force=False):
        if self.cancelled.is_set():
            raise StudioError('Processamento cancelado. Os dados anteriores foram preservados.', code='cancelled')
        if self.deadline and time.monotonic() > self.deadline:
            raise StudioError('O processamento excedeu o tempo permitido. Use uma cena ou um trecho menor.', code='timeout')
        if force or time.monotonic() - self.last_budget > 2:
            self.store.budget(self.project)
            self.last_budget = time.monotonic()

    def public(self):
        with self._lock:
            return {k: getattr(self, k) for k in ['id', 'project', 'kind', 'status', 'stage', 'progress', 'result', 'error']}


class Queue:
    def __init__(self, store):
        self.store, self.jobs = store, {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vos-media')
        self.lock = threading.Lock()

    def submit(self, project, kind, function):
        with self.lock:
            live = [j for j in self.jobs.values() if j.status in ['queued', 'running']]
            if any(j.project == project['id'] for j in live):
                raise StudioError('Este projeto já está sendo processado. Aguarde ou cancele a tarefa atual.', status=409)
            if len(live) >= self.store.settings.queue_size:
                raise StudioError('Fila cheia. Tente novamente após a tarefa atual terminar.', code='queue_full', status=503)
            job = Job(self.store, project, kind)
            self.jobs[job.id] = job
            self.store.active.add(project['id'])
            self.executor.submit(self._execute, job, function)
            return job

    def _execute(self, job, function):
        job.status, job.deadline = 'running', time.monotonic() + self.store.settings.timeout
        try:
            job.check(force=True)
            result = function(job)
            with job._lock:
                job.result, job.status, job.progress, job.stage = result, 'succeeded', 100, 'Concluído'
        except StudioError as e:
            job.error = {'message': e.message, 'code': e.code, 'details': e.details}
            job.status = 'cancelled' if e.code == 'cancelled' else 'failed'
        except Exception:
            import logging
            logging.exception('Falha na tarefa %s', job.id)
            job.error = {'message': 'Falha no processamento. O projeto anterior não foi substituído. Tente um trecho menor.', 'code': 'processing_failed', 'details': []}
            job.status = 'failed'
        finally:
            self.store.active.discard(job.project)

    def get(self, job_id, owner):
        job = self.jobs.get(job_id)
        if not job or job.owner != owner:
            raise StudioError('Tarefa não encontrada ou expirada.', status=404)
        return job

    def close(self):
        for job in self.jobs.values():
            if job.status in ['queued', 'running']:
                job.cancelled.set()
        self.executor.shutdown(wait=True, cancel_futures=True)


@contextmanager
def managed_process(command, job, *, text=False):
    """Bounded RAM and wall time. Never use shell=True for media inputs."""
    if Path(command[0]).name == Path(job.store.settings.ffmpeg).name:
        command = [command[0], '-filter_threads', '1', '-filter_complex_threads', '1', *command[1:]]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=text, start_new_session=True)
        stop, failure = threading.Event(), []
        def watch():
            while not stop.wait(.2):
                try:
                    job.check()
                except StudioError as e:
                    failure.append(e)
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            yield process
            code = process.wait()
            if failure:
                raise failure[0]
            if code:
                errors.seek(0, 2)
                errors.seek(max(0, errors.tell() - 3000))
                detail = errors.read().decode('utf-8', 'replace')
                raise StudioError('O FFmpeg/FFprobe não conseguiu processar este arquivo.', code='media_error', details=[detail])
        finally:
            stop.set()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            process.stdout.close()
            watcher.join(timeout=1)


def run_process(command, job, *, on_line=None):
    with managed_process(command, job, text=True) as process:
        for line in process.stdout:
            if on_line:
                on_line(line.rstrip('\n'))
