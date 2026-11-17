from contextlib import contextmanager
from pathlib import Path
import hashlib, hmac, json, os, re, secrets, shutil, threading, time
from .errors import StudioError

ID = re.compile(r'^[a-f0-9]{32}$')


def identifier():
    return secrets.token_hex(16)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + identifier() + '.tmp')
    try:
        with temp.open('w') as f:
            json.dump(value, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def directory_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.data_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.active = set()

    def owner(self, token):
        return hmac.new(self.settings.secret.encode(), token.encode(), hashlib.sha256).hexdigest()

    def path(self, project_id):
        if not ID.fullmatch(project_id):
            raise StudioError('Projeto não encontrado ou expirado.', status=404)
        return self.root / project_id

    def get(self, project_id, owner):
        path = self.path(project_id)
        try:
            project = read_json(path / 'project.json')
        except (OSError, ValueError):
            raise StudioError('Projeto não encontrado ou expirado. Envie o vídeo novamente.', status=404)
        if not hmac.compare_digest(project['owner'], owner):
            raise StudioError('Projeto não encontrado ou expirado.', status=404)
        os.utime(path, None)
        return project

    def save(self, project):
        atomic_json(self.path(project['id']) / 'project.json', project)

    def create(self, owner, name):
        project_id = identifier()
        path = self.path(project_id)
        path.mkdir(mode=0o700)
        for folder in ['originals', 'packages', 'imports', 'outputs', 'staging']:
            (path / folder).mkdir()
        project = {'id': project_id, 'owner': owner, 'name': Path(name).name[:180], 'created': time.time(), 'status': 'uploading', 'revision': 0, 'imports': [], 'outputs': []}
        self.save(project)
        return project

    def budget(self, project_id):
        s = self.settings
        if shutil.disk_usage(self.root).free < s.reserve_bytes:
            raise StudioError('Espaço temporário insuficiente. Baixe seus arquivos e remova projetos antigos.', code='disk_limit', status=507)
        if directory_size(self.path(project_id)) > s.project_bytes or directory_size(self.root) > s.total_bytes:
            raise StudioError('Limite de armazenamento temporário atingido. Reduza o trecho ou envie um vídeo menor.', code='disk_limit', status=507)

    def cleanup(self):
        cutoff = time.time() - self.settings.ttl
        with self.lock:
            for path in self.root.iterdir():
                if path.is_dir() and ID.fullmatch(path.name) and path.name not in self.active and path.stat().st_mtime < cutoff:
                    shutil.rmtree(path, ignore_errors=True)

    def list(self, owner):
        result = []
        for path in self.root.glob('*/project.json'):
            try:
                p = read_json(path)
                if hmac.compare_digest(p['owner'], owner):
                    result.append({k: p[k] for k in ['id', 'name', 'created', 'status']})
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda p: p['created'], reverse=True)
