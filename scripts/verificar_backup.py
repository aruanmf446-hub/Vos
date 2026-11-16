"""Verify byte-exact captured project assets without rendering or network use."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'manifests/files.json').read_text())
problems=[]
for item in manifest['files']:
    p=ROOT/item['path']
    if not p.is_file():problems.append(f'Missing: {item["path"]}');continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if p.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:
        problems.append(f'Changed: {item["path"]}')
if problems:raise SystemExit('\n'.join(problems))
print(f'OK: {len(manifest["files"])} arquivos conferidos por tamanho e SHA-256.')
