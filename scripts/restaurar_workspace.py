"""Restore captured assets into a new workspace; dry-run unless --apply."""
from pathlib import Path,PurePosixPath
import argparse,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dest',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
p.add_argument('--apply',action='store_true');args=p.parse_args()
dest=args.dest.expanduser().resolve()
manifest=json.loads((ROOT/'manifests/files.json').read_text())
def mapped(name):
    parts=PurePosixPath(name).parts
    if not parts or name.startswith('/') or '..' in parts:raise ValueError(name)
    if parts[0] in ['avenida-marvel','strange-wanda','generated_images']:return Path(*parts)
    if parts[:2]==('media','videos'):return Path('shared/downloads',*parts[2:])
    if parts[:2]==('media','aprovados'):return Path('shared/downloads',*parts[2:])
    if parts[0]=='referencias':return Path('anexos',*parts[1:])
    return Path('backup-info',*parts)
print(f'Destino: {dest}')
print(f'Ativos do Git: {len(manifest["files"])}; checkpoint: {args.checkpoint or "não informado"}')
if not args.apply:print('Simulação. Use --apply para restaurar em uma pasta nova.');raise SystemExit(0)
if dest.exists() and any(dest.iterdir()):raise SystemExit('O destino precisa estar vazio; use outra pasta para preservar arquivos existentes.')
if args.checkpoint:
    with zipfile.ZipFile(args.checkpoint) as z:
        bad=z.testzip()
        if bad:raise SystemExit(f'Checkpoint inválido: {bad}')
        for info in z.infolist():mapped(info.filename)
dest.mkdir(parents=True,exist_ok=True)
if args.checkpoint:
    with zipfile.ZipFile(args.checkpoint) as z:
        for info in z.infolist():
            if info.is_dir():continue
            target=dest/mapped(info.filename);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,target.open('wb') as out:shutil.copyfileobj(source,out)
for item in manifest['files']:
    target=dest/item['restore_path'];target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/item['path'],target)
old_attachments=manifest['original_attachment_prefix']
changed=0
for folder in ['avenida-marvel','strange-wanda']:
    for f in (dest/folder).rglob('*'):
        if not f.is_file() or f.suffix not in ['.py','.json','.md','.txt']:continue
        try:text=f.read_text()
        except UnicodeDecodeError:continue
        new=text.replace('/workspace',str(dest)).replace(old_attachments,str(dest/'anexos'))
        if new!=text:f.write_text(new);changed+=1
print(f'Restauração concluída. Caminhos ajustados em {changed} arquivos de trabalho.')
