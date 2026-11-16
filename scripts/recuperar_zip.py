#!/usr/bin/env python3
"""Remonta os ZIPs preservados em partes; também baixa partes ausentes do GitHub."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://raw.githubusercontent.com/aruanmf446-hub/Vos/main/backups/'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arquivo', choices=['entrega-e-projeto.zip', 'checkpoint-editavel.zip'], default='checkpoint-editavel.zip')
    p.add_argument('--dest', type=Path, default=Path.cwd())
    p.add_argument('--baixar', action='store_true', help='Baixar do GitHub as partes que faltarem ou falharem na verificação')
    args = p.parse_args()
    manifest = ROOT / 'backups/partes.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
    elif args.baixar:
        with urllib.request.urlopen(BASE + 'partes.json', timeout=60) as response:
            data = json.load(response)
    else:
        p.error('Manifesto ausente. Baixe o repositório ou use --baixar.')
    entry = next(x for x in data['archives'] if x['file'] == args.arquivo)
    args.dest.mkdir(parents=True, exist_ok=True)
    output = args.dest / args.arquivo
    if output.exists():
        if output.stat().st_size == entry['bytes'] and digest(output) == entry['sha256']:
            print('ZIP já está completo e verificado:', output)
            return
        raise SystemExit(f'O destino já existe e difere do backup: {output}. Escolha outra pasta.')
    parts = []
    for n, part in enumerate(entry['parts'], 1):
        name = part['file']
        if Path(name).name != name:
            raise SystemExit('Caminho inválido no manifesto.')
        path = ROOT / 'backups' / name
        if not path.exists():
            path = args.dest / 'partes' / name
        valid = path.exists() and path.stat().st_size == part['bytes'] and digest(path) == part['sha256']
        if not valid:
            if not args.baixar:
                raise SystemExit(f'Parte ausente ou inválida: {path}. Use --baixar para recuperar.')
            path = args.dest / 'partes' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + '.download')
            print(f'Baixando {n}/{len(entry["parts"])}: {name}', flush=True)
            with urllib.request.urlopen(BASE + name, timeout=120) as response, temp.open('wb') as dest:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    dest.write(block)
            if temp.stat().st_size != part['bytes'] or digest(temp) != part['sha256']:
                raise SystemExit(f'Falha de integridade: {name}. Execute novamente para tentar essa parte.')
            temp.replace(path)
        parts.append(path)
    temp = output.with_name(output.name + '.montando')
    h = hashlib.sha256()
    with temp.open('wb') as dest:
        for path in parts:
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    dest.write(block)
                    h.update(block)
    if temp.stat().st_size != entry['bytes'] or h.hexdigest() != entry['sha256']:
        raise SystemExit(f'Falha na montagem: {temp}')
    temp.replace(output)
    print('ZIP completo e SHA-256 verificado:', output)

if __name__ == '__main__':
    main()
