#!/usr/bin/env python3
"""Refresh the requested chat archive with the latest assets and video draft."""
from pathlib import Path
import zipfile, json, hashlib

ROOT=Path('/workspace')
ARCHIVE=ROOT/'shared/downloads/arquivos-do-chat.zip'


def main():
    additions={}
    attachment=Path('/tmp/codex-remote-attachments/6afa7d92-0128-8191-adc5-72526702d031/57AE234E-50CE-49F0-AE22-0489FBD4DEAE/1-Foto-1.jpg')
    if attachment.exists():additions['anexos/04_nova_referencia_doom/1-Foto-1.jpg']=attachment
    attachments=Path('/tmp/codex-remote-attachments/6afa7d92-0128-8191-adc5-72526702d031')
    for folder,label in [('FEE8B8BF-3C0A-48A9-B749-899179F9782B','05_doom_retrato'),('5A122554-FC6F-4C1E-813B-6F24C74AC12E','06_doom_mao'),('1DD8388A-55D6-468E-9290-018A245A61C8','07_referencias_doom')]:
        for p in (attachments/folder).glob('*.jpg'):
            additions[f'anexos/{label}/{p.name}']=p
    for p in (ROOT/'generated_images').glob('*.png'):
        additions[f'imagens_geradas/{p.name}']=p
    for p in (ROOT/'shared/downloads').glob('*.mp4'):
        if '.encoding.' not in p.name:
            additions[f'videos/{p.name}']=p
    project=ROOT/'avenida-marvel'
    for p in project.glob('*.py'):
        additions[f'projetos/02_cena_avenida/{p.name}']=p
    for p in (project/'work/anchors').glob('*.png'):
        additions[f'projetos/02_cena_avenida/quadros_aprovados/{p.name}']=p
    for directory in ['new_poseguides']:
        for p in (project/'work'/directory).glob('*'):
            if p.is_file():additions[f'projetos/02_cena_avenida/{directory}/{p.name}']=p
    for name in ['continuation_manifest.json','full_timeline_status.json']:
        p=project/'work'/name
        if p.exists():additions[f'projetos/02_cena_avenida/configuracoes/{name}']=p
    for name in ['delivery-report.json','export-validation.json']:
        p=project/'work/delivery'/name
        if p.exists():additions[f'projetos/02_cena_avenida/configuracoes/{name}']=p
    for name in ['doom-full-alpha.npy','doom-face-track.npy','source-grids.json','cleanplate-0000.png','cleanplate-0210.png','cleanplate-0302.png','cleanplate-0500.png']:
        p=project/'work/delivery'/name
        if p.exists():additions[f'projetos/02_cena_avenida/montagem_completa/{name}']=p
    temporary=ARCHIVE.with_name('arquivos-do-chat.atualizando.zip')
    hashes=[]
    with zipfile.ZipFile(ARCHIVE,'r') as old, zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as new:
        readme=old.read('LEIA-ME.txt').decode('utf-8')
        readme=readme.replace('O segundo vídeo completo de 25,93 segundos ainda NÃO foi concluído.',
            'A substituição de todos os personagens no segundo vídeo ainda NÃO foi concluída.')
        if 'ATUALIZAÇÃO DA MONTAGEM' in readme and 'Conteúdo' in readme:
            readme=readme.split('\n',1)[0]+'\n\nConteúdo'+readme.split('Conteúdo',1)[1]
        update=('ATUALIZAÇÃO DA MONTAGEM\n'
            'video-continuo-em-revisao-60fps.mp4 contém os 778 quadros originais de tempo, '
            '25,933 segundos em 60 FPS e sem áudio, sem remover planos. O cenário original é '
            'preservado fora da área de composição dos personagens. Novas poses foram adicionadas '
            'à levantada de Wanda e a duas reações de Doutor Estranho.\n'
            'IMPORTANTE: é um rascunho em revisão. O plano do Doom e o efeito/plano final '
            'continuam com os atores originais, pois as substituições não puderam ser geradas. '
            'Em algumas bordas de planos, o movimento foi aproximado por rastreamento.\n\n')
        if (ROOT/'shared/downloads/video-marvel-completo-60fps.mp4').exists():
            update=('ATUALIZAÇÃO DA MONTAGEM\n'
                'video-marvel-completo-60fps.mp4 é a montagem completa de 25,933 segundos, '
                '1556 quadros a 60 FPS e sem áudio. Contém Doutor Estranho, Wanda, Doutor Doom, '
                'Hulk e Sylvie, inclusive o plano do Doom e o retrato final. '
                'A levantada de Wanda e a correção de fundo aprovada foram preservadas.\n'
                'LIMITAÇÕES: o Doom e o retrato final foram animados a partir de imagens '
                'já geradas, usando movimento do vídeo original. Gestos e expressões '
                'continuam aproximados; a montagem não reproduz integralmente a atuação original. '
                'Os vídeos anteriores são versões de revisão e testes.\n\n')
            readme=readme.replace('A substituição de todos os personagens no segundo vídeo ainda NÃO foi concluída.',
                'A montagem completa está em video-marvel-completo-60fps.mp4; veja as limitações acima.')
        readme=readme.split('\n',1)[0]+'\n\n'+update+readme.split('\n',1)[1]
        if (ROOT/'shared/downloads/video-marvel-completo-60fps.mp4').exists():
            readme=('ARQUIVOS DESTE CHAT\n\n'+update+
                'ARQUIVO PRINCIPAL\n'
                'videos/video-marvel-completo-60fps.mp4\n\n'
                'PERSONAGENS\n'
                'Carminha -> Doutor Estranho\n'
                'Homem que se levanta -> Wanda\n'
                'Tufão -> Doutor Destino (Doom)\n'
                'Rapaz ao fundo -> Hulk\n'
                'Mulher sentada no sofá -> Sylvie\n\n'
                'CONTEÚDO\n'
                'anexos/: vídeos originais e imagens de referência enviados.\n'
                'videos/: montagem completa e versões anteriores de revisão e comparação.\n'
                'imagens_geradas/: imagens geradas durante o trabalho, incluindo intermediárias.\n'
                'projetos/: scripts, quadros de referência e configurações disponíveis.\n'
                'MANIFESTO.json: tamanho e SHA-256 de cada arquivo.\n\n'
                'Modelos de inferência e caches temporários de renderização não estão incluídos.\n')
        def add_bytes(name,data):
            new.writestr(name,data)
            hashes.append({'arquivo':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        add_bytes('LEIA-ME.txt',readme.encode('utf-8'))
        for info in old.infolist():
            if info.filename in ['LEIA-ME.txt','MANIFESTO.json'] or info.filename in additions:continue
            add_bytes(info.filename,old.read(info.filename))
        for name,p in sorted(additions.items()):add_bytes(name,p.read_bytes())
        new.writestr('MANIFESTO.json',json.dumps(hashes,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temporary) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'Invalid archive entry: {bad}')
        count=len(z.infolist())
    temporary.replace(ARCHIVE)
    print(json.dumps({'zip':str(ARCHIVE),'files':count,'bytes':ARCHIVE.stat().st_size,'crc':'OK'}))


if __name__=='__main__':main()
