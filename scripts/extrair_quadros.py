"""Re-extract the original source frames into an already restored workspace."""
from pathlib import Path
import argparse,subprocess,json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True,type=Path);a=p.parse_args();w=a.workspace.resolve()
inputs=[('avenida-marvel','1E20634F-D279-4FB4-8BDA-96DB412A04A7','1-ssstik.io_1794801823730-2.mp4',576,854,778)]
for project,group,name,width,height,expected in inputs:
    source=w/'anexos'/group/name
    if not source.exists():raise SystemExit(f'Vídeo original ausente: {source}')
    target=w/project/'work/source';target.mkdir(parents=True,exist_ok=True)
    if any(target.glob('*.png')):raise SystemExit(f'A pasta já contém quadros: {target}')
    subprocess.run(['ffmpeg','-v','error','-i',str(source),'-an','-vf',f'fps=30,scale={width}:{height}:flags=lanczos','-start_number','0',str(target/'%04d.png')],check=True)
    count=len(list(target.glob('*.png')))
    if count!=expected:raise SystemExit(f'{project}: esperados {expected} quadros, extraídos {count}; confira a fonte antes de renderizar.')
    print(f'{project}: {count} quadros extraídos.')
    if project=='avenida-marvel':
        active=w/project/'work/action';active.mkdir(exist_ok=True)
        subprocess.run(['ffmpeg','-v','error','-framerate','30','-i',str(target/'%04d.png'),'-vf','crop=576:364:0:230','-start_number','0',str(active/'%04d.png')],check=True)
print('Primeiro experimento: usar os 900 PNGs preservados no checkpoint editável.')
