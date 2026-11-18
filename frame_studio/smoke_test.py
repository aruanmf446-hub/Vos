"""Reproducible tiny synthetic transport test. NEVER calls SAM2/Wan generation.
Outputs only a report. Temporary colored video fixtures/receipts are deleted.
"""
import argparse
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
from PIL import Image,ImageDraw
try:
    from .wan_bridge import prepare,finish,read_vos,generation_hashes
    from . import target_masks as target
except ImportError:
    from wan_bridge import prepare,finish,read_vos,generation_hashes
    import target_masks as target


def run_smoke():
    results=[]
    for fps in (24,30,60):
        with tempfile.TemporaryDirectory(prefix='vos-synthetic-only-') as temporary:
            root=Path(temporary);frames=[];source=root/'synthetic-source.zip'
            with zipfile.ZipFile(source,'w') as z:
                for n in range(3):
                    name=f'scenes/scene_001/frame_{n:06d}.png';im=Image.new('RGB',(64,40),'gray');d=ImageDraw.Draw(im)
                    d.rectangle((5,10,15,30),fill='red');d.rectangle((45,10,55,30),fill='green');buf=io.BytesIO();im.save(buf,'PNG');z.writestr(name,buf.getvalue())
                    frames.append({'file':name,'order':n,'scene_id':'scene_001','width':64,'height':40,'timestamp':str(Fraction(n,fps)),'duration':str(Fraction(1,fps))})
                m={'schema':'vos-frame-studio/1','project_id':'synthetic-only','package_id':'not-a-real-vos-project','source':{'width':64,'height':40},'sampling':{'mode':'native'},'selection':{'duration':str(Fraction(3,fps))},'expected_frame_count':3,'frames':frames}
                raw=json.dumps(m).encode();z.writestr('manifest.json',raw)
            ref=root/'ref.png';Image.new('RGB',(64,40),'blue').save(ref);work=root/'job';prepare(source,ref,work)
            target.save_selection(work,{'name':'Pessoa sintetica','keyframes':{'0':{'points':[[.15,.5,1],[.8,.5,0]]}}});(work/'target_masks').mkdir()
            mask=Image.new('L',(64,40));ImageDraw.Draw(mask).rectangle((5,10,15,30),fill=255)
            for n in range(3):mask.save(target.mask_path(work,n))
            target.approve(work,[0,1,2]);stage=work/'wan_processed';stage.mkdir();count=json.loads((work/'bridge.json').read_text())['frame_count']
            mask.convert('RGB').save(root/'mask.png')
            subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-framerate','30','-i',str(root/'mask.png'),'-frames:v',str(count),'-c:v','libx264rgb','-crf','0',str(stage/'src_mask.mp4')],check=True)
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=blue:s=64x40:r=30','-frames:v',str(count),'-c:v','libx264',str(work/'wan_generated.mp4')],check=True)
            # Fixture receipt isolated in temporary work; explicitly not a successful model run.
            (work/'generation.json').write_text(json.dumps(generation_hashes(work)));output=root/'synthetic-edited.zip';finish(work,output)
            read_vos(output,root/'verified')
            with zipfile.ZipFile(source) as before,zipfile.ZipFile(output) as after:
                assert after.read('manifest.json')==raw
                for f in frames:
                    a=Image.open(io.BytesIO(before.read(f['file']))).convert('RGB');b=Image.open(io.BytesIO(after.read(f['file']))).convert('RGB')
                    assert all(a.getpixel((x,y))==b.getpixel((x,y)) for y in range(40) for x in range(64) if mask.getpixel((x,y))==0)
            results.append({'fps':fps,'native_frames':3,'duration':str(Fraction(3,fps)),'manifest_exact':True,'outside_mask_exact':True})
    return {'schema':'vos-synthetic-smoke/1','real_generation':False,'sam2_executed':False,'description':'Teste sintetico de transporte e composicao; nenhum modelo carregado; nao importar em projeto VOS real.','results':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);args=parser.parse_args()
    result=run_smoke();Path(args.report).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
