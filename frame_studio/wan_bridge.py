"""VOS native ZIP -> Wan2.2 Animate Replacement -> VOS native ZIP.
No external inference API. Requires Pillow, FFmpeg, OpenCV, numpy for finalization.
"""
import argparse, bisect, hashlib, json, math, shutil, subprocess, sys, zipfile
from fractions import Fraction
from io import BytesIO
from pathlib import Path, PurePosixPath
from PIL import Image

def cmd(args, cwd=None):
    print("+", " ".join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)), cwd=cwd, check=True)

def read_vos(zippath, work):
    work=Path(work);work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zippath) as z:
        members=[x for x in z.infolist() if not x.is_dir()]
        names=[x.filename for x in members]
        if len(names)!=len(set(names)) or len(names)>6010 or sum(x.file_size for x in members)>2*1024**3:
            raise ValueError("ZIP invalido: nomes duplicados ou limite")
        for x in members:
            p=PurePosixPath(x.filename)
            if p.is_absolute() or any(s in (".","..") for s in p.parts) or "\\" in x.filename or ":" in x.filename or x.flag_bits&1 or ((x.external_attr>>16)&0o170000)==0o120000:
                raise ValueError("ZIP com caminho perigoso")
        raw=z.read("manifest.json"); m=json.loads(raw)
        fs=m.get("frames",[])
        if m.get("schema")!="vos-frame-studio/1" or not fs or len(fs)!=m.get("expected_frame_count"):
            raise ValueError("Manifesto VOS desconhecido")
        if m.get("sampling",{}).get("mode")!="native":
            raise ValueError("Selecione Original - todos os quadros no VOS")
        if len({f["scene_id"] for f in fs})!=1:
            raise ValueError("Exporte UMA cena por vez")
        expected={f["file"] for f in fs}
        if len(expected)!=len(fs) or set(names)-{"manifest.json","INSTRUCOES_EDICAO.md"}!=expected:
            raise ValueError("ZIP com quadros extras/ausentes")
        w,h=int(m["source"]["width"]),int(m["source"]["height"])
        starts=[];durations=[]
        for i,f in enumerate(fs):
            name=f["file"];t=Fraction(f["timestamp"]);d=Fraction(f["duration"])
            if f["order"]!=i or f["width"]!=w or f["height"]!=h or not name.startswith("scenes/") or len(PurePosixPath(name).parts)!=3 or not name.endswith(".png") or t<0 or d<=0 or (starts and t!=starts[-1]+durations[-1]):
                raise ValueError("Quadro, nome ou tempo invalido")
            starts.append(t);durations.append(d)
            content=z.read(name)
            with Image.open(BytesIO(content)) as im:
                if im.format!="PNG" or im.size!=(w,h):raise ValueError("PNG/resolucao invalida")
                im.load()
            dest=work/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        total=Fraction(m["selection"]["duration"])
        if starts[0]!=0 or starts[-1]+durations[-1]!=total:raise ValueError("Duracao divergente")
        (work/"manifest.json").write_bytes(raw)
        if "INSTRUCOES_EDICAO.md" in names:(work/"INSTRUCOES_EDICAO.md").write_bytes(z.read("INSTRUCOES_EDICAO.md"))
    return m,fs,starts,durations

def prepare(archive,reference,work):
    work=Path(work).resolve()
    if work.exists() and any(work.iterdir()):raise ValueError("Diretorio de trabalho deve estar vazio")
    m,fs,starts,ds=read_vos(archive,work)
    total=Fraction(m["selection"]["duration"]);count=math.ceil(total*30)
    if not (1<=count<=6000):raise ValueError("Duracao invalida")
    sampled=work/"sampled";sampled.mkdir()
    for n in range(count):
        t=min(total-Fraction(1,1000000),Fraction(2*n+1,60))
        index=max(0,min(len(fs)-1,bisect.bisect_right(starts,t)-1))
        shutil.copyfile(work/fs[index]["file"],sampled/f"frame_{n:06d}.png")
    w,h=m["source"]["width"],m["source"]["height"]
    cmd(["ffmpeg","-v","error","-y","-framerate","30","-i",sampled/"frame_%06d.png","-an","-c:v","libx264","-pix_fmt","yuv420p" if w%2==0 and h%2==0 else "yuv444p",work/"driving.mp4"])
    with Image.open(reference) as im:im.convert("RGB").save(work/"reference.png")
    shutil.copyfile(archive,work/"original_vos.zip")
    info={"frame_count":count,"fps":30,"width":w,"height":h,"manifest_sha256":hashlib.sha256((work/"manifest.json").read_bytes()).hexdigest(),"project_id":m["project_id"],"package_id":m["package_id"]}
    (work/"bridge.json").write_text(json.dumps(info,indent=2))
    print("Preparado:",info)

def generate(work,wan,checkpoint):
    work=Path(work).resolve();wan=Path(wan).resolve();ck=Path(checkpoint).resolve()
    if not (wan/"generate.py").is_file():raise FileNotFoundError("Wan2.2 oficial ausente")
    for x in ("det/yolov10m.onnx","pose2d/vitpose_h_wholebody.onnx","sam2/sam2_hiera_large.pt"):
        if not (ck/"process_checkpoint"/x).is_file():raise FileNotFoundError("Checkpoint ausente: "+x)
    stage=work/"wan_processed";stage.mkdir(exist_ok=True)
    cmd([sys.executable,wan/"wan/modules/animate/preprocess/preprocess_data.py","--ckpt_path",ck/"process_checkpoint","--video_path",work/"driving.mp4","--refer_path",work/"reference.png","--save_path",stage,"--fps","30","--resolution_area","1280","720","--iterations","3","--k","7","--w_len","1","--h_len","1","--replace_flag"],cwd=wan)
    for name in ("src_pose.mp4","src_face.mp4","src_bg.mp4","src_mask.mp4"):
        if not (stage/name).is_file():raise RuntimeError("Preprocessamento incompleto: "+name)
    cmd([sys.executable,wan/"generate.py","--task","animate-14B","--ckpt_dir",ck,"--src_root_path",stage,"--refert_num","1","--replace_flag","--use_relighting_lora","--offload_model","True","--convert_model_dtype","--save_file",work/"wan_generated.mp4"],cwd=wan)
    if not (work/"wan_generated.mp4").is_file():raise RuntimeError("Sem MP4 gerado pelo Wan")

def video_frames(path):
    import cv2
    cap=cv2.VideoCapture(str(path));out=[]
    if not cap.isOpened():raise ValueError("Video nao abriu: "+str(path))
    while True:
        ok,f=cap.read()
        if not ok:break
        out.append(f)
    cap.release()
    if not out:raise ValueError("Video vazio")
    return out

def finish(work,output):
    import cv2,numpy as np
    work=Path(work).resolve();info=json.loads((work/"bridge.json").read_text())
    m,frames,starts,durations=read_vos(work/"original_vos.zip",work)
    if hashlib.sha256((work/"manifest.json").read_bytes()).hexdigest()!=info["manifest_sha256"]:
        raise ValueError("Manifesto modificado")
    generated=video_frames(work/"wan_generated.mp4")
    masks=video_frames(work/"wan_processed/src_mask.mp4")
    count=info["frame_count"]
    if len(generated)!=count or len(masks)!=count:
        raise ValueError(f"Contagem Wan divergente: {len(generated)} gerados / {len(masks)} mascaras / {count} esperados")
    w,h=info["width"],info["height"]
    for f,t,d in zip(frames,starts,durations):
        i=max(0,min(count-1,int((t+d/2)*30)));path=work/f["file"]
        base=cv2.imread(str(path),cv2.IMREAD_COLOR);gen=generated[i]
        if base is None or abs(gen.shape[1]/gen.shape[0]-w/h)>0.035:raise ValueError("Resolucao ou proporcao divergente")
        mask=cv2.resize(masks[i][:,:,0],(w,h),interpolation=cv2.INTER_NEAREST)>=128
        fraction=float(np.mean(mask))
        if fraction==0 or fraction>0.65:raise ValueError(f"Mascara suspeita ({fraction:.1%}); reveja o personagem")
        gen=cv2.resize(gen,(w,h),interpolation=cv2.INTER_LANCZOS4)
        edited=base.copy();edited[mask]=gen[mask]
        if not np.array_equal(edited[~mask],base[~mask]):raise AssertionError("Cenario alterado fora da mascara")
        ok,png=cv2.imencode(".png",edited)
        if not ok:raise RuntimeError("PNG nao gerado")
        path.write_bytes(png.tobytes())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",allowZip64=True) as z:
        z.write(work/"manifest.json","manifest.json",compress_type=zipfile.ZIP_DEFLATED)
        if (work/"INSTRUCOES_EDICAO.md").is_file():z.write(work/"INSTRUCOES_EDICAO.md","INSTRUCOES_EDICAO.md",compress_type=zipfile.ZIP_DEFLATED)
        for f in frames:z.write(work/f["file"],f["file"],compress_type=zipfile.ZIP_STORED)
    with zipfile.ZipFile(output) as z:
        if z.read("manifest.json")!=(work/"manifest.json").read_bytes():raise AssertionError("Manifesto mudou")
    print("ZIP de retorno:",output)

if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="action",required=True)
    for name in ("prepare","run","finish"):sub.add_parser(name)
    p.add_argument("--zip");p.add_argument("--reference");p.add_argument("--work",required=True)
    p.add_argument("--wan");p.add_argument("--checkpoint");p.add_argument("--output")
    a=p.parse_args()
    if a.action=="prepare":prepare(a.zip,a.reference,a.work)
    elif a.action=="run":generate(a.work,a.wan,a.checkpoint)
    else:finish(a.work,a.output)
