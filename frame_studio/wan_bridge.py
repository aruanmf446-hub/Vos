"""VOS native ZIP -> Wan2.2 Animate Replacement -> VOS native ZIP.
No external inference API. Requires Pillow and FFmpeg/FFprobe for finalization.
"""
import argparse, bisect, hashlib, json, math, os, shutil, subprocess, sys, tempfile, zipfile
from fractions import Fraction
from io import BytesIO
from pathlib import Path, PurePosixPath
from PIL import Image

def cmd(args, cwd=None):
    print("+", " ".join(map(str,args)), flush=True)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    subprocess.run(list(map(str,args)), cwd=cwd, check=True, env=env)

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
    info={"original_zip_sha256":file_hash(work/"original_vos.zip"),"frame_count":count,"fps":30,"width":w,"height":h,"manifest_sha256":hashlib.sha256((work/"manifest.json").read_bytes()).hexdigest(),"project_id":m["project_id"],"package_id":m["package_id"]}
    (work/"bridge.json").write_text(json.dumps(info,indent=2))
    print("Preparado:",info)

STAGE_FILES = ("src_pose.mp4", "src_face.mp4", "src_bg.mp4", "src_mask.mp4", "src_ref.png")


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def stage_hashes(work):
    return {name: file_hash(Path(work)/"wan_processed"/name) for name in STAGE_FILES}


def verify_model_manifest(checkpoint, manifest):
    ck = Path(checkpoint).resolve()
    data = json.loads(Path(manifest).read_text())
    for entry in data["files"]:
        path = (ck/entry["path"]).resolve()
        if not path.is_relative_to(ck) or not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError("Peso ausente/tamanho incorreto: " + entry["path"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError("SHA-256 incorreto: " + entry["path"])
    print("Pesos locais verificados por tamanho e SHA-256:", len(data["files"]))


def check_checkpoints(checkpoint):
    ck = Path(checkpoint)
    required = ["models_t5_umt5-xxl-enc-bf16.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "Wan2.1_VAE.pth", "relighting_lora.ckpt", "config.json",
                "google/umt5-xxl/tokenizer_config.json", "xlm-roberta-large/tokenizer.json"]
    index = ck/"diffusion_pytorch_model.safetensors.index.json"
    if index.is_file():
        required += sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        required += ["diffusion_pytorch_model.safetensors"]
    for name in required:
        part = ck/name
        if not part.is_relative_to(ck) or ".." in Path(name).parts or not part.is_file() or part.stat().st_size == 0:
            raise FileNotFoundError("Checkpoint incompleto: " + name)


def mask_statistics(files):
    """Detecta saltos grosseiros, nao identifica pessoas nem certifica rastreamento."""
    report = []
    previous = None
    for i, path in enumerate(files):
        with Image.open(path) as image:
            mask = image.convert("L").point(lambda level:255 if level>=128 else 0)
            coverage = mask.histogram()[255]/(mask.width*mask.height)
            box = mask.getbbox()
            if not box or not 0 < coverage <= 0.65:
                raise ValueError(f"Mascara suspeita no quadro {i}: {coverage:.1%}")
            center = ((box[0]+box[2])/2/mask.width, (box[1]+box[3])/2/mask.height)
            if previous and math.dist(center, previous) > 0.35:
                raise ValueError(f"Salto de rastreamento no quadro {i}; revise possivel troca de pessoa")
            previous = center
            report.append({"frame":i,"coverage":coverage,"box":box})
    return report


def review_mask(work):
    """Registra a confirmacao visual explicita; nao decide quem foi segmentado."""
    work = Path(work)
    (work/"mask_review.json").unlink(missing_ok=True)
    expected = json.loads((work/"bridge.json").read_text())["frame_count"]
    for name in STAGE_FILES:
        if name.endswith(".mp4"): video_info(work/"wan_processed"/name, expected)
    hashes = stage_hashes(work)
    preflight = work/"preprocess.json"
    if not preflight.is_file() or json.loads(preflight.read_text()) != hashes:
        raise ValueError("Execute preprocess com sucesso antes de revisar a mascara")
    (work/"mask_review.json").write_text(json.dumps(hashes, indent=2))


def video_info(path, expected):
    result = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path)], text=True)
    streams = json.loads(result)["streams"]
    if len(streams) != 1:
        raise ValueError("Video sem stream: " + str(path))
    stream = streams[0]
    if int(stream.get("nb_read_frames", 0)) != expected or Fraction(stream["avg_frame_rate"]) != 30:
        raise ValueError("Contagem/FPS divergente (esperado 30 FPS): " + str(path))
    timing = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)], text=True))["frames"]
    if len(timing) != expected or any(abs(Fraction(f["best_effort_timestamp_time"])-Fraction(i,30)) > Fraction(1,1000) for i,f in enumerate(timing)):
        raise ValueError("Timestamps Wan fora da grade de 30 FPS: " + str(path))
    return stream["width"], stream["height"]


def paths_for_wan(work,wan,checkpoint):
    work=Path(work).resolve()
    wan=Path(wan).resolve()
    ck=Path(checkpoint).resolve()
    if not (wan/"generate.py").is_file():
        raise FileNotFoundError("Wan2.2 oficial ausente")
    return work,wan,ck

def preprocess(work,wan,checkpoint):
    """Roda apenas o preprocessamento oficial. Permite revisar mascara antes da IA."""
    work=Path(work).resolve()
    for name in ("preprocess.json", "mask_review.json", "generation.json", "wan_generated.mp4"):
        (work/name).unlink(missing_ok=True)
    work,wan,ck=paths_for_wan(work,wan,checkpoint)
    for x in ("det/yolov10m.onnx","pose2d/vitpose_h_wholebody.onnx","sam2/sam2_hiera_large.pt"):
        model = ck/"process_checkpoint"/x
        if model.is_dir(): model = model/"end2end.onnx"
        if not model.is_file():
            raise FileNotFoundError("Checkpoint ausente: "+x)
    stage=work/"wan_processed"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir()
    cmd([sys.executable,wan/"wan/modules/animate/preprocess/preprocess_data.py",
         "--ckpt_path",ck/"process_checkpoint",
         "--video_path",work/"driving.mp4",
         "--refer_path",work/"reference.png",
         "--save_path",stage,"--fps","30",
         "--resolution_area","1280","720",
         "--iterations","3","--k","7","--w_len","1","--h_len","1",
         "--replace_flag"],cwd=wan)
    for name in ("src_pose.mp4","src_face.mp4","src_bg.mp4","src_mask.mp4","src_ref.png"):
        if not (stage/name).is_file():
            raise RuntimeError("Preprocessamento incompleto: "+name)
    expected=json.loads((work/"bridge.json").read_text())["frame_count"]
    sizes = {name: video_info(stage/name, expected) for name in STAGE_FILES if name.endswith(".mp4")}
    if len({sizes[name] for name in ("src_pose.mp4", "src_bg.mp4", "src_mask.mp4")}) != 1:
        raise ValueError("Dimensoes dos intermediarios divergentes")
    mask_files=extract_video_png(stage/"src_mask.mp4",work/"mask_preflight")
    if len(mask_files)!=expected:
        raise ValueError(f"Mask tem {len(mask_files)} quadros, esperados {expected}; nao iniciar IA")
    report = mask_statistics(mask_files)
    (work/"mask_report.json").write_text(json.dumps(report, indent=2))
    (work/"preprocess.json").write_text(json.dumps(stage_hashes(work), indent=2))
    print("Preprocessamento validado estruturalmente; REVISE VISUALMENTE src_mask.mp4 e src_pose.mp4!")

def infer(work,wan,checkpoint):
    """So gera depois de existirem os intermediarios e a mascara revisada."""
    (Path(work)/"generation.json").unlink(missing_ok=True)
    work,wan,ck=paths_for_wan(work,wan,checkpoint)
    stage=work/"wan_processed"
    for name in ("src_pose.mp4","src_face.mp4","src_bg.mp4","src_mask.mp4","src_ref.png"):
        if not (stage/name).is_file():
            raise FileNotFoundError("Execute preprocess e revise a mascara antes: "+name)
    review = work/"mask_review.json"
    if not review.is_file() or json.loads(review.read_text()) != stage_hashes(work):
        raise ValueError("Confirme a revisao visual da mascara atual com review-mask --confirm SIM")
    (work/"wan_generated.mp4").unlink(missing_ok=True)
    check_checkpoints(ck)
    cmd([sys.executable,wan/"generate.py","--task","animate-14B",
         "--ckpt_dir",ck,"--src_root_path",stage,"--refert_num","1",
         "--replace_flag","--use_relighting_lora",
         "--offload_model","True","--convert_model_dtype",
         "--save_file",work/"wan_generated.mp4"],cwd=wan)
    if not (work/"wan_generated.mp4").is_file():
        raise RuntimeError("Sem MP4 gerado pelo Wan")
    expected = json.loads((work/"bridge.json").read_text())["frame_count"]
    video_info(work/"wan_generated.mp4", expected)
    (work/"generation.json").write_text(json.dumps(generation_hashes(work), indent=2))

def generate(work,wan,checkpoint):
    preprocess(work,wan,checkpoint)
    raise ValueError("Execute review-mask --confirm SIM apos revisar os videos e depois infer")

def extract_video_png(path, directory):
    """Decodifica sequencialmente via FFmpeg: memoria constante, sem CV2 ou GPU."""
    directory=Path(directory)
    if directory.exists():shutil.rmtree(directory)
    directory.mkdir(parents=True)
    cmd(["ffmpeg","-v","error","-y","-i",path,"-map","0:v:0",
         "-an","-fps_mode","passthrough","-start_number","0",
         directory/"frame_%06d.png"])
    found=sorted(directory.glob("frame_*.png"))
    if not found:raise ValueError("Nenhum quadro decodificado de "+str(path))
    if [p.name for p in found]!=[f"frame_{n:06d}.png" for n in range(len(found))]:
        raise ValueError("Sequencia incompleta ou incorreta em "+str(path))
    return found

def generation_hashes(work):
    work = Path(work)
    return {"generated_sha256":file_hash(work/"wan_generated.mp4"),
            "mask_sha256":file_hash(work/"wan_processed/src_mask.mp4"),
            "manifest_sha256":file_hash(work/"manifest.json")}


def finish(work,output):
    """Restaura todo quadro VOS e sobrescreve SOMENTE a regiao mascarada.
    Usa Pillow + FFmpeg com consumo de RAM constante, preservando manifesto.
    """
    work=Path(work).resolve()
    info=json.loads((work/"bridge.json").read_text())
    if file_hash(work/"original_vos.zip") != info["original_zip_sha256"]:
        raise ValueError("ZIP original modificado")
    receipt = work/"generation.json"
    if not receipt.is_file() or json.loads(receipt.read_text()) != generation_hashes(work):
        raise ValueError("Geracao nao concluida ou arquivos alterados; execute infer com sucesso")
    m,frames,starts,durations=read_vos(work/"original_vos.zip",work)
    if hashlib.sha256((work/"manifest.json").read_bytes()).hexdigest()!=info["manifest_sha256"]:
        raise ValueError("Manifesto modificado")
    count=info["frame_count"]
    generated_size = video_info(work/"wan_generated.mp4", count)
    mask_size = video_info(work/"wan_processed/src_mask.mp4", count)
    if generated_size != mask_size:
        raise ValueError("Dimensoes Wan/mascara divergentes; nao deformar")
    gen_files=extract_video_png(work/"wan_generated.mp4",work/"decoded_wan")
    mask_files=extract_video_png(work/"wan_processed/src_mask.mp4",work/"decoded_mask")
    count=info["frame_count"]
    if len(gen_files)!=count or len(mask_files)!=count:
        raise ValueError(f"Contagem Wan divergente: {len(gen_files)} gerados / {len(mask_files)} mascaras / {count} esperados")
    mask_statistics(mask_files)
    w,h=info["width"],info["height"]
    for f,t,d in zip(frames,starts,durations):
        i=max(0,min(count-1,int((t+d/2)*30)))
        path=work/f["file"]
        with Image.open(path) as original,Image.open(gen_files[i]) as result,Image.open(mask_files[i]) as raw_mask:
            base=original.convert("RGB")
            if abs(result.width/result.height-w/h)>0.035:
                raise ValueError("Proporcao do video Wan mudou; nao deformar")
            edited=result.convert("RGB").resize((w,h),Image.Resampling.LANCZOS)
            mask=raw_mask.convert("L").resize((w,h),Image.Resampling.NEAREST).point(
                lambda level:255 if level>=128 else 0)
            pixels=mask.histogram()[255]
            coverage=pixels/(w*h)
            if coverage==0 or coverage>0.65:
                raise ValueError(f"Mascara suspeita ({coverage:.1%}); revise personagem selecionado")
            # Somente pixels brancos da mascara recebem pixels gerados.
            base.paste(edited,(0,0),mask)
            base.save(path,format="PNG")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.resolve() == (work/"original_vos.zip").resolve():
        raise ValueError("Nao sobrescreva o ZIP original")
    handle, temporary = tempfile.mkstemp(prefix=output.name+".", suffix=".partial", dir=output.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary,"w",allowZip64=True) as z:
            z.write(work/"manifest.json","manifest.json",compress_type=zipfile.ZIP_DEFLATED)
            if (work/"INSTRUCOES_EDICAO.md").is_file():
                z.write(work/"INSTRUCOES_EDICAO.md","INSTRUCOES_EDICAO.md",compress_type=zipfile.ZIP_DEFLATED)
            for f in frames:z.write(work/f["file"],f["file"],compress_type=zipfile.ZIP_STORED)
        with zipfile.ZipFile(temporary) as z:
            if z.read("manifest.json")!=(work/"manifest.json").read_bytes():
                raise AssertionError("Manifesto mudou")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print("ZIP de retorno:",output)

if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="action",required=True)
    p_prepare=sub.add_parser("prepare",help="VOS ZIP para video de conducao")
    p_prepare.add_argument("--zip",required=True)
    p_prepare.add_argument("--reference",required=True)
    p_prepare.add_argument("--work",required=True)
    p_weights=sub.add_parser("verify-weights",help="Verificar pesos ModelScope locais por SHA-256")
    p_weights.add_argument("--checkpoint",required=True)
    p_weights.add_argument("--manifest",required=True)
    p_review=sub.add_parser("review-mask",help="Confirmar revisao visual dos intermediarios atuais")
    p_review.add_argument("--work",required=True)
    p_review.add_argument("--confirm",required=True,choices=["SIM"])
    for action in ("run","preprocess","infer"):
        stage=sub.add_parser(action,help="Wan2.2 Animate Replacement: "+action)
        stage.add_argument("--work",required=True)
        stage.add_argument("--wan",required=True)
        stage.add_argument("--checkpoint",required=True)
    p_finish=sub.add_parser("finish",help="Recompor ZIP VOS original")
    p_finish.add_argument("--work",required=True)
    p_finish.add_argument("--output",required=True)
    a=p.parse_args()
    if a.action=="prepare":prepare(a.zip,a.reference,a.work)
    elif a.action=="verify-weights":verify_model_manifest(a.checkpoint,a.manifest)
    elif a.action=="run":generate(a.work,a.wan,a.checkpoint)
    elif a.action=="review-mask":review_mask(a.work)
    elif a.action=="preprocess":preprocess(a.work,a.wan,a.checkpoint)
    elif a.action=="infer":infer(a.work,a.wan,a.checkpoint)
    else:finish(a.work,a.output)
