"""Regressao da ponte ZIP VOS/Wan, sem GPU e sem pesos generativos."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from PIL import Image
from frame_studio.wan_bridge import read_vos, prepare


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def fixture(self, *, sampled=False, traversal=False, scenes=False):
        frames=[]
        files={}
        for i in range(2):
            name=("../escape.png" if i==0 and traversal else
                  f"scenes/scene_00{2 if i and scenes else 1}/frame_{i:06d}.png")
            img=Image.new("RGB",(32,32),(20+i*80,40,90))
            content=io.BytesIO();img.save(content,"PNG")
            files[name]=content.getvalue()
            frames.append({"file":name,"order":i,"scene_id":("scene_002" if i and scenes else "scene_001"),
                           "width":32,"height":32,"timestamp":str(i)+"/30","duration":"1/30"})
        manifest={"schema":"vos-frame-studio/1","project_id":"project-test","package_id":"package-test",
                  "sampling":{"mode":"15" if sampled else "native"},
                  "source":{"width":32,"height":32},
                  "selection":{"duration":"1/15"},"expected_frame_count":2,"frames":frames}
        path=self.root/"source.zip"
        raw=json.dumps(manifest,indent=2).encode()
        with zipfile.ZipFile(path,"w") as z:
            z.writestr("manifest.json",raw)
            for name,content in files.items():z.writestr(name,content)
        return path,raw

    def test_manifest_bytes_and_rational_timestamps(self):
        z,raw=self.fixture()
        m,frames,starts,durations=read_vos(z,self.root/"decoded")
        self.assertEqual((self.root/"decoded/manifest.json").read_bytes(),raw)
        self.assertEqual([str(t) for t in starts],["0","1/30"])
        self.assertEqual([str(t) for t in durations],["1/30","1/30"])
        self.assertEqual(len(frames),2)

    def test_native_scene_only(self):
        for args in ({"sampled":True},{"traversal":True},{"scenes":True}):
            with self.subTest(args=args):
                z,_=self.fixture(**args)
                with self.assertRaises(ValueError):
                    read_vos(z,self.root/("decoded-"+str(len(list(self.root.iterdir())))))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg ausente")
    def test_finish_preserves_every_outside_mask_pixel_and_manifest(self):
        from PIL import ImageDraw
        from frame_studio.wan_bridge import finish
        z,raw=self.fixture()
        ref=self.root/"reference.jpg"
        Image.new("RGB",(32,32),(200,150,60)).save(ref)
        work=self.root/"job";prepare(z,ref,work)
        result_dir=work/"synthetic_generated";mask_dir=work/"synthetic_mask"
        processed=work/"wan_processed"
        for p in (result_dir,mask_dir,processed):p.mkdir(exist_ok=True)
        for i in range(2):
            Image.new("RGB",(32,32),(230,15+i*10,30)).save(result_dir/f"frame_{i:06d}.png")
            mask=Image.new("RGB",(32,32),(0,0,0))
            ImageDraw.Draw(mask).rectangle((10,10,20,20),fill=(255,255,255))
            mask.save(mask_dir/f"frame_{i:06d}.png")
        def encode(folder,dest):
            subprocess.run(["ffmpeg","-v","error","-y","-framerate","30",
                            "-i",str(folder/"frame_%06d.png"),"-c:v","libx264",
                            "-pix_fmt","yuv420p",str(dest)],check=True)
        encode(result_dir,work/"wan_generated.mp4")
        encode(mask_dir,processed/"src_mask.mp4")
        from frame_studio.wan_bridge import generation_hashes
        # Recibo de fixture sintetica, somente para testar composicao; nao e IA.
        (work/"generation.json").write_text(json.dumps(generation_hashes(work)))
        out=self.root/"edited.zip"
        finish(work,out)
        with zipfile.ZipFile(z) as original,zipfile.ZipFile(out) as edited:
            self.assertEqual(edited.read("manifest.json"),raw)
            self.assertEqual(set(edited.namelist()),set(original.namelist()))
            for name in (n for n in original.namelist() if n.endswith(".png")):
                with Image.open(io.BytesIO(original.read(name))) as a,Image.open(io.BytesIO(edited.read(name))) as b:
                    self.assertEqual(a.size,b.size)
                    for y in range(32):
                        for x in range(32):
                            if not (10<=x<=20 and 10<=y<=20):
                                self.assertEqual(a.getpixel((x,y)),b.getpixel((x,y)))
                    self.assertNotEqual(a.getpixel((15,15)),b.getpixel((15,15)))
        self.assertEqual(len(read_vos(out,self.root/"verified")[1]),2)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),"FFmpeg ausente")
    def test_prepare_without_gpu(self):
        z,raw=self.fixture()
        reference=self.root/"reference.jpg"
        Image.new("RGB",(32,32),(200,150,60)).save(reference)
        prepare(z,reference,self.root/"job")
        job=self.root/"job"
        self.assertTrue((job/"driving.mp4").is_file())
        self.assertTrue((job/"reference.png").is_file())
        self.assertEqual((job/"manifest.json").read_bytes(),raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),json.loads((job/"bridge.json").read_text())["manifest_sha256"])
        stream=subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
                                        "-count_frames","-show_entries","stream=nb_read_frames",
                                        "-of","default=nw=1:nk=1",str(job/"driving.mp4")],text=True)
        self.assertEqual(int(stream.strip()),2)





class BridgeFailureTests(unittest.TestCase):
    setUp = BridgeTests.setUp
    fixture = BridgeTests.fixture
    def test_weight_manifest_rejects_truncated_and_changed_files(self):
        from frame_studio.wan_bridge import verify_model_manifest
        content=b'local test fixture, not a model'
        weight=self.root/'tensor';weight.write_bytes(content)
        manifest=self.root/'weights.json'
        manifest.write_text(json.dumps({'files':[{'path':'tensor','size':len(content),'sha256':hashlib.sha256(content).hexdigest()}]}))
        verify_model_manifest(self.root,manifest)
        for bad in (content[:-1],b'x'*len(content)):
            weight.write_bytes(bad)
            with self.assertRaises(ValueError):verify_model_manifest(self.root,manifest)

    def test_mask_tracking_jump_and_bad_coverage_rejected(self):
        from PIL import ImageDraw
        from frame_studio.wan_bridge import mask_statistics
        paths=[]
        for n,box in enumerate(((1,1,4,4),(26,26,30,30))):
            path=self.root/f'mask{n}.png';im=Image.new('L',(32,32));ImageDraw.Draw(im).rectangle(box,fill=255);im.save(path);paths.append(path)
        with self.assertRaisesRegex(ValueError,'Salto'):mask_statistics(paths)
        for color in (0,255):
            Image.new('L',(32,32),color).save(paths[0])
            with self.assertRaises(ValueError):mask_statistics(paths[:1])

    def test_inference_requires_current_review_and_clears_stale_output(self):
        from unittest.mock import patch
        from frame_studio.wan_bridge import infer, STAGE_FILES, stage_hashes
        wan=self.root/'Wan';wan.mkdir();(wan/'generate.py').touch()
        stage=self.root/'wan_processed';stage.mkdir()
        for name in STAGE_FILES:(stage/name).write_bytes(b'test fixture')
        generated=self.root/'wan_generated.mp4';generated.write_bytes(b'old output')
        with patch('frame_studio.wan_bridge.cmd') as command:
            with self.assertRaisesRegex(ValueError,'revisao visual'):infer(self.root,wan,self.root/'weights')
            (self.root/'mask_review.json').write_text(json.dumps(stage_hashes(self.root)))
            (stage/'src_mask.mp4').write_bytes(b'different mask')
            with self.assertRaisesRegex(ValueError,'revisao visual'):infer(self.root,wan,self.root/'weights')
            (self.root/'mask_review.json').write_text(json.dumps(stage_hashes(self.root)))
            with self.assertRaises(FileNotFoundError):infer(self.root,wan,self.root/'weights')
            self.assertFalse(generated.exists())
            command.assert_not_called()

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg ausente')
    def test_finish_rejects_same_count_wrong_fps_without_zip(self):
        from frame_studio.wan_bridge import finish
        archive,_=self.fixture();reference=self.root/'ref.png';Image.new('RGB',(32,32)).save(reference)
        work=self.root/'job';prepare(archive,reference,work)
        subprocess.run(['ffmpeg','-v','error','-y','-framerate','15','-i',str(work/'sampled/frame_%06d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(work/'wan_generated.mp4')],check=True)
        output=self.root/'edited.zip'
        from frame_studio.wan_bridge import video_info
        with self.assertRaisesRegex(ValueError,'FPS'):video_info(work/"wan_generated.mp4",2)
        with self.assertRaisesRegex(ValueError,'Geracao nao concluida'):finish(work,output)
        self.assertFalse(output.exists())

    def test_review_is_bound_to_current_intermediates_and_failure_invalidates_result(self):
        from unittest.mock import patch
        from frame_studio.wan_bridge import STAGE_FILES, stage_hashes, infer, review_mask
        work=self.root/'job';work.mkdir();stage=work/'wan_processed';stage.mkdir()
        wan=self.root/'wan';wan.mkdir();(wan/'generate.py').touch()
        (work/'bridge.json').write_text(json.dumps({'frame_count':2}))
        for name in STAGE_FILES:(stage/name).write_bytes(b'test fixture')
        with patch('frame_studio.wan_bridge.video_info',return_value=(32,32)):
            with self.assertRaisesRegex(ValueError,'preprocess'):review_mask(work)
            (work/'preprocess.json').write_text(json.dumps(stage_hashes(work)))
            review_mask(work)
        with patch('frame_studio.wan_bridge.cmd') as runner:
            (stage/'src_mask.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'revisao visual'):infer(work,wan,self.root/'weights')
            runner.assert_not_called()
        (work/'mask_review.json').write_text(json.dumps(stage_hashes(work)))
        (work/'wan_generated.mp4').write_bytes(b'stale result')
        (work/'generation.json').write_text('{}')
        with patch('frame_studio.wan_bridge.check_checkpoints'),patch('frame_studio.wan_bridge.cmd',side_effect=subprocess.CalledProcessError(1,'test generator')):
            with self.assertRaises(subprocess.CalledProcessError):infer(work,wan,self.root/'weights')
        self.assertFalse((work/'wan_generated.mp4').exists())
        self.assertFalse((work/'generation.json').exists())

    def test_wan_subprocess_disables_hub_network(self):
        from unittest.mock import patch
        from frame_studio.wan_bridge import cmd
        with patch('frame_studio.wan_bridge.subprocess.run') as runner:
            cmd(['test-not-a-real-generator'])
        self.assertEqual(runner.call_args.kwargs['env']['HF_HUB_OFFLINE'],'1')
        self.assertEqual(runner.call_args.kwargs['env']['TRANSFORMERS_OFFLINE'],'1')


    def test_preprocess_accepts_official_external_data_onnx_directory(self):
        from unittest.mock import patch
        from PIL import ImageDraw
        from frame_studio.wan_bridge import preprocess, STAGE_FILES
        work=self.root/'job';work.mkdir();(work/'bridge.json').write_text('{"frame_count":2}')
        wan=self.root/'wan';wan.mkdir();(wan/'generate.py').touch()
        ck=self.root/'weights'
        for name in ('det/yolov10m.onnx','pose2d/vitpose_h_wholebody.onnx/end2end.onnx','sam2/sam2_hiera_large.pt'):
            file=ck/'process_checkpoint'/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'test')
        mask=self.root/'mask.png';im=Image.new('L',(32,32));ImageDraw.Draw(im).rectangle((8,8,15,15),fill=255);im.save(mask)
        def fake_preprocess(args, cwd=None):
            for name in STAGE_FILES:(work/'wan_processed'/name).write_bytes(b'structural fixture')
        with patch('frame_studio.wan_bridge.cmd',side_effect=fake_preprocess) as runner,patch('frame_studio.wan_bridge.video_info',return_value=(32,32)),patch('frame_studio.wan_bridge.extract_video_png',return_value=[mask,mask]):
            preprocess(work,wan,ck)
        self.assertIn('--replace_flag',runner.call_args.args[0])
        self.assertTrue((work/'preprocess.json').is_file())
        (work/'mask_review.json').write_text('{}');(work/'generation.json').write_text('{}')
        with patch('frame_studio.wan_bridge.cmd',side_effect=RuntimeError('failed preprocessor')):
            with self.assertRaises(RuntimeError):preprocess(work,wan,ck)
        self.assertFalse((work/'preprocess.json').exists())
        self.assertFalse((work/'mask_review.json').exists())
        self.assertFalse((work/'generation.json').exists())


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg ausente')
class BridgeEndToEndTests(unittest.TestCase):
    def test_vos_export_bridge_import_and_full_mp4_with_audio_and_cuts(self):
        """Synthetic colors exercise transport only; no generated performance claim."""
        from PIL import ImageDraw
        from frame_studio.tests.helpers import create_project, package, import_zip, job_for
        from frame_studio.packages import apply_import
        from frame_studio.render import build_video
        from frame_studio.wan_bridge import finish, generation_hashes
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);store,project=create_project(root)
            source=package(store,project,{'scope':'scene','scene_id':'scene_002'})
            reference=root/'reference.png';Image.new('RGB',(160,96),'green').save(reference)
            work=root/'bridge';prepare(source,reference,work)
            stage=work/'wan_processed';stage.mkdir()
            # Fixtures use real FFmpeg transport; they are not Wan inference.
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=red:s=160x96:r=30','-frames:v','6','-c:v','libx264',str(work/'wan_generated.mp4')],check=True)
            mask=Image.new('RGB',(160,96));ImageDraw.Draw(mask).rectangle((50,20,80,60),fill='white');mask.save(root/'mask.png')
            subprocess.run(['ffmpeg','-v','error','-y','-loop','1','-framerate','30','-i',str(root/'mask.png'),'-frames:v','6','-c:v','libx264','-pix_fmt','yuv420p',str(stage/'src_mask.mp4')],check=True)
            (work/'generation.json').write_text(json.dumps(generation_hashes(work)))
            edited=root/'edited.zip';finish(work,edited)
            iid,report=import_zip(store,project,edited)
            self.assertTrue(report['valid'],report['issues'])
            self.assertEqual(report['changed_files'],6)
            original=project['timeline'].copy();apply_import(store,project,iid)
            self.assertEqual(project['timeline'][:6],original[:6])
            self.assertEqual(project['timeline'][12:],original[12:])
            result=build_video(store,project,{'scope':'full'},'native',False,True,16,job_for(store,project))
            v=result['validation']
            self.assertEqual(v['decoded_frames'],18)
            self.assertEqual(v['duration'],'3/5')
            self.assertEqual(v['cuts'],['1/5','2/5'])
            self.assertEqual(v['resolution'],[160,96])
            self.assertTrue(v['audio'])
            self.assertTrue(v['complete_decode'])


if __name__=="__main__":unittest.main()
