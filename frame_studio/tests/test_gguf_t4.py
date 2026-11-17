"""GGUF CPU contracts and synthetic transport checks; NO real T4/model inference."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image,ImageDraw
from frame_studio.experimental.gguf_profile import diagnose,require_attempt
from frame_studio.experimental.gguf_t4 import manifest,verify_weights,geometry,fit,run_generation
from frame_studio.experimental.gguf_probe import read_header,require_animate

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'colab/wan22_t4_gguf_manifest.json'
BASELINE_SHA='3017dd54767683ee1b65eb176bfc965f62f820eb7ba88dd0aac24b90f3529f6c'


class GGUFTests(unittest.TestCase):
    def test_baseline_notebook_exactly_unchanged(self):
        self.assertEqual(hashlib.sha256((ROOT/'colab/VOS_Wan22_Animate_Replacement.ipynb').read_bytes()).hexdigest(),BASELINE_SHA)

    def test_modelscope_only_manifest_and_exact_same_animate_artifact(self):
        data=manifest(MANIFEST);model=next(f for f in data['files'] if f['role']=='animate_q3')
        self.assertEqual(model['repo'],'QuantStack/Wan2.2-Animate-14B-GGUF')
        self.assertEqual(model['size'],8630769472)
        self.assertEqual(model['sha256'],'62cd6067c96599ade5bdde54a49a5ef1a7fd4de449eaae56ff4fd61ccbb16678')
        self.assertLess(data['total_bytes'],20*2**30)
        self.assertFalse(data['enabled_by_default']);self.assertFalse(data['real_generation'])
        self.assertEqual({f['role'] for f in data['files']},{'animate_q3','t5','vae','clip','relight','preprocess'})
        for repo in data['repositories']:self.assertTrue(repo['url'].startswith('https://github.com/'));self.assertEqual(len(repo['commit']),40)
        self.assertFalse(any('diffusion_pytorch_model' in f['path'] for f in data['files']))

    def test_checksum_truncated_and_corrupt_quantized_file_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'test.gguf';raw=b'GGUF synthetic bytes, not model weights';path.write_bytes(raw)
            data={'files':[{'target':'test.gguf','path':'test.gguf','size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}
            verify_weights(root,data)
            for bad in (raw[:-1],b'x'*len(raw)):
                path.write_bytes(bad)
                with self.assertRaises(ValueError):verify_weights(root,data)

    def test_metadata_reader_no_weights_and_rejects_wrong_architecture(self):
        def text(s):b=s.encode();return struct.pack('<Q',len(b))+b
        b=b'GGUF'+struct.pack('<IQQ',3,3,1)+text('general.architecture')+struct.pack('<I',8)+text('wan')
        for name,kind in [('pose_patch_embedding.weight',0),('face_adapter.weight',0),('blocks.39.self_attn.q.weight',11)]:
            b+=text(name)+struct.pack('<IQQIQ',2,256,256,kind,0)
        header=read_header(b);require_animate(header)
        with self.assertRaises(ValueError):read_header(b[:-10])
        header['metadata']['general.architecture']='other'
        with self.assertRaises(ValueError):require_animate(header)

    def test_simulated_t4_is_not_a_gpu_proof_and_resource_shortage_blocks(self):
        t=SimpleNamespace(__version__='SIMULATED',version=SimpleNamespace(cuda='12.8'),cuda=SimpleNamespace(
            is_available=lambda:True,get_device_properties=lambda _:SimpleNamespace(name='T4 SIMULATED',total_memory=15*2**30),
            get_device_capability=lambda _:(7,5),mem_get_info=lambda _:(14*2**30,15*2**30)))
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=60*2**30)),patch('shutil.which',return_value='/fixture/nvcc'):
            report=diagnose('/tmp',torch_module=t,ram_bytes=12*2**30,probe=False)
        self.assertFalse(report['eligible_for_experimental_attempt']);self.assertFalse(report['real_generation']);self.assertFalse(report['wan_t4_proven'])
        for enabled in (False,True):
            with self.assertRaises(ValueError):require_attempt(report,enabled)
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=10*2**30)),patch('shutil.which',return_value=None):
            report=diagnose('/tmp',torch_module=t,ram_bytes=5*2**30,probe=False)
        self.assertTrue(any('RAM' in f for f in report['failures']));self.assertTrue(any('Disco' in f for f in report['failures']))

    def test_blocked_attempt_invalidates_stale_generation_without_touching_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            for name in ('generation.json','wan_generated.mp4','gguf_engine_receipt.json'):(work/name).write_text('stale')
            (work/'target.json').write_text('preserved')
            with self.assertRaises(ValueError):run_generation(work,work,{}, {'failures':['no GPU']},None,enabled=True)
            self.assertFalse((work/'generation.json').exists());self.assertFalse((work/'wan_generated.mp4').exists())
            self.assertEqual((work/'target.json').read_text(),'preserved')

    def test_internal_letterbox_preserves_native_aspect_without_cropping(self):
        for w,h in ((1920,1080),(720,1068),(161,97)):
            plan=geometry(w,h);self.assertTrue(all(x%16==0 for x in plan['internal_size']))
            self.assertLessEqual(max(plan['internal_size']),320)
            box=plan['content_box'];self.assertLess(abs((box[2]-box[0])/(box[3]-box[1])-w/h),.02)
            im=Image.new('L',(w,h));ImageDraw.Draw(im).rectangle((0,0,w-1,h-1),outline=255,width=5)
            adapted=fit(im,plan,mask=True);self.assertEqual(adapted.size,tuple(plan['internal_size']))
            self.assertEqual(adapted.getpixel((box[0],box[1])),255)

    def test_source_has_no_service_start_and_no_fake_receipt_bypass(self):
        code=(ROOT/'frame_studio/experimental/gguf_engine.py').read_text()
        self.assertNotIn('start_server',code);self.assertNotIn('ngrok',code);self.assertNotIn('http://',code)
        for node in ('WanVideoAnimateEmbeds','WanVideoSampler','CLIPLoaderGGUF','WanVideoLoraSelect'):
            self.assertIn(node,code)
        self.assertIn('merge_loras=False',code);self.assertIn("base_precision='fp16'",code)
        self.assertIn("model.model.diffusion_model.pose_patch_embedding",code)
        self.assertIn('force_offload=False',code) # avoid CPU spike at end of main sampling
        self.assertIn('images.shape[0]!=33',code);self.assertIn("'real_generation':True",code)

    def test_experimental_notebook_order_gates_and_python(self):
        nb=json.loads((ROOT/'colab/VOS_Wan22_Animate_GGUF_T4_EXPERIMENTAL.ipynb').read_text())
        codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        for n,s in enumerate(codes):compile(s,f'gguf-cell-{n}','exec')
        self.assertIn((ROOT/'frame_studio/experimental/gguf_profile.py').read_text().strip(),codes[0])
        self.assertNotIn('urlretrieve',codes[0]);self.assertNotIn('files.upload',codes[0])
        self.assertIn('LOW_VRAM_GGUF_EXPERIMENTAL=False',codes[1])
        self.assertRegex(codes[1],r"VOS_REF='[0-9a-f]{40}'")
        first=ast.parse(codes[1]).body
        self.assertTrue(any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and getattr(n.value.func,'id','')=='require_attempt' for n in first[:4]))
        for s in codes[2:]:
            guard=ast.parse(s).body[0];self.assertIsInstance(guard,ast.If)
            with self.assertRaises(RuntimeError):exec(compile(ast.Module(body=[guard],type_ignores=[]),'guard','exec'),{'GGUF_ENABLED':False})
        self.assertLess(next(i for i,s in enumerate(codes) if 'experiment.install(' in s),next(i for i,s in enumerate(codes) if 'files.upload' in s))
        self.assertTrue(all(not c.get('outputs') for c in nb['cells']))

    def test_kernel_failure_cannot_be_reclassified_as_eligible(self):
        t=SimpleNamespace(__version__='SIMULATED',version=SimpleNamespace(cuda='12.8'),cuda=SimpleNamespace(
            is_available=lambda:True,get_device_properties=lambda _:SimpleNamespace(name='SIMULATED T4',total_memory=15*2**30),
            get_device_capability=lambda _:(7,5),mem_get_info=lambda _:(14*2**30,15*2**30)))
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=60*2**30)),patch('shutil.which',return_value='fixture'),patch('frame_studio.experimental.gguf_profile.probe_kernels',side_effect=RuntimeError('no efficient kernel')):
            r=diagnose('/tmp',torch_module=t,ram_bytes=12*2**30)
        self.assertFalse(r['eligible_for_experimental_attempt']);self.assertFalse(r['kernel_probe_performed'])
        with self.assertRaises(ValueError):require_attempt(r,True)

    def test_experimental_report_in_sanitized_diagnostic_zip(self):
        import zipfile
        from frame_studio.colab_support import Session
        with tempfile.TemporaryDirectory() as tmp:
            session=Session(tmp);(Path(tmp)/'low_vram_report.json').write_text('{"real_generation":false,"token":"secret-example-value"}')
            (Path(tmp)/'private.mp4').write_bytes(b'private')
            with zipfile.ZipFile(session.bundle()) as z:
                self.assertIn('low_vram_report.json',z.namelist());self.assertNotIn('private.mp4',z.namelist())
                self.assertNotIn(b'secret-example-value',z.read('low_vram_report.json'))

    def test_receipt_rejects_simulation_changed_inputs_and_wrong_dimensions(self):
        from frame_studio.experimental.gguf_t4 import accept_result
        from frame_studio.wan_bridge import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            w=Path(tmp);(w/'wan_generated.mp4').write_bytes(b'NOT A REAL VIDEO; receipt validation fixture')
            plan={'stage_hashes':{'same':True},'native_target':{'same':True},'count':2,'stage_size':[32,32]}
            (w/'gguf_plan.json').write_text(json.dumps(plan))
            receipt={'real_generation':False,'model':'Wan2.2-Animate-14B-Q3_K_M','plan_sha256':file_hash(w/'gguf_plan.json'),'video_sha256':file_hash(w/'wan_generated.mp4')}
            r=w/'receipt.json';r.write_text(json.dumps(receipt))
            with patch('frame_studio.experimental.gguf_t4.target.require_review'),patch('frame_studio.experimental.gguf_t4.target.signature',return_value={'same':True}),patch('frame_studio.experimental.gguf_t4.stage_hashes',return_value={'same':True}),patch('frame_studio.experimental.gguf_t4.video_info',return_value=(64,32)):
                with self.assertRaisesRegex(ValueError,'Recibo'):accept_result(w,r)
                receipt['real_generation']=True # synthetic validator fixture only, never published as generation
                r.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError,'Dimensoes'):accept_result(w,r)
                self.assertFalse((w/'generation.json').exists())
