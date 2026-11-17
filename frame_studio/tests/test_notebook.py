"""Notebook contract and failure gates; no Colab, model downloads or GPU."""
import json
from pathlib import Path
from unittest import TestCase, mock

NOTEBOOK = Path(__file__).resolve().parents[2]/'colab/VOS_Wan22_Animate_Replacement.ipynb'


class NotebookTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(NOTEBOOK.read_text())
        cls.sources = [''.join(c['source']) for c in cls.data['cells'] if c['cell_type']=='code']

    def test_python_cells_and_required_steps(self):
        self.assertEqual(self.data['nbformat'],4)
        for n,source in enumerate(self.sources):
            compile(source,f'{NOTEBOOK}:cell{n}','exec')
        combined='\n'.join(self.sources)
        for step in ('prepare','preprocess','review-mask','infer','finish','verify-weights'):
            self.assertIn(repr(step),combined)
        for requirement in ('requirements_animate.txt','moviepy==1.0.3','HF_HUB_OFFLINE','WAN_REV=','allow_patterns='):
            self.assertIn(requirement,combined)
        self.assertTrue(all(not c.get('outputs') for c in self.data['cells']))

    def test_failed_resource_gate_prevents_installation_and_download(self):
        source=next(s for s in self.sources if 'WAN_REV=' in s)
        runner=mock.Mock()
        def unavailable():raise RuntimeError('test: insufficient resources')
        with self.assertRaisesRegex(RuntimeError,'insufficient'):
            exec(source,{'check_resources':unavailable,'subprocess':runner})
        runner.run.assert_not_called()

    def test_weight_manifest_has_local_tokenizers_and_external_onnx_data(self):
        data=json.loads((NOTEBOOK.parent/'wan22_modelscope_manifest.json').read_text())
        entries={e['path']:e for e in data['files']}
        self.assertEqual(len(entries),len(data['files']))
        self.assertIn('process_checkpoint/pose2d/vitpose_h_wholebody.onnx/end2end.onnx',entries)
        self.assertIn('xlm-roberta-large/tokenizer.json',entries)
        self.assertGreater(len(entries),400)
        for e in entries.values():
            self.assertGreater(e['size'],0)
            self.assertRegex(e['sha256'],r'^[0-9a-f]{64}$')


    def test_fast_gpu_check_is_first_action_before_upload(self):
        sources = self.sources
        diagnosis = next(i for i, s in enumerate(sources) if 'def check_resources():' in s)
        upload = next(i for i, s in enumerate(sources) if 'files.upload()' in s)
        install = next(i for i, s in enumerate(sources) if 'WAN_REV=' in s)
        self.assertLess(diagnosis, upload)
        self.assertLess(upload, install)
        self.assertIn('VOS_GPU_DIAGNOSTICO.json', sources[diagnosis])
        self.assertIn('if not gpu_present:', sources[diagnosis])
        self.assertIn('passou_bloqueio_preventivo', sources[diagnosis])

    def test_resource_gate_rejects_t4_and_cpu(self):
        from types import SimpleNamespace
        cuda=SimpleNamespace(is_available=lambda:False)
        source=next(s for s in self.sources if 'def check_resources' in s)
        with mock.patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=cuda),'psutil':SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError,'CUDA indisponivel'):exec(source,{})
        cuda.is_available=lambda:True
        cuda.get_device_properties=lambda _:SimpleNamespace(total_memory=16*2**30,name='T4 fixture')
        cuda.is_bf16_supported=lambda:False
        with mock.patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=cuda),'psutil':SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=32*2**30))}),mock.patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*2**30)):
            with self.assertRaisesRegex(RuntimeError,'Recursos insuficientes'):exec(source,{})
            cuda.get_device_properties=lambda _:SimpleNamespace(total_memory=32*2**30,name='V100 fixture')
            with self.assertRaisesRegex(RuntimeError,'BF16'):exec(source,{})
