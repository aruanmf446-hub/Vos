"""Notebook order, zero-download diagnosis, explicit gates and immutable toolkit."""
import ast
import json
from pathlib import Path
import re
import unittest

NOTEBOOK=Path(__file__).resolve().parents[2]/'colab/VOS_Wan22_Animate_Replacement.ipynb'


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(NOTEBOOK.read_text());cls.sources=[''.join(c['source']) for c in cls.data['cells'] if c['cell_type']=='code']

    def test_python_cells_required_steps_and_outputs(self):
        self.assertEqual(self.data['nbformat'],4)
        for n,source in enumerate(self.sources):compile(source,f'{NOTEBOOK}:{n}','exec')
        combined='\n'.join(self.sources)
        for step in ('prepare','preprocess','review-mask','infer','finish','verify-weights','track'):self.assertIn(repr(step),combined)
        for value in ('requirements_animate.txt','moviepy==1.0.3','HF_HUB_OFFLINE','WAN_REV=','allow_patterns=','wan_audit.py','smoke_test.py','src_face.mp4'):self.assertIn(value,combined)
        self.assertTrue(all(not c.get('outputs') for c in self.data['cells']))

    def test_diagnosis_before_any_download_and_install_before_private_upload(self):
        diagnosis=next(i for i,s in enumerate(self.sources) if 'def check_resources' in s)
        bootstrap=next(i for i,s in enumerate(self.sources) if 'VOS_REF=' in s)
        install=next(i for i,s in enumerate(self.sources) if 'WAN_REV=' in s)
        upload=next(i for i,s in enumerate(self.sources) if 'files.upload()' in s)
        self.assertLess(diagnosis,bootstrap);self.assertLess(bootstrap,install);self.assertLess(install,upload)
        self.assertIn('VOS_GPU_DIAGNOSTICO.json',self.sources[diagnosis]);self.assertIn('VOS_PREFLIGHT_LOGS.zip',self.sources[diagnosis])
        self.assertIn('PREFLIGHT_OK=False',self.sources[diagnosis])

    def test_all_dependent_cells_stop_before_side_effects_when_blocked(self):
        diagnosis=next(i for i,s in enumerate(self.sources) if 'def check_resources' in s)
        for source in self.sources[diagnosis+1:]:
            first=ast.parse(source).body[0]
            self.assertIsInstance(first,ast.If)
            with self.assertRaisesRegex(RuntimeError,'Diagnostico bloqueado'):
                exec(compile(ast.Module(body=[first],type_ignores=[]),'guard','exec'),{'PREFLIGHT_OK':False})

    def test_early_diagnose_is_identical_to_audited_module(self):
        source=next(s for s in self.sources if 'def check_resources' in s)
        module=(NOTEBOOK.parents[1]/'frame_studio/preflight.py').read_text()
        self.assertIn(module.strip(),source)
        self.assertNotIn('urlretrieve',source);self.assertNotIn('files.upload',source)

    def test_toolkit_commit_and_experimental_defaults(self):
        combined='\n'.join(self.sources)
        self.assertRegex(combined,r"VOS_REF='[0-9a-f]{40}'")
        self.assertIn('USE_TARGET_SELECTION=False',combined);self.assertIn('LOW_MEMORY_EXPERIMENTAL=False',combined)
        self.assertIn('RESTORE_PROGRESS=False',combined)
        self.assertIn('restore_progress(WORK,archive)',combined);self.assertIn("backup('antes_da_inferencia')",combined)
        self.assertNotIn("'pip','freeze'",combined)

    def test_weights_include_local_tokenizers_and_external_onnx_parts(self):
        data=json.loads((NOTEBOOK.parent/'wan22_modelscope_manifest.json').read_text());entries={e['path']:e for e in data['files']}
        self.assertEqual(len(entries),len(data['files']));self.assertGreater(len(entries),400)
        for name in ('process_checkpoint/pose2d/vitpose_h_wholebody.onnx/end2end.onnx','xlm-roberta-large/tokenizer.json'):self.assertIn(name,entries)
        for entry in entries.values():self.assertGreater(entry['size'],0);self.assertRegex(entry['sha256'],r'^[0-9a-f]{64}$')
