"""Preflight, sanitized diagnostics, exact-export recovery and synthetic smoke."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from frame_studio.preflight import diagnose
from frame_studio.colab_support import Session,Pipeline
from frame_studio.recovery import export_progress,restore_progress
from frame_studio.tests import test_target_masks as fixtures
from frame_studio import target_masks as target


def torch_fixture(total=40,free=39,cap=(8,0),cuda_version='12.4',context=True):
    def memory(*args):
        if not context:raise RuntimeError('driver fixture incompatible')
        return free*2**30,total*2**30
    return SimpleNamespace(__version__='fixture',version=SimpleNamespace(cuda=cuda_version),cuda=SimpleNamespace(
        is_available=lambda:True,get_device_properties=lambda _:SimpleNamespace(name='GPU fixture',total_memory=total*2**30),
        get_device_capability=lambda _:cap,mem_get_info=memory,is_bf16_supported=lambda **kwargs:True))


class FinalPreparationTests(unittest.TestCase):
    def test_full_preflight_and_failures_never_claim_inference(self):
        ram=SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=32*2**30))
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*2**30)):
            report=diagnose(torch_fixture(),ram,root='/tmp');self.assertTrue(report['passou_bloqueio_preventivo'])
            for fixture in (torch_fixture(total=16,free=15,cap=(7,5)),torch_fixture(free=8),torch_fixture(cuda_version=None),torch_fixture(context=False)):
                bad=diagnose(fixture,ram,root='/tmp');self.assertFalse(bad['passou_bloqueio_preventivo']);self.assertTrue(bad['failures'])
            # Emulated BF16 on pre-Ampere is not native BF16.
            self.assertFalse(diagnose(torch_fixture(cap=(7,5)),ram,root='/tmp')['bf16_nativo'])
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=20*2**30)):
            self.assertFalse(diagnose(torch_fixture(),ram,root='/tmp')['passou_bloqueio_preventivo'])

    def test_sam_coordinates_use_absolute_pixels_and_clamp_boundary(self):
        from frame_studio.wan_target import sam_points
        points,labels=sam_points([[0,0,1],[1,1,0],[.5,.5,1]],(160,90))
        self.assertEqual(points.tolist(),[[0,0],[159,89],[80,45]])
        self.assertEqual(labels.tolist(),[1,0,1])

    def test_stage_failure_prevents_later_actions_but_allows_corrective_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline=Pipeline(Session(temporary))
            with self.assertRaisesRegex(RuntimeError,'Diagnostico'):pipeline.begin('install',False)
            pipeline.begin('install');pipeline.failure()
            with self.assertRaisesRegex(RuntimeError,'anterior falhou'):pipeline.begin('prepare')
            pipeline.begin('install');pipeline.complete('install');pipeline.begin('prepare');pipeline.complete('prepare')
            pipeline.begin('selection');pipeline.failure();pipeline.begin('prepare')
            self.assertFalse(pipeline.ready['selection'])

    def test_diagnostics_small_redacted_and_no_private_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);session=Session(folder)
            (folder/'01.log').write_text('x'*200000+'\nBearer token-fixture-xyz\npassword=fixture-pass\nhf_0123456789abcdef\nhttps://user:password@example.com/file?token=fixture-query\nENV_VALUE_FIXTURE123')
            (folder/'private.mp4').write_bytes(b'private video');(folder/'private.txt').write_text('private arbitrary file')
            (folder/'weights.pth').write_bytes(b'not weights')
            with patch.dict('os.environ',{'FAKE_API_TOKEN':'ENV_VALUE_FIXTURE123'}):output=session.bundle()
            with zipfile.ZipFile(output) as z:
                text=z.read('01.log').decode()
                for secret in ('token-fixture-xyz','fixture-pass','hf_0123456789abcdef','fixture-query','ENV_VALUE_FIXTURE123','user:password'):self.assertNotIn(secret,text)
                self.assertNotIn('private.mp4',z.namelist());self.assertNotIn('private.txt',z.namelist());self.assertLess(sum(i.file_size for i in z.infolist()),2*1024**2)


class RecoveryTests(unittest.TestCase):
    mask=fixtures.TargetTests.mask
    def setUp(self):
        fixtures.TargetTests.setUp(self)
        (self.work/'bridge.json').write_text(json.dumps({'original_zip_sha256':'exact-original-zip-sha'}))

    def test_restore_reviewed_and_unreviewed_selection_without_inference_receipt(self):
        backup=export_progress(self.work,self.work/'progress.zip')
        self.assertEqual(restore_progress(self.work,backup),'selection_unreviewed')
        target.approve(self.work,[0,1,2]);export_progress(self.work,backup)
        target.invalidate(self.work)
        self.assertEqual(restore_progress(self.work,backup),'masks_reviewed');target.require_review(self.work)
        self.assertFalse((self.work/'generation.json').exists());self.assertFalse((self.work/'mask_review.json').exists())

    def test_wrong_original_and_path_traversal_rejected(self):
        backup=export_progress(self.work,self.work/'progress.zip')
        (self.work/'bridge.json').write_text('{"original_zip_sha256":"different export"}')
        with self.assertRaises(ValueError):restore_progress(self.work,backup)
        with zipfile.ZipFile(backup,'a') as z:z.writestr('../escape.txt','fixture')
        with self.assertRaises(ValueError):restore_progress(self.work,backup)
        self.assertFalse((self.work.parent/'escape.txt').exists())


    def test_selection_free_backup_cannot_silently_keep_existing_actor(self):
        cfg=self.work/'target.json';saved=cfg.read_bytes();cfg.unlink()
        archive=export_progress(self.work,self.work/'no-selection.zip');cfg.write_bytes(saved)
        with self.assertRaisesRegex(ValueError,'alvo existente'):restore_progress(self.work,archive)
        self.assertEqual(cfg.read_bytes(),saved)
