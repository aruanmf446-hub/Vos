"""Mock SAM2 contract only. Uses real native PNG I/O, no weights/GPU/model call."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
from frame_studio import target_masks as target
from frame_studio.wan_target import track
from frame_studio.wan_bridge import infer,STAGE_FILES,stage_hashes


class FakeTensor:
    def __init__(self,array):self.array=array
    def __getitem__(self,index):return FakeTensor(self.array[index])
    def __gt__(self,value):return FakeTensor(self.array>value)
    def cpu(self):return self
    def numpy(self):return self.array


class SamContractTests(unittest.TestCase):
    def test_prompted_sam2_offloads_state_and_consumes_mask_keyframes(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary);wan=work/'wan';pre=wan/'wan/modules/animate/preprocess';pre.mkdir(parents=True);(pre/'sam_utils.py').touch()
            frames=[{'file':f'{n}.png','timestamp':str(n)+'/60'} for n in range(3)]
            (work/'manifest.json').write_text(json.dumps({'source':{'width':64,'height':40},'frames':frames}))
            mask=Image.new('L',(64,40));ImageDraw.Draw(mask).rectangle((5,10,15,30),fill=255)
            for f in frames:Image.new('RGB',(64,40),'red').save(work/f['file'])
            target.save_selection(work,{'keyframes':{'0':{'points':[[.15,.5,1],[.8,.5,0]]},'2':{'points':[[.15,.5,1]]}}})
            (work/'target_seeds').mkdir();mask.save(work/'target_seeds/frame_000001.png')
            class Predictor:
                def __init__(self):self.prompts=[];self.seeds=[];self.state_options={}
                def init_state(self,**kwargs):self.state_options=kwargs;return {'mock':True}
                def add_new_points(self,**kwargs):self.prompts.append(kwargs)
                def add_new_mask(self,**kwargs):self.seeds.append(kwargs)
                def propagate_in_video(self,state):
                    for n in range(3):yield n,[1],FakeTensor((np.array(mask,dtype=np.float32)[None,None]-128))
            predictor=Predictor();fake_torch=SimpleNamespace(inference_mode=nullcontext,autocast=lambda *a,**k:nullcontext(),bfloat16='mock dtype')
            fake_sam=SimpleNamespace(build_sam2_video_predictor=lambda *a,**k:predictor)
            with patch.dict('sys.modules',{'torch':fake_torch,'sam_utils':fake_sam}):track(work,wan,work/'checkpoints')
            self.assertTrue(predictor.state_options['offload_video_to_cpu']);self.assertTrue(predictor.state_options['offload_state_to_cpu'])
            self.assertEqual([p['frame_idx'] for p in predictor.prompts],[0,2])
            self.assertEqual(predictor.prompts[0]['labels'].tolist(),[1,0])
            self.assertEqual(predictor.seeds[0]['frame_idx'],1)
            self.assertFalse((work/'target_review.json').exists())
            target.approve(work,[0,1,2])
            # Partial propagation must fail closed and remove a previous approval.
            predictor.propagate_in_video=lambda state:iter([(0,[1],FakeTensor(np.array(mask,dtype=np.float32)[None,None]-128))])
            with patch.dict('sys.modules',{'torch':fake_torch,'sam_utils':fake_sam}):
                with self.assertRaisesRegex(ValueError,'incompleto'):track(work,wan,work/'checkpoints')
            self.assertFalse((work/'target_review.json').exists())

    def test_target_bbox_is_absolute_xyxy_even_when_character_is_off_center(self):
        """Regressao: o ViTPose interpreta os ultimos dois valores como x2,y2."""
        from frame_studio.wan_target import bbox_xyxy_from_mask
        mask=np.zeros((90,160),dtype=bool)
        mask[31:78,92:141]=True
        box=bbox_xyxy_from_mask(mask).tolist()
        self.assertEqual(box,[92.0,31.0,141.0,78.0])
        self.assertGreater(box[2],box[0])
        self.assertGreater(box[3],box[1])
        with self.assertRaisesRegex(ValueError,'Mascara vazia'):
            bbox_xyxy_from_mask(np.zeros((90,160),dtype=bool))
        # Regression guard for the handoff to upstream ViTPose.
        code=(Path(__file__).resolve().parents[1]/'wan_target.py').read_text()
        self.assertIn("prepare_pose_tensor(pose.model,isolated,bbox)",code)
        self.assertNotIn("xx.max()-xx.min()+1",code)

    def test_memory_profile_only_changes_official_flags_when_opted_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary);wan=work/'wan';wan.mkdir();(wan/'generate.py').touch();(work/'wan_processed').mkdir()
            (work/'manifest.json').write_text('{}');(work/'bridge.json').write_text('{"frame_count":2}')
            for name in STAGE_FILES:(work/'wan_processed'/name).write_bytes(b'fixture')
            (work/'mask_review.json').write_text(json.dumps(stage_hashes(work)))
            commands=[]
            def fake_model(args,cwd=None):commands.append(list(map(str,args)));(work/'wan_generated.mp4').write_bytes(b'not a real video: mock fixture')
            with patch('frame_studio.wan_bridge.check_checkpoints'),patch('frame_studio.wan_bridge.video_info',return_value=(64,40)),patch('frame_studio.wan_bridge.cmd',side_effect=fake_model):
                infer(work,wan,work/'weights');infer(work,wan,work/'weights',low_memory=True)
            self.assertNotIn('--t5_cpu',commands[0]);self.assertNotIn('--frame_num',commands[0])
            self.assertIn('--t5_cpu',commands[1]);self.assertEqual(commands[1][commands[1].index('--frame_num')+1],'33')
            for command in commands:self.assertIn('animate-14B',command);self.assertIn('--offload_model',command)
