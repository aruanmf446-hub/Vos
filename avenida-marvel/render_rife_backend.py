"""Official Practical-RIFE 4.25 Lite for temporal video interpolation on CPU."""
from pathlib import Path
import importlib.util
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent


class Rife:
    def __init__(self, threads=2):
        torch.set_num_threads(threads)
        directory=ROOT/'models/rife425'
        if 'model' not in sys.modules:
            sys.modules['model']=types.ModuleType('model')
        spec=importlib.util.spec_from_file_location('model.warplayer',directory/'warplayer.py')
        warp=importlib.util.module_from_spec(spec)
        sys.modules['model.warplayer']=warp
        spec.loader.exec_module(warp)
        spec=importlib.util.spec_from_file_location('rife425_ifnet',directory/'train_log/IFNet_HDv3.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.net=module.IFNet().eval()
        state=torch.load(directory/'train_log/flownet.pkl',map_location='cpu',weights_only=True)
        state={key.removeprefix('module.'):value for key,value in state.items()}
        expected=self.net.state_dict()
        self.net.load_state_dict({key:value for key,value in state.items() if key in expected})
        self.inputs={}

    def prepare(self, frame):
        h,w=frame.shape[:2]
        x=torch.from_numpy(np.ascontiguousarray(frame[...,::-1])).permute(2,0,1).float()[None]/255
        return F.pad(x,(0,(-w)%128,0,(-h)%128),mode='replicate')

    def between(self, a, b, timestep, pair=None):
        h,w=a.shape[:2]
        if pair is not None and pair in self.inputs:
            images=self.inputs[pair]
        else:
            images=torch.cat([self.prepare(a),self.prepare(b)],1)
            if pair is not None:
                self.inputs={pair:images}
        with torch.inference_mode():
            out=self.net(images,timestep,[32,16,8,4,1])[-1][-1]
        rgb=out[0,:,:h,:w].permute(1,2,0).numpy()
        return np.clip(rgb[...,::-1]*255+.5,0,255).astype(np.uint8)
