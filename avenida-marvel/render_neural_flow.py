"""Optional RAFT-small optical flow backend for local video interpolation."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights, raft_large, Raft_Large_Weights


class RaftFlow:
    def __init__(self, threads=2, updates=16, large=False):
        torch.set_num_threads(threads)
        self.model = (raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2,progress=False) if large
                      else raft_small(weights=Raft_Small_Weights.DEFAULT,progress=False)).eval()
        self.updates = updates

    def both(self, a, b, size):
        """Return a->b and b->a fields on the requested flow grid."""
        def tensor(im):
            im = cv2.resize(im,size,interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(im[...,::-1])
            value = torch.from_numpy(rgb).permute(2,0,1).float()/127.5-1
            return value
        aa,bb = tensor(a),tensor(b)
        h,w = aa.shape[-2:]
        pad = (0,(-w)%8,0,(-h)%8)
        one = F.pad(torch.stack([aa,bb]),pad,mode='replicate')
        two = F.pad(torch.stack([bb,aa]),pad,mode='replicate')
        with torch.inference_mode():
            flows = self.model(one,two,num_flow_updates=self.updates)[-1]
        fields=flows[:,:,:h,:w].permute(0,2,3,1).numpy().copy()
        return fields[0],fields[1]
