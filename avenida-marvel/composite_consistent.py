#!/usr/bin/env python3
"""Original-background video compositor for already generated character layers.

Usage from a timeline renderer:
    c = OriginalBackgroundCompositor()
    frame, diagnostics = c.process(original, generated, source_index, shot_key)

Inputs and outputs have the same size. This module generates no character
content; the source scene outside the replacement matte is retained exactly.
"""
import cv2
import numpy as np

from composite_consistent_test import PersonMatte, soft_mask, mix
from composite_consistent_alignment import fit_background, fit_color, warp_image, apply_color


class OriginalBackgroundCompositor:
    def __init__(self, registration=True, color_match=True, temporal_weight=.20,
                 segmentation_stride=2, registration_stride=8):
        self.model=PersonMatte()
        self.registration=registration
        self.color_match=color_match
        self.temporal_weight=temporal_weight
        self.segmentation_stride=max(1,segmentation_stride)
        self.registration_stride=max(1,registration_stride)
        self.flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.reset(None)

    def reset(self,shot_key):
        self.shot_key=shot_key
        self.frame_number=0
        self.source_index=None
        self.source_mask=None
        self.previous_gray=None
        self.previous_raw_mask=None
        self.previous_alpha=None
        self.matrix=np.eye(2,3,dtype=np.float32)
        self.gain=np.ones(3,np.float32)
        self.bias=np.zeros(3,np.float32)
        self.registration_info={'accepted':False,'reason':'initial'}

    @staticmethod
    def _transport(mask,flow):
        h,w=mask.shape
        y,x=np.mgrid[:h,:w].astype(np.float32)
        return cv2.remap(mask.astype(np.float32),x+flow[...,0],y+flow[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    def process(self,source,generated,source_index,shot_key):
        if source.shape!=generated.shape:
            raise ValueError(f'Input dimensions differ: {source.shape} / {generated.shape}')
        if shot_key!=self.shot_key or (self.source_index is not None and source_index<self.source_index):
            self.reset(shot_key)
        h,w=source.shape[:2]
        if source_index!=self.source_index:
            _,self.source_mask=self.model(source)
            self.source_index=source_index
        gray=cv2.cvtColor(generated,cv2.COLOR_BGR2GRAY)
        current_to_previous=None
        if self.previous_gray is not None:
            current_to_previous=self.flow.calc(gray,self.previous_gray,None)
        if self.previous_raw_mask is None or self.frame_number%self.segmentation_stride==0:
            _,new_mask=self.model(generated)
        else:
            new_mask=(self._transport(self.previous_raw_mask,current_to_previous)>.35).astype(np.uint8)
        self.previous_raw_mask=new_mask

        # Fit only against original/generated background. Small updates are
        # smoothed; changes of shot reset all state so no cut is morphed.
        if self.registration and self.frame_number%self.registration_stride==0:
            candidate,info=fit_background(source,generated,self.source_mask,new_mask)
            if info.get('accepted'):
                self.matrix=candidate if self.frame_number==0 else .75*self.matrix+.25*candidate
            self.registration_info=info
        aligned=warp_image(generated,self.matrix,(w,h)) if self.registration else generated
        aligned_mask=warp_image(new_mask,self.matrix,(w,h),True) if self.registration else new_mask
        # Exposure/color remains constant within this shot to avoid pumping.
        if self.color_match and self.frame_number==0:
            self.gain,self.bias=fit_color(source,aligned,self.source_mask,aligned_mask)
        if self.color_match:
            aligned=apply_color(aligned,self.gain,self.bias)

        union=np.maximum(self.source_mask,aligned_mask)
        alpha=soft_mask(union,3,3)
        # Alpha transport uses aligned layer flow, not the unaligned generated
        # layer, when registration is active.
        aligned_gray=cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY)
        if self.previous_alpha is not None and self.temporal_weight:
            backward=self.flow.calc(aligned_gray,self.previous_aligned_gray,None)
            prior=self._transport(self.previous_alpha,backward)
            stable=(1-self.temporal_weight)*alpha+self.temporal_weight*prior
            alpha=np.maximum(.85*alpha,stable)
            alpha[alpha<.015]=0
        result=mix(source,aligned,alpha)
        outside=alpha==0
        before_error=float(np.abs(generated.astype(np.float32)-source.astype(np.float32))[outside].mean()) if outside.any() else None
        after_error=float(np.abs(result.astype(np.float32)-source.astype(np.float32))[outside].mean()) if outside.any() else None
        self.previous_alpha=alpha
        self.previous_gray=gray
        self.previous_aligned_gray=aligned_gray
        self.frame_number+=1
        diagnostics={
            'coverage':float((alpha>0).mean()),
            'before_background_mae':before_error,
            'after_background_mae':after_error,
            'registration':self.registration_info,
            'matrix':self.matrix.tolist(),
            'color_gain':self.gain.tolist(),
            'color_bias':self.bias.tolist(),
        }
        return result,diagnostics
