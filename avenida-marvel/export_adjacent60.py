#!/usr/bin/env python3
"""Interpolate adjacent finished frames at 60 FPS, preserving exact hard cuts."""
from pathlib import Path
import argparse, subprocess, time
import cv2
import numpy as np

CUTS={39,76,173,241,272,349,404,666,749}


def remap(a,m):
    return cv2.remap(a,m[...,0],m[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    cv2.setNumThreads(2)
    h,w=854,576;fw,fh=384,243
    yy,xx=np.mgrid[:fh,:fw].astype(np.float32);grid=np.dstack([xx,yy])
    fy,fx=np.mgrid[:364,:w].astype(np.float32);fullgrid=np.dstack([fx,fy])
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setUseMeanNormalization(True)
    def inverse(f):
        m=grid-.5*f
        for _ in range(5):m=.35*m+.65*(grid-.5*remap(f,m))
        return m
    def midpoint(a,b):
        aa=a[230:594];bb=b[230:594]
        ga=cv2.resize(cv2.cvtColor(aa,cv2.COLOR_BGR2GRAY),(fw,fh))
        gb=cv2.resize(cv2.cvtColor(bb,cv2.COLOR_BGR2GRAY),(fw,fh))
        ab=dis.calc(ga,gb,None);ba=dis.calc(gb,ga,None)
        ma=inverse(ab);mb=inverse(ba)
        ca=np.exp(-np.linalg.norm(ab+remap(ba,grid+ab),axis=2)/1.8)
        cb=np.exp(-np.linalg.norm(ba+remap(ab,grid+ba),axis=2)/1.8)
        ca=.02+.98*remap(ca,ma);cb=.02+.98*remap(cb,mb)
        alpha=cb/(ca+cb)
        alpha=cv2.resize(cv2.GaussianBlur(alpha,(3,3),0),(w,364))[...,None]
        fa=cv2.resize(ma-grid,(w,364));fb=cv2.resize(mb-grid,(w,364))
        fa[...,0]*=w/fw;fa[...,1]*=364/fh
        fb[...,0]*=w/fw;fb[...,1]*=364/fh
        wa=remap(aa,fullgrid+fa);wb=remap(bb,fullgrid+fb)
        out=cv2.addWeighted(a,.5,b,.5,0)
        out[230:594]=np.clip(wa*(1-alpha)+wb*alpha+.5,0,255).astype(np.uint8)
        return out
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(args.input),'-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE)
    temporary=args.output.with_name(args.output.stem+'.adjacent-encoding.mp4')
    encoder=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','bgr24','-video_size','576x854','-framerate','60','-i','pipe:0','-vf','scale=720:1068:flags=lanczos,setsar=1','-an','-c:v','libx264','-preset','medium','-crf','18','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
    size=h*w*3
    def read():
        buf=bytearray()
        while len(buf)<size:
            chunk=decoder.stdout.read(size-len(buf))
            if not chunk:break
            buf.extend(chunk)
        if not buf:return None
        if len(buf)!=size:raise RuntimeError('Truncated decoded frame')
        return np.frombuffer(buf,dtype=np.uint8).reshape(h,w,3).copy()
    started=time.monotonic();a=read();count=0
    try:
        while a is not None:
            b=read()
            encoder.stdin.write(a.tobytes())
            if b is None or count+1 in CUTS:intermediate=a
            else:intermediate=midpoint(a,b)
            encoder.stdin.write(intermediate.tobytes())
            count+=1
            if count%60==0:print(f'60FPS export: {count}/778 input frames; {time.monotonic()-started:.1f}s',flush=True)
            a=b
        encoder.stdin.close();decoder.stdout.close()
        if decoder.wait() or encoder.wait():raise RuntimeError('Video codec failed')
    except BaseException:
        for proc in [decoder,encoder]:
            if proc.poll() is None:proc.kill();proc.wait()
        raise
    if count!=778:raise RuntimeError(f'Expected778sourceframes,got{count}')
    temporary.replace(args.output)
    print(f'Completed1556framesat60FPS: {args.output}',flush=True)


if __name__=='__main__':main()
