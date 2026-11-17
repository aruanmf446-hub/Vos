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


if __name__=="__main__":unittest.main()
