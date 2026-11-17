from pathlib import Path
import tempfile,time,unittest,zipfile,json,shutil
from fastapi.testclient import TestClient
from frame_studio.app import create_app
from frame_studio.config import Settings
from frame_studio.tests.helpers import make_video


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.settings=Settings(data_dir=self.root/'data',reserve_bytes=1024**2,max_seconds=10,max_frames=500,password='segredo-de-teste')
        self.app=create_app(self.settings);self.client=TestClient(self.app);self.client.__enter__()
        self.csrf=self.client.get('/api/session').json()['csrf'];r=self.client.post('/api/login',json={'password':'segredo-de-teste'},headers={'X-CSRF-Token':self.csrf});self.assertEqual(r.status_code,200)
    def tearDown(self):self.client.__exit__(None,None,None);self.temp.cleanup()
    def post(self,path,json_body=None,content=None):
        return self.client.post(path,json=json_body,content=content,headers={'X-CSRF-Token':self.csrf,'Content-Type':'application/octet-stream' if content is not None else 'application/json'})
    def wait(self,jid):
        for _ in range(400):
            r=self.client.get('/api/jobs/'+jid).json()
            if r['status'] not in ['queued','running']:return r
            time.sleep(.025)
        self.fail('job timeout')
    def upload(self):
        path=make_video(self.root/'api-source.mp4',audio=False)
        r=self.post('/api/projects?filename=teste.mp4',content=path.read_bytes());self.assertEqual(r.status_code,200,r.text)
        data=r.json();job=self.wait(data['job']);self.assertEqual(job['status'],'succeeded',job);return data['project_id']
    def test_auth_csrf_health_and_static_ui(self):
        self.assertEqual(self.client.get('/healthz').status_code,200)
        self.assertIn('Vos Frame Studio',self.client.get('/').text)
        self.assertEqual(self.client.get('/static/app.js').status_code,200)
        self.assertEqual(self.client.post('/api/projects',content=b'bad').status_code,403)
        with TestClient(self.app) as other:
            self.assertEqual(other.get('/api/projects').status_code,401)
    def test_isolation_between_sessions_and_artifacts(self):
        pid=self.upload();r=self.post('/api/projects/'+pid+'/packages',{'selection':{'scope':'full'},'fps':'native'});j=self.wait(r.json()['job']);self.assertEqual(j['status'],'succeeded',j);aid=j['result']['artifact']
        other=TestClient(self.app);token=other.get('/api/session').json()['csrf'];other.post('/api/login',json={'password':'segredo-de-teste'},headers={'X-CSRF-Token':token})
        self.assertEqual(other.get('/api/projects/'+pid).status_code,404)
        self.assertEqual(other.get('/api/projects/'+pid+'/source').status_code,404)
        self.assertEqual(other.get('/api/projects/'+pid+'/artifacts/'+aid+'.zip').status_code,404)
        self.assertEqual(other.get('/api/jobs/'+j['id']).status_code,404)
        self.assertEqual(other.get('/api/projects').json(),[])
    def test_end_to_end_scene_import_render_and_download(self):
        pid=self.upload();base='/api/projects/'+pid
        r=self.post(base+'/scenes',{'cuts':[6,12]});self.assertEqual(r.status_code,200,r.text)
        r=self.post(base+'/packages',{'selection':{'scope':'scene','scene_id':'scene_002'},'fps':'native'});job=self.wait(r.json()['job']);self.assertEqual(job['status'],'succeeded',job)
        zip_data=self.client.get(base+'/artifacts/'+job['result']['artifact']+'.zip');self.assertEqual(zip_data.status_code,200)
        r=self.post(base+'/imports',content=zip_data.content);iid=r.json()['import_id'];job=self.wait(r.json()['job']);self.assertTrue(job['result']['valid'],job)
        report=self.client.get(base+'/imports/'+iid).json();self.assertEqual(len(report['manifest']['frames']),6)
        self.assertEqual(self.client.get(base+'/frames/source/6').status_code,200)
        self.assertEqual(self.client.get(base+'/frames/import/'+iid+'/0').status_code,200)
        self.assertEqual(self.post(base+'/imports/'+iid+'/apply',{}).status_code,200)
        self.assertEqual(self.post(base+'/scenes',{'cuts':[5]}).status_code,409)
        r=self.post(base+'/renders',{'selection':{'scope':'full'},'fps':'60','preserve_audio':False});job=self.wait(r.json()['job']);self.assertEqual(job['status'],'succeeded',job)
        validation=job['result']['validation'];self.assertEqual(validation['decoded_frames'],36);self.assertEqual(len(validation['scene_counts']),3)
        out=base+'/artifacts/'+job['result']['artifact'];self.assertEqual(self.client.get(out+'.mp4').status_code,200);self.assertEqual(self.client.get(out+'.json').json()['passed'],True)
        partial=self.client.get(out+'.mp4?download=false',headers={'Range':'bytes=0-99'});self.assertEqual(partial.status_code,206);self.assertEqual(len(partial.content),100)
    def test_revision_changed_blocks_old_package(self):
        pid=self.upload();base='/api/projects/'+pid
        r=self.post(base+'/packages',{'selection':{'scope':'full'}});j=self.wait(r.json()['job']);data=self.client.get(base+'/artifacts/'+j['result']['artifact']+'.zip').content
        self.post(base+'/scenes',{'cuts':[6]});r=self.post(base+'/imports',content=data);j=self.wait(r.json()['job']);self.assertFalse(j['result']['valid']);self.assertEqual(j['result']['issues'][0]['code'],'revision_changed')
    def test_upload_size_limit_is_enforced(self):
        self.settings.video_bytes=10
        r=self.post('/api/projects?filename=large.mp4',content=b'x'*100);self.assertEqual(r.status_code,413,r.text)
    def test_expired_project_removed(self):
        pid=self.upload();p=self.settings.data_dir/pid
        import os
        os.utime(p,(time.time()-100,time.time()-100));self.settings.ttl=1;self.app.state.store.cleanup()
        self.assertEqual(self.client.get('/api/projects/'+pid).status_code,404)
    def test_detect_scenes_returns_suggestions_without_changing_cuts(self):
        pid=self.upload();base='/api/projects/'+pid
        r=self.post(base+'/detect-scenes',{'threshold':.25});job=self.wait(r.json()['job']);self.assertEqual(job['status'],'succeeded',job)
        self.assertTrue(job['result']['suggested_cuts']);self.assertEqual(len(self.client.get(base).json()['scenes']),1)
    def test_invalid_cut_order_rejected(self):
        pid=self.upload();self.assertEqual(self.post('/api/projects/'+pid+'/scenes',{'cuts':[12,6]}).status_code,400)


if __name__=='__main__':unittest.main()
