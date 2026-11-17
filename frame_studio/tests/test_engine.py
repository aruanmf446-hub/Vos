from pathlib import Path
from fractions import Fraction
import json,tempfile,unittest,zipfile,subprocess,shutil,io,warnings,os,time
from PIL import Image
from frame_studio.errors import StudioError
from frame_studio.storage import read_json,sha256
from frame_studio.timeline import rational,scenes_for,native_timeline,validate_coverage
from frame_studio.packages import apply_import,ensure_originals,export_package,validate_import
from frame_studio.render import build_video,render_plan
from frame_studio.probe import probe_project
from frame_studio.tests.helpers import create_project,job_for,package,rewrite_zip,import_zip,png,make_video


def decode_rgb(path):
    data=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-an','-fps_mode','passthrough','-threads','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
    size=160*96*3
    return [data[i:i+size] for i in range(0,len(data),size)]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.store,self.project=create_project(self.root)
    def tearDown(self):self.temp.cleanup()
    def job(self):return job_for(self.store,self.project)
    def manifest(self,path):
        with zipfile.ZipFile(path) as z:return json.loads(z.read('manifest.json'))
    def output(self,result):return self.store.path(self.project['id'])/'outputs'/(result['artifact']+'.mp4')
    def test_native_roundtrip_every_frame_audio_cuts_duration_and_resolution(self):
        archive=package(self.store,self.project);manifest=self.manifest(archive)
        self.assertEqual(manifest['expected_frame_count'],18)
        self.assertEqual([f['global_index'] for f in manifest['frames']],list(range(18)))
        self.assertEqual([s['expected_frames'] for s in manifest['scenes']],[6,6,6])
        iid,report=import_zip(self.store,self.project,archive);self.assertTrue(report['valid']);self.assertEqual(report['changed_files'],0)
        apply_import(self.store,self.project,iid)
        result=build_video(self.store,self.project,{'scope':'full'},'native',False,True,16,self.job());v=result['validation']
        self.assertEqual(v['decoded_frames'],18);self.assertEqual(v['duration'],'3/5');self.assertEqual(v['cuts'],['1/5','2/5']);self.assertEqual(v['resolution'],[160,96]);self.assertTrue(v['audio']);self.assertTrue(v['complete_decode'])
        before=decode_rgb(self.store.path(self.project['id'])/'source.media');after=decode_rgb(self.output(result))
        self.assertEqual(len(before),len(after))
        for a,b in zip(before,after):self.assertLess(sum(abs(x-y) for x,y in zip(a,b))/len(a),6)
    def test_only_one_edited_frame_changes(self):
        archive=package(self.store,self.project);m=self.manifest(archive);name=m['frames'][8]['file']
        edited=rewrite_zip(archive,self.root/'edited.zip',modify={name:png((220,20,210))})
        iid,report=import_zip(self.store,self.project,edited);self.assertTrue(report['valid']);self.assertEqual(report['changed_orders'],[8])
        apply_import(self.store,self.project,iid);result=build_video(self.store,self.project,{'scope':'full'},'native',False,False,16,self.job())
        before=decode_rgb(self.store.path(self.project['id'])/'source.media');after=decode_rgb(self.output(result))
        differences=[i for i,(a,b) in enumerate(zip(before,after)) if sum(abs(x-y) for x,y in zip(a,b))/len(a)>10]
        self.assertEqual(differences,[8])
    def test_scene_export_replacement_and_full_reassembly(self):
        archive=package(self.store,self.project,{'scope':'scene','scene_id':'scene_002'});m=self.manifest(archive)
        self.assertEqual([f['global_index'] for f in m['frames']],list(range(6,12)))
        iid,report=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'scene.zip',modify={m['frames'][2]['file']:png((210,20,200))}))
        original=self.project['timeline'].copy();apply_import(self.store,self.project,iid)
        self.assertEqual(self.project['timeline'][:6],original[:6]);self.assertEqual(self.project['timeline'][12:],original[12:])
        scene=build_video(self.store,self.project,{'scope':'scene','scene_id':'scene_002'},'native',False,False,16,self.job())
        self.assertEqual(scene['validation']['decoded_frames'],6);self.assertEqual(scene['validation']['duration'],'1/5')
        full=build_video(self.store,self.project,{'scope':'full'},'native',False,False,16,self.job());self.assertEqual(full['validation']['scene_counts'],{'scene_001':6,'scene_002':6,'scene_003':6})
    def test_replace_subset_keeps_previous_edits_outside_interval(self):
        full=package(self.store,self.project);m=self.manifest(full)
        iid,_=import_zip(self.store,self.project,rewrite_zip(full,self.root/'first.zip',modify={m['frames'][1]['file']:png((10,210,210))}));apply_import(self.store,self.project,iid)
        before=self.project['timeline'][1].copy();scene=package(self.store,self.project,{'scope':'scene','scene_id':'scene_002'})
        iid2,_=import_zip(self.store,self.project,scene);apply_import(self.store,self.project,iid2)
        self.assertEqual(self.project['timeline'][1],before);self.assertEqual(len(self.project['timeline']),18)
    def test_missing_frame_is_identified_and_never_applied(self):
        archive=package(self.store,self.project);name=self.manifest(archive)['frames'][7]['file']
        iid,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'bad.zip',missing=name))
        self.assertFalse(r['valid']);self.assertIn(name,[i.get('file') for i in r['issues']])
        with self.assertRaises(StudioError):apply_import(self.store,self.project,iid)
        self.assertTrue(all(f['asset']['kind']=='original' for f in self.project['timeline']))
    def test_wrong_resolution(self):
        archive=package(self.store,self.project);name=self.manifest(archive)['frames'][2]['file']
        _,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'bad.zip',modify={name:png('red',(80,96))}))
        self.assertFalse(r['valid']);self.assertIn('wrong_resolution',[i['code'] for i in r['issues']])
    def test_transparent_frame_rejected(self):
        archive=package(self.store,self.project);name=self.manifest(archive)['frames'][2]['file']
        _,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'bad.zip',modify={name:png((255,0,0,100),mode='RGBA')}))
        self.assertFalse(r['valid']);self.assertIn('transparent_image',[i['code'] for i in r['issues']])
    def test_duplicate_member(self):
        archive=package(self.store,self.project);name=self.manifest(archive)['frames'][2]['file']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');bad=rewrite_zip(archive,self.root/'bad.zip',duplicate=name)
        _,r=import_zip(self.store,self.project,bad);self.assertFalse(r['valid']);self.assertIn('duplicate_file',[i['code'] for i in r['issues']])
    def test_extra_frame_rejected(self):
        archive=package(self.store,self.project)
        _,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'bad.zip',extra=[('scenes/scene_001/frame_999999.png',png('red'))]))
        self.assertFalse(r['valid']);self.assertIn('extra_file',[i['code'] for i in r['issues']])
    def test_unordered_zip_members_and_outer_folder_are_valid(self):
        archive=package(self.store,self.project);_,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'reversed.zip',reverse=True,wrapper='vos-projeto/'))
        self.assertTrue(r['valid']);self.assertEqual(r['normalized_outer_folder'],'vos-projeto/')
    def test_renamed_files_are_not_silently_sorted(self):
        archive=package(self.store,self.project);name=self.manifest(archive)['frames'][0]['file']
        _,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'bad.zip',missing=name,extra=[('scenes/scene_001/new_name.png',png('red'))]))
        self.assertFalse(r['valid']);self.assertEqual({i['code'] for i in r['issues']},{'missing_frame','extra_file'})
    def test_different_project_manifest_rejected(self):
        archive=package(self.store,self.project);_,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'bad.zip',manifest_change=lambda m:m.update(project_id='a'*32)))
        self.assertFalse(r['valid']);self.assertEqual(r['issues'][0]['code'],'wrong_project')
    def test_changed_timestamps_rejected(self):
        archive=package(self.store,self.project)
        def change(m):m['frames'][2]['timestamp']='0'
        _,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'bad.zip',manifest_change=change))
        self.assertFalse(r['valid']);self.assertEqual(r['issues'][0]['code'],'manifest_changed')
    def test_path_traversal_and_symlink_rejected(self):
        archive=package(self.store,self.project)
        _,r=import_zip(self.store,self.project,rewrite_zip(archive,self.root/'traversal.zip',extra=[('../../outside.png',png('red'))]));self.assertFalse(r['valid'])
        destination=self.root/'symlink.zip'
        with zipfile.ZipFile(archive) as original,zipfile.ZipFile(destination,'w') as out:
            for i in original.infolist():out.writestr(i.filename,original.read(i))
            i=zipfile.ZipInfo('link');i.create_system=3;i.external_attr=(0o120777<<16);out.writestr(i,'/etc/passwd')
        _,r=import_zip(self.store,self.project,destination);self.assertFalse(r['valid']);self.assertEqual(r['issues'][0]['code'],'unsafe_zip')
    def test_invalid_zip(self):
        p=self.root/'fake.zip';p.write_bytes(b'not a zip');_,r=import_zip(self.store,self.project,p);self.assertFalse(r['valid'])
    def test_original_30_to_60_duplicates_without_new_movement_or_time_change(self):
        package(self.store,self.project)
        plan=render_plan(self.project,{'scope':'full'},'60',False);mapping=[f['source_index'] for f in plan[4]]
        self.assertEqual(mapping,[i for i in range(18) for _ in range(2)])
        result=build_video(self.store,self.project,{'scope':'full'},'60',False,False,16,self.job());v=result['validation']
        self.assertEqual(v['decoded_frames'],36);self.assertEqual(v['duration'],'3/5');self.assertEqual(v['sampling']['duplicated_frames'],18);self.assertEqual(v['cuts'],['1/5','2/5'])
    def test_sampled_package_can_be_imported_without_duration_loss(self):
        archive=package(self.store,self.project,fps='15');m=self.manifest(archive);self.assertEqual(m['expected_frame_count'],9)
        iid,r=import_zip(self.store,self.project,archive);self.assertTrue(r['valid']);apply_import(self.store,self.project,iid)
        result=build_video(self.store,self.project,{'scope':'full'},'native',False,False,16,self.job());self.assertEqual(result['validation']['decoded_frames'],9);self.assertEqual(result['validation']['duration'],'3/5')
    def test_sampling_requires_explicit_confirmation(self):
        with self.assertRaises(StudioError):export_package(self.store,self.project,{'scope':'full'},'15',False,self.job())
    def test_downsampling_output_requires_confirmation(self):
        with self.assertRaises(StudioError):render_plan(self.project,{'scope':'full'},'15',False)
    def test_trim_is_not_mislabeled_as_full(self):
        result=build_video(self.store,self.project,{'scope':'range','first_frame':3,'end_frame':15},'native',False,True,16,self.job());v=result['validation']
        self.assertEqual(v['scope'],'range');self.assertEqual(v['decoded_frames'],12);self.assertEqual(v['duration'],'2/5');self.assertEqual(v['cuts'],['1/10','3/10'])
    def test_corrupt_timeline_blocks_export(self):
        self.project['timeline']=self.project['timeline'][1:]
        with self.assertRaises(StudioError):build_video(self.store,self.project,{'scope':'full'},'native',False,False,16,self.job())
    def test_low_duration_and_disk_limits_fail_clearly(self):
        self.store.settings.project_bytes=1
        with self.assertRaises(StudioError) as e:self.job().check(force=True)
        self.assertEqual(e.exception.code,'disk_limit')


class TimingAndGeometryTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_fractional_fps_and_60_reference_preserve_cuts_and_duration(self):
        store,p=create_project(self.root,rate='30000/1001',audio=False)
        result=build_video(store,p,{'scope':'full'},'native',False,False,16,job_for(store,p));self.assertEqual(result['validation']['duration'],'3003/5000')
        result=build_video(store,p,{'scope':'full'},'60',False,False,16,job_for(store,p));self.assertEqual(result['validation']['duration'],'3003/5000');self.assertEqual(result['validation']['cuts'],['1001/5000','1001/2500'])
    def test_non_square_pixels_preserved(self):
        store,p=create_project(self.root,audio=False,sar='4/3');r=build_video(store,p,{'scope':'full'},'native',False,False,16,job_for(store,p));self.assertEqual(r['validation']['sample_aspect_ratio'],'4/3')
        metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(store.path(p['id'])/'outputs'/(r['artifact']+'.mp4'))]));self.assertEqual(metadata['streams'][0]['sample_aspect_ratio'],'4:3')
    def test_odd_resolution_never_cropped_or_padded(self):
        store,p=create_project(self.root,audio=False,size=(161,97));r=build_video(store,p,{'scope':'full'},'native',False,False,16,job_for(store,p));self.assertEqual(r['validation']['resolution'],[161,97]);self.assertEqual(r['validation']['pixel_format'],'yuv444p')
    def test_offset_audio_and_video_are_rebased_together(self):
        store,p=create_project(self.root,offset=2);self.assertEqual(p['source']['first_timestamp'],'2')
        r=build_video(store,p,{'scope':'scene','scene_id':'scene_002'},'native',False,True,16,job_for(store,p));self.assertEqual(r['validation']['duration'],'1/5');self.assertTrue(r['validation']['audio'])
    def test_rotation_preserves_display_geometry(self):
        store,p=create_project(self.root,audio=False);root=store.path(p['id']);rotated=self.root/'rotated.mp4'
        help_text=subprocess.check_output(['ffmpeg','-h','full'],stderr=subprocess.DEVNULL)
        command=['ffmpeg','-v','error']
        if b'-display_rotation' in help_text:command+=['-display_rotation:v:0','90','-i',str(root/'source.media'),'-c','copy']
        else:command+=['-i',str(root/'source.media'),'-c','copy','-metadata:s:v:0','rotate=90']
        subprocess.run(command+[str(rotated)],check=True)
        (root/'source.media').write_bytes(rotated.read_bytes());p['source_sha256']=sha256(root/'source.media');probe_project(store,p,job_for(store,p));self.assertEqual((p['source']['width'],p['source']['height']),(96,160))
        r=build_video(store,p,{'scope':'full'},'native',False,False,16,job_for(store,p));self.assertEqual(r['validation']['resolution'],[96,160])
    def test_variable_frame_rate_roundtrip(self):
        store,p=create_project(self.root,audio=False);root=store.path(p['id']);folder=self.root/'vfr';folder.mkdir();durations=['0.04','0.08','0.04','0.12','0.04','0.08']
        lines=['ffconcat version 1.0']
        for n,d in enumerate(durations):
            (folder/f'{n}.png').write_bytes(png((n*40,180,40)))
            lines += [f"file '{folder/n.__str__()}.png'",'option framerate 25','duration '+d]
        listing=folder/'list.ffconcat';listing.write_text('\n'.join(lines)+'\n')
        vfr=self.root/'vfr.mp4';subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(listing),'-fps_mode','vfr','-enc_time_base','1:1000','-c:v','libx264','-bf','0','-threads','1','-pix_fmt','yuv420p',str(vfr)],check=True)
        (root/'source.media').write_bytes(vfr.read_bytes());p['source_sha256']=sha256(root/'source.media');probe_project(store,p,job_for(store,p));self.assertTrue(p['source']['is_vfr']);self.assertEqual(len(p['frames']),6)
        p['scenes']=scenes_for(p['frames'],rational(p['source']['duration']),[2,4]);p['timeline']=native_timeline(p);store.save(p)
        r=build_video(store,p,{'scope':'full'},'native',False,False,16,job_for(store,p));self.assertEqual(r['validation']['decoded_frames'],6);self.assertEqual(r['validation']['duration'],p['source']['duration']);self.assertTrue(r['validation']['timestamps_exact'])


if __name__=='__main__':unittest.main()
