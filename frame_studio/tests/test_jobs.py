from pathlib import Path
import sys,tempfile,time,unittest
from frame_studio.config import Settings
from frame_studio.storage import Store
from frame_studio.jobs import Job,Queue,run_process
from frame_studio.errors import StudioError


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=Store(Settings(data_dir=Path(self.temp.name),reserve_bytes=1024))
        self.project=self.store.create('owner','source.mp4')
    def tearDown(self):self.temp.cleanup()
    def test_timeout_kills_child_and_preserves_project(self):
        job=Job(self.store,self.project,'test');job.deadline=time.monotonic()+.1
        before=time.monotonic()
        with self.assertRaises(StudioError) as error:
            run_process([sys.executable,'-c','import time; time.sleep(10)'],job)
        self.assertEqual(error.exception.code,'timeout')
        self.assertLess(time.monotonic()-before,3)
        self.assertEqual(self.store.get(self.project['id'],'owner')['status'],'uploading')
    def test_cancelled_job_never_runs_work(self):
        job=Job(self.store,self.project,'test');job.cancelled.set()
        with self.assertRaises(StudioError) as error:job.check()
        self.assertEqual(error.exception.code,'cancelled')
    def test_only_one_job_per_project(self):
        import threading
        queue=Queue(self.store);entered=threading.Event();finish=threading.Event()
        def work(job):entered.set();finish.wait(2);return {'done':True}
        try:
            queue.submit(self.project,'test',work);self.assertTrue(entered.wait(1))
            with self.assertRaises(StudioError) as error:queue.submit(self.project,'second',work)
            self.assertEqual(error.exception.status,409)
        finally:finish.set();queue.close()


if __name__=='__main__':unittest.main()
