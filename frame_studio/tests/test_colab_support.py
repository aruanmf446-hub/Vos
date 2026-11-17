"""Actual CPU subprocess failure/interrupt handling; no inference models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from frame_studio.colab_support import Session,install_error_hook


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.downloads=[];self.session=Session(self.root/'logs',download=self.downloads.append)

    def test_success_and_failed_process_bundle_stdout_stderr(self):
        self.session.run([sys.executable,'-c','print("cpu success fixture")'])
        with self.assertRaises(subprocess.CalledProcessError):
            self.session.run([sys.executable,'-c','import sys; print("fixture OOM diagnostic", file=sys.stderr); sys.exit(9)'])
        self.assertEqual(len(self.downloads),1)
        with zipfile.ZipFile(self.session.bundle()) as z:
            self.assertIn('fixture OOM diagnostic',z.read('02.log').decode())
            self.assertEqual(json.loads(z.read('02.json'))['status'],'failed')
            self.assertIn('finished_utc',json.loads(z.read('02.json')))
            self.assertEqual(json.loads(z.read('01.json'))['status'],'success')

    def test_launch_failure_is_logged(self):
        with self.assertRaises(FileNotFoundError):self.session.run([str(self.root/'does-not-exist')])
        self.assertTrue((self.session.folder/'VOS_LOGS.zip').is_file())
        self.assertEqual(json.loads((self.session.folder/'01.json').read_text())['status'],'failed')

    def test_keyboard_interrupt_terminates_process_group(self):
        child_pid=self.root/'child.pid'
        code='import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); Path(sys.argv[1]).write_text(str(p.pid)); print("ready",flush=True); time.sleep(60)'
        real_popen=subprocess.Popen;processes=[]
        class InterruptPipe:
            def __init__(self,stream):self.stream=stream;self.first=True
            def __iter__(self):return self
            def __next__(self):
                if self.first:self.first=False;return self.stream.readline()
                raise KeyboardInterrupt('test fixture interrupt')
            def close(self):self.stream.close()
        def launch(*args,**kwargs):
            p=real_popen(*args,**kwargs);processes.append(p);p.stdout=InterruptPipe(p.stdout);return p
        with patch('frame_studio.colab_support.subprocess.Popen',side_effect=launch):
            with self.assertRaises(KeyboardInterrupt):self.session.run([sys.executable,'-c',code,str(child_pid)])
        self.assertIsNotNone(processes[0].poll())
        pid=int(child_pid.read_text());proc=Path(f'/proc/{pid}/stat')
        if proc.exists():self.assertIn(proc.read_text().split()[2],('Z','X'))
        self.assertEqual(json.loads((self.session.folder/'01.json').read_text())['status'],'interrupted')

    def test_notebook_hook_collects_sdk_errors_and_replaces_old_hook(self):
        from unittest.mock import Mock
        ipython=SimpleNamespace(events=Mock())
        first=install_error_hook(self.session,ipython);second=install_error_hook(self.session,ipython)
        ipython.events.unregister.assert_called_once_with('post_run_cell',first)
        second(SimpleNamespace(error_in_exec=RuntimeError('SDK fixture download failed'),error_before_exec=None))
        self.assertEqual(len(self.downloads),1)
        with zipfile.ZipFile(self.session.bundle()) as z:self.assertIn('SDK fixture download failed',z.read(next(n for n in z.namelist() if n.startswith('notebook_failure-'))).decode())


    def test_reopened_session_preserves_numbered_logs(self):
        self.session.run([sys.executable,'-c','print("first fixture")'])
        reopened=Session(self.session.folder)
        reopened.run([sys.executable,'-c','print("second fixture")'])
        self.assertIn('first fixture',(self.session.folder/'01.log').read_text())
        self.assertIn('second fixture',(self.session.folder/'02.log').read_text())

    def test_bundle_disk_failure_does_not_hide_process_error(self):
        with patch.object(self.session,'bundle',side_effect=OSError('fixture disk full')):
            with self.assertRaises(subprocess.CalledProcessError):
                self.session.run([sys.executable,'-c','raise SystemExit(7)'])
        self.assertEqual(json.loads((self.session.folder/'01.json').read_text())['status'],'failed')
