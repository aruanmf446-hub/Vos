"""CPU-only regression fixtures: backend routing, never proof of T4 execution."""
import ast
from contextlib import contextmanager,nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from frame_studio.experimental.gguf_profile import diagnose,require_attempt


class Backend(Enum):
    EFFICIENT_ATTENTION=1
    MATH=2


def fixture(fail=None,nonfinite=None):
    state={'backend':None,'calls':[]}
    @contextmanager
    def kernel(backend):
        old=state['backend'];state['backend']=backend
        try:yield
        finally:state['backend']=old
    def attention(q,k,v):
        pair=(q.dtype,state['backend']);state['calls'].append(pair)
        # Simulate a GPU without FP32 efficient support, which is NOT required.
        if pair==('fp32',Backend.EFFICIENT_ATTENTION) or pair==fail:raise RuntimeError('kernel unavailable')
        return SimpleNamespace(finite=pair!=nonfinite)
    t=SimpleNamespace(__version__='SYNTHETIC',version=SimpleNamespace(cuda='synthetic'),float16='fp16',float32='fp32',
        inference_mode=nullcontext,randn=lambda *a,**k:SimpleNamespace(dtype=k['dtype']),
        isfinite=lambda r:SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:r.finite)),
        nn=SimpleNamespace(functional=SimpleNamespace(scaled_dot_product_attention=attention)),
        cuda=SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,is_available=lambda:True,
            get_device_properties=lambda _:SimpleNamespace(name='SIMULATED GPU',total_memory=15*2**30),
            get_device_capability=lambda _:(7,5),mem_get_info=lambda _:(14*2**30,15*2**30)))
    return t,state,SimpleNamespace(sdpa_kernel=kernel,SDPBackend=Backend)


class AttentionProbeTests(unittest.TestCase):
    def report(self,fail=None,nonfinite=None):
        t,state,module=fixture(fail,nonfinite)
        with patch.dict('sys.modules',{'torch.nn.attention':module}),patch('shutil.disk_usage',return_value=SimpleNamespace(free=60*2**30)),patch('shutil.which',return_value='fixture'):
            report=diagnose('/tmp',torch_module=t,ram_bytes=12*2**30)
        return report,state

    def test_missing_fp32_efficient_kernel_does_not_reject_eligible_hardware(self):
        report,state=self.report()
        self.assertEqual(state['calls'],[('fp16',Backend.EFFICIENT_ATTENTION),('fp32',Backend.MATH)])
        self.assertIsNone(state['backend']) # scoped probes restore caller backend policy
        self.assertTrue(report['eligible_for_experimental_attempt']);require_attempt(report,True)
        self.assertFalse(report['real_generation']);self.assertFalse(report['wan_t4_proven'])
        self.assertEqual(report['kernel_checks']['aux_fp32_math']['backend'],'MATH')

    def test_fp16_efficient_failure_blocks_without_trying_math_fallback(self):
        report,state=self.report(fail=('fp16',Backend.EFFICIENT_ATTENTION))
        self.assertNotIn(('fp16',Backend.MATH),state['calls'])
        self.assertFalse(report['eligible_for_experimental_attempt'])
        self.assertTrue(report['kernel_checks']['aux_fp32_math']['verified'])
        self.assertTrue(any('dit_fp16_efficient' in f for f in report['failures']))
        with self.assertRaises(ValueError):require_attempt(report,True)

    def test_fp32_math_failure_blocks_even_when_dit_probe_succeeds(self):
        report,_=self.report(fail=('fp32',Backend.MATH))
        self.assertTrue(report['kernel_checks']['dit_fp16_efficient']['verified'])
        self.assertFalse(report['eligible_for_experimental_attempt'])
        self.assertTrue(any('aux_fp32_math' in f for f in report['failures']))
        with self.assertRaises(ValueError):require_attempt(report,True)

    def test_nonfinite_outputs_block_either_phase(self):
        for pair in [('fp16',Backend.EFFICIENT_ATTENTION),('fp32',Backend.MATH)]:
            with self.subTest(pair=pair):
                report,_=self.report(nonfinite=pair)
                self.assertFalse(report['eligible_for_experimental_attempt'])
                with self.assertRaises(ValueError):require_attempt(report,True)

    def test_incomplete_or_legacy_probe_receipt_is_not_authorized(self):
        report,_=self.report();del report['kernel_checks']['dit_fp16_efficient']
        with self.assertRaises(ValueError):require_attempt(report,True)
        report.pop('kernel_checks')
        with self.assertRaises(ValueError):require_attempt(report,True)

    def test_sampler_still_disables_math_and_flash_after_probe_changes(self):
        # Execute the actual engine backend-switch statements without CUDA imports.
        source=Path(__file__).parents[1]/'experimental/gguf_engine.py'
        tree=ast.parse(source.read_text());boot=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='boot')
        switches=[n for n in boot.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr.startswith('enable_') and n.value.func.attr.endswith('_sdp')]
        self.assertEqual(len(switches),3)
        for sampler in (False,True):
            flags={}
            cuda=SimpleNamespace(**{f'enable_{name}_sdp':lambda value,key=name:flags.update({key:value}) for name in ('math','flash','mem_efficient')})
            exec(compile(ast.Module(body=switches,type_ignores=[]),'engine-switches','exec'),{'torch':SimpleNamespace(backends=SimpleNamespace(cuda=cuda)),'efficient_only':sampler})
            self.assertEqual(flags,{'math':not sampler,'flash':False,'mem_efficient':True})
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='boot' and any(k.arg=='efficient_only' for k in n.keywords))
        condition=next(k.value for k in call.keywords if k.arg=='efficient_only')
        self.assertEqual(ast.unparse(condition),"stage == 'sample'")
