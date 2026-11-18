"""Independent Turing gate: no BF16/32-GiB baseline waiver and no network."""
import datetime
import json
from pathlib import Path
import shutil


def diagnose(root='/content',min_disk_gib=33,torch_module=None,ram_bytes=None,probe=True):
    report={'schema':'vos-t4-gguf-report/1','profile':'LOW_VRAM_GGUF_EXPERIMENTAL',
        'status':'IMPLEMENTADO_MAS_SEM_VALIDAR_EM_T4','real_generation':False,'wan_t4_proven':False,
        'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'gpu':None,'vram_gib':0,'free_vram_gib':0,'available_ram_gib':0,'disk_free_gib':0,
        'bf16_required':False,'generator_precision':'fp16','text_precision':'fp32 (separate process)',
        'attention':'DiT FP16: EFFICIENT_ATTENTION only; isolated FP32: MATH permitted; FlashAttention disabled','kernel_checks':{},'kernel_probe_performed':False,'nvcc_available':bool(shutil.which('nvcc') or Path('/usr/local/cuda/bin/nvcc').is_file()),'failures':[]}
    if ram_bytes is None:
        try:
            import psutil
            ram_bytes=psutil.virtual_memory().available
        except ImportError:ram_bytes=0
    report['available_ram_gib']=ram_bytes/2**30
    report['disk_free_gib']=shutil.disk_usage(root).free/2**30
    try:
        if torch_module is None:import torch as torch_module
        t=torch_module;report['torch']=str(t.__version__);report['cuda_runtime']=t.version.cuda
        if not t.cuda.is_available():raise RuntimeError('CUDA indisponivel')
        p=t.cuda.get_device_properties(0);report['gpu']=p.name
        capability=t.cuda.get_device_capability(0);report['compute_capability']=list(capability)
        report['vram_gib']=p.total_memory/2**30;report['free_vram_gib']=t.cuda.mem_get_info(0)[0]/2**30
        if capability<(7,5):report['failures'].append('GPU abaixo de capability 7.5')
        if probe:
            report['kernel_checks']=probe_kernels(t)
            report['kernel_probe_performed']=True
            for name,check in report['kernel_checks'].items():
                if not check['verified']:report['failures'].append(name+': '+check['error'])
        else:report['failures'].append('Probe de kernels nao executado; apenas simulacao/inspecao')
    except Exception as error:report['failures'].append('GPU/CUDA/SDPA: '+str(error)[:500])
    if not report['nvcc_available']:report['failures'].append('Compilador NVCC ausente; SAM2 fixado exige extensao CUDA. Nao baixar pesos.')
    if report['vram_gib']<14 or report['free_vram_gib']<13:report['failures'].append('Exige pelo menos 14 GiB VRAM total e 13 GiB livre')
    if report['available_ram_gib']<10:report['failures'].append('RAM disponivel abaixo de 10 GiB; offload nao autorizado')
    if report['disk_free_gib']<min_disk_gib:report['failures'].append(f'Disco livre abaixo de {min_disk_gib} GiB')
    report['eligible_for_experimental_attempt']=not report['failures']
    report['warning']='Limiar de tentativa, nao minimo certificado. 8.63 GB de arquivo nao equivale ao pico de VRAM/RAM. Sem promessa de conclusao no Colab gratuito.'
    return report


def probe_kernels(t):
    # Match the isolated engine phases, not an unnecessary FP32 efficient kernel.
    # DiT may NEVER fall back to MATH; FP32 VAE/T5 already permit MATH in boot().
    from torch.nn.attention import sdpa_kernel,SDPBackend
    checks={}
    for name,dtype,backend in (
        ('dit_fp16_efficient',t.float16,SDPBackend.EFFICIENT_ATTENTION),
        ('aux_fp32_math',t.float32,SDPBackend.MATH),
    ):
        q=result=None
        try:
            with t.inference_mode(),sdpa_kernel(backend):
                q=t.randn(1,4,32,128,device='cuda',dtype=dtype)
                result=t.nn.functional.scaled_dot_product_attention(q,q,q)
                if not t.isfinite(result).all().item():raise RuntimeError('SDPA produziu nao-finitos')
                t.cuda.synchronize()
            checks[name]={'verified':True,'backend':backend.name,'error':None}
        except Exception as error:
            checks[name]={'verified':False,'backend':backend.name,'error':str(error)[:500]}
        finally:
            del q,result
            t.cuda.empty_cache()
    return checks


def require_attempt(report,enabled=False):
    if not enabled:raise ValueError('Perfil experimental desligado. Leia o diagnostico antes de ativar conscientemente.')
    checks=report.get('kernel_checks',{})
    if (not report.get('eligible_for_experimental_attempt') or not report.get('kernel_probe_performed')
        or any(not checks.get(name,{}).get('verified') for name in ('dit_fp16_efficient','aux_fp32_math'))):
        raise ValueError('Diagnostico GGUF bloqueado: '+ '; '.join(report.get('failures',[])))
