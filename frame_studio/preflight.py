"""Zero-network runtime diagnosis. This source is embedded in the first notebook cell."""
import datetime
import json
from pathlib import Path
import platform
import shutil
import sys
import subprocess


def diagnose(torch_module=None,psutil_module=None,root='/content'):
    report={'schema':'vos-preflight/2','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python':sys.version.split()[0],'platform':platform.system(),'gpu':'CUDA indisponivel',
        'vram_gib':0,'free_vram_gib':0,'available_ram_gib':0,'free_disk_gib':0,
        'cuda_runtime':None,'compute_capability':None,'bf16_nativo':False,'cuda_context_ok':False,'failures':[]}
    try:
        if psutil_module is None:import psutil as psutil_module
        report['available_ram_gib']=round(psutil_module.virtual_memory().available/2**30,2)
        report['free_disk_gib']=round(shutil.disk_usage(root).free/2**30,2)
    except Exception as error:report['failures'].append('Nao foi possivel medir RAM/disco: '+str(error)[:300])
    try:
        if torch_module is None:import torch as torch_module
        report['torch_version']=str(getattr(torch_module,'__version__','indisponivel'))
        report['cuda_runtime']=getattr(getattr(torch_module,'version',None),'cuda',None)
        gpu_present=bool(torch_module.cuda.is_available())
        if gpu_present:
            device=torch_module.cuda.get_device_properties(0);report['gpu']=device.name
            report['vram_gib']=round(device.total_memory/2**30,2)
            capability=torch_module.cuda.get_device_capability(0);report['compute_capability']=list(capability)
            free,total=torch_module.cuda.mem_get_info(0) # initializes CUDA context, not inference
            report['free_vram_gib']=round(free/2**30,2);report['cuda_context_ok']=True
            try:supported=torch_module.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:supported=torch_module.cuda.is_bf16_supported()
            report['bf16_nativo']=bool(supported and capability[0]>=8)
        else:report['failures'].append('CUDA indisponivel: selecione uma GPU gratuita ou encerre a tentativa.')
    except Exception as error:report['failures'].append('Falha ao inicializar CUDA/GPU: '+str(error)[:300])
    if not report['cuda_runtime']:report['failures'].append('PyTorch sem runtime CUDA utilizavel.')
    if report['vram_gib']<32 or report['free_vram_gib']<32:report['failures'].append('Recursos insuficientes: precisa de 32 GiB de VRAM total e livre nesta tentativa preventiva.')
    if report['available_ram_gib']<24:report['failures'].append('Recursos insuficientes: RAM disponivel abaixo de 24 GiB.')
    if report['free_disk_gib']<85:report['failures'].append('Recursos insuficientes: disco livre abaixo de 85 GiB.')
    if not report['bf16_nativo']:report['failures'].append('GPU sem BF16 nativo para esta configuracao.')
    if not report['cuda_context_ok']:report['failures'].append('Contexto CUDA nao foi confirmado.')
    if shutil.which('nvidia-smi'):
        try:report['driver_version']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=3).strip()[:200]
        except (OSError,subprocess.SubprocessError):report['driver_version']='indisponivel'
    report['passou_bloqueio_preventivo']=not report['failures']
    report['observacao']='Passar permite tentar; nao comprova Wan no Colab gratuito e nao autoriza GPU paga.'
    return report
