# Preparação para GPU gratuita: Wan2.2 Animate 14B quantizado (estudo técnico / NÃO EXECUTADO)

## Por que esta investigação é necessária

O usuário **não possui GPU local**, não pagará GPU, não usará Hugging Face nem API externa de inferência. A arquitetura continua **VOS no Render → ZIP VOS → Colab gratuito → Wan2.2 Animate Replacement → ZIP editado → VOS**. Não substituir o modelo por outro, nem fazer merge automático do PR #2.

O notebook atual usa o checkpoint completo e **bloqueia T4 16 GB / BF16 ausente / VRAM < 32 GiB**. Isso é uma política conservadora para **aquele motor**, não prova científica de que o Wan2.2 Animate 14B é impossível em hardware menor. **Não remover esse bloqueio no caminho já testado.**

### Evidência externa concreta

1. ModelScope contém um repositório **QuantStack/Wan2.2-Animate-14B-GGUF**, convertido do mesmo Wan2.2 Animate 14B, sem mudar para T2V/I2V/5B: <https://www.modelscope.cn/models/QuantStack/Wan2.2-Animate-14B-GGUF>.
2. O espelho público de Git LFS desse repositório contém o objeto **Wan2.2-Animate-14B-Q3_K_M.gguf**, com **8.630.769.472 bytes** e SHA-256 LFS **62cd6067c96599ade5bdde54a49a5ef1a7fd4de449eaae56ff4fd61ccbb16678**: <https://gitserver.onethingai.com/ai-models/Wan2.2-Animate-14B-GGUF/src/branch/master/Wan2.2-Animate-14B-Q3_K_M.gguf>. A fonte de download permitida continua sendo **o ModelScope**, não o espelho nem Hugging Face. Verifique se o arquivo baixado corresponde ao SHA; falhas devem impedir geração.
3. Há relato público de execução **Wan2.2 Animate 14B GGUF** com **2× T4 e 32 GB de RAM** em um notebook Kaggle (11/11/2026): <https://github.com/kelvinweijun/wan-2.2-animate-comfyui-kaggle>. **Não demonstra execução em um único T4 nem no Colab gratuito.** Não transferir para Kaggle automaticamente: a decisão do usuário é Colab gratuito.
4. ModelScope/DiffSynth mantém um exemplo do **mesmo Wan2.2 Animate 14B** com offload disco/CPU e quantização: <https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_inference_low_vram/Wan2.2-Animate-14B.py>. Esse exemplo utiliza BF16 e não deve ser anunciado como compatível com T4 Turing sem validação.

## Tarefa para Codex: implementar um caminho EXPERIMENTAL opcional, não uma promessa

Trabalhe em commits pequenos na **branch do PR #2**, preserve todos os testes e mantenha o motor original desligado de mudanças. Novo perfil deve se chamar `LOW_VRAM_GGUF_EXPERIMENTAL` e vir **DESLIGADO** por padrão; pode existir em notebook separado para evitar complicar a experiência padrão.

1. **Auditar hardware/viabilidade antes de codar geração:** pesquisar GPU T4 (CUDA capability 7.5, sem BF16 nativo) e verificar se o motor GGUF selecionado opera com FP16 e PyTorch SDPA/xFormers, **sem exigir FlashAttention2**. Verificar suporte real no projeto ComfyUI-GGUF + WanVideoWrapper, usando versões pinadas e licença. Se não houver caminho compatível, entregar análise objetiva em vez de criar código fictício.
2. **Auditar fontes 100% permitidas:** Q3_K_M do ModelScope com hash indicado; T5, CLIP, VAE, LoRA de relighting e ferramentas de pré-processamento também **via ModelScope/GitHub**, nunca Hugging Face, espelho desconhecido ou API de geração. As versões pinadas precisam existir. Não baixar pesos sem manifesto e espaço suficientes. `HF_HUB_OFFLINE=1` e `TRANSFORMERS_OFFLINE=1` na execução. O fato de usar bibliotecas locais não é autorização para baixar do Hub.
3. **Montar infraestrutura headless/interna para Colab** se tecnicamente viável: ComfyUI local como subprocesso ou biblioteca, com qualquer interface/API restrita a **localhost no runtime**, sem ngrok, túnel, web público, API externa, conta paga nem transmissão de imagens. Extrair dados VOS como já existente; reusar `src_pose.mp4`, `src_face.mp4`, `src_bg.mp4`, `src_mask.mp4`, `src_ref.png` ou mapear explicitamente aos nodes exigidos. Nunca ignorar a máscara escolhida nem perder o frame nativo.
4. **Controle de memória:** carregar SAM2/ViTPose somente em etapa isolada; descarregar antes do gerador. Q3_K_M quantizado, T5/CLIP carregados sob demanda, VAE tiled, CPU offload **apenas quando RAM permitir**, resolução interna reduzida para um primeiro teste de ~1 s. É permitido mudar a resolução **interna** do motor, mas o arquivo VOS final deve conservar dimensões e tempo de origem; documentar possível perda de qualidade. Não afirmar que 8,63 GB é o consumo total de VRAM.
5. **Gate específico para GGUF**, totalmente separado do gate atual 32/24/85/BF16. Checar: CUDA, VRAM/RAM/disco disponíveis, suporte real do motor Turing, pesos essenciais e SHA, armazenamento para checkpoints temporários, instalação. Somente liberar download e inferência se a configuração experimental for tecnicamente elegível. **Não pular gates de modo indiscriminado**, nem apenas remover o assert BF16 atual.
6. **VOS round-trip:** gerar `wan_generated.mp4` e os sidecars assinados que a ponte atual espera, ou adicionar adaptador de recibos separado com testes; retornar pelo `finish` existente. ZIP inválido ou máscara não revisada deve abortar. Preservar `manifest.json` byte a byte, FPS/duração originais, quadros fora da máscara e áudio na montagem VOS. Não marcar frame sintético como saída do modelo.
7. **Testes sem GPU:** schema/assinaturas oficiais/nodes, T4 no diagnóstico simulado, falta de memória e disco, checksum de GGUF falso/truncado, processos interrompidos, bloqueio de saída antiga, mapeamento FPS 24/30/60 e ZIP de volta. GitHub Actions deve validar novas células sem precisar de GPU. Não afirmar sucesso de Wan ou teste em T4 com simulação.
8. **Ergonomia:** oferecer modo experimental só para usuário com GPU disponível, depois do diagnóstico. Não fazer downloads enormes automaticamente ou selecionar arquivos privados do Render antes de preparar ambiente. Incluir `low_vram_report.json` sanitizado e passos para Colab gratuito.

### Critérios de aceite

- Um caminho **realista e auditável** para testar o mesmo Wan2.2 Animate em T4, sem depender do checkpoint BF16 integral de 53,65 GiB, ou uma demonstração explícita e fundamentada de que o caminho não é viável sob as restrições.
- Todas as saídas sem GPU marcadas `real_generation=false`.
- Se o caminho não puder ser validado sem GPU, distinguir claramente `IMPLEMENTADO MAS SEM VALIDAR EM T4` de `COMPROVADO`.
- Não alterar o notebook de baseline nem o Render em produção inadvertidamente. Não fazer merge.
- Informar commits, arquivos alterados, resultados dos testes e procedimento de decisão: começar pelo diagnóstico; em T4 usar **somente se** o perfil novo estiver validado o suficiente para uma tentativa.
