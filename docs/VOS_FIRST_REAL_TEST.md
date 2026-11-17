# Primeiro teste real: preparação final do PR #2

O notebook prepara o runtime antes de pedir vídeo privado ou um projeto novo no Render. A integração existente foi preservada, incluindo a correção XYXY do commit `c470d59`. Não há merge/deploy automático.

## Correções e verificações reais sem GPU

- **Canais da pose:** o adaptador passa pela mesma conversão RGB→BGR de `Pose2d.load_images` antes de `ViTPose.preprocess`. O recorte XYXY fora do centro foi preservado. `wan_audit.py` carrega diretamente `pose2d_utils.py` e `utils.py` oficiais e executa a função real `ViTPose.preprocess` extraída do fonte, sem inicializar ONNX, Torch ou pesos. Compara o tensor do adaptador com o oficial e exercita rosto/normalização de coordenadas. Não é um teste com preditor falso.
- **SAM2:** pontos normalizados da interface viram XY absoluto em pixels, limitados ao quadro; `normalize_coords=True` fica explícito. Isso corresponde à divisão por largura/altura e escala interna no SAM2 fixado. Assinaturas de pontos/máscaras/estado/propagação e formato dos checkpoints continuam sendo os oficiais.
- **Dependências antes dos pesos:** após instalar, o notebook importa as bibliotecas reais, incluindo FlashAttention/SAM2/FluxKontextPipeline, lista providers ONNX e executa a auditoria CPU. Se houver erro ABI/import, para antes de baixar os ~54 GiB de pesos. A instalação CUDA completa ainda não foi comprovada neste ambiente.
- **Diagnóstico zero-download:** GPU, VRAM total e livre, RAM disponível, disco, versão CUDA do PyTorch, capability, BF16 nativo e criação do contexto CUDA via consulta de memória. BF16 emulado em GPU antiga não é aceito. Mantidos 32 GiB VRAM (agora também livre), 24 GiB RAM disponível e 85 GiB disco. Passar só permite tentar; não comprova compatibilidade.
- **Bloqueio de células:** diagnóstico recusado impede qualquer célula posterior. O controlador de etapas também impede preparar/upload após instalação falhar e impede inferência após pré-processamento/revisão falhar. Corrija e reexecute a etapa que falhou ou uma anterior; não há continuação silenciosa de “Executar tudo”.

Código oficial Wan: `1ea34ff48f87168174e12956e200b1d908b1c5ff`. SAM2: `0e78a118995e66bb27d78518c4bd9a3e95b4e266`. Os checks do PR executam a auditoria com checkout real dessa revisão. O notebook fixa o toolkit VOS em um SHA completo, separado do commit do próprio notebook, para evitar referência circular/mutável. Não é necessário editar `VOS_REF` à noite.

## Teste mínimo reproduzível

A instalação termina com `smoke_test.py`: três quadros sintéticos, duas pessoas desenhadas, em 24/30/60 FPS. Usa FFmpeg real, seleção/máscara nativa, revisão, composição, manifesto byte a byte e comparação de todos os pixels fora da máscara. Saídas coloridas e recibos de fixture ficam somente em diretório temporário, apagado depois; o relatório diz `real_generation=false` e `sam2_executed=false`. Nenhum modelo é carregado e nenhum ZIP sintético é oferecido como entrega real.

A reimportação pelo VOS e montagem completa com cortes/áudio continuam verificadas pela suíte Python/Docker existente. O smoke do Colab verifica transporte local da ponte; não afirma criar um projeto real no Render.

```bash
python -m frame_studio.wan_audit --wan /caminho/Wan2.2
python -m frame_studio.smoke_test --report /tmp/synthetic-smoke.json
python -m unittest discover -s frame_studio/tests -t . -v
```

## Progresso versus diagnóstico

**VOS_LOGS.zip:** somente texto sanitizado e limitado (até 2 MiB descompactados, caudas de 32 KiB por arquivo, até 50 arquivos), versões conhecidas, recursos e etapa interrompida. Não inclui arquivos arbitrários `.txt`, dumps de ambiente, máscaras, pesos, vídeos ou frames privados. Valores de ambiente com nomes de segredo, padrões comuns de tokens, Bearer, senhas e credenciais em URLs são removidos. Logs completos locais podem ter mais conteúdo; compartilhe o ZIP sanitizado, não a pasta inteira. O diagnóstico anterior ao bootstrap também gera seu ZIP pequeno.

**VOS_PROGRESSO_*.zip:** backup separado com manifesto, pontos, máscaras/sementes disponíveis e estado de revisão. Não contém o vídeo-fonte, pesos nem imagens do personagem. Guarde também o ZIP original e a referência que você já possui no computador. Downloads são pedidos após preparação, seleção, revisão de máscaras e antes da inferência. Se o navegador bloquear múltiplos downloads, salve pelo painel Arquivos antes de continuar.

Para recuperar em runtime novo: passe novamente pelo diagnóstico e instalação, prepare o **mesmo ZIP original** e referência, ative `RESTORE_PROGRESS=True` na célula de seleção e envie o último backup. O código valida manifesto byte a byte, hash do ZIP original, caminhos, limites, máscaras e revisão. Seleção ainda não revisada volta como não revisada. Máscaras aprovadas e válidas dispensam novo SAM2; pose/rosto precisam ser preparados e revisados novamente. Não se restaura `generation.json`, estado de denoising, inferência parcial ou progresso GPU.

Um backup de outra exportação/projeto é recusado. Nunca edite manifestos para torná-lo compatível. Destruição da VM impede executar o download depois do fato; só backups já guardados fora do runtime sobrevivem com certeza.

## Risco Render verificado no fonte

`PROJECT_TTL_SECONDS` tem padrão de **7200 s sem acesso**. `Store.get` atualiza o acesso, mas a limpeza pode remover projetos inativos. Hibernação/reinício/redeploy do Render gratuito pode apagar o diretório temporário independentemente desse TTL. A instalação/pesos/smoke agora vêm antes da exportação VOS para reduzir a espera; isso não garante sobrevivência durante uma geração longa. Mantenha o mesmo navegador/projeto e guarde arquivos fora do Render. O backup Colab não restaura a sessão/projeto Render. Se expirou, um novo projeto/exportação e ciclo compatível serão necessários.

## Três passos iniciais à noite

1. Abra o notebook exato indicado no PR e selecione uma **GPU gratuita**. Execute o diagnóstico. Se bloquear, baixe `VOS_PREFLIGHT_LOGS.zip` e pare; não envie vídeo, não pule limites e não contrate GPU.
2. Se aprovado, execute as ferramentas fixadas e a instalação/auditoria/smoke. Aguarde “Runtime preparado”. Ainda não crie o projeto real no Render. Um smoke aprovado não é geração Wan.
3. Só então exporte uma cena nova de **1–3 segundos**, uma cena e cadência Original no VOS, guarde o ZIP e envie ZIP/referência. Para alvo explícito, ative a opção já existente e siga as revisões. Guarde os backups antes de iniciar inferência; perfil experimental de memória fica desligado no primeiro teste.

## O que permanece dependente de GPU real

Instalação completa na imagem atual do Colab, imports CUDA/FlashAttention reais, download e carregamento integral dos pesos, providers/modelos reais, integração do iframe Colab, rastreamento/identidade/oclusões, pose/expressão, inferência, VRAM/RAM/tempo e qualidade/continuidade. T4 16 GB continua bloqueada. A auditoria CPU, os mocks SAM2 e o smoke sintético não substituem essas evidências. O pipeline definitivo e as restrições sem APIs externas de inferência, Hugging Face, assinatura, GPU paga ou outro modelo foram mantidos.
