# VOS + Wan2.2 Animate Replacement no Colab gratuito

O VOS do Render continua responsável por extração, manifest.json, validação e reconstrução. Nenhum endpoint, esquema ou projeto anterior é substituído. O notebook apenas converte o ZIP do VOS para vídeo de condução de 30 FPS, executa Wan e devolve um ZIP compatível.

## Requisitos reais
O Wan2.2-Animate-14B é pesado. O Colab gratuito normalmente oferece NVIDIA T4 16 GB quando houver GPU, sem garantia. Não foi comprovado que o Wan2.2 Animate Replacement rode nessa configuração. O notebook bloqueia GPUs com menos de 32 GiB de VRAM, menos de 24 GiB de RAM ou menos de 85 GiB de espaço livre ANTES de baixar checkpoints. São limiares conservadores arbitrados para evitar desperdício, não mínimos certificados nem garantia de êxito com hardware superior. Não há modelo alternativo nem GPU paga.

Documentação oficial: https://github.com/Wan-Video/Wan2.2 e https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md. Limites gratuitos: https://research.google.com/colaboratory/faq.html .

## Fluxo exato
1. Na mesma sessão VOS/Render, envie seu vídeo, selecione UMA CENA e exporte ZIP com cadência Original - todos os quadros. Guarde o ZIP.
2. Abra https://colab.research.google.com/github/aruanmf446-hub/Vos/blob/main/colab/VOS_Wan22_Animate_Replacement.ipynb e selecione Ambiente de execução > Alterar tipo > GPU (gratuita quando disponível).
3. Execute as células na ordem. Faça upload do ZIP e de uma imagem PNG/JPG do novo personagem. A preparação não usa IA e mantém manifesto intacto.
4. Confira o diagnóstico. Se aparecer T4 16 GB ou outro recurso insuficiente, interrompa: NÃO houve geração nem ZIP editado.
5. Em runtime suficiente, o notebook clona o Wan oficial do GitHub e baixa os checkpoints exclusivamente do ModelScope (https://www.modelscope.cn/models/Wan-AI/Wan2.2-Animate-14B). Download de arquivos não é API externa de inferência. Não usa Hugging Face nem serviço de geração pago.
6. Execute **preprocess** primeiro (preprocess_data.py com --replace_flag). Assista a `src_mask.mp4` e `src_pose.mp4` no notebook e confirme por escrito **SIM** que somente o personagem selecionado foi marcado. O notebook bloqueia máscaras vazias/muito grandes e frames faltantes, mas só a revisão visual identifica um ator errado. Depois execute **infer** (generate.py --task animate-14B --replace_flag). Se a geração falhar por GPU, RAM, disco ou incompatibilidade, não há substituição falsificada.
7. Depois da geração e da conferência de contagem, será gerado VOS_EDITADO.zip. Faça download.
8. De volta ao mesmo projeto/sessão no VOS: Importar edições > ZIP editado > Validar > Revisar > Aplicar pacote > Montar vídeo com cadência original.

## Preservação e limites
- Mesmos nomes de PNGs, mesmo número de quadros, dimensões e manifest.json byte a byte. O VOS conserva os timestamps, FPS declarado, duração e o áudio na montagem. A saída H.264 tem perdas.
- Os quadros fora da cena não são modificados; dentro da cena, pixels fora da máscara são copiados do PNG original. O modelo produz o conteúdo dentro da máscara.
- O Wan opera com quadros de condução a 30 FPS. A ponte mapeia quadros por timestamp original na volta, sem interpolação. Não garante continuidade perceptual em vídeos VFR ou FPS diferentes.
- A máscara automática oficial foi pensada para uma pessoa. Cenas com várias pessoas podem segmentar a pessoa errada ou abranger outras figuras, mesmo que a saída seja válida. É obrigatório revisar o resultado; preservação visual de não selecionados só é garantida FORA da máscara, não dentro de uma máscara incorreta.
- O Render gratuito perde seus arquivos em reinicializações. Um manifesto de outra sessão/projeto não pode ser forçado na importação.
- Sem chamada a API de inferência externa; o download inicial de código/pesos via GitHub/ModelScope usa acesso à internet.

Script de integração: `frame_studio/wan_bridge.py`. O script `finish` usa FFmpeg + Pillow em processamento quadro a quadro, evitando carregar o vídeo inteiro na RAM. O ZIP desktop CineSwap_Local_Python.zip da conversa original permanece preservado separadamente e suas rotinas inspiraram o pré-processamento da nova ponte, sem substituir o VOS já implantado.

## Checklist para a validação noturna

- [ ] Colab mostra a GPU, VRAM, RAM e disco disponíveis (antes de baixar pesos)
- [ ] Se for **T4 16 GB**, o notebook interrompe por insuficiência preventiva: não interpretar como resultado de substituição
- [ ] ZIP original tem **uma cena** e **amostragem native**; vídeo de condução toca antes da geração
- [ ] Em hardware viável, etapa **preprocess** gera `src_mask.mp4` com a pessoa correta e `src_pose.mp4` com movimentos adequados
- [ ] Responder **SIM** somente após ver a máscara inteira; se houver duas pessoas sobrepostas, recortar a cena ou revisar antes de inferir
- [ ] Etapa **infer** produz `wan_generated.mp4` realmente gerado; só depois executar **finish**
- [ ] O `VOS_EDITADO.zip` mantém manifesto inalterado, quantidade de PNGs e resolução
- [ ] No VOS/Render, importar e aplicar no **mesmo projeto**, montar vídeo completo com FPS original e conferir cortes, duração e personagens intocados

Diagnóstico: **falha de GPU no Colab gratuito não tem solução garantida que respeite ao mesmo tempo Wan 14B completo, sem GPU paga e sem outro modelo**. Neste caso, guardar logs e ZIPs e não simular geração. Nenhum novo serviço de inferência/API foi adicionado.

## Revisão técnica desta integração

Para validar este PR **antes do merge**, abra o [notebook da branch de revisão](https://colab.research.google.com/github/aruanmf446-hub/Vos/blob/codex/finalize-wan22-validation/colab/VOS_Wan22_Animate_Replacement.ipynb). A variável `VOS_REF` aponta para essa mesma branch; não misture o notebook novo com a ponte antiga da `main`. Depois do merge, a branch continua sendo uma referência válida enquanto existir; para usar `main`, mude explicitamente `VOS_REF='main'` em uma cópia do notebook.

- Código Wan fixado em `1ea34ff48f87168174e12956e200b1d908b1c5ff`. Pesos ModelScope fixados na revisão `4cfe61a9`, com 413 arquivos selecionados (~53,65 GiB) e hashes publicados pelo ModelScope em `colab/wan22_modelscope_manifest.json`. O download evita cópias extras de modelos que não são usadas em Replacement. Cada arquivo é conferido por tamanho e SHA-256 antes do pré-processamento.
- Instalação inclui `requirements_animate.txt`, SAM2 na revisão oficial, `moviepy==1.0.3` e `diffusers==0.35.1`, que oferece `FluxKontextPipeline` importado pelo preprocessador. A compatibilidade completa com a imagem CUDA oferecida pelo Colab ainda exige o teste real.
- O caminho `pose2d/vitpose_h_wholebody.onnx` oficial é uma **pasta**, contendo `end2end.onnx` e centenas de partes externas. A ponte aceita esse formato. Não renomeie nem copie somente o arquivo `.onnx`.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` e `HF_DATASETS_OFFLINE=1` bloqueiam downloads implícitos pelo Hub. Transformers/Diffusers são bibliotecas locais exigidas pelo Wan oficial; os pesos/tokenizadores vêm exclusivamente do ModelScope. Nenhuma API de inferência é chamada.
- RAM medida passa a ser **disponível**, não RAM total. GPUs sem suporte BF16 também são bloqueadas. O bloqueio de recursos é reexecutado na célula de instalação para evitar downloads após uma célula anterior falhar.
- Pré-processamento bem-sucedido registra hashes dos intermediários e `mask_report.json` com cobertura e bounding box por quadro. Máscaras vazias, acima de 65% do quadro, com contagem/FPS/geometria inválidos ou salto de centro maior que 0,35 da escala normalizada são bloqueadas. Este último é um detector conservador de saltos grosseiros; pode rejeitar movimentos rápidos legítimos e não comprova identidade nem estabilidade fina.
- Após responder **SIM**, `review-mask --confirm SIM` vincula a revisão aos intermediários atuais. Alterá-los exige repetir `preprocess` e a revisão. O comando antigo `run` agora termina após preprocessamento e pede revisão, em vez de iniciar IA automaticamente.
- `infer` apaga o recibo anterior antes de tentar gerar. O MP4 anterior é removido antes da chamada ao modelo. Só uma execução bem-sucedida, com vídeo decodificável e contagem/30 FPS/timestamps corretos, grava `generation.json`. `finish` exige esse recibo e os hashes correspondentes; não usa saída antiga de uma tentativa que falhou.
- Na volta, máscaras e vídeo gerado devem ter dimensões idênticas. O ZIP é publicado por troca atômica somente após composição completa, preservando o ZIP de origem e o manifesto byte a byte.

Veja o [relatório técnico e roteiro noturno](VOS_WAN22_VALIDATION.md) para evidências e limites.

## Selecionar uma pessoa específica

Há um modo opcional e experimental, desligado por padrão, para pontos positivos/negativos, SAM2 e correção/revisão de máscaras nativas. Veja [instruções, limites e roteiro atualizado](VOS_TARGET_SELECTION.md). O diagnóstico ainda vem antes de uploads/pesos. O fluxo automático anterior continua disponível.

## Preparação final do primeiro teste real

A ordem atual instala/audita e executa o smoke antes de uploads reais e usa referências imutáveis, controlador de etapas e recuperação de pontos/máscaras. As instruções atuais e três passos iniciais estão em [VOS_FIRST_REAL_TEST.md](VOS_FIRST_REAL_TEST.md); essa ordem substitui os roteiros anteriores de uploads antes da instalação.
