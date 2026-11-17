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
