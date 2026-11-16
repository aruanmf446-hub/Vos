# Revisão de continuidade v3

Entrega: `media/videos/video-marvel-continuidade-v3-60fps.mp4`.
Comparação: `media/videos/comparacao-continuidade-v2-v3.mp4`, com a versão anterior em cima e a revisão embaixo. Os quatro trechos têm o mesmo enquadramento e tempo; a comparação roda em velocidade normal.

## O que mudou

- Oito planos foram refeitos: abertura do Estranho, diálogos do Estranho, Wanda com Hulk, caminhada do Doom e diálogo longo da Wanda.
- Cada plano de diálogo usa uma imagem aprovada por personagem como textura persistente. Movimentos da cabeça e expressões vêm de medidas de 478 pontos do rosto no original, com transferência relativa da abertura dos olhos e da boca. Isso reduz a troca de aparência entre imagens diferentes.
- A roupa recebe movimento denso medido no vídeo original. A malha triangular permanece fixa durante o plano; o renderizador não troca sua triangulação entre quadros. O tecido inferior permanece dentro do enquadramento, evitando prolongamentos por colunas de pixels.
- O Doom recebe movimento dos braços articulado por segmentos, além do acompanhamento da cabeça; continua partindo de uma única imagem aprovada.
- O Hulk tem imagem e máscara próprias, mantidas durante todo o plano, sob a Wanda. A máscara dele não é novamente classificada a cada quadro. A Wanda pode ocultá-lo naturalmente.
- A abertura e a caminhada do Doom usam um único fundo recuperado dos pixels visíveis do vídeo original, com os buracos preenchidos pela base existente. O restante dos diálogos mantém os fundos contínuos v2.
- Nos meios-quadros a 60 FPS, a posição dos vértices e a câmera são interpoladas separadamente. O cenário não atravessa fluxo óptico junto com o personagem. Cada imagem é amostrada a partir da textura original, evitando deformações acumuladas por sucessivos remapeamentos.

## Limites reais

É uma adaptação de movimento em 2D. Não houve nova geração de imagens ou vídeo nesta revisão. Não há simulação física do tecido, superfícies novas das mãos nem reconstrução exata do interior da boca.

A abertura ainda usa nove vistas aprovadas para a virada do Estranho. A levantada da Wanda com a Sylvie e o retrato final mantêm a montagem v2. Os quadros de transição 747 e 748 também são preservados. O Doom conserva a aparência da pose aprovada; os movimentos são aproximados. O Hulk tem pose persistente e movimento de câmera, sem recriar todos os gestos do ator ao fundo.

O teste alternativo que encaixava diretamente o esqueleto da atriz na abertura deformou a capa e foi rejeitado. Ele não está no MP4 entregue. O usuário preferiu continuar apenas com esta revisão; nenhuma geração do Higgsfield foi solicitada ou consumida.

## Reprodução

Restaurar o checkpoint v2 e depois sobrepor os arquivos atuais do repositório, conforme a ferramenta de restauração e o manifesto atualizado. Os dados novos e necessários da v3 estão no Git, em `avenida-marvel/work/continuity-v3/`; o ZIP arquivado continua sendo o snapshot v2 e não foi substituído.

Com pesos, dependências e quadros de origem disponíveis:

```bash
python avenida-marvel/render_continuity_v3.py --shots 0,2,3,4,5,6,7,8 --fps 60
python avenida-marvel/finalize_continuity_v3.py
```

O renderizador usa os vídeos v2 e as referências preservadas. Ao restaurar em outro caminho, conferir o caminho `O` nos scripts de renderização e finalização, além dos caminhos ajustados pelo restaurador. Manter a versão v2 disponível: é uma entrada da v3.

As medidas do original e das referências estão em `source-geometry.npz` e `anchor-geometry.npz`. Os dados do plano da levantada, quadros 39–75, foram marcados como ausentes: há duas pessoas sobrepostas e a seleção do ator que se levanta precisa ser validada antes de usar esse rastreamento. Não usar o rosto da mulher sentada para dirigir a Wanda. O Doom usa os pontos de pose e o rastreamento anterior do rosto, pois seu rosto de perfil não foi identificado pelo novo detector.

Para recalcular as medidas, instalar as dependências opcionais de `requirements-motion-extraction.txt` e a biblioteca do sistema `libGLESv2.so.2` (pacote Debian `libgles2`). Os modelos do MediaPipe não estão no Git:

- <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
- <https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task>

Guardá-los em `avenida-marvel/models/mediapipe/`. Hashes em `manifests/models.json`. O renderizador pode usar as medidas e as máscaras já preservadas sem instalar o MediaPipe; ainda precisa dos pesos RVM/LRASPP quando gerar uma máscara ausente.

## Validação

`avenida-marvel/work/continuity-v3/export-validation.json` registra tamanho e SHA-256, decodificação integral, ausência de áudio, resolução 720 × 1068, 1.556 quadros, 60 FPS e os nove cortes originais. Também confere que o meio-quadro anterior a cada corte não mistura os dois planos. Esses testes verificam integridade e montagem, sem afirmar que toda a atuação é idêntica à original.
