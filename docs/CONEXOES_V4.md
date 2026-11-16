# Revisão das conexões v4

A versão v3 continua preservada no commit `3836fe2` e em seu MP4 original. Esta revisão trabalha especificamente nas passagens entre quadros.

## Diagnóstico

A auditoria `avenida-marvel/work/continuity-v4/audit-v3.json` encontrou triângulos invertidos ou quase colapsados em partes visíveis da imagem. Na amostra do rosto da Wanda, havia entre 3 e 43 triângulos invertidos por quadro inspecionado. Abertura e Doom também apresentaram inversões em alguns quadros. Desenhar esses triângulos pode sobrepor partes da textura.

Na abertura, o renderizador v3 também misturava duas poses por transparência. Quando seus contornos não coincidiam, isso produzia uma exposição dupla.

## Alteração testada

- As trajetórias são filtradas nas coordenadas do cenário; o movimento de câmera é reaplicado depois.
- O renderizador substitui o desenho dos triângulos deformados por um campo de velocidade contínuo, integrado em oito etapas de composição de coordenadas. A textura do personagem e sua máscara são amostradas juntas uma única vez.
- Na troca de poses da abertura, o RIFE recebe quatro canais: RGB com transparência pré-multiplicada e alpha. O mesmo fluxo é usado para cor e máscara. O fundo não passa pelo RIFE.
- Os fundos, cortes e personagens permanecem baseados na v3. A levantada e o retrato final mantêm a montagem anterior.

A primeira tentativa de corrigir a posição dos vértices diretamente foi descartada: em alguns quadros reduzia demais a deformação expressiva. `connections_v4.py` registra esse experimento; o renderizador atual usa `continuous_warp_v4.py`.

## Teste curto

`media/videos/teste-conexoes-v4-60fps.mp4`: 382 quadros, 6,3667 s, 720 × 1068, 60 FPS, sem áudio. Reúne a abertura, a caminhada do Doom e um trecho da Wanda. Não é o vídeo inteiro.

O relatório `pilot-report.json` registrou 232 amostras de mapas de camadas, sem pixels de orientação invertida nas regiões opacas medidas; o menor jacobiano foi aproximadamente 0,0778. Isso verifica a transformação medida antes da mistura RIFE; não prova reprodução exata da atuação ou naturalidade do resultado final.

## Reprodução

Restaurar os ativos v2/v3, as dependências e os pesos RIFE já documentados. Não há nova geração de imagens, uso de créditos ou serviço de vídeo externo.

```bash
python avenida-marvel/render_connections_v4.py          # teste curto
python avenida-marvel/render_connections_v4.py --full   # montagem inteira
```

As coordenadas filtradas usadas pelo novo renderizador são os arquivos `velocity-*.npz` em `work/continuity-v4/trajectories/`. Os outros arquivos de trajetória pertencem ao experimento descartado e não são entradas desta montagem.

O resultado continua sendo uma adaptação em 2D, com gestos e superfícies não vistos aproximados. A correção das conexões não transforma as imagens aprovadas em uma atuação nova filmada.
