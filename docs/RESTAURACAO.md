# Restaurar o trabalho

**Base atual retomada: v2. V3/v4 são experimentos rejeitados e continuam preservados.** Os ZIPs em partes são o snapshot v2. Usar o repositório atual junto com o checkpoint para recuperar os arquivos novos; a versão v2 também é entrada da v3. Ver [conexões v4](CONEXOES_V4.md) e [revisão v3](REVISAO_V3.md).

O Git guarda código, referências, entregas e quadros aprovados. A pasta `backups/` preserva os seguintes pacotes, divididos em partes menores:

- `entrega-e-projeto.zip`: cópia organizada dos arquivos do repositório.
- `checkpoint-editavel.zip`: os mesmos arquivos e o estado editável, incluindo os quadros gerados intermediários e recortes RGBA.
- `SHA256SUMS.txt`: hashes dos arquivos dos pacotes.

Primeiro, [reconstruir o ZIP a partir das partes](../backups/README.md). O utilitário `scripts/recuperar_zip.py` verifica os hashes e também permite baixar as partes ausentes.

## Verificar a cópia

```bash
python scripts/verificar_backup.py
```

O comando verifica os hashes dos ativos registrados em `manifests/files.json`. O histórico Git preserva também os documentos e utilitários adicionados ao repositório.

## Preparar outra pasta

Primeiro, simular:

```bash
python scripts/restaurar_workspace.py --dest /caminho/novo-workspace
```

Depois, copiar de fato para uma pasta nova:

```bash
python scripts/restaurar_workspace.py --dest /caminho/novo-workspace --apply
```

Se também baixou o checkpoint:

```bash
python scripts/restaurar_workspace.py --dest /caminho/novo-workspace \
  --checkpoint /caminho/checkpoint-editavel.zip --apply
```

Os caminhos `/workspace` e o prefixo dos anexos usados na conversa são ajustados apenas nas cópias restauradas. O repositório original permanece como registro fiel.

## Quadros do vídeo original

Os quadros de trabalho do primeiro experimento estão incluídos no checkpoint. Para o segundo vídeo, os PNGs extraídos e os caches de fluxo óptico podem ser reconstruídos a partir do vídeo original preservado. Após restaurar, extrair os quadros do segundo vídeo com:

```bash
python scripts/extrair_quadros.py --workspace /caminho/novo-workspace
```

O segundo vídeo usa 778 quadros de 576 × 854, a 30 FPS. A área ativa é o recorte `576:364:0:230`. A ferramenta confere as dimensões e contagens esperadas. O primeiro projeto contém 900 PNGs de trabalho de 576 × 1024: eles foram preservados no checkpoint, pois não se deve assumir que uma nova extração com FPS fixo reproduzirá a contagem e a amostragem desse estado de trabalho.

## Dependências e modelos

`ffmpeg` e `ffprobe` precisam estar no PATH. As versões observadas de Python e das bibliotecas estão em `manifests/environment.json`; `requirements.txt` indica as famílias de dependências. A restauração não instala pacotes nem inicia geração de imagens.

Os pesos dos modelos não foram incluídos. Tamanhos e SHA-256 usados no trabalho estão em `manifests/models.json`.

- RVM MobileNetV3 FP32 TorchScript: <https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0>
- RIFE **4.25 Lite**, incluindo `train_log/flownet.pkl` e os arquivos de arquitetura: <https://github.com/hzwer/Practical-RIFE>
- YuNet 2023mar: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- LRASPP MobileNetV3: baixado automaticamente pelo Torchvision ao instanciar o modelo usado no script.

Usar as versões e os hashes registrados, em vez de assumir que o modelo mais recente produz os mesmos resultados. Os scripts ainda refletem o ambiente de trabalho da conversa; não são um pacote de instalação portátil já testado em todas as plataformas.

## Não perder os dados editáveis

O checkpoint inclui, em especial:

- `work/generated_native_v3/` e `work/review_rife/`: quadros gerados já existentes.
- `work/delivery/`: quadros adicionais, máscaras, câmera e montagens nativas usadas nas revisões.
- `work/continuity-v2/canonical/`: recortes com transparência registrados para a última revisão.
- `work/continuity-v2/dense-anchors/` e `raw-overrides/`: novos quadros e pontes da piscada.
- `work/dense_motion_test/`: levantada da Wanda e Sylvie.
- `strange-wanda/work/source/`: os 900 quadros do estado de trabalho do primeiro experimento.

Há intermediários e testes que não são entregas finais. A referência atual continua sendo o MP4 `video-marvel-continuidade-v2-60fps.mp4`.
