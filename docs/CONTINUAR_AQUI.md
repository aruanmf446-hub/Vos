# Continuar deste ponto

## Entrega de referência

Usar `media/videos/video-marvel-continuidade-v2-60fps.mp4`, e não as prévias antigas ou o arquivo `video-marvel-completo-60fps.mp4`.

Validações da entrega mais recente:

- 1.556 quadros a 60 FPS, 25,933333 segundos, 720 × 1068, H.264, sem áudio.
- Decodificação completa sem erros.
- Nove cortes originais preservados.
- Inserção da camada do Hulk conferida contra o trecho corrigido.
- Relatório: `avenida-marvel/work/continuity-v2/export-validation.json`.

## O problema que motivou a última revisão

O usuário percebeu o personagem colado no cenário, partes do fundo mudando entre imagens, posição/escala inconsistentes e reações encurtadas. Apenas aumentar o FPS por interpolação não resolvia isso.

A revisão v2 usa:

1. Uma base contínua de fundo em cada plano de diálogo, sem alternar retalhos de cenários gerados.
2. Rastreamento da câmera do vídeo original.
3. Registro do rosto em coordenadas da cena para cancelar diferenças de posição e escala dos quadros gerados.
4. Robust Video Matting com estado recorrente para as bordas e a cor do primeiro plano.
5. Interpolação do RGBA do personagem separada da projeção do fundo.
6. Um tratamento separado para o Hulk, que o modelo de recorte inicialmente classificou como fundo.

Não aplicar novamente fluxo óptico ao cenário e ao personagem como se fossem uma única superfície. Não voltar a suavizar só a posição em tela do personagem enquanto o fundo acompanha outra trajetória de câmera.

## Scripts relevantes

- `rebuild_continuity_v2.py`: renderização atual; os scripts antigos continuam para registrar o histórico.
- `--full`: montagem completa. A versão arquivada já contém a proteção adicional da camada do Hulk.
- `--only-shot 3`: refaz somente o plano Wanda/Hulk.
- `finalize_continuity_v2.py`: registro da finalização que foi usada na entrega. Ele espera o intermediário `full-with-hulk.mp4`; não é um comando genérico idempotente.
- `finish_doom_pose.py`: correção anterior que substituiu uma deformação do Doom por rastreamento estável da posição do rosto.
- `deliver_full_video.py`: montagem anterior; não usar sozinha como se fosse a versão atual.

A entrega atual foi feita com uma renderização completa e um patch específico do Hulk, unidos por corte exato nos quadros 346 a 481 da saída a 60 FPS. O comando de união está em `docs/MONTAGEM.md`.

## Poses e limitações

- O quadro aprovado do Doom é `generated_images/exec-bf3d6e55-f887-4ac8-a0c8-5b855eca5571.png`, usando a foto de cosplay no corredor: `referencias/1DD8388A-55D6-468E-9290-018A245A61C8/6-Foto-6.jpg`.
- O Doom mantém uma pose única; os gestos das mãos não reproduzem integralmente o ator original.
- O retrato final usa uma imagem previamente aprovada do Doutor Estranho em preto e branco.
- Os seis quadros realmente novos da piscada são os quadros nativos 575 a 580, preservados em `work/continuity-v2/dense-anchors/`.
- Os quadros 581 e 582 seguram a pose fechada; 583 a 585 usam interpolação de reabertura, seguida de poses existentes. Não chamar todos esses quadros de novas gerações.
- A tentativa de gerar a continuação foi bloqueada com `moderation_blocked`, categoria `other`, sem justificativa específica. Não afirmar que existe uma proibição geral ao personagem ou inventar um motivo.
- Os planos 0, 1, 5 e 9, mais o efeito de transição dos quadros 747/748, foram reaproveitados da versão anterior.

## Próxima revisão útil

Avaliar trechos de quadros consecutivos, em escala e enquadramento iguais. Verificar o encaixe nas passagens entre poses, o cabelo, o contorno da roupa e a relação com os elementos arquitetônicos. Alguns movimentos e expressões continuam aproximados; ainda não há uma recriação integral da atuação original.

O arquivo de comparação do usuário é `referencias/C54788EA-B8E1-412B-8BE2-FCE92F3E2696/1-copy_41706C62-DA87-44CB-B34E-AA0CF2DADCF5.mov`. Ele é um comparativo em HEVC a 25 FPS, não a fonte de quadros para reconstrução. Para os movimentos, usar o vídeo original a 30 FPS.
