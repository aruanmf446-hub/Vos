# Registro da finalização v2

O arquivo atual preserva os 778 tempos nativos a 30 FPS e produz 1.556 quadros a 60 FPS. Os cortes originais começam nos quadros nativos:

```text
39, 76, 173, 241, 272, 349, 404, 666, 749
```

A montagem entregue recebeu um patch específico do Hulk, porque a primeira passagem do recorte temporal o classificou como fundo. A versão atual do script principal já contém a camada semântica adicional para ele.

Com os intermediários disponíveis, a união usada foi:

```bash
ffmpeg -i video-marvel-continuidade-v2-60fps.mp4 \
  -i patch-shot-3-60fps.mp4 \
  -filter_complex \
  '[0:v]trim=start_frame=0:end_frame=346,setpts=PTS-STARTPTS[a];[1:v]trim=start_frame=0:end_frame=136,setpts=PTS-STARTPTS[b];[0:v]trim=start_frame=482:end_frame=1556,setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0,fps=60,setsar=1[v]' \
  -map '[v]' -frames:v 1556 -an -c:v libx264 -preset fast -crf 18 \
  -pix_fmt yuv420p -movflags +faststart full-with-hulk.mp4
```

O script `finalize_continuity_v2.py` validou e publicou essa união. O arquivo intermediário foi movido para o nome final; a ausência de `full-with-hulk.mp4` depois da finalização é esperada.

A comparação em câmera lenta contém dois trechos: quadros nativos 94–129 e 570–593. São 2 segundos em velocidade normal e uma repetição de 8 segundos a 0,25×. Ela não substitui o vídeo completo.
