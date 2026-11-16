# Vos — projeto de vídeo Marvel

Registro do trabalho realizado nesta conversa, preservado em 16/11/2026.

**Versão mais recente:** [baixar o MP4 com continuidade v3](media/videos/video-marvel-continuidade-v3-60fps.mp4).

[Comparar v2 e v3](media/videos/comparacao-continuidade-v2-v3.mp4) · [O que mudou e limites da revisão](docs/REVISAO_V3.md).

O vídeo tem **25,93 segundos, 1.556 quadros, 60 FPS, 720 × 1068 e nenhum áudio**.

- [Comparação de trechos com câmera lenta](media/videos/comparacao-encaixe-v2-camera-lenta.mp4)
- [Primeiro experimento: Doutor Estranho e Wanda](media/videos/doutor-estranho-wanda.mp4)
- [Backup completo e checkpoint editável](backups/README.md)
- [MP4: download direto](https://raw.githubusercontent.com/aruanmf446-hub/Vos/main/media/videos/video-marvel-continuidade-v3-60fps.mp4)
- [Por onde continuar](docs/CONTINUAR_AQUI.md)
- [Como restaurar os arquivos](docs/RESTAURACAO.md)

[Novo teste das conexões v4](media/videos/teste-conexoes-v4-60fps.mp4) · [Diagnóstico e alteração](docs/CONEXOES_V4.md).

## Personagens

| Na cena original | Substituição |
| --- | --- |
| Carminha | Doutor Estranho |
| Homem que se levanta | Wanda |
| Tufão, que se aproxima para interrogar | Doutor Destino / Doom |
| Rapaz ao fundo | Hulk, mantendo o tamanho gigante |
| Mulher sentada no sofá | Sylvie |

## O que está preservado

- Scripts dos dois projetos, referências enviadas, vídeos originais e comparação feita no CapCut.
- Imagens geradas e quadros aprovados, inclusive o quadro do Doom que funcionou e os seis quadros consecutivos da piscada da Wanda.
- Vídeos entregues, versões de revisão, relatórios de validação e rastreamento da câmera.
- Um checkpoint em partes na pasta `backups/` com os quadros gerados, recortes com transparência e estado editável que são grandes demais para manter no histórico Git.
- Manifestos com tamanhos e SHA-256 para conferir a integridade dos arquivos.

## Estado real da versão

A v3 refaz oito planos com textura de rosto persistente, movimentos medidos no original, braços do Doom articulados, camada independente do Hulk e malha de topologia fixa. A levantada e o retrato final mantêm a v2. Veja [revisão v3](docs/REVISAO_V3.md) para reprodução e limitações.

### Histórico v2

A continuidade v2 refez o encaixe dos planos de diálogo: fundo contínuo por plano, posição e escala do personagem alinhadas ao original, recorte temporal com RVM e interpolação separada do personagem. O Hulk foi conservado numa camada própria atrás da Wanda.

O início, a levantada, o plano do Doom e o encerramento mantêm as versões anteriores. Algumas reações e gestos continuam aproximados. O Doom usa uma pose aprovada, acompanhando a posição do ator. Apenas seis quadros novos foram gerados para a piscada; a continuação da geração foi bloqueada e outras poses são reaproveitadas ou interpoladas. Os fundos reconstruídos reproduzem o cenário de forma aproximada.

O MP4 final foi decodificado inteiro; a contagem de quadros e os nove cortes originais foram verificados. Isso confirma a integridade do arquivo, não que toda a atuação original foi reproduzida.

No CapCut, manter **60 FPS** na exportação. O `.mov` de comparação enviado estava em 25 FPS.

## Organização

| Pasta | Conteúdo |
| --- | --- |
| `avenida-marvel/` | Scripts e dados principais da cena de 25,93 s |
| `strange-wanda/` | Scripts e quadros do primeiro experimento |
| `generated_images/` | Saídas originais da geração de imagens |
| `referencias/` | Fotos, vídeos originais e comparação enviada |
| `media/videos/` | Entregas e testes; usar o arquivo `continuidade-v3` como referência atual |
| `manifests/` | Integridade dos arquivos e versões dos modelos |
| `scripts/` | Verificação e restauração do backup |

Os scripts foram preservados com seus caminhos de trabalho originais. A ferramenta de restauração cria uma cópia com os caminhos ajustados; os arquivos registrados no Git permanecem intactos. Modelos baixáveis e caches de fluxo óptico não estão incluídos no checkpoint. Os quadros de trabalho do primeiro experimento estão preservados nele; os quadros extraídos do segundo vídeo podem ser reconstruídos da fonte. Veja [restauração](docs/RESTAURACAO.md).
