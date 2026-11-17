# Vos Frame Studio

Aplicativo web para **extrair todos os frames, editar externamente e reconstruir cenas ou o vídeo completo**, com manifesto, validação obrigatória e montagem sem deformação ou interpolação. Frontend e backend Python no **mesmo serviço Docker gratuito do Render**.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/aruanmf446-hub/Vos)

- [Como usar e executar localmente](frame_studio/README.md)
- [Configurar o Render, senha e limites](docs/RENDER_DEPLOY.md)
- [Formato do ZIP, timestamps e montagem exata](docs/VOS_FRAME_STUDIO.md)
- [Testes e validação](docs/FRAME_STUDIO_TESTS.md)

**Fluxo:** vídeo → seleção/cortes → ZIP com PNGs + manifesto → edição externa → importação validada → MP4 H.264. O modo original preserva a cadência; 30/60 FPS usa somente seleção/repetição temporal explícita, mantendo duração e cortes. A edição com IA fica fora do aplicativo.

Os projetos no Render são temporários e podem desaparecer após reinício/hibernação ou expiração. Guarde ZIPs e resultados fora do serviço. Não há dependência de banco, Redis, GitHub Pages, Supabase ou IA paga. O deploy requer conectar sua conta Render ao repositório; os arquivos não criam uma URL pública sozinhos.

## Projetos de vídeo anteriores — arquivo preservado

O aplicativo está isolado em `frame_studio/` e não utiliza as rotinas experimentais abaixo. O histórico e os backups anteriores permanecem preservados.

# Vos — histórico do projeto de vídeo Marvel

Registro do trabalho realizado nesta conversa, preservado em 16/11/2026.

**Base de trabalho retomada:** [continuidade v2](media/videos/video-marvel-continuidade-v2-60fps.mp4), o rascunho que o usuário avaliou em cerca de 75%.

**As revisões v3 e v4 foram consideradas piores pelo usuário.** Permanecem preservadas como experimentos; não devem ser apresentadas como melhorias aprovadas nem substituir a v2. Os testes de integridade e de geometria não demonstraram naturalidade visual.

O vídeo tem **25,93 segundos, 1.556 quadros, 60 FPS, 720 × 1068 e nenhum áudio**.

- [Comparação de trechos com câmera lenta](media/videos/comparacao-encaixe-v2-camera-lenta.mp4)
- [Primeiro experimento: Doutor Estranho e Wanda](media/videos/doutor-estranho-wanda.mp4)
- [Backup completo e checkpoint editável](backups/README.md)
- [MP4: download direto](https://raw.githubusercontent.com/aruanmf446-hub/Vos/main/media/videos/video-marvel-continuidade-v2-60fps.mp4)
- [Por onde continuar](docs/CONTINUAR_AQUI.md)
- [Como restaurar os arquivos](docs/RESTAURACAO.md)

[Experimento v4 rejeitado](media/videos/teste-conexoes-v4-60fps.mp4) · [Diagnóstico e alteração](docs/CONEXOES_V4.md).

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

A v4 tentou corrigir conexões com uma deformação contínua e uma interpolação compartilhada da cor e da máscara entre as poses da abertura. O resultado foi rejeitado pelo usuário e não é a base atual. O fundo continua separado. Não há nova geração de imagens; gestos e superfícies não vistos permanecem aproximados. Veja [conexões v4](docs/CONEXOES_V4.md).

### Experimento v3 — não usar como base

A v3 refaz oito planos com textura de rosto persistente, movimentos medidos no original, braços do Doom articulados, camada independente do Hulk e malha de topologia fixa. A levantada e o retrato final mantêm a v2. Veja [revisão v3](docs/REVISAO_V3.md) para reprodução e limitações.

### V2 — base retomada

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
| `media/videos/` | Entregas e testes; usar o arquivo `continuidade-v2` como base atual |
| `manifests/` | Integridade dos arquivos e versões dos modelos |
| `scripts/` | Verificação e restauração do backup |

Os scripts foram preservados com seus caminhos de trabalho originais. A ferramenta de restauração cria uma cópia com os caminhos ajustados; os arquivos registrados no Git permanecem intactos. Modelos baixáveis e caches de fluxo óptico não estão incluídos no checkpoint. Os quadros de trabalho do primeiro experimento estão preservados nele; os quadros extraídos do segundo vídeo podem ser reconstruídos da fonte. Veja [restauração](docs/RESTAURACAO.md).
