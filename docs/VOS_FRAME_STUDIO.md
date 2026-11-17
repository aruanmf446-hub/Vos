# Contrato de frames e montagem

## Separação dos experimentos anteriores

Foram consultados `README.md`, `docs/CONTINUAR_AQUI.md`, `docs/REVISAO_V3.md`, `docs/CONEXOES_V4.md`, `avenida-marvel/render60.py`, `avenida-marvel/rebuild_continuity_v2.py` e `scripts/extrair_quadros.py`. A v2 permanece a base histórica; v3/v4 são experimentos rejeitados. Nenhuma dessas rotinas é importada pelo aplicativo independente em `frame_studio/`.

O novo serviço transporta imagens completas e tempos. Não reconstrói fundos, não recorta personagens, não cria atuação e não corrige deformações já presentes em uma edição externa.

## Estrutura do ZIP

```text
vos-projeto-<id>.zip
├── manifest.json
├── INSTRUCOES_EDICAO.md
└── scenes/
    ├── scene_001/
    │   ├── frame_000000.png
    │   └── frame_000001.png
    └── scene_002/
        └── frame_000002.png
```

O manifesto usa `schema: vos-frame-studio/1` e deve voltar **sem alterações de conteúdo**. A ordem das chaves JSON e dos membros do ZIP é irrelevante. Uma única pasta externa envolvendo o pacote é aceita e informada no relatório. Não renomeie os PNGs.

| Campo | Conteúdo |
| --- | --- |
| `project_id`, `package_id`, `project_revision` | vínculo privado ao projeto, pacote e revisão de cortes |
| `source` | SHA-256 da origem, dimensões codificadas e exibidas, rotação, SAR, time base, primeiro PTS, FPS declarado, duração, quadros decodificados e áudio |
| `selection` | escopo, índices originais de início/fim exclusivo, intervalo racional e duração |
| `scenes` | ID, ordem, intervalos originais e selecionados, contagem esperada |
| `frames` | lista explícita e ordenada de todos os quadros do pacote |
| `sampling` | FPS escolhido, descartes e repetições previstos |
| `audio_sync` | origem temporal do vídeo e início/fim absolutos da seleção |
| `export_defaults` | montagem exata, FPS original, H.264, dimensões e opção de áudio |

Cada item de `frames` tem `id`, `order`, `global_index`, `source_index`, `original_pts`, `scene_id`, `file`, `source_start`, `timestamp`, `duration`, `width`, `height` e `original_sha256`. Tempos são strings racionais, por exemplo `1001/30000`, evitando arredondamento decimal acumulado. `timestamp` é relativo à seleção; `source_start` é relativo ao primeiro quadro exibido da origem.

No modo nativo, `global_index` identifica cada quadro decodificado original, sem saltos. Em amostragem explícita, um índice original pode repetir ou ser descartado; `order` e `id` continuam únicos. Os nomes nesse modo usam ordinais únicos do pacote. Os relatórios deixam a alteração de cadência visível.

## Importação e aplicação

O servidor compara o manifesto ao exemplar canônico que guardou na exportação. Outro projeto, revisão de cortes ou alteração dos tempos bloqueia a aplicação. Nomes desordenados dentro do ZIP não alteram a sequência: a montagem segue o manifesto.

Arquivos ausentes, duplicados, extras, dimensões incorretas, PNGs inválidos/animados, transparência real, orientação EXIF, links simbólicos e caminhos inválidos são rejeitados. Há limites de upload, número de entradas, tamanho expandido e razão de compressão. O servidor nunca executa `extractall` com caminhos fornecidos pelo ZIP.

A validação não aplica nada automaticamente. A interface mostra problemas e comparação lado a lado. Após **Aplicar pacote**, apenas seu intervalo substitui a composição. Outras cenas usam imagens originais ou edições aplicadas anteriormente. Os PNGs não editados precisam voltar no pacote. Não existe preenchimento de quadros ausentes.

Os pacotes sempre extraem os **quadros da fonte original**. Para revisar uma edição anterior, trabalhe sobre o ZIP dessa edição guardado externamente. Não há exportação implícita de uma composição editada como se fosse o original.

Alterar os cortes exige remover as edições aplicadas; os pacotes de uma revisão antiga tornam-se incompatíveis. Isso impede que uma cena seja silenciosamente encaixada em cortes diferentes.

## Montagem exata e FPS

No modo original, cada imagem ocupa exatamente o intervalo de apresentação registrado. São preservados ordem, geometria, enquadramento e temporização. A codificação usa PyAV/libx264 com PTS e duração de cada pacote definidos explicitamente. O FFmpeg fornece decodificação, PNGs, áudio e conferência; não há concatenação de PNGs com FPS presumido.

Escolher 30 ou 60 FPS ativa somente **seleção/repetição temporal**. Por exemplo, 30 → 60 repete cada imagem duas vezes sem criar movimentos. Reduções exigem confirmação do descarte. O primeiro e o último quadro de cada cena/seleção são preservados. Se uma duração ou corte não cair na grade do FPS escolhido, o último intervalo é fracionado: o FPS é uma referência e o MP4 pode ser VFR. Essa escolha preserva cortes e duração em vez de arredondá-los para um CFR incompatível. Não se promete simultaneamente CFR estrito e cortes fora de sua grade.

Nem a montagem nativa nem a conversão de FPS usam fluxo óptico, mistura entre imagens, deformação, geração de frames ou recorte espacial. Para amostragem reduzida, os descartes são intencionais e contabilizados. **H.264 é com perdas**; pixels podem mudar por compressão. CRF 16/18/20 controla essa perda, sem promessa de identidade pixel a pixel.

A orientação de exibição da origem é aplicada na extração e registrada junto às dimensões codificadas. O SAR é preservado, inclusive em imagens anamórficas. Dimensões ímpares usam H.264 YUV 4:4:4, sem arredondar, cortar ou preencher a imagem; alguns navegadores/aparelhos não reproduzem esse perfil. O aplicativo avisa antes da exportação. O player original depende dos codecs suportados pelo navegador; MKV/AVI ou certos MOV podem ser extraídos mesmo sem preview HTML5.

O áudio, quando solicitado, vem somente da fonte disponível: é recortado pelo intervalo absoluto do manifesto, sincronizado, preenchido com silêncio quando necessário e recodificado em AAC. Não se promete cópia binária do áudio comprimido. O vídeo é copiado sem nova recodificação nessa etapa.

## Validação antes de liberar o MP4

1. A linha do tempo deve cobrir todo o escopo solicitado, sem lacunas, sobreposições ou cenas fora de ordem.
2. O encoder recebe uma imagem por entrada do plano, sem interpolação. Cada PNG é carregado individualmente.
3. FFprobe confere quantidade e **todos** os PTS e durações racionais dos pacotes, resolução, SAR, presença de áudio e todos os cortes.
4. FFmpeg decodifica o MP4 inteiro. Cada quadro é comparado com uma assinatura RGB reduzida do PNG esperado. Primeira e última imagem também são verificadas. Essa assinatura tolera compressão; não constitui prova matemática de identidade visual nem avaliação da edição externa.
5. Somente depois dessas verificações o resultado entra no registro de downloads. O relatório JSON inclui escopo, contagens por cena, cortes, tempos, amostragem, dimensões e SHA-256 do MP4.

Exportar um trecho não é reconstruir o vídeo completo. A interface e o relatório nomeiam o escopo explicitamente. Um ZIP parcial não pode ser apresentado como cobertura do vídeo inteiro: as cenas restantes só são montadas se os dados originais/edições continuarem disponíveis.

## Recursos e arquivos temporários

Uploads são transmitidos para arquivos; ZIPs e mídia não são carregados inteiros na RAM. Uma fila em memória executa uma tarefa por vez, com limites de tempo, disco, duração, resolução e quantidade. FFmpeg usa uma thread de codec e filtros limitados. Cancelamento/timeout encerram o grupo do subprocesso e removem o estágio em andamento. Resultados anteriores só são substituídos após validação.

Não há armazenamento persistente, autenticação de conta recuperável ou retomada após perda do serviço. O cookie da sessão e os arquivos originais/canônicos ainda disponíveis são necessários para voltar a um projeto. Consulte [deploy e limites](RENDER_DEPLOY.md).
