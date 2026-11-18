# Seleção explícita de personagem — opcional e experimental

Continuação do PR #2 na branch `codex/finalize-wan22-validation`. O modo automático anterior continua disponível. **Seleção explícita e perfil de memória são desligados por padrão**, pois a execução real SAM2/Wan ainda precisa de GPU. Nenhuma dependência de GPU foi adicionada ao Render; NumPy adicional é somente dependência de teste/Colab.

## O que foi implementado

O editor local do notebook permite nomear o alvo, navegar por todos os quadros originais, marcar pontos positivos no alvo e negativos em outra pessoa/cenário e adicionar pontos em quadros-chave. Não envia imagens a outro serviço. Depois do rastreamento, mostra sobreposição da máscara, pincel para incluir pixels e borracha para preservar regiões. É necessário marcar cada quadro como revisado. Editar o quadro invalida sua aprovação.

Exemplo: em uma cena com Wanda e Estranho, marque verde no corpo/rosto da Wanda e vermelho no Estranho. Use novos quadros-chave nos cruzamentos ou quando o alvo mudar de pose. Pontos selecionam regiões; **o nome digitado não identifica semanticamente o personagem**.

O adaptador usa o SAM2 já fixado pelo Wan, com `obj_id=1`, `add_new_points` positivo/negativo, `add_new_mask` para quadros-chave corrigidos e `propagate_in_video`. As assinaturas dessas APIs foram conferidas no fonte SAM2 `0e78a118995e66bb27d78518c4bd9a3e95b4e266`. Rastreamento ocorre na sequência **nativa**, não no vídeo de 30 FPS. Frames/estado SAM2 são descarregados para RAM da CPU; o processo termina antes de carregar pose/Wan, liberando seus recursos. O editor e o rastreamento experimental limitam-se a 600 quadros nativos por cena. Esse limite não certifica que a RAM será suficiente.

Máscaras corrigidas podem ser usadas como novos quadros-chave ao executar SAM2 novamente. Não há interpolação entre máscaras. Mudança dos pontos do primeiro quadro que define o alvo descarta sementes antigas; alteração de um quadro-chave posterior invalida a semente daquele quadro. Correções de outros quadros permanecem preservadas.

Após aprovação nativa, o condicionamento de pose e rosto usa **somente pixels dentro da máscara escolhida** e sua bounding box, com ViTPose e utilitários oficiais. A seleção automática do detector da pessoa maior/central não é usada nesse modo. O cenário de `src_bg.mp4` vem da cena original, com apenas a máscara removida. Isso ainda precisa de verificação visual real de pose/rosto no Colab, especialmente sob oclusão.

## Bloqueios e limites reais

- Máscaras devem ser PNG binário opaco, uma por quadro nativo, na mesma resolução e ligadas por hash ao manifesto original.
- Máscara vazia ou acima de 65% do quadro é rejeitada. Entre quadros, IoU abaixo de 0,05, salto de centro acima de 0,25 da escala normalizada ou mudança de área maior que 3× são sinalizados. Pontos positivos precisam estar dentro da máscara e negativos fora, nos quadros onde foram marcados.
- Esses critérios são **heurísticas conservadoras**, não reconhecimento de identidade. Podem bloquear movimento legítimo/oclusão e podem deixar passar uma troca suave. Não há garantia de detecção de todas as trocas. Revisar todos os quadros é obrigatório; um alerta bloqueia a aprovação até a correção.
- Qualquer alteração de alvo/máscara invalida revisão, pré-processamento e recibo de geração. Sequência incompleta e interrupção do SAM2 não liberam aprovação. O fluxo não cai silenciosamente no modo automático quando existe uma seleção explícita inválida.
- O `manifest.json` original permanece byte a byte. A composição usa a máscara **do próprio quadro nativo**, mesmo quando o Wan repete uma imagem gerada de 30 FPS para dois quadros de uma fonte 60 FPS. Assim a máscara de outro instante não amplia a região editável.
- Fora da máscara nativa, todos os pixels dos PNGs originais são preservados. Personagens dentro de uma máscara incorreta não podem ser protegidos por essa garantia. O MP4 H.264 final tem perdas.

## Sincronização 24/30/60 FPS

`temporal_map.json` documenta cada seleção VOS → Wan e Wan → VOS usando frações exatas. O Wan recebe intervalos de 1/30 s. Cada amostra de condução seleciona um quadro original; na volta, cada quadro original recebe a amostra Wan correspondente ao centro do seu intervalo. Não são criadas poses ou imagens por interpolação.

Fontes de 24 FPS podem repetir imagens na condução. Em 60 FPS, alguns quadros não passam pelo modelo e a região editada recebe repetição temporal da saída Wan. A máscara nativa ainda acompanha cada quadro. O último intervalo Wan pode ultrapassar a duração da cena por menos de 1/30 s; a montagem restaura apenas os timestamps e a duração do manifesto original. O relatório registra o erro entre centros (até 1/60 s), não promete continuidade perceptual. Duração, cortes, dimensões e timestamps não são reescritos.

Testes CPU com duas pessoas sintéticas percorrem exportação VOS, seleção/correção, composição, ZIP, importação e MP4 completo em 24, 30 e 60 FPS. Comparam todos os pixels fora das máscaras e a integridade do manifesto. Isso comprova transporte/composição, **não qualidade de movimento Wan nem confiabilidade do SAM2**.

## Logs e interrupções

O diagnóstico continua antes de uploads e downloads. Falha de GPU/RAM/disco cria `VOS_GPU_DIAGNOSTICO.json`, `VOS_PREFLIGHT_FALHA.log` e `VOS_PREFLIGHT_LOGS.zip`. Não pule esse bloqueio.

Depois da preparação, `Session.run` captura stdout/stderr em arquivos numerados, comandos, status e traceback. Falha ou interrupção encerra o grupo de processos (inclusive filhos do gerador), gera `VOS_LOGS.zip` e tenta iniciar seu download. Um hook do notebook também captura falhas do SDK ModelScope/editor. Se o navegador impedir o download automático, use o painel Arquivos. O pacote inclui os sidecars de seleção/tempo/recibos, sem pesos, vídeos ou dump de variáveis de ambiente.

Após a revisão nativa, há download separado de `VOS_MASCARAS_REVISAO.zip`, com máscaras, manifesto e metadados. **Esse ZIP é backup de revisão, não um pacote de importação VOS**. A importação usa somente `VOS_EDITADO.zip` com imagens completas.

Uma desconexão total ou destruição da VM não pode disparar código/download depois de acontecer. Salve os artefatos antes da inferência e guarde uma cópia do notebook com saídas. O backup não implementa restauração automática de runtime; não promete retomada de uma geração interrompida.

## Otimizações legítimas de memória

O perfil padrão mantém Wan2.2 Animate 14B, resolução de condicionamento e comportamento anterior. `offload_model=True` e conversão para o dtype oficial já eram usados.

`LOW_MEMORY_EXPERIMENTAL=True` acrescenta apenas flags oficiais `--t5_cpu --frame_num 33`: o fonte mantém o encoder T5 na CPU e transfere embeddings, e reduz a janela temporal de 77 para 33 quadros. A saída oficial é truncada ao número real de quadros; o CLI não reduz a duração VOS. Janelas menores podem aumentar o número de transições e piorar continuidade; CPU pode aumentar RAM/tempo. `memory_profile.json` registra a opção.

Inspeção do fonte mostra que DiT, CLIP e VAE ainda são carregados na GPU inicialmente. Portanto essa opção **não resolve o requisito inicial de VRAM**, não permite ignorar o diagnóstico e não demonstra que T4 16 GB funcione. Não há quantização nova, modelo alternativo, API de inferência, GPU paga ou download pelo Hugging Face.

## Roteiro para esta noite

1. Abra [o notebook desta branch](https://colab.research.google.com/github/aruanmf446-hub/Vos/blob/codex/finalize-wan22-validation/colab/VOS_Wan22_Animate_Replacement.ipynb). Para reprodutibilidade, use no `VOS_REF` o SHA revisado do PR.
2. Execute o diagnóstico antes de enviar vídeo. Se bloquear, baixe os diagnósticos e encerre: nenhuma geração foi comprovada.
3. Exporte no VOS uma cena curta de 1–3 segundos, cadência Original, e guarde o ZIP. Use a mesma sessão/projeto no retorno.
4. Envie ZIP e referência, execute preparação e confira a condução. Ative `USE_TARGET_SELECTION=True` somente para testar este recurso.
5. No editor, marque positivo no alvo no quadro 0 e negativo na outra pessoa. Adicione pontos nos quadros-chave necessários e salve.
6. Instale/baixe conforme o notebook e aguarde os hashes dos pesos. Execute a célula SAM2. No editor de revisão, examine cada quadro, corrija com pincel/borracha e marque como revisado. Se alterar pontos ou houver alertas, salve e execute SAM2 novamente; reabra a revisão. Não aprove a pessoa errada para contornar um bloqueio.
7. Guarde `VOS_MASCARAS_REVISAO.zip`. Execute o pré-processamento e assista integralmente máscara, pose e rosto (`wan_processed/src_face.mp4`, pelo painel Arquivos se necessário). A pose/expressão precisam seguir o alvo, não a outra pessoa.
8. Confirme `SIM` somente após essa revisão. Faça a primeira inferência com `LOW_MEMORY_EXPERIMENTAL=False`. Só compare o perfil experimental depois de um teste controlado, no mesmo hardware/cena, guardando ambos os logs; não o use para forçar hardware bloqueado.
9. Assista ao MP4 realmente gerado. Execute `finish`; baixe `VOS_EDITADO.zip` e `VOS_LOGS.zip`. Importe no mesmo VOS, valide e revise os PNGs antes de aplicar. Monte o MP4 com FPS original e confira cortes, áudio e duração.
10. Registre GPU, RAM/disco, SHA VOS, revisão Wan/pesos, perfil, duração, eventuais picos/erros e aprovação visual. Teste primeiro sem cruzamento e depois uma cena com oclusão/cruzamento. Para 60 FPS, examine também os quadros entre as amostras de 30 FPS.

## Reproduzir sem GPU

```bash
pip install -r frame_studio/requirements.txt -r frame_studio/requirements-test.txt
python -m unittest discover -s frame_studio/tests -t . -v
npm install --prefix /tmp/vos-browser --no-package-lock playwright@1.62.1
/tmp/vos-browser/node_modules/.bin/playwright install --with-deps chromium
NODE_PATH=/tmp/vos-browser/node_modules node frame_studio/tests/target_editor_browser.cjs
```

Se houver Chromium de sistema, defina `VOS_CHROMIUM_PATH=/caminho/do/chromium` no último comando. O GitHub Actions executa os testes Python no Docker, o editor em Chromium e o healthcheck da imagem de produção. Testes SAM2 usam um preditor simulado explicitamente para verificar contrato/estado incompleto, não baixam pesos nem afirmam inferência real. O editor em Chromium local foi testado; integração do iframe com o notebook real do Colab continua sendo parte da validação manual.

## Preparação final do primeiro teste real

A ordem atual instala/audita e executa o smoke antes de uploads reais e usa referências imutáveis, controlador de etapas e recuperação de pontos/máscaras. As instruções atuais e três passos iniciais estão em [VOS_FIRST_REAL_TEST.md](VOS_FIRST_REAL_TEST.md); essa ordem substitui os roteiros anteriores de uploads antes da instalação.
