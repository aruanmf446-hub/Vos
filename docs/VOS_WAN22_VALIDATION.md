# Validação técnica — VOS / Wan2.2 Animate Replacement

Base: `main` em `d5b4ccdba4fb78d6075939bc14123fdb2f859431`. Branch: `codex/finalize-wan22-validation`. O aplicativo existente, seus endpoints, o histórico e as mídias anteriores permanecem como base; a alteração concentra-se na ponte, notebook, testes e CI. Não há merge automático nem implantação no Render nesta revisão.

## O que os testes demonstram

A suíte Python usa FFmpeg/FFprobe reais, Pillow, PyAV e o importador/montador VOS existentes. Testa manifesto, tempos racionais, cadência original, cenas, cortes, áudio, geometria, amostragem explícita, importações inválidas e limites do serviço. Os testes novos cobrem:

- compilação de todas as células Python e contrato de instalação/etapas do notebook;
- impedimento de instalação/download quando o diagnóstico de recursos falha;
- hashes e tamanhos de pesos locais, incluindo rejeição de arquivo truncado ou alterado (fixtures pequenas);
- máscara vazia, excessiva e salto grosseiro de rastreamento;
- formato oficial VitPose em pasta ONNX com dados externos (pré-processador simulado explicitamente no teste);
- confirmação vinculada aos intermediários e invalidação de resultados após falha;
- variáveis offline nos subprocessos Wan;
- comparação de **todos** os pixels fora da máscara de uma fixture, manifesto byte a byte e mesmos nomes/dimensões;
- exportação real de uma cena VOS → preparação → composição com MP4s sintéticos de teste → importação validada → MP4 completo com 18 quadros, 160×96, 3/5 s, cortes em 1/5 e 2/5 s e áudio.

Os MP4s coloridos e as máscaras desenhadas são fixtures de transporte/composição. Não foram gerados pelo Wan e não demonstram qualidade de substituição, atuação ou desempenho de GPU. Os testes de subprocesso usam mocks para não executar modelos.

O código oficial Wan foi inspecionado na revisão fixada. Argumentos de preprocessamento e geração, nomes dos intermediários, FPS de saída, polaridade branca da máscara, requisitos Animate e caminhos dos checkpoints foram conferidos no fonte. A listagem ModelScope na revisão `4cfe61a9` respondeu com sucesso; os 413 caminhos/tamanhos/SHA-256 selecionados foram conferidos nessa listagem. O arquivo pequeno `config.json` também foi baixado diretamente dessa revisão ModelScope, e tamanho/SHA-256 coincidiram com o manifesto. Isso não equivale a baixar ou carregar os pesos.

## Resultado local registrado

- Suíte final Python: **55 testes, todos passaram**, incluindo transporte VOS → ponte com fixtures sintéticas → importador → MP4 completo com áudio/cortes.
- Notebook: todas as células Python compiladas, quatro testes de contrato/bloqueios passaram, sem saídas de geração anexadas.
- Workflow YAML: estrutura verificada sem chaves duplicadas; `git diff --check` passou.
- O resultado das construções Docker e da execução remota final fica nos checks do PR. Um teste local anterior de 53 testes já passou no Docker; a imagem final inclui 55 testes.

## Reproduzir as verificações sem GPU

Na raiz do repositório, Python 3.12 e FFmpeg/FFprobe:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --only-binary=:all: -r frame_studio/requirements.txt -r frame_studio/requirements-test.txt
python -m unittest discover -s frame_studio/tests -t . -v
git diff --check
docker build --target test -t vos-frame-studio:test .
docker build --target production -t vos-frame-studio .
docker run -d --name vos-check --memory=512m --cpus=1 -p 10000:10000 vos-frame-studio
curl --fail http://localhost:10000/healthz
docker logs vos-check
docker rm -f vos-check
```

No ambiente Cloud usado nesta revisão, pip dentro do Docker precisou da CA do proxy. O Dockerfile já oferece `--secret id=build_ca,src=/etc/ssl/certs/ca-certificates.crt`. Esse segredo é montado somente durante a construção; a verificação TLS permanece ligada. Em ambiente sem esse proxy, use os comandos normais acima.

O workflow `.github/workflows/frame-studio.yml` executa os testes do notebook, a suíte inteira no estágio Docker `test` e uma imagem de produção com 512 MiB/1 CPU e `/healthz`. Executa em PRs, pushes `codex/**` e por `workflow_dispatch`; o checkout esparso evita baixar as mídias históricas grandes. Consulte a execução associada ao commit final do PR, não apenas uma execução antiga da `main`.

## O que exige Colab/GPU real

Não há CUDA/PyTorch/GPU disponível no ambiente de verificação desta revisão. Permanecem sem comprovação: instalação completa do Wan/SAM2/FlashAttention na imagem atual do Colab, download integral e hashes reais dos pesos, providers ONNX, consumo máximo de VRAM/RAM/disco, tempo de execução, pré-processamento real, identidade e estabilidade da máscara, inferência Animate Replacement e qualidade visual resultante. Os limiares 32 GiB VRAM / 24 GiB RAM disponível / 85 GiB disco livre são preventivos, não requisitos mínimos certificados.

O Colab gratuito pode fornecer apenas T4 16 GB ou não disponibilizar GPU. Nessa situação o diagnóstico deve bloquear antes de baixar pesos. Não existe resultado de geração e a validação GPU fica **bloqueada**, sem alternativa paga, Hugging Face, outro modelo ou API de inferência. Executar à noite não garante disponibilidade ou capacidade maior.

Preservação exata vale para os PNGs fora da máscara usada na composição. Em fontes 60 FPS/VFR, o Wan continua trabalhando a 30 FPS; mapear a máscara pelo tempo pode não acompanhar perfeitamente a silhueta entre amostras. Isso precisa ser revisado quadro a quadro. A máscara automática é para uma pessoa; a cena com vários personagens não está tecnicamente resolvida apenas porque o ZIP passa. O MP4 final H.264 tem perdas e não conserva pixels exatos como os PNGs.

## Acesso rápido — sem upload nem instalação

**Prioridade:** abra o notebook da branch do PR com GPU gratuita selecionada e execute as **duas primeiras células de código** (importações leves e diagnóstico, antes do upload). A primeira verificação grava `/content/VOS_GPU_DIAGNOSTICO.json` e imprime GPU, VRAM, RAM livre, disco e BF16. Se o bloqueio impedir prosseguir, **não envie vídeo e não baixe nenhum peso**: guarde apenas o diagnóstico. O mesmo diagnóstico roda novamente antes da instalação. Esses limiares são preventivos, não mínimos certificados.

## Procedimento exato de validação noturna

1. **Preparar e guardar:** no VOS/Render já existente, crie o projeto com o vídeo real. Para a primeira tentativa escolha 2–3 segundos, uma única cena, preferencialmente uma única pessoa visível e sem corte interno. Exporte com **Original — todos os quadros**. Guarde o ZIP original, o vídeo-fonte e a imagem do personagem fora do Render. Mantenha o mesmo navegador/sessão/projeto; hibernação/reinício pode apagar o projeto.
2. **Escolher a revisão:** abra [este notebook da branch do PR](https://colab.research.google.com/github/aruanmf446-hub/Vos/blob/codex/finalize-wan22-validation/colab/VOS_Wan22_Animate_Replacement.ipynb), não a versão antiga da main. Confira `VOS_REF='codex/finalize-wan22-validation'`. Se a branch tiver recebido novos commits desde a revisão, use em `VOS_REF` o SHA exato aprovado no PR. Não use “Executar tudo”: execute célula por célula para inspecionar cada resultado.
3. **Runtime gratuito:** selecione Ambiente de execução → Alterar tipo de ambiente → GPU disponível no plano gratuito. Não contrate plano nem GPU. Anote GPU, VRAM, RAM disponível, disco e horário local (America/Sao_Paulo), e salve uma cópia do notebook para manter os logs.
4. **Preparar sem IA:** depois que o diagnóstico inicial passou, envie o ZIP e a referência. Execute `prepare` e assista ao `driving.mp4`: cena, pessoa, enquadramento e movimento devem corresponder. A pasta de trabalho deve estar vazia; para reiniciar, use um runtime limpo e repita os uploads.
5. **Diagnóstico:** execute o bloqueio de recursos. Se falhar, registre a saída e **encerre a tentativa GPU**. Não pule o bloqueio e não trate esse evento como geração. O teste noturno termina com “bloqueado por recursos”.
6. **Instalar e baixar:** somente com diagnóstico aprovado, instale as dependências fixadas e baixe os arquivos selecionados via ModelScope. Espere `verify-weights` concluir todos os tamanhos e SHA-256. Guarde revisão Wan, revisão dos pesos e `pip freeze` impressos. Qualquer erro de instalação/download/hash bloqueia as etapas posteriores; guarde o traceback inteiro.
7. **Pré-processar:** execute `preprocess`. Todos os quatro vídeos precisam ter a contagem esperada e 30 FPS, com os timestamps corretos. Guarde `mask_report.json`. Assista integralmente `driving.mp4`, `src_mask.mp4` e `src_pose.mp4`, pausando em cruzamentos, oclusões, mãos/rosto e movimento rápido. Branco é área que será sobrescrita. Se incluir outra pessoa, perder o alvo ou saltar, interrompa e recorte/reexporte uma cena adequada. Uma confirmação não corrige rastreamento errado.
8. **Confirmar e gerar:** responda **SIM** somente depois da inspeção. O notebook grava `mask_review.json` vinculado aos arquivos revisados. Execute `infer`. Aguarde o término sem exceção e a criação de `generation.json`; OOM/queda/desconexão não produz recibo válido. Guarde o log, a duração e o pico de recursos observado no Colab.
9. **Conferir a geração:** reproduza `wan_generated.mp4` inteiro antes de terminar. Confira identidade do personagem, gestos, expressão, movimentos de câmera, cenário e pessoas não selecionadas. Se reprovado, guarde os arquivos como tentativa rejeitada; não apresente como entrega aprovada.
10. **Recompor e baixar:** execute `finish`, que valida hashes, contagem, FPS, timestamps, proporção e máscaras. Baixe `VOS_EDITADO.zip`. Pelo painel Arquivos do Colab, baixe também `bridge.json`, `preprocess.json`, `mask_report.json`, `mask_review.json`, `generation.json`, os vídeos intermediários/gerado e a cópia do notebook com os logs. Não acrescente esses arquivos ao ZIP importado pelo VOS.
11. **Retornar ao VOS:** no mesmo projeto/sessão original, envie `VOS_EDITADO.zip`, valide, compare os PNGs, confira quantidade/resolução e aplique somente após revisar. Se o projeto expirou, não altere o manifesto para forçar a importação: reinicie o ciclo com um novo pacote.
12. **MP4 final:** monte o vídeo completo com FPS original e áudio preservado. Baixe MP4 e relatório. Confira decodificação completa, duração, resolução, contagem, cortes e sincronismo de áudio no relatório. Assista ao vídeo completo, especialmente às entradas/saídas da cena e a todos os personagens não selecionados. Para 60 FPS/VFR, examine os quadros intermediários por causa do mapeamento a 30 FPS.
13. **Registrar o resultado:** use “aprovado”, “rejeitado visualmente” ou “bloqueado por recursos/instalação/inferência”, incluindo commit VOS, revisão Wan/pesos, hardware, tempos, logs e arquivos. Só marque geração real como comprovada quando houver `wan_generated.mp4` produzido pelo modelo, recibo, ZIP importado e MP4 final revisado.

## Continuação: alvo específico

Os resultados de 55 testes acima pertencem à primeira revisão. A continuação preserva o diagnóstico antecipado adicionado depois e implementa seleção opcional, máscaras nativas, testes 24/30/60 FPS, logs e perfil experimental. Veja [seleção explícita e roteiro atualizado](VOS_TARGET_SELECTION.md). Os checks do commit atual do PR registram a suíte ampliada; nenhum teste simulado demonstra geração real.
