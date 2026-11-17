# Verificações do Vos Frame Studio

## Executar a suíte

Use Python 3.12 e FFmpeg/FFprobe, na raiz do repositório:

```bash
pip install --only-binary=:all: -r frame_studio/requirements.txt -r frame_studio/requirements-test.txt
python -m unittest discover -s frame_studio/tests -t . -v
```

Para executar a mesma suíte no ambiente que será publicado:

```bash
docker build --target test -t vos-frame-studio:test .
docker build --target production -t vos-frame-studio .
```

O estágio de testes roda como usuário sem privilégios. A imagem de produção não instala HTTPX nem Playwright. O workflow `.github/workflows/frame-studio.yml` repete a suíte Docker e verifica a saúde da imagem de produção com limite de 512 MB.

## Cobertura

As fixtures são vídeos reais curtos gerados com FFmpeg a partir de PNGs numerados visualmente. O encoder da fixture é independente do montador PyAV em teste. Os testes conferem conteúdo decodificado, não apenas existência do MP4.

- Roundtrip sem edição: todos os quadros, resolução, áudio, duração, três cenas e dois cortes.
- Edição artificial de um quadro: apenas essa posição muda no conteúdo decodificado.
- Exportação/importação de cena isolada e recolocação no vídeo inteiro; edições anteriores fora do intervalo permanecem.
- ZIP com frame ausente, repetido, extra, renomeado, resolução errada, transparência, manifesto alterado ou pertencente a outro projeto.
- Membros em ordem invertida e pasta externa: ordem do manifesto preservada e normalização informada.
- ZIP inválido, path traversal e link simbólico bloqueados.
- 30 FPS, 60 FPS com repetição temporal, 29,97 FPS, vídeo VFR e amostragem explícita de 15 FPS.
- Primeiro/último quadro, duração exata, cortes fora da grade do FPS e ausência de lacunas na composição.
- SAR não quadrado, dimensões ímpares, rotação e PTS inicial deslocado com áudio.
- Autenticação opcional, CSRF, isolamento de projetos/artefatos/tarefas entre sessões, downloads HTTP Range.
- Limites de upload/duração/disco, limpeza por expiração, cancelamento, timeout que encerra o subprocesso e bloqueio de tarefas concorrentes no mesmo projeto.

## Navegador e reprodução normal

O teste opcional `frame_studio/tests/browser_flow.py` usa Chromium com Playwright. Não é dependência do servidor. Para reproduzir em uma máquina com Chromium instalado em `/usr/bin/chromium`:

```bash
pip install playwright
APP_PASSWORD=teste-ui COOKIE_SECURE=false uvicorn frame_studio.app:app --host 127.0.0.1 --port 8765
```

Em outro terminal:

```bash
STUDIO_URL=http://127.0.0.1:8765 STUDIO_PASSWORD=teste-ui \
  BROWSER_RESULTS=/tmp/vos-browser-results python frame_studio/tests/browser_flow.py
```

O teste verifica senha incorreta/correta, upload, cortes manuais, ZIP de cena, edição real de um PNG, comparação, aplicação, prévia de 0,2 s, montagem completa em 60 FPS e reprodução com `playbackRate=1` até o evento de término. Também salva capturas desktop/mobile, mede overflow horizontal e captura erros JavaScript. Não desabilita a CSP do aplicativo.

## Resultado observado nesta implementação

Na execução local: **39 testes aprovados**. O navegador completou o fluxo: 18 quadros de origem a 30 FPS, três cenas, uma imagem alterada, cena com 6 quadros e 0,2 s; vídeo completo com 36 quadros em 60 FPS e 0,6 s. Reprodução normal concluída, zero erros JavaScript e largura mobile de 390 px sem overflow.

No Docker configurado para o Render: **39 testes aprovados**, com Python 3.12, FFmpeg 5.1.9 e PyAV 19.0.2, executados como usuário `vos`, sem privilégios. A imagem de produção iniciou e passou no health check.

Também foi concluído um fluxo HTTP real pela imagem de produção com o vídeo fornecido no chat: **778 quadros, 576 × 854, 25,933333… s**, nove cenas configuradas, exportação de uma cena, alteração de um único PNG e importação validada. As montagens completas nativa (778 quadros) e 60 FPS (1.556 quadros) passaram em todos os timestamps, cortes, assinaturas e decodificação completa, com áudio. Em seguida, um segundo vídeo independente foi enviado e montado no mesmo serviço.

A rede deste ambiente de teste exige uma CA específica. O build foi executado com `--secret id=build_ca,src=/etc/ssl/certs/ca-certificates.crt`. Esse segredo de build é opcional, não é copiado para a imagem e não desabilita a verificação HTTPS. No Render e no GitHub Actions, o comando padrão usa as CAs normais da imagem.

A URL Render precisa de verificação separada após conexão da conta: nenhum teste local equivale a um deploy público. Não havia conexão autorizada com uma conta Render neste ambiente.
