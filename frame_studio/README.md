# Vos Frame Studio

Extrator e montador de quadros para um único Web Service Docker gratuito no Render. Frontend HTML/CSS/JS e backend FastAPI na mesma origem. Sem IA interna ou dependências de serviços pagos.

## Executar localmente

Python 3.12 e FFmpeg/FFprobe no PATH:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --only-binary=:all: -r frame_studio/requirements.txt
uvicorn frame_studio.app:app --host 127.0.0.1 --port 10000
```

Abra http://localhost:10000. No Windows, ative o ambiente com `.venv\Scripts\activate`.

Com Docker:

```bash
docker build -t vos-frame-studio .
docker run --rm --name vos-frame-studio -p 10000:10000 --memory=512m --cpus=1 \
  -e PORT=10000 -e COOKIE_SECURE=false -e APP_PASSWORD=troque-esta-senha \
  vos-frame-studio
```

Dados locais e do container também são temporários por padrão. Não coloque credenciais no repositório. Em HTTPS, use `COOKIE_SECURE=true`; o Blueprint já configura isso.

## Fluxo de uso

1. Envie o vídeo e aguarde a leitura de **todos os timestamps decodificados**.
2. Avance quadro a quadro, marque início/fim e adicione cortes. O índice de fim é exclusivo. A detecção automática apenas sugere cortes para revisão.
3. Escolha vídeo completo, uma cena, um trecho ou todas as cenas em pastas. Mantenha **Original — todos os quadros** para não amostrar.
4. Baixe e guarde o ZIP. Envie-o ao editor externo com as referências e leia `INSTRUCOES_EDICAO.md`. Não há promessa de geração de centenas de imagens de uma vez.
5. Volte ao **mesmo projeto e sessão do navegador**, envie o ZIP devolvido e confira a validação e as imagens lado a lado. Aplique o pacote somente após revisar.
6. Monte a cena, o trecho ou o vídeo completo. As regiões fora do pacote continuam usando os originais, ou edições anteriores já aplicadas. O download só é liberado após a validação completa.
7. Baixe MP4, relatório e ZIP antes de sair. Para outro vídeo, clique em **Outro vídeo**.

Uma imagem faltando bloqueia o pacote inteiro: o aplicativo não preenche faltas. Para editar apenas uma cena, exporte essa cena e devolva **todos** os seus PNGs, incluindo os não modificados.

## Documentação

- [Deploy gratuito no Render](../docs/RENDER_DEPLOY.md)
- [Manifesto, temporização e limitações](../docs/VOS_FRAME_STUDIO.md)
- [Testes e reprodução das verificações](../docs/FRAME_STUDIO_TESTS.md)

Os projetos antigos continuam nas próprias pastas. Este aplicativo não importa suas rotinas de deformação, interpolação, composição de personagens ou fluxo óptico.

## Wan2.2 Animate Replacement via Colab

Novo fluxo opcional e externo de ida e volta ZIP, sem alterações de endpoints ou montagem: [guia do VOS e Colab](../docs/VOS_WAN22_COLAB.md). O hardware do Colab gratuito pode impedir geração.
