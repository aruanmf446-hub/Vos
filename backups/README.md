# Backups editáveis preservados

Os ZIPs estão divididos em partes de até 48 MiB, para caberem no próprio Git sem depender do servidor de anexos do GitHub Releases. Não abra uma parte isolada como ZIP.

- `checkpoint-editavel.zip`: 1.010.177.902 bytes, 4.229 arquivos de trabalho e o manifesto interno. Inclui o projeto principal e os quadros intermediários.
- `entrega-e-projeto.zip`: 143.064.178 bytes, 377 arquivos e o manifesto interno. Inclui código, documentação, referências, imagens geradas e vídeos.

## Recuperar no computador

Depois de clonar ou baixar este repositório, execute:

```bash
python scripts/recuperar_zip.py --dest /caminho/backup
```

Para o pacote menor:

```bash
python scripts/recuperar_zip.py --arquivo entrega-e-projeto.zip --dest /caminho/backup
```

O script confere cada parte e o SHA-256 do ZIP final. Para baixar somente o script e recuperar as partes pela internet, acrescente `--baixar`. Downloads já concluídos e válidos são reaproveitados em uma nova tentativa. Precisa de Python 3; não precisa instalar bibliotecas.

O checkpoint pode então ser usado por `scripts/restaurar_workspace.py`. Leia [restauração](../docs/RESTAURACAO.md).

## No celular

O [MP4 atual](https://raw.githubusercontent.com/aruanmf446-hub/Vos/main/media/videos/video-marvel-continuidade-v2-60fps.mp4) está disponível separadamente, sem precisar abrir ZIP. Para preservar tudo, o botão **Code → Download ZIP** do GitHub baixa uma cópia do repositório com todas as partes; a reconstrução do checkpoint é feita depois no computador.

Os pacotes refletem o primeiro commit de preservação (`e03fe68`). Os utilitários de recuperação das partes e estas instruções foram adicionados depois, sem alterar os ZIPs ou seus hashes.
