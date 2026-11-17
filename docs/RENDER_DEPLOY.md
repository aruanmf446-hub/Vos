# Publicar Vos Frame Studio no Render

O repositório contém frontend e backend em um único serviço Web Docker. Não há GitHub Pages, banco, Redis, worker separado, armazenamento externo ou serviço de IA.

## Blueprint recomendado

1. Entre em https://dashboard.render.com com a sua conta.
2. Escolha **New → Blueprint** e autorize o Render a acessar `aruanmf446-hub/Vos`.
3. Selecione o repositório e a branch **main**, usando `render.yaml` na raiz.
4. Confira que há **somente um Web Service**, `vos-frame-studio`, com plano **Free**. Confirme a criação.
5. Em **Environment**, adicione opcionalmente `APP_PASSWORD` com uma senha privada. `APP_SECRET` será gerado pelo Blueprint; mantenha-o estável e secreto.
6. Aguarde o build e o deploy. Abra a URL `https://<nome-atribuido>.onrender.com` mostrada pelo próprio Render. Não existe uma URL pública reservada pelo código.
7. Abra `/healthz` nessa URL: deve responder `status: ok`. Faça um envio curto, extraia o ZIP, importe-o sem editar e monte o MP4 antes de usar um projeto importante.

O botão de criação também pode abrir o Blueprint: [Deploy to Render](https://render.com/deploy?repo=https://github.com/aruanmf446-hub/Vos). A criação e a conexão da conta precisam ocorrer no painel autorizado.

## Configuração manual equivalente

Em **New → Web Service**, conecte o mesmo repositório e configure:

| Campo | Valor |
| --- | --- |
| Branch | `main` |
| Root Directory | vazio, raiz do repositório |
| Language / Runtime | Docker |
| Dockerfile Path | `./Dockerfile` |
| Docker Build Context | `.` |
| Instance Type | Free |
| Health Check Path | `/healthz` |
| Auto Deploy | após commit em main |
| Build / Start Command | usar os comandos do Dockerfile |

Defina `APP_SECRET` com um valor aleatório privado de pelo menos 32 bytes, `COOKIE_SECURE=true` e, se desejar, `APP_PASSWORD`. O Render fornece `PORT`; o Dockerfile usa essa porta, com fallback local 10000. Não configure outros serviços.

## Limites configuráveis

| Variável | Padrão | Significado |
| --- | --- | --- |
| `DATA_DIR` | `/tmp/vos-frame-studio` | arquivos temporários |
| `MAX_VIDEO_MB` | 100 | tamanho máximo do vídeo enviado |
| `MAX_ZIP_MB` | 512 | tamanho máximo do ZIP enviado |
| `MAX_DURATION_SECONDS` | 120 | duração máxima da origem |
| `MAX_FRAMES` | 6000 | quantidade máxima de quadros decodificados |
| `MAX_PIXELS` | 2073600 | largura × altura máxima |
| `MAX_PROJECT_MB` | 1024 | limite de arquivos por projeto |
| `MAX_TOTAL_MB` | 2048 | limite de arquivos de todas as sessões |
| `MIN_FREE_MB` | 128 | reserva mínima de espaço livre |
| `JOB_TIMEOUT_SECONDS` | 300 | tempo máximo de uma tarefa |
| `PROJECT_TTL_SECONDS` | 7200 | expiração após inatividade |
| `MAX_QUEUED_JOBS` | 4 | tarefas admitidas, incluindo a em execução |

Esses valores são tetos, **não capacidade garantida**. PNGs completos podem ocupar muito mais que o vídeo comprimido. No plano gratuito, prefira cenas curtas; diminua os limites se necessário. O processamento é serial, com um processo Uvicorn e uma fila em memória. Não aumente `workers`.

O Render gratuito pode hibernar; a primeira abertura pode demorar. Reinício, deploy ou hibernação pode remover os temporários. O aplicativo também os apaga após o TTL. **Não há recuperação garantida**. Guarde os resultados fora do serviço. O ZIP de frames não inclui o áudio nem restaura sozinho uma sessão perdida: o vídeo original e o manifesto canônico precisam continuar disponíveis no projeto.

Se o Render encerrar o container inteiro por falta de memória, o processo não consegue mostrar uma mensagem antes de morrer. Na próxima visita, será necessário reenviar o material. Os limites e a fila reduzem esse risco, mas não substituem memória disponível. Erros detectados de espaço, duração, tempo, mídia ou validação aparecem no aplicativo sem liberar um MP4 incompleto.

## Acesso e downloads

Os projetos pertencem a uma sessão de navegador com cookie privado. Saber um ID não permite baixar arquivos de outra sessão. A senha opcional restringe a entrada; ela não reúne os projetos de usuários diferentes. Use o mesmo navegador para voltar ao projeto. Downloads são servidos pela mesma URL, com suporte a pedidos HTTP Range.

## Verificação após deploy

Faça o fluxo completo com um vídeo seu, teste outro vídeo, confira a comparação de frames e reproduza o MP4. Os testes locais/Docker não substituem essa verificação da URL pública. Se houver falha de build, consulte **Logs**; se um projeto expirar, envie-o novamente. Não altere o plano para pago para concluir este setup.
