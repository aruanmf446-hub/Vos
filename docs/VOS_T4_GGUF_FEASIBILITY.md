# Wan2.2 Animate 14B GGUF Q3_K_M no Colab gratuito

**IMPLEMENTADO MAS SEM VALIDAR EM T4.** Não houve inferência, instalação CUDA nem medição de pico de memória neste ambiente sem GPU. Este fluxo é independente: o notebook baseline permanece byte a byte intacto e seu bloqueio BF16/32 GiB continua válido para aquele motor. Nenhuma alteração de produção no Render.

Notebook: [VOS_Wan22_Animate_GGUF_T4_EXPERIMENTAL.ipynb](../colab/VOS_Wan22_Animate_GGUF_T4_EXPERIMENTAL.ipynb). Perfil `LOW_VRAM_GGUF_EXPERIMENTAL=False` por padrão. Não precisa de GPU local; não contrate GPU se o Colab gratuito não disponibilizar recursos.

## Evidência verificável, sem geração

- A API de arquivos do ModelScope confirmou todos os caminhos, revisões, tamanhos e hashes do manifesto [wan22_t4_gguf_manifest.json](../colab/wan22_t4_gguf_manifest.json). Downloads usam somente `model_file_download` do ModelScope; não usam snapshots integrais, Hugging Face nem espelhos.
- Uma requisição HTTP Range ao **ModelScope** retornou apenas os primeiros 2 MiB do Q3_K_M. [Evidência do cabeçalho](VOS_T4_GGUF_HEADER_EVIDENCE.json): GGUF v3, arquitetura `wan`, 1.441 tensores, `pose_patch_embedding` e `face_adapter`, tipos Q3_K e BF16 de armazenamento. Isso identifica componentes Animate, mas **não verifica o SHA do arquivo inteiro nem sua qualidade**. O download real deve conferir tamanho 8.630.769.472 e SHA-256 `62cd6067c96599ade5bdde54a49a5ef1a7fd4de449eaae56ff4fd61ccbb16678` antes de carregar.
- Corrigido também um erro preexistente na passagem de prompts SAM2: uma função duplicada devolvia `(pontos, labels)` e o chamador entregava a tupla inteira como coordenadas. Agora separa corretamente pixels Nx2 e labels; teste de contrato rejeita a forma errada.
- Auditoria AST executada nos fontes reais pinados confirma assinaturas dos nodes chamados, entradas Animate, FP16, SDPA e decodificadores Q3/BF16-storage. Não equivale a importar ou executar os nodes em CUDA.
- ComfyUI `10edeb91317107db6fe1184c16ec4a6e9608cc9e` (GPL-3.0), WanVideoWrapper `a7439db30bed0c5b5fa437d77ae8a7bb220dfd00` (Apache-2.0), ComfyUI-GGUF `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` (Apache-2.0), Wan oficial `1ea34ff48f87168174e12956e200b1d908b1c5ff` (Apache-2.0). URLs e pins estão no manifesto. Mantidas as licenças dos projetos; pesos continuam sujeitos à licença de cada repositório de origem.
- O relato externo com **duas T4/32 GB RAM** citado no estudo inicial não comprova uma T4/Colab gratuito. O motor aqui não usa nós multigpu nem o workflow de downloads daquele relato.

## Caminho implementado e memória

O processo usa ComfyUI como biblioteca local, sem servidor, túnel ou API. DiT é o mesmo Wan2.2 Animate 14B quantizado Q3_K_M. LoRA oficial de relighting FP16 é aplicada sem merge; não há distillation LoRA nem substituição por I2V/T2V/5B.

T4 tem capability 7.5 e não possui BF16 nativo. DiT calcula em **FP16**; SDPA **efficient** é exigido durante sampling, FlashAttention e xFormers ficam desativados. O primeiro diagnóstico executa kernels reais FP16/FP32 com o PyTorch instalado, antes de uploads/downloads. Depois da instalação, outro subprocesso importa os motores reais e testa dequantização Q3_K CUDA com bloco sintético, antes dos pesos. Falhar em qualquer probe bloqueia a tentativa; não existe fallback indiscriminado para atenção matemática no DiT. Preview `auto` escolhe o callback core: evita o preview customizado que pressupõe servidor ComfyUI ativo.

Texto UMT5 GGUF (FP32) → condicionamento CLIP/VAE → DiT FP16 → decode VAE rodam em **quatro processos distintos**. SAM2 FP16 e pose/rosto também saem da memória antes da geração. VAE armazenado BF16 é convertido para FP32 e usa tiles; não exige cálculo BF16 no VAE. O GGUF também contém tensores BF16 de armazenamento e o carregador os dequantiza/converte. A eficiência numérica e de memória de todas essas conversões ainda exige GPU real. SAM2 pinado constrói extensão CUDA: NVCC deve existir; sua ausência bloqueia sem baixar pesos.

Offload de 30 blocos DiT para CPU, sem prefetch nem transferência assíncrona, evita deslocar todos os blocos ao mesmo tempo. O sampler não tenta mover o modelo inteiro à RAM ao terminar: o processo encerra antes do decode. Isso reduz sobreposição, **não certifica** que cabe em 12–13 GB de RAM. Quantização não significa VRAM igual ao tamanho do arquivo.

Pesos selecionados: **18.149.539.986 bytes (16,903 GiB)**, incluindo pré-processamento. Gate inicial: 33 GiB livres de disco (mais 16 GiB de reserva no download), 10 GiB RAM disponível, 14 GiB VRAM total/13 GiB livre. Depois dos pesos, inferência exige mais 8 GiB livres. São limites conservadores de tentativa, não mínimos medidos/certificados. Pip preserva versões Torch/vision/audio do Colab; não instala FlashAttention2. Não é possível prometer que a combinação de pacotes se instalará em toda imagem futura do Colab: imports/kernels reais bloqueiam antes dos pesos.

## Integridade VOS e tempo

O primeiro experimento limita-se a **uma cena de até 1 segundo**. Reutiliza seleção, correção/revisão nativa, pré-processamento e `finish` existentes. Não aceita alvo sem revisão nem intermediários alterados. Máscara branca significa área substituída; o node Animate inverte internamente somente para condicionar o fundo.

O stage mantém a grade Wan de 30 FPS definida pela ponte existente. Condições são preenchidas até 33 quadros repetindo o último (contrato `4n+1` do node), numa única janela; depois do decode devem existir exatamente 33 quadros, sem referência extra. Remove-se exclusivamente a cauda de preenchimento, sem interpolação. A ponte existente faz a correspondência por timestamps para origens 24/30/60 FPS. Testes sintéticos desses FPS exercitam transporte/reconstrução, **não geração Wan**.

Resolução interna tem no máximo 320 px, múltiplos de 16, com letterbox e inversão do mesmo recorte. O resultado volta ao tamanho do stage antes do `finish`. O ZIP final mantém `manifest.json` original byte a byte, frames/timestamps nativos e pixels fora da máscara. A qualidade dentro da máscara pode cair por quantização e resolução interna reduzida. Contagem/FPS/dimensões incompatíveis, saída não finita, checksum inválido e recibo inconsistente abortam; nenhuma saída antiga é reutilizada após falha.

## Recuperação e diagnóstico

O notebook oferece downloads de pontos e máscaras pelo exportador existente. Guarde também ZIP original e referência no computador. Não há promessa de retomada da inferência: um runtime perdido exige reinstalar/recarregar e gerar novamente. Backups de progresso não são ZIPs de importação final. A sessão/projeto temporário do Render pode expirar ou desaparecer em reinício; mantenha o projeto original aberto e não altere manifesto para forçar importação em outro projeto.

Falhas produzem `VOS_LOGS.zip`, pequeno, com logs sanitizados, versões, recursos, etapa interrompida e `low_vram_report.json`; não inclui pesos, vídeos privados, máscaras ou credenciais. Sem RAM/disco, falha abrupta do kernel ou desconexão total podem impedir o último download: os backups prévios são necessários. O relatório nunca marca prova T4 automaticamente; recibo real só nasce após execução e decode completos.

## Validação noturna e decisão

1. Abra o notebook experimental no **Colab gratuito**, selecione runtime GPU e execute somente o diagnóstico. Baixe `low_vram_report.json`. Sem GPU, RAM, disco, NVCC ou kernel eficiente, **pare**; não altere os gates. Um ambiente elegível permite uma tentativa, não prova sucesso.
2. Ative conscientemente a caixa experimental e execute instalação/probe; só então baixe pesos verificados. Se imports, dependências ou kernels falharem, guarde `VOS_LOGS.zip` e pare. Prepare um projeto novo VOS de uma cena até 1 s, preferencialmente duas pessoas; guarde ZIP/ref e faça seleção/revisão/backup das máscaras.
3. Execute pré-processamento, revise máscara/pose/rosto, tente a geração e importe o ZIP somente após sucesso. Compare personagem selecionado, rosto, outras pessoas, cenário, FPS/duração/cortes e áudio no VOS. Guarde relatório, recibo, logs e resultado visual. Uma execução finita sem OOM, no hardware documentado, e validação visual humana são necessárias para considerar viabilidade; não anunciar compatibilidade geral com T4 a partir de mocks.

Pendências exclusivas da GPU: instalação/imports completos na imagem Colab, kernels reais incluindo SAM2, carregamento integral e hashes dos pesos, pico de RAM/VRAM/disco, tempo dentro da sessão gratuita, ausência de OOM/NaN, contrato real de 33 quadros e qualidade/identidade visual. Nenhum impedimento incontornável foi demonstrado por leitura de código; tampouco foi comprovada viabilidade de memória em uma T4. O teste real pode revelar um impedimento nessa configuração.
