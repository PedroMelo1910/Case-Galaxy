# Limitações e desvios declarados

Esta lista é para ser lida junto com os [resultados](resultados.md). Nada aqui afirma conformidade integral ao TAPI nem promete um percentual de acerto.

## Desvios em relação ao TAPI

1. **Camada 1 local (TAPI §6.3) não atendida; desvio declarado e não resolvido.** A Camada 1 "mínima, obrigatória, 100% offline" prevê frames, SigLIP ou CLIP, faster-whisper, OCR, rosto, YOLO, OpenCV e detecção de cortes sobre o vídeo. O projeto não baixa vídeo do YouTube (D-B2) e não obteve os arquivos originais dos Shorts da base (a investigação de 07/10 não achou fonte adequada). A rota disponível é **capa local (CLIP, OCR, rosto, cor) + legenda local (e5) + Gemini por URL pública** (rubrica e transcrição). A ablação mede só esses atributos e **não** é avaliação da Camada 1.
2. **Gemini no lugar do Whisper.** A transcrição vem do Gemini e foi **validada só qualitativamente**, em exemplos inspecionados pelo autor, sem taxa de erro. Estados: `ok`, `sem_fala`, `incerta` (tratada como ausente) e `falha_extracao`.
3. **Validação temporal retrospectiva aproximada.** As curtidas do histórico e dos rótulos foram lidas na coleta, não na data simulada. A sensibilidade (histórico com 90 dias de defasagem; intervalo treino–teste de 90 dias) ficou inconclusiva contra o principal; é compatível com efeito pequeno da aproximação e não é prova de ausência.
4. **Sem conformal.** A faixa é de quantis (sem CQR); o TAPI trata o conformal como extensão.
5. **Sem MLflow.** O registro de experimentos é feito por manifestos com SHA-256 (ver [registro_de_experimentos.md](registro_de_experimentos.md)).
6. **Base abaixo da meta do briefing** (16 canais; 82% automotivo; beleza com 4 canais), com a exceção de país para três canais.

## Limitações do modelo e da avaliação

- Vantagem clara só sobre a média aritmética. Sobre a mediana é pequena; sobre a média móvel exponencial, inconclusiva em erro. Em canais não vistos o MAE sobe para 1,642 e a vantagem em erro some.
- A faixa de 80% cobre 64,2% (modelo completo), 63,1% (imagem) e 63,1% (texto) no teste; em beleza 50,0%; na dobra mais recente 49,7%. A cobertura declarada é a do procedimento, **não foi medida no artefato servido** (retreinado com todos os dados) e não é garantida.
- **Regra de baixa confiança por largura** (limiar 2,9733 em log, n = 100): validade **presumida** no pacote servido; o diagnóstico no pacote de desenvolvimento foi **inconclusivo** (IC 95% da diferença de −0,148 a 2,215); não foi avaliada no teste.
- Imagem e texto: sem ganho de ordenação demonstrado; sempre carregam o aviso de evidência limitada. O vídeo que cai na contingência usa o modelo de imagem, com a versão e a cobertura dele.
- Ganho multimodal na ordenação: inconclusivo; a ablação tem pouco poder (menor efeito detectável de cerca de 0,21 no Spearman intra-canal).
- O alvo é a curtida acumulada de vídeos de 30 dias a cerca de 1 ano de idade: não é uma previsão exata aos 30 dias. Metadados podem ter sido editados e vídeos apagados depois da coleta.
- Temperatura 1,0 no Gemini: uma chamada por previsão; a variação não entra na faixa.
- Miniatura de URL nunca vista: `hqdefault` pode ter resolução diferente da miniatura da API usada no treino (diferença de domínio não medida).

## Uso e operação

- **Upload de vídeo local: não verificado ao vivo.** Implementado e verificado offline com a Files API simulada; a verificação real depende de um vídeo local autorizado e ainda não foi feita.
- Gemini e YouTube ao vivo: não verificados com o pacote final (ver [verificado_ao_vivo_offline_sintetico.md](verificado_ao_vivo_offline_sintetico.md)).
- O histórico fornecido no pedido sem `video_id` só protege o próprio vídeo pela defasagem de 30 dias; a saída avisa.
- O servidor MCP atende um pedido por vez; a espera do upload pode bloquear os demais.
- O bloqueio de 30 minutos depois de cota diária esgotada do Gemini existe só em memória e o cliente do YouTube não tem bloqueio equivalente; reiniciar o servidor repete uma requisição que falharia.
- Fora do recorte: tela, RAG, personas, outros setores (moda é recusado), previsão de visualizações.

## Retenção, licenças e termos

- Menor prazo: **2026-11-03T23:46:26Z** (pacotes de modelo, dataset de referência, caches e respostas salvas); base 2026-11-13T02:14:37Z; de-para 2026-11-17T21:45:43Z. Depois deles, a demonstração, o servidor e a reprodução dependem de nova autorização ou nova coleta. O autor pode estender se a regra de origem permitir; nada é presumido.
- Licenças dos pesos de e5-small, CLIP, EasyOCR e MediaPipe: **NÃO CONFIRMADAS**; os pesos não vão na distribuição ([dados_e_retencao.md](dados_e_retencao.md)).
- Métricas derivadas e agregação entre canais (YouTube) e o uso do Gemini gratuito sobre URLs do YouTube: questões de plataforma pendentes (N9, N10).

## Demonstração

4 dos 6 canais sem confirmação de país ou idioma do Brasil, conteúdo em inglês e espanhol em parte, tipo de canal ausente e um canal prolífico sem histórico utilizável ([demonstracao.md](demonstracao.md)). É só ilustrativa da mecânica.
