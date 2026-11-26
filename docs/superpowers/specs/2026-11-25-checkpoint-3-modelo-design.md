# Checkpoint 3 · Detector de lances treinado do zero

**Data:** 25/11/2026 · **Entrega:** 04/12/2026 (Checkpoint 3 e apresentação final)
**Status:** revisado, aguardando aprovação

## Intenção

Treinar modelos próprios, sem pesos pré-treinados, que recebem **só o áudio** de uma partida inédita e devolvem os horários dos lances importantes. A pergunta do Checkpoint 3 é se um modelo que aprende os parâmetros com os dados do SoccerNet encontra mais lances do que a regra de volume e tom e o wav2vec dos Checkpoints 1 e 2, nos mesmos jogos e com o mesmo protocolo.

O sucesso é medido pelo **F1**, pela **precisão** e pelo **recall** no lote de teste, comparados lado a lado com os métodos anteriores. Também entram os acertos por tipo de lance.

## Objetivos

- Construir três modelos sobre a mesma base de dados e a mesma avaliação:
  1. ML clássico com atributos acústicos manuais.
  2. Rede temporal 1D (TCN) sobre log-mel em baixa resolução.
  3. CNN 2D sobre o espectrograma log-mel completo.
- Comparar os três com a regra de volume e tom, o wav2vec e a fusão nos mesmos jogos de teste.
- Reportar precisão, recall, F1, acertos por classe e resultados por jogo.

## Fora do escopo

- Wav2vec, PANNs ou qualquer modelo pré-treinado como entrada ou inicialização.
- Vídeo, transcrição ou texto como entrada.
- Classificar o tipo do lance. O detector emite só o horário, como no CP2.
- As 17 classes do SoccerNet e a métrica oficial average-mAP.
- **Acurácia.** Quase todo instante é "sem lance", então a acurácia fica perto de 100% mesmo para um modelo inútil.

## Anti-objetivos (falham mesmo que "funcionem")

- Usar o lote de teste para escolher hiperparâmetros, limiar ou época. O teste é aberto uma única vez, no final.
- Comparar métodos em conjuntos de jogos diferentes.
- Vazar um jogo entre treino e validação. As dobras separam por **jogo**, nunca por janela ou tempo.
- Lotar o disco ou versionar mídia, pesos ou a credencial do Hugging Face.

## Restrições

- **Prazo:** 9 dias, de 25/11 a 04/12.
- **Máquina:** Mac com MPS e **83 GB livres**. Não cabe guardar o áudio dos 500 jogos: os WAVs de 16 kHz somam cerca de 86 GB.
- **Dados:** SoccerNet via Hugging Face, com conta autorizada e token só no cache local do `hf`. Licença de uso para pesquisa.
- **Referência de custo:** a rodada integral do CP2 baixou e processou 448 jogos em cerca de 15 h, de 23/11 23:00 a 24/11 13:44, incluindo a inferência do wav2vec.

## Classes-alvo e protocolo de avaliação (igual ao CP2)

- Lances-alvo: `Goal`, `Penalty`, `Shots on target`, `Shots off target`, `Red card`, `Yellow->red card`. É o conjunto `KEY_EVENTS` de `ouvindo_o_gol.py`, e as duas expulsões contam juntas como "Expulsão".
- Anotações duplicadas de mesmo rótulo a menos de 2 s viram uma só, como no CP1/CP2.
- Um alerta é um horário. Cada alerta é associado à anotação-alvo disponível mais próxima em até **±15 s**, em ordem de prioridade do alerta, e cada anotação recebe no máximo um acerto. **Reutilizar o código de pareamento de `avaliar_checkpoint2_integral.py` e `ouvindo_o_gol.py`**, sem reescrever.
- Métricas: acertos, alertas, anotações, precisão, recall e F1, todos os alertas contados. Acertos por classe vêm do pareamento conjunto.
- **Dois pontos de operação, com duas tabelas** (decidido com o Pedro):
  1. **Principal, pelo F1.** O percentil e *d* de cada modelo são escolhidos pelo F1 fora da dobra, como no CP2. A tabela traz os três modelos novos, a regra, o wav2vec e a fusão.
  2. **Mesmo número de alertas que a fusão.** Em cada tempo de jogo do teste, cada modelo novo emite os *k* picos mais fortes após a supressão, sendo *k* o número de alertas que a fusão do CP2 emitiu naquele tempo. A fusão entra como referência. A pergunta aqui é se o modelo acha mais lances com o mesmo trabalho de revisão. A mesma tabela é calculada também fora da dobra, só para leitura.

## Divisão dos dados

- Universo: os 493 jogos com áudio validado no CP2. Os 7 de `outputs/checkpoint-2-integral/pendencias.csv` ficam de fora.
- **Teste:** 100 jogos sorteados entre os 395 do lote de avaliação do CP2, estratificados por competição, com semente fixa. A lista é gravada em arquivo **antes** de qualquer treino e não muda depois.
- **Treino:** os outros 393 jogos (98 do ajuste do CP2 + 295 da avaliação).
- **Validação cruzada:** 5 dobras dentro do treino, separadas por jogo e estratificadas por competição. As previsões fora da dobra escolhem hiperparâmetros, época, limiar e distância de supressão de cada modelo.
- **Modelo final:** média das previsões dos 5 modelos das dobras aplicada ao teste. *Decisão do Claude:* evita um sexto treino; o construtor pode trocar por um retreino nos 393 se sobrar tempo.
- **Linhas de base no teste:** regra (P97), wav2vec (P98) e fusão (75%) nos mesmos 100 jogos, com os parâmetros do CP2 congelados. Não há vazamento: esses parâmetros foram ajustados nos 98 jogos de ajuste, que agora estão no treino. Fonte: `/Users/pedromatos/.cache/pdsi-checkpoint-2-runtime/rodada/` (escores dos 986 tempos) e `outputs/checkpoint-2-integral/por_jogo.csv` (resultados por jogo dos 493 jogos). Os caches dentro do repositório cobrem só 448 jogos e não bastam. O teste é sorteado entre os **395** jogos de avaliação.
- **Conferência:** somar as linhas de `por_jogo.csv` dos 100 jogos de teste e checar se batem com o recálculo a partir dos escores.

## Base de dados de atributos (compartilhada pelos três modelos)

Fluxo por jogo, em série e em segundo plano:

1. Obter os WAVs mono de 16 kHz dos dois tempos com `acquire_game()` de `adquirir_midias_checkpoint2_integral.py`, que já valida a sincronia e apaga o `.mkv` na pasta temporária. Reaproveitar os ~94 WAVs que já estão em `.cache/`.
2. Calcular o **log-mel**: 64 bandas, `n_fft` 1024, hop de 320 amostras (20 ms, 50 quadros/s), escala log, `float16`. Gravar um `.npy` por tempo em `.cache/checkpoint-3/logmel/<jogo>/<tempo>.npy`.
3. Na mesma passada, a partir da **STFT linear**, calcular e gravar em `.cache/checkpoint-3/quadros/<jogo>/<tempo>.npz`, a 50 quadros/s, `float16`: RMS, centroide espectral, rolloff (85%), fluxo espectral, taxa de cruzamentos por zero e energia nas faixas 0–300 Hz, 300–3400 Hz e 3400–8000 Hz. Os 20 MFCC não precisam ser gravados: saem da DCT do log-mel. O tom (pYIN) fica de fora por custo. *Decisão do Claude.*
4. **Apagar o WAV** só depois de os dois arquivos estarem gravados e conferidos (formato e valores finitos). O WAV pré-existente em `.cache/` não é apagado. Depois que o áudio é apagado, nenhum atributo novo que dependa dele pode ser calculado sem baixar tudo de novo. Por isso a lista acima é fechada antes de iniciar a extração.

Ocupação esperada: ~17 GB de log-mel mais ~2 GB de atributos de quadro para 986 tempos, mais um jogo em trânsito (~1 GB). Download total estimado em ~180 GB de tráfego, sem ficar no disco. O script retoma de onde parou, pulando tempos já gravados.

Os modelos 1 e 2 usam uma versão reduzida a **passos de 0,5 s** (média e máximo por banda), derivada do log-mel completo.

## Rótulo de treino

- Para cada passo de tempo, o alvo é uma gaussiana centrada em cada anotação-alvo, com σ escolhido na validação cruzada entre {2, 4, 6} s e pico 1. Para o modelo 1, um passo é positivo se o alvo passa de 0,5.
- O modelo vê contexto antes e depois do instante, já que a inferência é offline. Assim ele aprende sozinho o atraso entre o lance e a reação sonora.

## Pós-processamento comum

1. Suavizar a curva de probabilidade com média móvel de 2 s.
2. O limiar é um **percentil por tempo de jogo**, como os P97 e P98 do CP2, e não uma probabilidade absoluta. A média dos 5 modelos muda a escala da curva, mas não a fração de instantes acima do percentil. Assim, o número de alertas no teste fica coerente com o da validação cruzada. *Decisão do Claude.*
3. Pegar os picos acima do limiar e suprimir picos a menos de *d* s de um pico mais forte, com *d* ∈ {10, 20, 30}.
4. Cada pico restante vira um alerta. O percentil e *d* são escolhidos nas previsões fora da dobra.

## Modelos

### 1 · ML clássico (LightGBM)

- Janela de 2 s, a cada 0,5 s.
- Atributos: média e desvio das 64 bandas log-mel, 20 MFCC com deltas (média e desvio), energia, centroide espectral, fluxo espectral, rolloff e fração de energia em bandas baixa, média e alta.
- Contexto: diferença de cada atributo principal em relação à mediana dos 30 s anteriores e dos 10 s seguintes, e o máximo de energia nos próximos 10 s.
- Classificador: LightGBM com `scale_pos_weight` para o desbalanceamento. Se o LightGBM não instalar, usar `HistGradientBoostingClassifier` do scikit-learn.
- Entregar a **importância dos atributos** para o slide de discussão.

### 2 · Rede temporal 1D (TCN)

- Entrada: a sequência de 0,5 s (128 canais, média e máximo de 64 bandas) do tempo inteiro, normalizada por tempo com mediana e MAD.
- Arquitetura: blocos residuais de convolução 1D dilatada, não causal, com campo receptivo de cerca de ±60 s e saída sigmoide por passo.
- Treino: recortes aleatórios de 5 min, perda BCE ponderada ou focal contra o alvo gaussiano, AdamW, parada antecipada pela validação da dobra. Roda em MPS.

### 3 · CNN 2D sobre o espectrograma

- Entrada: recortes de 20 s do log-mel completo (1000 × 64), normalizados por tempo.
- Arquitetura: CNN pequena tipo VGG (4 a 5 blocos conv-BN-ReLU-pool), que reduz a frequência a 1 e o tempo a passos de 0,5 s, com cabeça sigmoide por passo.
- Treino: amostragem balanceada, com metade dos recortes contendo um lance-alvo, e aumento por SpecAugment (máscaras de tempo e frequência) e ganho aleatório.
- Inferência: janelas deslizantes com sobreposição de 50%, médias nas regiões sobrepostas.
- É o modelo mais caro. Se o prazo apertar, treinar com menos épocas, mas **manter as 5 dobras** para a comparação ser justa.

## Ordem e cronograma

| Quando | Trabalho |
| --- | --- |
| 25–26/11 | Sortear e gravar o teste e as dobras; extrair log-mel em segundo plano (~15 h); começar com os jogos já disponíveis. |
| 26–28/11 | Avaliação comum e linhas de base no teste; modelo 1 completo. |
| 28–30/11 | Modelo 2. |
| 30/11–02/12 | Modelo 3. |
| 02–03/12 | Avaliação final no teste (uma vez), tabelas, gráficos e slides. |
| Extra | Cortes de 30 s, se sobrar tempo. |

A prioridade é 1 → 2 → 3. Cada modelo concluído já é um resultado apresentável.

## Extra opcional: cortes de 30 s

Dado o vídeo de um jogo de teste e os alertas do melhor modelo, gerar um clipe por alerta, de *t* − 15 s a *t* + 15 s, com `ffmpeg`. Só para a demonstração na apresentação final.

## Saídas esperadas

- `outputs/checkpoint-3/`: lista de teste e dobras, configuração escolhida por modelo, `metricas.csv`, `por_classe.csv`, `por_jogo.csv`, `alertas.csv`, curvas fora da dobra, importância de atributos e figuras.
- Os scripts no repositório; log-mel e modelos treinados ficam em `.cache/checkpoint-3/`, fora do git.

## Abordagens consideradas

| Abordagem | Troca | Decisão |
| --- | --- | --- |
| ML clássico com atributos manuais | Rápido, interpretável; teto limitado pelos atributos | Feito (modelo 1) |
| TCN 1D sobre log-mel reduzido | Contexto longo, treino barato; menos interpretável | Feito (modelo 2) |
| CNN 2D sobre espectrograma completo | Mais expressivo; mais caro e arriscado no prazo | Feito (modelo 3) |
| Treinar só nos 98 jogos de ajuste e testar nos 395 | Comparação idêntica ao CP2, mas pouco dado de treino | Descartado |
| Validação cruzada nos 493 sem teste separado | Usa tudo, mas não deixa um teste intocado | Descartado |

Decidido com o Pedro: as mesmas classes do CP1/CP2, a avaliação por precisão, recall e F1, a divisão com ~100 jogos de teste e validação cruzada em 5 dobras no treino, e os três modelos. Decisões do Claude, abertas a correção: log-mel 64 × 50 quadros/s, alvo gaussiano, média das dobras como modelo final, apagar o WAV após extrair e a ordem 1 → 2 → 3.

## Fica a cargo do construtor

Hiperparâmetros exatos dentro das grades, tamanho das redes, taxa de aprendizado, número de épocas, organização dos scripts e formato dos gráficos.
