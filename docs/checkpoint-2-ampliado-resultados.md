# Signal FC · Checkpoint 2 reavaliado

A primeira rodada usava 12 jogos. Com acesso ao SoccerNet, reavaliamos os métodos em **42 partidas inéditas para o projeto**, de seis competições. O sistema recebe a trilha sonora. Usa características acústicas, como volume, tom e intensidade da reação. A narração e a torcida já estão misturadas nesse sinal. Palavras, transcrições e imagens ficam fora do detector.

O seminário compara a regra de volume e tom, o Wav2vec de 4 s e a fusão dos dois com **todos os alertas gerados**. A fusão encontrou **382 dos 753 lances anotados**. O relatório também registra as variantes exploratórias e a comparação com lista curta.

## Amostra e protocolo

| Competição | Ajuste · train | Avaliação · valid |
| --- | ---: | ---: |
| Inglaterra | 2 | 5 |
| Champions League | 2 | 5 |
| França | 2 | 5 |
| Alemanha | 2 | 5 |
| Itália | 2 | 5 |
| Espanha | 2 | 5 |
| **Total** | **12** | **30** |

Os jogos de ajuste têm 292 lances-alvo e os de avaliação, 753. Juntos, somam **84 tempos** e **64,1 horas de áudio**. O conjunto de teste do SoccerNet ficou reservado para o Checkpoint 3. A seleção seguiu uma ordem fixa por identificador dentro de cada competição. Em 4 jogos, a diferença entre as durações da trilha e da gravação passou de 3 s mesmo após a correção de sincronização. Nesses casos, usamos a próxima reserva da mesma competição. [Plano anterior à execução](checkpoint-2-ampliado-plano.md) · [Candidatos e substituições](../outputs/checkpoint-2-ampliado/amostra_candidatos.csv) · [Tempos e duração](../outputs/checkpoint-2-ampliado/amostra_midias.csv)

Um alerta é um horário para revisar um possível lance. Em ordem de prioridade, associamos cada alerta à anotação disponível mais próxima, até **15 s** de distância. Cada anotação pode receber um acerto. Precisão conta os acertos entre os alertas. Recall conta os acertos entre os lances anotados. F1 combina as duas medidas. São métricas do projeto para este protocolo. [Tarefa oficial do SoccerNet](https://www.soccer-net.org/tasks/action-spotting)

## O que testamos e ajustamos

1. **Referências.** A regra inicial usa volume, com uma variante que dá bônus quando o tom sobe. O controle calcula volume RMS nas mesmas janelas de 4 s usadas pelo modelo.
2. **Ativação.** Usamos o wav2vec 2.0 pré-treinado da audEERING para dar uma nota ao áudio. Pontuamos **115.218 janelas de 4 s** e **115.302 de 2 s**, com passo de 2 s. Janelas consecutivas acima do limiar formam um alerta no seu pico.
3. **Combinações.** A fusão combina os rankings da regra e do wav2vec. A subida privilegia mudanças em relação ao som dos segundos anteriores. A mistura distribui as três vagas entre a fusão e a subida de ativação.

Só os jogos de ajuste participaram da escolha dos parâmetros. Comparamos os percentis 95, 97, 98 e 99 e escolhemos **P98 para a ativação** e **P98 para o controle RMS**. As janelas de 2 s receberam o mesmo limiar das de 4 s. A fusão ficou com **50,0% de peso para a ativação**, e os contextos escolhidos foram **30 s** para subida de ativação e **20 s** para subida RMS. A divisão das três vagas da mistura foi **1 da fusão e 2 da subida**. [Opções e métricas de ajuste](../outputs/checkpoint-2-ampliado/calibracao.csv) · [Configuração congelada](../outputs/checkpoint-2-ampliado/configuracao_ajuste.json)

## Avaliação em 30 jogos novos

### Todos os alertas

| Método | Acertos | Alertas | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Só volume | 173 | 533 | 32,5% | 23,0% | 0,269 |
| Regra de volume + tom | 193 | 570 | 33,9% | 25,6% | 0,292 |
| Controle RMS · 4 s | 251 | 1014 | 24,8% | 33,3% | 0,284 |
| wav2vec · 4 s | 324 | 1038 | 31,2% | 43,0% | 0,362 |
| wav2vec · 2 s | 367 | 1304 | 28,1% | 48,7% | 0,357 |
| Fusão | 382 | 1382 | 27,6% | 50,7% | 0,358 |
| Subida de ativação | 250 | 941 | 26,6% | 33,2% | 0,295 |
| Subida de volume RMS | 165 | 908 | 18,2% | 21,9% | 0,199 |

**Fusão** encontrou mais lances, com **382** em **1382 alertas**. O maior F1 foi o de **wav2vec · 4 s**, com **0,362**. Nas janelas iguais de 4 s, a ativação fez **324 acertos em 1038 alertas** (31,2% de precisão e 43,0% de recall), frente a **251 em 1014** (24,8% de precisão e 33,3% de recall). Ao passar de 4 s para 2 s, fez **367 acertos em 1304 alertas** (28,1% de precisão e 48,7% de recall), frente a **324 em 1038** (31,2% de precisão e 43,0% de recall).

### Análise complementar com até três alertas por tempo

| Método | Acertos | Alertas | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Só volume | 88 | 180 | 48,9% | 11,7% | 0,189 |
| Regra de volume + tom | 85 | 180 | 47,2% | 11,3% | 0,182 |
| Controle RMS · 4 s | 72 | 180 | 40,0% | 9,6% | 0,154 |
| wav2vec · 4 s | 99 | 180 | 55,0% | 13,1% | 0,212 |
| wav2vec · 2 s | 98 | 180 | 54,4% | 13,0% | 0,210 |
| Fusão | 101 | 180 | 56,1% | 13,4% | 0,217 |
| Subida de ativação | 91 | 180 | 50,6% | 12,1% | 0,195 |
| Subida de volume RMS | 43 | 180 | 23,9% | 5,7% | 0,092 |
| Mistura de listas | 96 | 180 | 53,3% | 12,7% | 0,206 |

Na lista curta, **Fusão** obteve **101 acertos em 180 alertas**. A mistura ficou 11 acertos à frente da regra, com **14 ganhos, 6 empates e 10 perdas** na comparação por jogo. [Comparação por partida](../outputs/checkpoint-2-ampliado/comparacao_lista_curta.csv) A análise avalia horários de lances em geral. A avaliação auxiliar de gols usa um pareamento próprio com o gabarito restrito a gols. [Resultados completos](../outputs/checkpoint-2-ampliado/metricas.csv) · [Resultados por jogo](../outputs/checkpoint-2-ampliado/por_jogo.csv) · [Horários e associações](../outputs/checkpoint-2-ampliado/alertas.csv)

### Onde apareceram os acertos

| Tipo | Regra | wav2vec · 4 s | wav2vec · 2 s | Fusão |
| --- | ---: | ---: | ---: | ---: |
| Gol | 64/95 | 66/95 | 64/95 | 83/95 |
| Pênalti | 0/9 | 2/9 | 2/9 | 2/9 |
| Finalização no alvo | 74/340 | 163/340 | 190/340 | 183/340 |
| Finalização para fora | 54/304 | 92/304 | 108/304 | 112/304 |
| Expulsão | 1/5 | 1/5 | 3/5 | 2/5 |

Os valores são acertos sobre anotações de cada tipo, considerando todos os alertas e o pareamento conjunto. Leia pênaltis e expulsões junto com seus denominadores. Sons de torcida, impedimentos e outras reações podem gerar alertas sem correspondência nas cinco categorias-alvo. [Resultados por tipo, todos os métodos](../outputs/checkpoint-2-ampliado/por_classe.csv)

## Reprodução

Da raiz do repositório, use o Python do ambiente do Checkpoint 2 e uma conta autorizada a baixar os jogos. A seleção usa `data/games_valid.txt`, `data/games_valid_holdout.txt` e as anotações de treino e validação em `data/soccernet/videos-224p/`. A execução mantém a mídia em `.cache/checkpoint-2-ampliado/`. [Preparação do ambiente](checkpoint-2-execucao.md#como-reproduzir)

```sh
pdsi_python=.cache/checkpoint-2/venv313/bin/python
hf download audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim \
  --revision 6eba34a2485ea31cb03600241787c3a5edab8626 \
  --local-dir .cache/checkpoint-2/model
"$pdsi_python" preparar_midias_checkpoint2_ampliado.py
"$pdsi_python" pontuar_checkpoint2_ampliado.py --precisao float16 --lote 24 --lote-2s 24
"$pdsi_python" avaliar_checkpoint2_ampliado.py --somente-ajuste
"$pdsi_python" avaliar_checkpoint2_ampliado.py
```

Nesta rodada os vídeos 224p aceitos somaram **15,3 GB transferidos**. O perfil de execução usa **float16 em MPS**. No piloto técnico em um tempo já conhecido, a comparação com os escores de 32 bits teve erro absoluto máximo de **0,00118** e manteve os horários dos três alertas mais bem colocados nos quatro percentis e nas duas durações. [Perfil de pontuação](../outputs/checkpoint-2-ampliado/perfil_escores.json) · [Verificação numérica](../outputs/checkpoint-2-ampliado/piloto-precisao.json)
