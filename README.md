# Ouvindo o gol · Signal FC

INF0413 · Processamento Digital de Sinais e Imagens · UFG

O projeto localiza horários candidatos a clipes de futebol **pela trilha de áudio**. A narração, a torcida e os demais sons chegam misturados na transmissão. As entradas do detector são medidas acústicas desse sinal. Palavras, transcrições e imagens ficam fora do método. As anotações do SoccerNet servem de gabarito para a avaliação.

## Equipe

| Matrícula | Nome completo |
| --- | --- |
| 202503999 | Felipe Orlow de Oliveira Sene |
| 202504023 | Pedro Augusto Ferreira Matos |
| 202504021 | Murilo Uchôa Lima |
| 202505701 | Gustavo Henrique Barros da Silva |

## Apresentações

- [Prévia do seminário de 06/11](slides/Ouvindo%20o%20gol%20%E2%80%94%20Pr%C3%A9via%20do%20Semin%C3%A1rio.pdf)
- [Checkpoint 1 de 13/11 · PDF apresentado](slides/checkpoint-1/Signal-FC-Checkpoint-1.pdf) · [Registro da primeira etapa](docs/checkpoint-1.md)
- Checkpoint 2 de 27/11 · [PDF](slides/checkpoint-2/Signal-FC-Checkpoint-2.pdf) · [PowerPoint editável](slides/checkpoint-2/Signal-FC-Checkpoint-2.pptx)

## Metodologia

### Dados e avaliação

Reunimos as 500 partidas anotadas das divisões de origem do [SoccerNet-v2](https://www.soccer-net.org/tasks/action-spotting) e criamos uma divisão própria por partida, estratificada por competição. Processamos **493 partidas** em seis competições, com **98 para ajuste** e **395 para avaliação**. Sete partidas ficaram de fora após a verificação da mídia. [Partidas e motivos](outputs/checkpoint-2-integral/pendencias.csv)

Um **alerta** é um horário sugerido para revisar um lance. Um acerto associa o alerta à anotação disponível mais próxima em até **15 s**. Cada anotação pode receber um único acerto. Avaliamos todos os alertas gerados, sem limite por tempo de jogo ou por partida.

### Métodos

1. **Regra de volume e tom.** Mede o volume RMS e dá um bônus quando o tom, estimado com pYIN, sobe.
2. **Wav2vec sozinho.** Estima a intensidade da reação em janelas de 4 s de áudio. [Modelo utilizado](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
3. **Fusão da regra com o Wav2vec.** Agrupa alertas próximos em até 4 s e usa um peso por método e a posição no ranking para escolher um horário por grupo. Alertas isolados também entram.

Nos 98 jogos de ajuste, o F1 selecionou o percentil **97** para a regra, o percentil **98** para o Wav2vec e o peso de **75%** do Wav2vec na fusão. Mantivemos esses parâmetros nos 395 jogos de avaliação. [Calibração](outputs/checkpoint-2-integral/calibracao.csv)

## Checkpoint 2 · Resultados finais

Os 395 jogos de avaliação somaram **10.120 lances anotados**.

| Método | Alertas | Lances encontrados | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regra de volume e tom | 10.990 | 3.214 | 29,2% | 31,8% | 0,305 |
| Wav2vec sozinho | 13.562 | 4.293 | 31,7% | 42,4% | 0,363 |
| Fusão da regra com o Wav2vec | 21.054 | 5.530 | 26,3% | 54,6% | 0,355 |

O áudio ajudou a localizar lances. O Wav2vec obteve o maior F1, e a fusão encontrou a maior parcela dos lances anotados. Muitos alertas ficaram sem correspondência com esses lances.

[Relatório final](outputs/checkpoint-2-integral/resumo.md) · [Resultados e configuração](outputs/checkpoint-2-integral/) · [Rodada de 42 jogos](docs/checkpoint-2-ampliado-resultados.md) · [Primeira rodada de 12 jogos](docs/checkpoint-2-resultados.md)

## Como reproduzir

O [protocolo](outputs/checkpoint-2-integral/protocolo.json) registra a divisão por partida e a revisão do modelo. A execução usa as anotações em `data/soccernet/videos-224p/`, FFmpeg e as [dependências Python](docs/checkpoint-2-execucao.md#como-reproduzir). Use uma conta do Hugging Face com acesso aprovado ao SoccerNet para obter o áudio. A mídia e o cache de processamento ficam locais.

```bash
python executar_checkpoint2_integral.py --somente-planejar
hf auth login
hf download audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim \
  --revision 6eba34a2485ea31cb03600241787c3a5edab8626 \
  --local-dir .cache/checkpoint-2/model
python executar_checkpoint2_integral.py
```

Para recalcular as métricas a partir dos escores salvos localmente, use `python avaliar_checkpoint2_integral.py --manifesto outputs/checkpoint-2-integral/manifesto.csv`.

## Cronograma

| Data | Marco | Trabalho do grupo |
| --- | --- | --- |
| 13/11/2026 | Checkpoint 1 | Regras de volume e tom, com avaliação nos 12 jogos disponíveis |
| 27/11/2026 | Checkpoint 2 | Ativação acústica, comparação por tipo de lance e experimentos de seleção de alertas |
| 04/12/2026 | Checkpoint 3 e apresentação final | Testar um classificador de atributos acústicos com e sem ativação, fazer um teste em outra transmissão e demonstrar a geração de clipes pelos horários |

## Referências

- DELIÈGE, Adrien et al. *SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos*. CVPR Workshops, 2021. [Artigo](https://doi.org/10.1109/CVPRW53098.2021.00508)
- MAUCH, Matthias; DIXON, Simon. *pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions*. ICASSP, 2014. [Artigo](https://doi.org/10.1109/ICASSP.2014.6853678)
- WAGNER, Johannes et al. *Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence Gap*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. [Artigo dos autores](https://arxiv.org/abs/2203.07378)
- XIONG, Ziyou et al. *Audio Events Detection Based Highlights Extraction from Baseball, Golf and Soccer Games in a Unified Framework*. ICME, 2003. [Artigo](https://doi.org/10.1109/ICME.2003.1221333)
