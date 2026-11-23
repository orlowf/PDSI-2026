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

Na reavaliação do Checkpoint 2, usamos **42 partidas que ficaram fora da primeira rodada**, com **12 jogos de ajuste** da divisão de treino e **30 jogos inéditos de avaliação** da divisão de validação. Os jogos abrangem seis competições e o lote de avaliação tem **753 lances-alvo anotados**. [SoccerNet-v2](https://www.soccer-net.org/tasks/action-spotting)

Um **alerta** é um horário sugerido para revisar um lance. Um acerto associa o alerta à anotação disponível mais próxima em até **15 s**. Cada anotação pode receber um único acerto. A comparação apresentada no seminário usa todos os alertas gerados em cada partida.

### Métodos

1. **Regra de volume e tom.** Mede o volume RMS e dá um bônus quando o tom, estimado com pYIN, sobe.
2. **Wav2vec sozinho.** Estima a intensidade da reação em janelas de 4 s de áudio. [Modelo utilizado](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)
3. **Fusão da regra com o Wav2vec.** Reúne alertas próximos das duas listas e mantém os horários sugeridos por apenas um método. [Plano e protocolo](docs/checkpoint-2-ampliado-plano.md)

## Checkpoint 2 · Resultados reavaliados

| Método | Alertas | Lances encontrados | Precisão | Recall |
| --- | ---: | ---: | ---: | ---: |
| Regra de volume e tom | 570 | 193 | 33,9% | 25,6% |
| Wav2vec sozinho | 1.038 | 324 | 31,2% | 43,0% |
| Fusão da regra com o Wav2vec | 1.382 | 382 | 27,6% | 50,7% |

A regra teve a maior precisão. A fusão encontrou 58 lances a mais que o Wav2vec sozinho e gerou 344 alertas adicionais para revisão.

[Relatório ampliado com todos os testes](docs/checkpoint-2-ampliado-resultados.md) · [Métricas e calibração](outputs/checkpoint-2-ampliado/) · [Primeira rodada em 12 jogos](docs/checkpoint-2-resultados.md)

## Como reproduzir

O [relatório ampliado](docs/checkpoint-2-ampliado-resultados.md#reprodução) reúne os comandos, a revisão do modelo, os jogos selecionados e a configuração de execução. A mídia do SoccerNet fica local, em diretório ignorado pelo Git.

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
