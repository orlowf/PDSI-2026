# Checkpoint 2 · Rodada ampliada

Foram processadas **493 de 500 partidas elegíveis** do manifesto. A lista de 7 partidas sem resultado e seus motivos está em [pendencias.csv](pendencias.csv).

| Lote | Partidas processadas | Partidas elegíveis |
| --- | ---: | ---: |
| Ajuste | 98 | 100 |
| Avaliação | 395 | 400 |

## Partidas sem resultado nesta rodada

Cada partida continua no manifesto. A tabela agrupa os motivos registrados. O arquivo por partida traz o texto registrado quando disponível. [Partidas e motivos](pendencias.csv)

| Categoria | Divisão de origem | Lote | Partidas |
| --- | --- | --- | ---: |
| Lacunas no áudio | train | avaliacao | 1 |
| Sincronização sem comprovação | test | ajuste | 1 |
| Sincronização sem comprovação | train | ajuste | 1 |
| Sincronização sem comprovação | train | avaliacao | 2 |
| Sincronização sem comprovação | valid | avaliacao | 2 |

No ajuste, o F1 selecionou o percentil **97** para a regra, o percentil **98** para o wav2vec e o peso de **75%** do wav2vec na fusão. Esses parâmetros foram fixados antes de calcular as métricas de avaliação.

As contagens consideram todos os alertas. Cada anotação pode receber um acerto, com tolerância de 15 segundos. [Tabela de calibração](calibracao.csv)

## Avaliação separada

Partidas do lote reservado para avaliar os métodos.

| Método | Acertos | Alertas | Anotações | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regra de volume e tom | 3.214 | 10.990 | 10.120 | 29,2% | 31,8% | 0,305 |
| wav2vec · 4 s | 4.293 | 13.562 | 10.120 | 31,7% | 42,4% | 0,363 |
| Fusão | 5.530 | 21.054 | 10.120 | 26,3% | 54,6% | 0,355 |

## Conjunto geral

Reúne as partidas processadas dos lotes de ajuste e avaliação.

| Método | Acertos | Alertas | Anotações | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regra de volume e tom | 4.058 | 13.670 | 12.723 | 29,7% | 31,9% | 0,308 |
| wav2vec · 4 s | 5.409 | 16.939 | 12.723 | 31,9% | 42,5% | 0,365 |
| Fusão | 6.942 | 26.168 | 12.723 | 26,5% | 54,6% | 0,357 |

[Métricas completas](metricas.csv) · [Métricas por classe](por_classe.csv)
