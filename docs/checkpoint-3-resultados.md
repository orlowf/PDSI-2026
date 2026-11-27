# Checkpoint 3 · Modelos treinados do zero

Neste checkpoint, treinamos três modelos do zero, só com o áudio dos jogos do SoccerNet-v2. Eles detectam as mesmas classes de lance do CP1 e do CP2: Goal, Penalty, Shots on target, Shots off target, Red card e Yellow->red card. A avaliação é a mesma do CP2: pareamento um a um com tolerância de ±15 s e métricas de precisão, recall e F1.

Design e plano: `docs/superpowers/specs/2026-11-25-checkpoint-3-modelo-design.md` e `docs/superpowers/plans/2026-11-25-checkpoint-3-modelo.md`.

## Dados

| | Jogos | Tempos | Lances anotados |
|---|---:|---:|---:|
| Treino (validação cruzada em 5 dobras) | 393 | 786 | 10 244 |
| Teste (avaliado uma única vez) | 100 | 200 | 2 479 |

- A divisão está em `outputs/checkpoint-3/divisao.json`. A semente é 2026; as dobras são feitas por jogo e estratificadas por competição.
- A extração está registrada em `outputs/checkpoint-3/extracao.csv`. Cada tempo virou um log-mel (16 kHz, 64 bandas, 50 quadros/s) mais 8 atributos por quadro. O áudio baixado foi apagado logo depois da extração.
- 14 jogos falharam na conferência de sincronia da versão HQ (12 de treino e 2 de teste). Todos foram recuperados no modo de recuperação do CP2 (`acquire_game(..., recovery=True)`), então nenhum jogo ficou de fora.
- Mídia, WAVs, atributos e pesos ficam em `.cache/`, fora do git, por causa do NDA do SoccerNet.

## Modelos

| Modelo | Entrada | Código |
|---|---|---|
| Gradient boosting (`HistGradientBoostingClassifier`) | 235 atributos manuais por passo de 0,5 s: média e desvio em 2 s do mel, dos MFCC e dos deltas, mais atributos de contexto (subida em relação aos 30 s anteriores, diferença em relação aos 10 s seguintes) | `cp3/classico.py` |
| TCN 1D | média e máximo do log-mel por passo (128 valores) | `cp3/redes.py`, `cp3/treino_rede.py` |
| CNN 2D | espectrograma de 20 s | `cp3/redes.py`, `cp3/treino_rede.py` |

O alvo de treino é uma gaussiana em torno de cada lance, com σ ∈ {2, 4, 6} s. Depois do treino, a curva de escores é suavizada em 2 s e os picos acima de um percentil viram alertas, mantendo uma distância mínima entre eles. O σ, o percentil e a distância foram escolhidos pelo F1 nas previsões fora da dobra (`outputs/checkpoint-3/config_*.json`). No teste, cada método usa a média dos 5 modelos das dobras.

O plano previa LightGBM, mas a libomp do LightGBM e a do PyTorch derrubam o processo quando carregadas juntas no macOS. Por isso trocamos pelo gradient boosting do scikit-learn, que cumpre o mesmo papel.

## Resultados no teste (100 jogos, 2 479 lances)

Ponto de operação escolhido pelo F1 (`outputs/checkpoint-3/teste/comparacao.png`):

| Método | Alertas | Acertos | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| **Gradient boosting** | 2 020 | 1 140 | **56,4%** | 46,0% | **0,507** |
| TCN 1D | 2 932 | 1 200 | 40,9% | 48,4% | 0,444 |
| CNN 2D | 3 023 | 1 018 | 33,7% | 41,1% | 0,370 |
| Regra (CP1) | 2 849 | 834 | 29,3% | 33,6% | 0,313 |
| Wav2vec (CP2) | 3 395 | 1 103 | 32,5% | 44,5% | 0,376 |
| Fusão (CP2) | 5 337 | 1 424 | 26,7% | 57,4% | 0,364 |

Com o mesmo número de alertas que a fusão do CP2 (`comparacao_mesmo_k.png`):

| Método | Alertas | Acertos | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| **Gradient boosting** | 5 337 | 1 692 | 31,7% | **68,3%** | 0,433 |
| TCN 1D | 5 337 | 1 536 | 28,8% | 62,0% | 0,393 |
| CNN 2D | 5 337 | 1 368 | 25,6% | 55,2% | 0,350 |
| Fusão (CP2) | 5 337 | 1 424 | 26,7% | 57,4% | 0,364 |

Recall por tipo de lance com o mesmo número de alertas (`por_classe.png`):

| Classe (lances) | Gradient boosting | TCN 1D | CNN 2D | Fusão (CP2) |
|---|---:|---:|---:|---:|
| Gol (343) | 93% | 93% | 83% | 93% |
| Finalização no alvo (1 092) | 61% | 52% | 48% | 56% |
| Finalização para fora (1 003) | 69% | 63% | 54% | 47% |
| Pênalti (31) | 39% | 29% | 35% | 45% |
| Expulsão (10) | 60% | 80% | 70% | 80% |

Os demais arquivos do teste ficam em `outputs/checkpoint-3/teste/`: `metricas.csv`, `por_classe.csv`, `por_jogo.csv`, `alertas.csv` (cada alerta com o lance pareado) e `linha_do_tempo.png`. O `trava.json` registra o hash da configuração usada.

### Leitura

- **O gradient boosting é o melhor modelo.** No ponto de melhor F1, tem precisão de 56% contra 27% da fusão. Com o mesmo número de alertas, acha 268 lances a mais que a fusão (68% contra 57%).
- **O ganho vem das finalizações.** Gol todos já acertam (~93%). As finalizações para fora sobem de 47% para 69%.
- **Os pênaltis continuam difíceis.** Muitos vêm colados no gol, e o pareamento um a um só deixa um alerta pagar um lance.
- **A CNN 2D ficou melhor no teste (F1 0,370) do que na validação cruzada (0,288).** No teste, ela usa a média de 5 modelos, o que reduz o ruído da curva.

## Validação cruzada (393 jogos de treino, fora da dobra)

`outputs/checkpoint-3/cv_resumo.csv`

| Modelo | Ponto | Alertas | Precisão | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Gradient boosting | melhor F1 | 8 199 | 57,0% | 45,6% | 0,507 |
| Gradient boosting | mesmo nº de alertas | 20 831 | 32,9% | 67,0% | 0,442 |
| TCN 1D | melhor F1 | 11 112 | 38,6% | 41,9% | 0,402 |
| TCN 1D | mesmo nº de alertas | 20 831 | 28,4% | 57,8% | 0,381 |
| CNN 2D | melhor F1 | 22 468 | 20,9% | 45,9% | 0,288 |
| CNN 2D | mesmo nº de alertas | 20 831 | 22,3% | 45,4% | 0,299 |
| Fusão (CP2) | referência | 20 831 | 26,5% | 53,9% | 0,355 |

### Atributos mais importantes do gradient boosting

`outputs/checkpoint-3/importancia_classico.csv` (importância por permutação, medida pela queda da precisão média):

1. `diferenca10_centroide`: brilho do som agora menos a mediana dos 10 s seguintes.
2. `diferenca10_e_voz`: energia na faixa de voz agora menos a dos 10 s seguintes.
3. `subida30_e_voz`: energia na faixa de voz agora menos a dos 30 s anteriores.
4. `max10_rms`: pico de volume nos 10 s seguintes.

O modelo se apoia na reação de torcida e narrador em torno do lance, e não no timbre de um instante isolado.

## Recall em função do número de alertas

Se a ideia é que uma pessoa revise todos os alertas e escolha os cortes, a métrica que importa é o recall para um número fixo de alertas. Os números abaixo são do gradient boosting (σ=2, distância mínima de 10 s), nos jogos de treino, fora da dobra. **Não foram medidos no teste.**

| Alertas por tempo (~45 min) | 15 | 27 | 45 (1 por min) | 90 |
|---|---:|---:|---:|---:|
| Recall, pareamento um a um (métrica do CP2) | 54% | 68% | 79% | 91% |
| Recall por cobertura (há um alerta a até 15 s) | 68% | 80% | 89% | 98% |
| Recall de alertas sorteados ao acaso | — | 23% | 34% | 57% |

O recall por cobertura é o mais próximo de uma aplicação com cortes de 30 s: um alerta que cobre dois lances próximos (pênalti seguido de gol, rebote) mostra os dois. Do total de lances, 17% ficam a menos de 10 s do lance anterior.

Recall um a um por classe nos mesmos dados:

| Alertas por tempo | Gol | No alvo | Para fora | Pênalti | Expulsão |
|---|---:|---:|---:|---:|---:|
| 27 | 93% | 62% | 67% | 42% | 66% |
| 45 | 95% | 74% | 80% | 60% | 83% |
| 90 | 97% | 87% | 95% | 81% | 98% |

Combinar o gradient boosting com a TCN (média das posições) não melhorou o recall.

**Atenção:** o teste já foi avaliado uma vez. Um novo ponto de operação, por exemplo um orçamento de alertas, tem de ser escolhido só nos dados de treino fora da dobra. Ao reportá-lo no teste, deixe claro que foi definido depois da primeira avaliação.

## Como reproduzir

Requisitos: o ambiente do CP2 mais `pip install -e '.[cp3]'`, e o token do Hugging Face no cache local do `hf`, nunca em arquivo.

```bash
python -m cp3.divisao                       # gera divisao.json (recusa sobrescrever)
python -m cp3.extrair --faixas 8            # baixa, extrai atributos e apaga o áudio (~1 dia)
python -m cp3.classico                      # validação cruzada do gradient boosting (~3 h)
python -m cp3.treino_rede --modelo tcn      # validação cruzada da TCN (~15 min no MPS)
python -m cp3.treino_rede --modelo cnn2d    # validação cruzada da CNN 2D (~3,5 h no MPS)
python -m cp3.relatorio_cv                  # cv_resumo.csv
python -m cp3.teste                         # avaliação única no teste + figuras
python -m cp3.graficos                      # refaz só as figuras a partir dos CSVs
python -m cp3.cortes --video <tempo.mkv> --jogo "<jogo>" --tempo 1   # cortes de 30 s por alerta
pytest                                      # 69 testes
```

O `cp3.teste` se recusa a rodar se alguma configuração mudar depois da primeira avaliação (trava por hash em `outputs/checkpoint-3/teste/trava.json`).
