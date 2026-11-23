# Execução do Checkpoint 2

O experimento usa o modelo audEERING wav2vec 2.0 fixado na revisão
`6eba34a2485ea31cb03600241787c3a5edab8626`. A entrada é o WAV mono de 16 kHz
da transmissão. Medimos características do som, com a torcida e a
narração misturadas na trilha. Palavras, transcrições e imagens ficam
fora do detector. O estudo inicial usa janelas de 4 s com avanço de 2 s.
O CSV registra início, centro, fim e ativação de cada janela.
[Modelo e método dos autores](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)

## Execução dos 12 jogos em 19/11

O lote completo pontuou 32.851 janelas em 24 tempos. A execução desta
etapa, com reaproveitamento do primeiro tempo do piloto, levou 29 min 51 s
incluindo a avaliação. O percentil 98 foi fixado no ajuste. As contagens
das regras de referência bateram com as do Checkpoint 1.
[Relatório e material para 27/11](checkpoint-2-resultados.md)

## Janelas de 2 s em 22/11

A execução pontuou 32.875 janelas nos 24 tempos em 20 min 40 s, incluindo
o carregamento do modelo nos dois lotes. Usamos o mesmo modelo e o mesmo
passo de 2 s, com PyTorch 2.12.1, MPS e lotes de 8. As duas durações foram
avaliadas com P98 e pareamento de ±15 s.
[Comparação das janelas e resultado da mistura](checkpoint-2-ajustes-finais.md)

## Piloto de 18/11

- Burnley × Arsenal, primeiro tempo, de 7:00 a 12:00. Foram pontuadas
  149 janelas em 25,30 s com PyTorch 2.12.1, MPS e lotes de 8, após
  carregar o modelo.
- Os pesos foram conferidos pelo SHA-256 publicado. A execução em CPU
  reproduziu os três valores do exemplo de silêncio dos autores com
  diferença absoluta inferior a 0,000001.
- O [gráfico do piloto](../outputs/checkpoint-2/piloto/linha-do-tempo.png)
  mostra a ativação e o RMS em janelas de 4 s. As linhas tracejadas marcam
  a finalização para fora aos 427,911 s e o gol aos 662,787 s. O RMS do
  gráfico é o volume da trilha inteira e serve para inspecionar o trecho.
  A comparação de desempenho usa a regra original e os dois lotes.
- O primeiro tempo completo gerou 1.349 janelas em 148,74 s. As
  149 janelas que também aparecem no trecho curto deram os mesmos
  valores. A taxa observada no tempo completo dá uma estimativa de
  cerca de uma hora de inferência para as 32.851 janelas do projeto,
  sujeita a variação entre jogos e máquinas.
- No ensaio do avaliador, o percentil 98 foi fixado para verificar o
  funcionamento no primeiro tempo. A regra de volume + tom associou
  1 de 3 lances em 12 alertas. O wav2vec 2.0 associou 1 em 15 alertas.
  Esse ensaio antecedeu a calibração e a avaliação do lote completo.
  [Tabela do ensaio](../outputs/checkpoint-2/ensaio-primeiro-tempo/metricas.csv)

## Como reproduzir

Na raiz do projeto, coloque os arquivos autorizados do `data.zip` em
`data/`. São necessários os manifestos dos dois lotes, os WAVs em
`data/audio/` e as anotações em `data/soccernet/videos-224p/`.

```sh
uv venv --python 3.13 .cache/checkpoint-2/venv313
uv export --locked --format requirements-txt --no-hashes --no-emit-project \
  -o .cache/checkpoint-2/base-requirements.txt
uv pip install --python .cache/checkpoint-2/venv313/bin/python \
  -r .cache/checkpoint-2/base-requirements.txt 'torch==2.12.1' 'transformers==4.57.6'

pdsi_python=.cache/checkpoint-2/venv313/bin/python
pdsi_jogo="$(head -n 1 data/games_valid.txt)"

"$pdsi_python" pontuar_arousal.py --jogo "$pdsi_jogo" --tempo 1 \
  --inicio 420 --duracao 300 --saida outputs/checkpoint-2/piloto
"$pdsi_python" pontuar_arousal.py --jogo "$pdsi_jogo" --tempo 1 \
  --saida outputs/checkpoint-2/escores

"$pdsi_python" pontuar_arousal.py --lista data/games_valid.txt \
  --saida outputs/checkpoint-2/escores
"$pdsi_python" pontuar_arousal.py --lista data/games_valid_holdout.txt \
  --saida outputs/checkpoint-2/escores
"$pdsi_python" avaliar_arousal.py --ajuste data/games_valid.txt \
  --avaliacao data/games_valid_holdout.txt --escores outputs/checkpoint-2/escores \
  --saida outputs/checkpoint-2/avaliacao
"$pdsi_python" avaliar_rms_checkpoint2.py
"$pdsi_python" graficos_checkpoint2.py
"$pdsi_python" avaliar_alternativas_checkpoint2.py

"$pdsi_python" pontuar_arousal.py --janela 2 --lista data/games_valid.txt \
  --saida outputs/checkpoint-2/escores-2s
"$pdsi_python" pontuar_arousal.py --janela 2 --lista data/games_valid_holdout.txt \
  --saida outputs/checkpoint-2/escores-2s
"$pdsi_python" avaliar_ajustes_finais_checkpoint2.py
```

O primeiro comando de pontuação recorta de 7:00 a 12:00 do primeiro tempo
de Burnley × Arsenal. O restante da execução usa tempos completos.
O comando de pontuação aceita `--modelo-local` para uma cópia da revisão
acima e valida os CSVs que já existem ao retomar um lote.

A avaliação escolhe um dos percentis 95, 97, 98 e 99 usando apenas
`games_valid.txt`. Aplica o valor escolhido em `games_valid_holdout.txt`.
O pareamento usa ±15 s, um acerto por anotação e a prioridade dos alertas
do Checkpoint 1. Os relatórios incluem todas as detecções e até três alertas
por tempo, além dos acertos por família de lances. O relatório
`calibracao.csv` registra o critério escolhido.

O último avaliador combina os rankings de fusão e de subida de ativação,
selecionando as vagas de cada lista em `games_valid.txt`. Também compara a
duração das janelas com o P98 fixo para os dois comprimentos. Para
reproduzir só a mistura, após gerar `alternativas/alertas.csv`, rode
`avaliar_ajustes_finais_checkpoint2.py --somente-mistura` com o mesmo
Python. A saída dessa opção é `outputs/checkpoint-2/mistura/`, que pode
ser alterada com `--saida`.
[Detalhes e resultados dos dois testes](checkpoint-2-ajustes-finais.md)
