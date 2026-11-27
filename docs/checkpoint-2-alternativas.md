# Signal FC · Dois experimentos complementares

[Continuação com a mistura de listas e janelas de 2 s](checkpoint-2-ajustes-finais.md)

A fusão e a subida de ativação encontraram **19 lances em 36 alertas**.
As duas referências encontraram **18 em 36**. A fusão também teve a
maior cobertura quando usamos todos os alertas, com **78 dos 157 lances**
encontrados em **259 alertas**.

## Como testamos

1. **Fusão de volume + tom e ativação.** Reaproveitamos os dois rankings
   de picos do Checkpoint 2. Agrupamos os picos em intervalos de no máximo
   4 s. Em cada grupo, cada método contribui com seu pico mais bem
   colocado. A contribuição é o peso do método dividido pela posição
   no ranking mais três. Somamos as contribuições e usamos o horário
   do pico de maior contribuição como horário do alerta. O ajuste
   comparou pesos de 25%, 50% e 75% para a ativação e escolheu **50%**.
2. **Subida em relação ao contexto local.** Na mesma grade de janelas de
   4 s com passo de 2 s, subtraímos do escore atual a mediana das janelas
   anteriores que não se sobrepõem à janela atual. A busca comparou
   12, 20 e 30 s de contexto e escolheu **30 s** tanto para ativação quanto
   para volume RMS. Mantivemos o P98 por tempo sobre as subidas positivas.

Escolhemos os parâmetros pelo F1 com até três alertas por tempo nos seis
jogos de ajuste. A tolerância de **±15 s** e o pareamento um a um seguiram
o experimento original. Os seis jogos de avaliação já tinham sido
inspecionados quando pensamos nas alternativas. Os números a seguir são
uma comparação exploratória nesse mesmo lote. [Calibração](../outputs/checkpoint-2/alternativas/calibracao.csv)

## Resultado com 36 alertas

| Método | Acertos | Alertas | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volume + tom | 18 | 36 | 50,0% | 11,5% | 0,187 |
| wav2vec 2.0 | 18 | 36 | 50,0% | 11,5% | 0,187 |
| Fusão | 19 | 36 | 52,8% | 12,1% | 0,197 |
| Subida de ativação | 19 | 36 | 52,8% | 12,1% | 0,197 |
| Subida de volume RMS | 11 | 36 | 30,6% | 7,0% | 0,114 |

Os dois métodos com 19 acertos compartilharam só 8 desses lances. Na
análise só de gols, com os mesmos 36 alertas, a fusão encontrou **10 de 19**
e a subida de ativação encontrou **7 de 19**. Isso repete os números
das respectivas referências, volume + tom e wav2vec 2.0.

## Resultado com todos os alertas

| Método | Acertos em 157 | Alertas | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volume + tom | 42 | 111 | 37,8% | 26,8% | 0,313 |
| wav2vec 2.0 | 59 | 189 | 31,2% | 37,6% | 0,341 |
| Fusão | 78 | 259 | 30,1% | 49,7% | 0,375 |
| Subida de ativação | 50 | 183 | 27,3% | 31,8% | 0,294 |
| Subida de volume RMS | 34 | 180 | 18,9% | 21,7% | 0,202 |

A fusão encontrou **19 lances a mais que o wav2vec**, com **70 alertas
adicionais**. É o melhor candidato para uma triagem com lista ampla
neste recorte. Na lista curta, a diferença se resume a **um acerto** e
varia por partida. A subida de ativação ajudou nessa lista, mas perdeu
cobertura com todos os alertas. A subida de volume RMS ficou abaixo do
controle de volume na mesma grade, que encontrou **14 em 36** na lista
curta. [Controle RMS](../outputs/checkpoint-2/controle-rms/avaliacao_metricas.csv)

[Métricas completas](../outputs/checkpoint-2/alternativas/metricas.csv) ·
[Acertos por tipo de lance](../outputs/checkpoint-2/alternativas/por_classe.csv) ·
[Alertas e anotações associadas](../outputs/checkpoint-2/alternativas/alertas.csv)

Da raiz do projeto, o comando abaixo reproduz as tabelas a partir dos
escores, dos alertas anteriores e dos WAVs.

```sh
.cache/checkpoint-2/venv313/bin/python avaliar_alternativas_checkpoint2.py
```
