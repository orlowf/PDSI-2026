# Signal FC · Mistura de listas e duração das janelas

Experimentos de 22/11/2026 para o Checkpoint 2, com **apenas áudio**. O
detector usa a trilha mono da transmissão e calcula medidas acústicas.
O conteúdo da fala, transcrições e imagens ficam fora do método.

Os seis jogos de ajuste somam 153 lances e os seis de avaliação somam 157.
As listas são disjuntas e ambas vieram da divisão de validação do SoccerNet.
Aplicamos pareamento um a um em ±15 s. As duas experiências reutilizam um
lote de avaliação já inspecionado e devem ser lidas como exploratórias.

## Mistura de rankings

Cada lista isolada acertava 19 lances em 36 alertas, com oito acertos em
comum. Testamos duas cotas por tempo. As duas deram 21 de 36 e F1 0,222
no ajuste. Como critério de desempate, mantivemos a primeira configuração,
com **duas vagas da fusão e uma da subida de ativação**.

Preenchemos primeiro as duas vagas da fusão, em ordem do ranking, e depois
buscamos o melhor candidato disponível na lista de subida. Um pico a até
4 s de outro já escolhido é ignorado e a busca segue dentro da lista. Se
ela acabar, a outra pode preencher a vaga. A mistura acertou **21 em 36**
na avaliação, com **58,3% de precisão**, **13,4% de recall** e F1 **0,218**.
Cada método isolado obteve 52,8% de precisão, 12,1% de recall e F1 0,197.
Na análise só de gols, a mistura encontrou 9 de 19, com os mesmos 36
alertas. A fusão isolada encontrou 10 e a subida encontrou 7.
[Calibração da mistura](../outputs/checkpoint-2/ajustes-finais/calibracao_mistura.csv)

## Janelas de 4 s e de 2 s

O wav2vec 2.0 pontuou **32.851 janelas de 4 s** e **32.875 janelas de 2 s**
nos mesmos 24 tempos. Ambas as grades avançam a cada 2 s, usando janelas
completas e o centro de cada janela como horário do escore. Usamos os mesmos
pesos, o mesmo P98 por tempo e o mesmo agrupamento de picos. O P98 foi
selecionado nos seis jogos de ajuste com a duração original e aplicado
diretamente a 2 s, para comparar apenas a duração.

| Janela e seleção | Acertos em 157 | Alertas | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 s, todos | 59 | 189 | 31,2% | 37,6% | 0,341 |
| 2 s, todos | 71 | 253 | 28,1% | 45,2% | 0,346 |
| 4 s, três por tempo | 18 | 36 | 50,0% | 11,5% | 0,187 |
| 2 s, três por tempo | 17 | 36 | 47,2% | 10,8% | 0,176 |

As duas grades tiveram 329 janelas acima do P98 no lote de avaliação.
Com 2 s, elas formaram 253 grupos, frente a 189 com 4 s. O saldo foi de
64 alertas e 12 acertos a mais. O ganho
de cobertura veio principalmente de finalizações no alvo. Com três
alertas por tempo, a janela menor teve um acerto a menos.

| Tipo de lance | 4 s, todos | 2 s, todos |
| --- | ---: | ---: |
| Finalização no alvo | 21/69 | 33/69 |
| Finalização para fora | 21/66 | 20/66 |
| Gol | 15/19 | 16/19 |
| Pênalti | 2/3 | 2/3 |
| Expulsão | — | — |

As contagens por classe vêm da associação de todos os lances em conjunto.
Na avaliação secundária usando só gols como alvo, as janelas de 4 s
encontraram 16 de 19 e as janelas de 2 s encontraram 17 de 19.

[Métricas dos dois lotes](../outputs/checkpoint-2/ajustes-finais/metricas.csv) ·
[Acertos por classe](../outputs/checkpoint-2/ajustes-finais/por_classe.csv) ·
[Alertas e anotações associadas](../outputs/checkpoint-2/ajustes-finais/alertas.csv) ·
[Como reproduzir](checkpoint-2-execucao.md)
