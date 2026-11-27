# Signal FC · Resultados do Checkpoint 2

Primeira rodada, em 12 partidas. [Reavaliação em partidas novas](checkpoint-2-ampliado-resultados.md).

[Fusão e subida local](checkpoint-2-alternativas.md) ·
[Mistura de listas e janelas de 2 s](checkpoint-2-ajustes-finais.md) ·
[Slides de 27/11](../slides/checkpoint-2/Signal-FC-Checkpoint-2.pdf)

O wav2vec 2.0 encontrou **59 dos 157 lances** do lote de avaliação. A regra de volume + tom encontrou **42**. A ativação trouxe mais cobertura e passou de F1 **0,313** para **0,341**, com **189** alertas em vez de **111**. Com três alertas por tempo, os métodos empataram em **18 acertos em 36 alertas**.

## Como medimos

Foram pontuadas **32.851 janelas de áudio**, com 4 s de duração e avanço de 2 s, em 24 tempos de 12 jogos do SoccerNet. Os seis jogos de ajuste têm 153 eventos anotados. Os seis de avaliação interna têm 157. As duas listas são disjuntas e pertencem à divisão oficial de validação do SoccerNet.

Usamos a saída de ativação do modelo da audEERING e normalizamos os escores pela mediana e pelo desvio absoluto mediano de cada tempo. O detector lê só a trilha sonora. A análise usa características acústicas, sem palavras, transcrições ou imagens. Janelas consecutivas acima do limiar formam um alerta no centro da janela de maior escore do grupo. O pareamento associa cada alerta à anotação disponível mais próxima, dentro de ±15 s, em ordem de prioridade. Cada anotação recebe no máximo um acerto.

O limiar escolhido no lote de ajuste foi o **percentil 98 por tempo**. Entre os quatro percentis testados, ele deu o maior F1, **0,395**, com 69 acertos e 196 alertas. A escolha foi registrada antes do pareamento no lote de avaliação. [Calibração](../outputs/checkpoint-2/avaliacao/calibracao.csv) · [Registro da execução](checkpoint-2-execucao.md)

## Comparação nos seis jogos de avaliação

| Método | Acertos em 157 | Alertas | Precisão | Revocação | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Só volume | 39 | 117 | 33,3% | 24,8% | 0,285 |
| Volume + tom | 42 | 111 | 37,8% | 26,8% | 0,313 |
| wav2vec 2.0 | 59 | 189 | 31,2% | 37,6% | 0,341 |

A ativação cobriu **10,8 pontos percentuais** a mais de lances do que volume + tom, com **6,6 pontos a menos** de precisão. Houve **130 alertas sem associação** com a ativação e **69** com volume + tom. Os métodos encontraram 25 lances em comum. A ativação encontrou 34 que a regra deixou passar e a regra encontrou outros 17. [Métricas](../outputs/checkpoint-2/avaliacao/avaliacao_metricas.csv) · [Comparação visual](../outputs/checkpoint-2/figuras/comparacao.png)

Com o limite de **três alertas por tempo**, volume + tom e wav2vec 2.0 fizeram **18/36**, com 50% de precisão e F1 **0,187**. Nos dois casos acertaram 18 dos 157 lances anotados. A regra de só volume fez **17/36**. Na análise secundária que usa **apenas gols como alvo**, o wav2vec encontrou 16 de 19 gols com todos os alertas, frente a 14 da regra de volume + tom. Com 36 alertas, a regra encontrou 10 gols e o wav2vec encontrou 7.

O controle de RMS na mesma grade de 4 s com passo de 2 s, com ajuste independente nos mesmos seis jogos, também selecionou o percentil 98. No lote de avaliação, encontrou **47 lances em 191 alertas**, com F1 **0,270**. Com quase o mesmo número de alertas, a ativação encontrou **12 lances a mais** do que esse controle de volume. [Controle RMS](../outputs/checkpoint-2/controle-rms/avaliacao_metricas.csv)

## Resultado por tipo de lance

| Anotação | Só volume | Volume + tom | wav2vec 2.0 |
| --- | ---: | ---: | ---: |
| Finalização no alvo | 15/69 | 16/69 | 21/69 |
| Finalização para fora | 10/66 | 11/66 | 21/66 |
| Gol | 13/19 | 14/19 | 15/19 |
| Pênalti | 1/3 | 1/3 | 2/3 |
| Expulsão | — | — | — |

A maior diferença veio das finalizações, que responderam por **15 dos 17 acertos líquidos adicionais** em relação a volume + tom. Os valores vêm do pareamento de todos os lances juntos. Esse pareamento pode atribuir um alerta próximo a outro evento, por isso a linha de gols difere da análise que usa apenas gols como alvo. Há três pênaltis e nenhuma expulsão no lote de avaliação, o que limita a leitura dessas categorias. [Gráfico por tipo](../outputs/checkpoint-2/figuras/por-tipo.png) · [Contagens por tipo](../outputs/checkpoint-2/avaliacao/avaliacao_por_classe.csv)

## Três trechos para a apresentação

| Caso | Partida e tempo do áudio | O que aparece no pareamento |
| --- | --- | --- |
| Acerto em comum | PSG × Dijon, 1º tempo, 14:24 | Gol associado pelos dois métodos. Picos aos 864,00 s e 865,25 s. |
| Lance encontrado pelo wav2vec | Crystal Palace × Liverpool, 1º tempo, 20:52 | Finalização no alvo associada pela ativação. |
| Alerta fora das classes-alvo | Real Madrid × Sevilla, 2º tempo, 15:42 | Pico da ativação junto a uma anotação de impedimento, fora das cinco categorias avaliadas. |

Os três trechos WAV estão em `data/checkpoint-2-exemplos/` para escuta local da equipe. O índice registra os segundos exatos do início, do fim e do pico. [Índice dos trechos](../outputs/checkpoint-2/exemplos.csv) · [Linha do tempo do piloto](../outputs/checkpoint-2/piloto/linha-do-tempo.png)

## Sequência para apresentar em 27/11

Os sete slides têm uma previsão de 6 min 10 s de fala nas notas do PowerPoint.

1. Retomar a regra de volume + tom e mostrar, com o exemplo de PSG × Dijon, como um alerta vira acerto.
2. Explicar o wav2vec como um modelo que dá uma nota para cada trecho de áudio. Contar como combinamos os horários e por que encurtamos os trechos.
3. Mostrar o resultado com todos os alertas e comparar a regra inicial com a mistura no limite de três sugestões por tempo.
4. Fechar com o próximo passo de 04/12. Os testes detalhados e os trechos de áudio servem de apoio para perguntas.

Na conferência, os dois métodos do Checkpoint 1 reproduziram as contagens publicadas. Foram recalculadas as métricas, os pareamentos e as contagens por tipo nos dois lotes. A ativação aparece como um sinal complementar para testar no Checkpoint 3. A vantagem observada aqui depende do número de alertas admitidos e do tipo de lance procurado.
