# Roteiro de fala do Checkpoint 1

**Ensaio sugerido de 6min30s.** As referências ficam como slide de apoio. A ordem das falas pode ser dividida entre os integrantes no ensaio.

## 1. Abertura · 0:00 a 0:25

> A gente quer encontrar o momento em que ocorreu um gol a partir do áudio de uma transmissão. Neste checkpoint, vamos mostrar a regra inicial, onde ela funcionou, onde errou e como vamos testar uma rede pré-treinada.

## 2. Problema e dados · 0:25 a 1:15

> A transmissão tem vários momentos de voz e torcida intensas. Nossa hipótese é que parte dessas mudanças coincide com lances importantes. Usamos as anotações do SoccerNet como gabarito. Os 12 jogos disponíveis correspondem a 24 tempos e têm 55 gols anotados. São dois lotes de seis partidas sem sobreposição, dentro da divisão oficial de validação de 100 jogos. Trabalhos anteriores já usaram eventos sonoros para extrair melhores momentos, e estudos no SoccerNet também mediram ganhos ao combinar áudio e vídeo.

## 3. Como funciona o Nível 0 · 1:15 a 2:25

> Primeiro convertemos a transmissão em áudio mono a 16 kHz e aplicamos um filtro de 300 a 3.400 Hz. Medimos intensidade com RMS a partir da STFT e estimamos o tom com pYIN. Normalizamos essas medidas dentro de cada tempo de jogo. A intensidade determina a base da pontuação, e a subida do tom dá um bônus limitado. Se o sinal suavizado fica acima do limiar por pelo menos dois segundos, registramos o pico daquele trecho. Na avaliação de gols, associamos cada pico a até um gol anotado dentro de mais ou menos 15 segundos.

## 4. Resultados · 2:25 a 3:35

> No primeiro lote, encontramos 22 dos 36 gols entre 115 detecções. No segundo, encontramos 14 de 19 entre 111. Isso dá revocação de 61% e 74%, com precisão de 19% e 13%. A combinação gera candidatos úteis, mas ainda exige muita revisão. Quando pegamos só os três picos mais fortes de cada tempo de jogo, a precisão sobe e a revocação cai. Vamos usar o F1 para comparar esse equilíbrio com os próximos modelos.

## 5. Demonstração · 3:35 a 4:35

> Os dois gráficos vêm do primeiro tempo de Burnley contra Arsenal. A linha azul indica a detecção e a pontilhada indica a anotação do SoccerNet. O gol foi anotado aos 11:03 e o pico mais forte saiu aos 11:05. O segundo pico apareceu aos 5:14, perto de uma anotação de bola fora aos 5:15. Na avaliação da classe gol, ele entra como falso positivo. O filtro foi aplicado à trilha inteira, então tanto a voz quanto a torcida ou outros sons podem contribuir para os picos.

## 6. Próximo experimento · 4:35 a 5:55

> Vamos comparar a regra com um classificador baseado em atributos como MFCC e com uma rede pré-treinada chamada PANNs CNN14. Na proposta neural, a rede recebe janelas de áudio e gera uma representação de cada uma. Um classificador leve aprende a distinguir janelas próximas de gols de outras janelas, incluindo lances que também geram picos fortes. O treino usará jogos separados. Na comparação, vamos usar os mesmos 12 jogos, a mesma tolerância e a mesma classe-alvo do Nível 0. A meta é aumentar a precisão preservando a capacidade de encontrar gols, o que deve aparecer no F1.

## 7. Encerramento · 5:55 a 6:30

> Hoje mostramos o Nível 0 como referência. Para o segundo checkpoint, em 27 de novembro, o plano é mostrar os resultados parciais do classificador acústico e o andamento do PANNs. Em 4 de dezembro, no terceiro checkpoint e apresentação final, planejamos comparar os métodos avaliados e demonstrar clipes com alguns segundos de contexto em torno do lance.

## Respostas para perguntas

- **Por que ±15 segundos?** É a tolerância usada pela avaliação inicial do grupo. A reação sonora pode começar ou atingir o pico alguns segundos depois da anotação. Mantemos a tolerância fixa na comparação entre modelos.
- **O áudio é só do narrador?** A entrada é a trilha completa da transmissão. A faixa de frequência favorece componentes da voz, mas outros sons também passam pelo filtro.
- **Já avaliamos a validação inteira?** Os dois lotes cobrem os 12 jogos disponíveis no arquivo do grupo. A divisão oficial tem 100 jogos anotados.
- **A rede neural já foi treinada?** O checkpoint apresenta o protocolo proposto para o experimento e os resultados medidos do Nível 0.
- **Qual é a diferença entre precisão e revocação?** Precisão é quantos dos candidatos apontados eram gols. Revocação é quantos dos gols reais o detector encontrou.
- **Como evitaremos ajustar no teste?** Partidas da divisão de treino servirão para treinamento e ajuste interno. A divisão oficial de teste fica reservada para a avaliação final.
- **Os resultados foram reproduzidos?** Sim. Repetimos a avaliação nos 24 tempos com o ambiente do repositório e os quatro CSVs saíram idênticos aos de referência.
