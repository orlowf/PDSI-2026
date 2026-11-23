# Signal FC · Plano da reavaliação do Checkpoint 2

**Pergunta.** Nos jogos novos, a ativação acústica e as combinações de alertas ajudam a encontrar mais lances do que a regra de volume e tom? Para quem pode revisar três sugestões por tempo, a lista mista é a comparação principal. A rodada de 12 jogos serve como ponto de partida histórico.

## 1. Selecionar partidas antes de medir

- Usar 12 partidas da divisão `train` para ajuste e 30 de `valid` para avaliação, com duas e cinco partidas por competição, respectivamente. Excluir os 12 jogos já usados. A divisão `test` fica reservada para o Checkpoint 3.
- Dentro de cada competição, ordenar o identificador de cada jogo pelo SHA-256 de `pdsi-cp2-v2|divisão|jogo`. Em caso de empate, ordenar pelo identificador. Guardar a lista completa de candidatos e a ordem das reservas. As anotações identificam a divisão e cada tempo. Contagens e tipos de lance ficam fora da seleção.
- Validar as duas trilhas de cada jogo. Registrar falhas de mídia, falhas de acesso e substituições por ordem. Se a transferência falhar, repetir pelo caminho HTTP homologado. Usar reserva da mesma competição após uma falha objetiva persistente ou ausência de áudio. Registrar as cotas efetivamente preenchidas.
- Extrair a faixa original para mono a 16 kHz. Usar o acesso pessoal autorizado ao SoccerNet. Os arquivos de mídia e caches de execução permanecem em diretórios ignorados pelo Git. [SoccerNet Raw HQ](https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ) · [Action Spotting](https://www.soccer-net.org/tasks/action-spotting)

## 2. Pontuar do áudio e congelar as escolhas

- Rodar do áudio a regra de só volume, volume + tom, o controle de volume nas mesmas janelas, ativação de 4 s e 2 s, fusão, subida de ativação, subida de volume e mistura de listas. Para as duas durações, o início avança a cada 2 s. Usar os pesos audEERING já verificados no Checkpoint 2. O detector usa apenas medidas do áudio, sem texto ou imagem.
- Reaproveitar as funções existentes de volume, tom, pontuação e pareamento. Calcular os atributos clássicos uma vez por tempo e salvar escores completos com seus horários. Se necessário, testar lotes e formas de baixar em um piloto técnico com áudio antigo. Esse piloto verifica velocidade, estabilidade e fidelidade numérica.
- Nos 12 jogos de ajuste, escolher o percentil da ativação de 4 s entre 95, 97, 98 e 99 pelo F1 com todos os alertas. Aplicar o percentil escolhido também às janelas de 2 s para isolar a duração. Calibrar o controle de volume de forma independente. Manter P98 nas subidas locais, como no Checkpoint 2.
- Escolher pesos da fusão, contexto da subida e cota da mistura no ajuste, usando F1 com até três alertas por tempo. Em empates, manter a ordem predefinida das opções. Gravar as escolhas antes de associar alertas a anotações no lote de avaliação.

## 3. Medir nos jogos inéditos

- Aplicar a configuração congelada. Registrar acertos, alertas, precisão, recall e F1 com todos os alertas e com até três por tempo. A mistura é avaliada na lista de até três. Associar o pico à anotação disponível mais próxima em ±15 s, com um acerto por anotação.
- Usar os cinco grupos do projeto como alvo principal. Guardar métricas auxiliares só para gols, contagens por tipo e as associações por jogo. Para regra versus mistura na lista curta, mostrar a diferença pareada no número de acertos por jogo e a quantidade de partidas com ganho, empate ou perda.
- Conferir tamanhos dos lotes, número de escores por tempo, validade dos horários, ausência de cruzamento com os jogos antigos e consistência das métricas com os alertas. Registrar a amostra real, parâmetros, arquivos descarregados e eventuais limitações.

## 4. Preparar a entrega

- Atualizar o README e elaborar um relatório com a pergunta, o que foi testado, todas as comparações e o significado dos resultados. Guardar a rodada anterior e os resultados separados.
- Refazer os slides locais em PDF e PowerPoint no estilo do Checkpoint 1. A fala conta de onde partimos, por que a abertura do acervo importa, como os testes viraram alertas e o que se repetiu nos jogos novos. Mostrar poucos números essenciais e concentrar as comparações no final. Usar notas com roteiro de cerca de seis minutos, com margem para o limite de sete.
- Validar o PDF e o PowerPoint renderizados e cruzar os números do material com os CSVs. Preservar a versão anterior dos slides no backup local. As mídias e os caches de execução ficam locais.
