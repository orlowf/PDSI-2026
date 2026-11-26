# Checkpoint 3 · Detector de lances treinado do zero · Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Três detectores de lances treinados do zero só com áudio (LightGBM, TCN 1D, CNN 2D), comparados com a regra, o wav2vec e a fusão do CP2 nos mesmos 100 jogos de teste.

**Architecture:** Um pacote `cp3/` na raiz do repositório. Ele baixa cada jogo com o pipeline do CP2, grava log-mel e atributos de quadro e apaga o WAV. Os três modelos leem esses arquivos, fazem validação cruzada em 5 dobras por jogo no treino, escolhem o ponto de operação fora da dobra e passam pela mesma avaliação de ±15 s do CP2, reaproveitando `measure()`. O teste é aberto uma vez só, por um script que trava a configuração.

**Tech Stack:** Python 3.13 (`.cache/checkpoint-2/venv313`), numpy, librosa, soundfile, scikit-learn, LightGBM, PyTorch com MPS, pytest e ffmpeg.

**Spec:** `docs/superpowers/specs/2026-11-25-checkpoint-3-modelo-design.md`

## Global Constraints

- Python: `PY=.cache/checkpoint-2/venv313/bin/python`. Rodar todo comando **da raiz do repositório** (`python -m cp3.<modulo>`), porque os módulos do CP2 estão na raiz.
- Lances-alvo: `KEY_EVENTS` de `ouvindo_o_gol.py`, com deduplicação por `dedupe_events` (2 s). As famílias vêm de `FAMILIES` (`avaliar_arousal.py`).
- Pareamento: `measure()` de `avaliar_alternativas_checkpoint2.py`, com tolerância de 15 s. Não reescrever.
- Log-mel: 16 kHz, `n_fft` 1024, hop 320 (20 ms, 50 quadros/s), 64 bandas, `power_to_db`, gravado em `float16`.
- Passo dos modelos e dos alertas: 0,5 s. O passo `i` tem horário `(i + 0.5) * 0.5` s.
- Divisão: 100 jogos de teste sorteados entre os 395 do lote `avaliacao` do CP2, estratificados por competição, semente `2026`. Os 393 restantes são treino, com 5 dobras por jogo estratificadas por competição. Os 7 jogos de `outputs/checkpoint-2-integral/pendencias.csv` ficam de fora.
- Grades fora da dobra: σ ∈ {2, 4, 6} s; percentil ∈ {95, 97, 98, 99, 99.5}; *d* ∈ {10, 20, 30} s. Escolher pelo F1; em empate, menos alertas.
- Modelo final no teste: média das previsões dos 5 modelos das dobras.
- Linhas de base: parâmetros congelados de `outputs/checkpoint-2-integral/configuracao.json` (P97, P98, 0,75). Escores em `~/.cache/pdsi-checkpoint-2-runtime/rodada/`.
- Locais: atributos em `.cache/checkpoint-3/` e resultados versionáveis em `outputs/checkpoint-3/`. Mídia, WAV, pesos e token **nunca** no git.
- Disco: parar a extração se o espaço livre ficar abaixo de **10 GB**.
- Git: trabalhar no branch `checkpoint-3`. Fazer `git add` só dos arquivos da tarefa, porque há trabalho do CP2 não commitado na árvore. Commitar só depois que o Pedro autorizar commits no início da execução.

## Review Focus

1. **Extração interrompida no meio de um jogo** (rede cai, processo morto). Rodar de novo retoma; nenhum `.npy` pela metade conta como pronto; nenhum WAV fica esquecido no disco. Teste em Task 3.
2. **Tempo sem nenhum lance-alvo.** O alvo é todo zero, o modelo ainda emite alertas pelo percentil e as métricas aceitam `anotados = 0`. Testes em Task 4 e Task 5.
3. **Anotação depois do fim do áudio.** Fica fora do alvo de treino, mas continua contando na avaliação, como no CP2. Teste em Task 4.
4. **Trecho de silêncio digital no áudio.** O log-mel continua finito (sem `-inf`) e os atributos sem `nan`. Teste em Task 2.
5. **Reabrir o teste depois de mudar a configuração.** O script recusa rodar. Teste em Task 11.

---

### Task 0: Ambiente e branch

**Files:**
- Modify: `pyproject.toml`
- Create: `cp3/__init__.py`, `tests/__init__.py`, `tests/cp3/__init__.py`, `tests/conftest.py`

- [ ] **Step 1:** Criar o branch: `git switch -c checkpoint-3`.
- [ ] **Step 2:** O venv já tem torch 2.12.1, scikit-learn 1.9.1 e librosa 1.0.0. Instalar só o que falta: `uv pip install --python .cache/checkpoint-2/venv313/bin/python lightgbm pytest`. Esperado: `$PY -c "import lightgbm, sklearn, torch, librosa; print(torch.backends.mps.is_available())"` imprime `True`.
- [ ] **Step 3:** Em `pyproject.toml`, adicionar `[project.optional-dependencies] cp3 = ["lightgbm", "scikit-learn", "torch", "pytest"]` e `[tool.pytest.ini_options] testpaths = ["tests"]`. Em `tests/conftest.py`, inserir a raiz do repositório em `sys.path`.
- [ ] **Step 4:** Rodar `$PY -m pytest -q`. Esperado: `no tests ran` sem erro de importação.
- [ ] **Step 5:** Commit: `git add pyproject.toml cp3/__init__.py tests/ && git commit -m "cp3: estrutura e dependências"`.

### Task 1: Divisão treino/teste e dobras

**Files:**
- Create: `cp3/divisao.py`, `tests/cp3/test_divisao.py`

**Interfaces:**
- Consumes: `read_manifest(path) -> dict[jogo, (divisao_origem, lote)]` de `avaliar_checkpoint2_integral.py`.
- Produces: `montar_divisao(manifesto: Path, pendencias: Path, semente: int = 2026, n_teste: int = 100, n_dobras: int = 5) -> dict`, com as chaves `teste: list[str]`, `treino: list[str]` e `dobras: dict[str, int]` (jogo de treino → dobra 0–4); `ler_divisao(path: Path) -> dict`. A CLI `python -m cp3.divisao` grava `outputs/checkpoint-3/divisao.json` e recusa sobrescrever.

- [ ] **Step 1: Testes que falham** (manifesto sintético com 6 competições):
  - `test_teste_vem_so_da_avaliacao`: todo jogo de `teste` tem lote `avaliacao`; `len(teste) == 100`.
  - `test_sem_sobreposicao_e_sem_pendentes`: `set(teste) & set(treino) == set()`; nenhum jogo de pendências aparece; `len(teste) + len(treino) == total - pendentes`.
  - `test_estratificado`: a contagem de teste por competição difere da proporcional em no máximo 1.
  - `test_dobras_equilibradas`: cada dobra tem `len(treino) // 5` ou `+1` jogos, e cada competição se espalha com diferença ≤ 1 entre as dobras.
  - `test_deterministico`: a mesma semente dá saída idêntica; outra semente dá teste diferente.
  - `test_cli_nao_sobrescreve`: a segunda execução sai com erro e o arquivo não muda.
- [ ] **Step 2:** `$PY -m pytest tests/cp3/test_divisao.py -v`. Esperado: FAIL (módulo ausente).
- [ ] **Step 3:** Implementar `montar_divisao` (competição = primeira parte do caminho; embaralhar com `numpy.random.default_rng(semente)`; cotas pelo maior resto) e a CLI.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Gerar a divisão real: `$PY -m cp3.divisao --manifesto outputs/checkpoint-2-integral/manifesto.csv --pendencias outputs/checkpoint-2-integral/pendencias.csv`. Esperado: 100 de teste, 393 de treino.
- [ ] **Step 6:** Commit de `cp3/divisao.py`, do teste e de `outputs/checkpoint-3/divisao.json`.

### Task 2: Log-mel e atributos de quadro

**Files:**
- Create: `cp3/extrair.py`, `tests/cp3/test_extrair.py`

**Interfaces:**
- Produces: as constantes `SR = 16000`, `N_FFT = 1024`, `HOP = 320`, `N_MELS = 64` e `FPS = 50`; `logmel(y: np.ndarray, sr: int) -> np.ndarray` (float16, forma `(n_quadros, 64)`); `atributos_quadro(y, sr) -> dict[str, np.ndarray]`, com as chaves `rms`, `centroide`, `rolloff`, `fluxo`, `zcr`, `e_baixa` (0–300 Hz), `e_voz` (300–3400 Hz) e `e_alta` (3400–8000 Hz), cada uma float16 com o mesmo `n_quadros`; `extrair_tempo(wav: Path, destino: Path) -> int`, que grava `destino/logmel.npy` e `destino/quadros.npz` de forma atômica (arquivo temporário e `os.replace`) e devolve `n_quadros`; `tempo_pronto(destino: Path) -> bool`.

- [ ] **Step 1: Testes que falham:**
  - `test_logmel_forma_e_tipo`: seno de 1 kHz, 10 s → forma `(501, 64)` (`center=True`), dtype `float16`, tudo finito.
  - `test_logmel_banda_do_seno`: a banda de maior energia média é a que contém 1 kHz (`librosa.mel_frequencies`).
  - `test_atributos_seno`: centroide mediano a ±100 Hz de 1000; `e_voz` > `e_baixa` e `e_alta`; zcr mediano ≈ 2000/16000 ± 0,02.
  - `test_silencio_finito` (Review Focus 4): 5 s de zeros → log-mel e todos os atributos finitos.
  - `test_extrair_tempo_atomico`: depois de `extrair_tempo`, `tempo_pronto` é True e não sobra arquivo temporário; com um `logmel.npy` truncado escrito à mão, `tempo_pronto` é False.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar com `librosa.stft` e `librosa.feature.melspectrogram(S=…)`. Calcular os dois a partir da mesma STFT. Usar `power_to_db(ref=1.0, amin=1e-10)`. O fluxo é a norma L2 da diferença positiva entre magnitudes consecutivas. `tempo_pronto` confere se os arquivos abrem, se as formas batem e se os valores são finitos.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit.

### Task 3: Pipeline baixar → extrair → apagar

**Files:**
- Modify: `cp3/extrair.py`
- Create: `tests/cp3/test_pipeline.py`

**Interfaces:**
- Consumes: `games_from_annotations(root) -> list[Game]`, `cached_wavs(game)` e `ANNOTATIONS_DIR` de `executar_checkpoint2_integral.py`/`ouvindo_o_gol.py`; `acquire_game(...)` e `resolve_revision()` de `adquirir_midias_checkpoint2_integral.py`; `ler_divisao`.
- Produces: `processar_jogo(jogo: Game, raiz: Path, revisao: str, faixas: int, adquirir=acquire_game) -> str`, que devolve `"pronto"`, `"reaproveitado"` ou `"ja_existia"`; os arquivos ficam em `raiz/<jogo>/<tempo>/`. A CLI `python -m cp3.extrair [--faixas 8] [--limite N]` processa primeiro os jogos de treino e depois os de teste, anexa uma linha por jogo a `outputs/checkpoint-3/extracao.csv` (`jogo,estado,segundos,motivo`) e segue para o próximo jogo quando um falha.

- [ ] **Step 1: Testes que falham** (com `adquirir` falso que escreve WAVs sintéticos em `tmp_path`):
  - `test_apaga_wav_baixado`: depois de `processar_jogo`, os dois `tempo_pronto` são True e os WAVs criados por `adquirir` não existem mais.
  - `test_nao_apaga_wav_preexistente`: com `cached_wavs` falso apontando para WAVs existentes, eles continuam no disco e o estado é `"reaproveitado"`.
  - `test_retoma` (Review Focus 1): com um tempo já pronto e outro com arquivo truncado, só o truncado é refeito; se `adquirir` levantar exceção, nenhum WAV temporário sobra em `raiz`.
  - `test_ja_existia`: com os dois tempos prontos, `adquirir` não é chamado.
  - `test_para_com_disco_cheio`: com `shutil.disk_usage` falso (livre < 10 GB), a CLI encerra antes de baixar e registra o motivo.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar. O diretório de trabalho do `acquire_game` é `.cache/checkpoint-3/transferencias/`. Apagar a pasta do jogo (`audio/<jogo>`) dentro desse diretório no `finally`. Nunca apagar caminhos devolvidos por `cached_wavs`.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Teste real em 1 jogo de treino: `$PY -m cp3.extrair --limite 1`. Esperado: estado `pronto`, nenhum `.wav` em `.cache/checkpoint-3/` e ~35 MB de log-mel por tempo.
- [ ] **Step 6:** Disparar a extração completa **em segundo plano** (cerca de 15 h): `nohup $PY -m cp3.extrair --faixas 8 > .cache/checkpoint-3/extracao.log 2>&1 &`. As tarefas 4 a 7 seguem em paralelo com os jogos já prontos.
- [ ] **Step 7:** Commit.

### Task 4: Rótulos e pós-processamento

**Files:**
- Create: `cp3/rotulos.py`, `cp3/posproc.py`, `tests/cp3/test_rotulos.py`, `tests/cp3/test_posproc.py`

**Interfaces:**
- Consumes: `read_source_annotations`, `dedupe_events`, `KEY_EVENTS`.
- Produces:
  - `eventos_alvo(anotacoes: dict, divisao_origem: str, jogo: str, tempo: int) -> list[tuple[float, str]]`
  - `alvo_gaussiano(eventos_s: list[float], n_passos: int, passo_s: float, sigma_s: float) -> np.ndarray` (float32)
  - `suavizar(p: np.ndarray, passo_s: float, janela_s: float = 2.0) -> np.ndarray`
  - `picos(p, passo_s, percentil: float, d_s: float) -> list[tuple[float, float]]` (horário, escore), em ordem decrescente de escore
  - `top_k(p, passo_s, k: int, d_s: float) -> list[tuple[float, float]]`

- [ ] **Step 1: Testes que falham:**
  - `test_alvo_pico_no_evento`: evento em 100 s, passo 0,5 → o valor 1,0 fica no índice 199; em 100 ± σ, `≈ exp(-0.5)` com tolerância de 1e-3.
  - `test_alvo_dois_eventos_maximo`: eventos a 1 s de distância → máximo ≤ 1 (usa máximo, não soma).
  - `test_alvo_sem_evento` (Review Focus 2): lista vazia → tudo zero.
  - `test_alvo_evento_apos_fim` (Review Focus 3): evento além do último passo → não aparece no alvo; `eventos_alvo` continua devolvendo esse evento.
  - `test_eventos_alvo_filtra_e_deduplica`: `Goal` duplicado a 1 s vira um; `Throw-in` sai.
  - `test_picos_supressao`: picos de 0,9 em 100 s e 0,8 em 105 s, com `d_s=10` → só o de 100 s; com `d_s=3` → os dois, em ordem decrescente.
  - `test_picos_serie_constante`: série constante → lista vazia (limiar estrito `>`).
  - `test_top_k_exato`: 5 picos separados e `k=3` → 3 itens, os mais fortes; com `k` maior que o número de picos, devolve todos.
  - `test_horario_do_passo`: um pico no índice `i` tem horário `(i + 0.5) * passo_s`.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar. Máximos locais vêm de `scipy.signal.find_peaks`, seguidos de supressão gulosa por escore. `suavizar` usa `uniform_filter1d` com `max(1, round(janela_s / passo_s))`.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit.

### Task 5: Avaliação comum e linhas de base no teste

**Files:**
- Create: `cp3/avaliacao.py`, `tests/cp3/test_avaliacao.py`

**Interfaces:**
- Consumes: `measure`, `read_manifest`, `read_source_annotations`, `load_half`, `predictions`, `METHODS` (`avaliar_checkpoint2_integral.py`) e `ler_divisao`.
- Produces:
  - `Previsao = tuple[str, int, list[tuple[float, str]], list[tuple[float, float]]]` (jogo, tempo, eventos, alertas ordenados)
  - `avaliar(previsoes: list[Previsao]) -> dict`, que devolve as métricas de `measure` mais `por_classe` (família → acertos/anotados) e `detalhes`
  - `por_jogo(previsoes) -> list[dict]`
  - `linhas_de_base(jogos: list[str], raiz_cp2: Path = Path.home() / ".cache/pdsi-checkpoint-2-runtime/rodada") -> dict[str, list[Previsao]]`, com chaves iguais a `METHODS` e parâmetros de `outputs/checkpoint-2-integral/configuracao.json`
  - `alertas_da_fusao(previsoes_fusao) -> dict[tuple[str, int], int]`

- [ ] **Step 1: Testes que falham:**
  - `test_avaliar_caso_conhecido`: 2 tempos, eventos em 100 s e 500 s, alertas em 110 s, 400 s e 120 s → acertos 1, alertas 3, anotados 2, precisão 1/3, recall 1/2.
  - `test_avaliar_tempo_sem_evento` (Review Focus 2): tempo sem eventos e com 4 alertas → anotados 0, alertas 4, sem exceção.
  - `test_linhas_de_base_batem_com_cp2` (marca `lento`; `skip` se `raiz_cp2` não existir): para 5 jogos do lote `avaliacao`, acertos, alertas e anotados por jogo e método são iguais às linhas de `outputs/checkpoint-2-integral/por_jogo.csv`.
  - `test_alertas_da_fusao`: conta os alertas por `(jogo, tempo)`.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar como uma camada fina sobre `measure` e `predictions`. Não reescrever o pareamento.
- [ ] **Step 4:** `$PY -m pytest tests/cp3/test_avaliacao.py -v -m "lento or not lento"`. Esperado: PASS.
- [ ] **Step 5:** Commit.

### Task 6: Leitura dos tempos para os modelos

**Files:**
- Create: `cp3/conjunto.py`, `tests/cp3/test_conjunto.py`

**Interfaces:**
- Consumes: os arquivos da Task 2/3 e `eventos_alvo`.
- Produces:
  - `@dataclass Tempo(jogo: str, numero: int, logmel: np.ndarray, quadros: dict[str, np.ndarray], eventos: list[tuple[float, str]])`, com a propriedade `n_passos = n_quadros // 25`
  - `carregar_tempo(jogo, numero, raiz=Path(".cache/checkpoint-3/atributos"), anotacoes=None) -> Tempo`
  - `tempos_prontos(jogos: list[str], raiz) -> list[tuple[str, int]]`
  - `reduzido(t: Tempo) -> np.ndarray` (float32, forma `(n_passos, 128)`: média e máximo das 64 bandas em blocos de 25 quadros, normalizados por tempo com mediana e MAD)

- [ ] **Step 1: Testes que falham:**
  - `test_reduzido_forma`: log-mel com 1000 quadros → `(40, 128)`; as 25 colunas de um bloco com o mesmo valor dão média = máximo.
  - `test_reduzido_normaliza`: a mediana de cada canal sai ≈ 0.
  - `test_tempos_prontos_ignora_incompletos`: um tempo sem `quadros.npz` fica de fora.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit.

### Task 7: Validação cruzada e escolha do ponto de operação

**Files:**
- Create: `cp3/cv.py`, `tests/cp3/test_cv.py`

**Interfaces:**
- Consumes: `ler_divisao`, `tempos_prontos`, `carregar_tempo`, `alvo_gaussiano`, `suavizar`, `picos`, `avaliar`.
- Produces:
  - O protocolo do modelo: `class Modelo(Protocol)`, com `treinar(treino: list[Tempo], validacao: list[Tempo], sigma_s: float) -> object` e `prever(estado: object, t: Tempo) -> np.ndarray` (forma `(n_passos,)`, valores em [0, 1]).
  - `rodar_cv(nome: str, modelo: Modelo, divisao: dict, sigmas=(2, 4, 6)) -> Path`, que grava as curvas fora da dobra em `.cache/checkpoint-3/oof/<nome>/sigma<σ>/<jogo>/<tempo>.npy` e os estados das dobras em `.cache/checkpoint-3/modelos/<nome>/sigma<σ>/dobra<k>.*`; retoma dobras já gravadas.
  - `escolher_operacao(nome: str, tempos: list[Tempo], percentis=(95, 97, 98, 99, 99.5), ds=(10, 20, 30)) -> dict`, que devolve `{"sigma", "percentil", "d", "f1", "alertas"}` e grava a tabela completa em `outputs/checkpoint-3/calibracao_<nome>.csv` e a escolha em `outputs/checkpoint-3/config_<nome>.json`.
  - `tabela_mesmo_k(nome, tempos, k_por_tempo: dict, d: float) -> dict`, para a leitura fora da dobra.

- [ ] **Step 1: Testes que falham** (com um `Modelo` falso cuja curva é o próprio alvo gaussiano com ruído):
  - `test_oof_sem_vazamento`: o `Modelo` falso registra os jogos recebidos em `treinar`; nenhuma previsão fora da dobra é de um jogo que o modelo daquela dobra viu.
  - `test_oof_cobre_todo_treino`: existe uma curva para cada tempo de treino, nenhuma para o teste.
  - `test_escolhe_maior_f1_e_desempata`: com a métrica falsa empatada, vence o ponto com menos alertas.
  - `test_retoma_dobras`: na segunda chamada, `treinar` não é chamado para dobras já gravadas.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar. Na dobra *k*, `treinar` recebe os jogos das outras 4 dobras; `validacao` é 10% desses jogos, separados por jogo, só para parada antecipada. As previsões são feitas na dobra *k*.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Commit.

### Task 8: Modelo 1 · LightGBM com atributos manuais

**Files:**
- Create: `cp3/classico.py`, `tests/cp3/test_classico.py`

**Interfaces:**
- Consumes: `Tempo`, `Modelo` e `rodar_cv`/`escolher_operacao`.
- Produces: `atributos_janela(t: Tempo) -> tuple[np.ndarray, list[str]]` (forma `(n_passos, F)`); `class Classico(Modelo)`; a CLI `python -m cp3.classico`, que roda a CV, escolhe a operação e grava `outputs/checkpoint-3/importancia_classico.csv`.

Atributos por passo de 0,5 s, em janela de 2 s centrada:
- média e desvio das 64 bandas;
- 20 MFCC (`scipy.fft.dct(logmel, type=2, norm="ortho")[:, :20]`) e os deltas, cada um com média e desvio;
- média e desvio das 8 séries de `quadros`;
- contexto: para `rms`, `e_voz`, `e_alta`, `fluxo` e `centroide`, a diferença entre a média da janela e a mediana dos 30 s anteriores, e a diferença para a mediana dos 10 s seguintes;
- o máximo de `rms` nos 10 s seguintes.

O rótulo positivo é `alvo_gaussiano > 0.5`.

- [ ] **Step 1: Testes que falham:**
  - `test_atributos_forma_e_nomes`: `F == len(nomes)`, sem nomes repetidos, tudo finito num tempo sintético com trecho de silêncio.
  - `test_contexto_detecta_subida`: série com `rms` constante e um degrau em 60 s → o atributo "subida vs 30 s anteriores" é máximo perto de 60 s.
  - `test_classico_aprende_sintetico`: 8 tempos sintéticos (ruído com rajadas de energia nos eventos), `treinar` e depois `prever` → AUC > 0,9 em 2 tempos novos.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar. Usar `lightgbm.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg/pos)`, com parada antecipada em 100 rodadas na `validacao`. O estado é o modelo mais a ordem dos atributos.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Com a extração em ≥ 90% do treino, rodar `$PY -m cp3.classico`. Esperado: `config_classico.json` gravado e F1 fora da dobra impresso.
- [ ] **Step 6:** Commit do código e de `outputs/checkpoint-3/{calibracao,config,importancia}_classico.*`.

### Task 9: Modelos 2 e 3 · TCN 1D e CNN 2D

**Files:**
- Create: `cp3/redes.py`, `cp3/treino_rede.py`, `tests/cp3/test_redes.py`

**Interfaces:**
- Consumes: `Tempo`, `reduzido`, `alvo_gaussiano` e `Modelo`.
- Produces:
  - `class TCN(nn.Module)`, com `__init__(entrada=128, canais=64, blocos=7, kernel=3)`: blocos residuais com dilatação `2**b` para `b` de 0 a 6, convoluções não causais e campo receptivo de 255 passos (±64 s). Entrada `(B, 128, T)` → saída `(B, T)` em logits.
  - `class CNN2D(nn.Module)`: 5 blocos conv3×3-BN-ReLU com canais (16, 32, 64, 64, 128) e pooling (tempo, frequência) = (1,2), (5,2), (1,2), (5,2), (1,4). Entrada `(B, 1, 1000, 64)` → saída `(B, 40)` em logits.
  - `class RedeTCN(Modelo)`: recortes aleatórios de 600 passos (5 min).
  - `class RedeCNN(Modelo)`: recortes de 20 s; metade dos recortes de cada lote contém um evento; SpecAugment com 2 máscaras de tempo ≤ 2 s e 2 de frequência ≤ 8 bandas, mais ganho aleatório de ±6 dB. `prever` usa janelas de 20 s com passo de 10 s e média nas sobreposições.
  - Treino comum: AdamW (lr 1e-3, weight decay 1e-4), BCE com `pos_weight = neg/pos` dos passos, parada antecipada pela perda de validação com paciência 5, no máximo 60 épocas (TCN) ou 30 (CNN), dispositivo `mps` com recuo para `cpu`.
  - As CLIs `python -m cp3.treino_rede --modelo tcn|cnn2d` rodam a CV e escolhem a operação.

- [ ] **Step 1: Testes que falham** (rodam na CPU):
  - `test_tcn_forma`: `(2, 128, 300)` → `(2, 300)`.
  - `test_tcn_campo_receptivo`: o gradiente da saída no passo 150 em relação à entrada é zero nos passos com `|i - 150| > 127` e diferente de zero em `|i - 150| ≤ 127`.
  - `test_cnn_forma`: `(2, 1, 1000, 64)` → `(2, 40)`.
  - `test_cnn_prever_cobre_tempo`: para um tempo de 95 s com uma rede constante, `prever` devolve `n_passos` valores, todos iguais.
  - `test_treino_reduz_perda`: com dados sintéticos (rajadas nos eventos), a perda de treino da época 3 fica abaixo da época 1, nas duas redes.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** Rodar a CV da TCN: `$PY -m cp3.treino_rede --modelo tcn`. Esperado: `config_tcn.json`.
- [ ] **Step 6:** Rodar a CV da CNN 2D: `$PY -m cp3.treino_rede --modelo cnn2d`. Esperado: `config_cnn2d.json`. Se o tempo apertar, reduzir o máximo de épocas, mas manter as 5 dobras e as 3 σ.
- [ ] **Step 7:** Commit do código e das saídas de calibração.

### Task 10: Leitura fora da dobra com o mesmo número de alertas

**Files:**
- Modify: `cp3/cv.py`
- Create: `cp3/relatorio_cv.py`

**Interfaces:**
- Consumes: `tabela_mesmo_k`, `linhas_de_base` (só para os jogos de **treino**, para obter *k*) e as configurações escolhidas.
- Produces: `outputs/checkpoint-3/cv_resumo.csv`, com uma linha por modelo e ponto de operação ("f1" e "mesmo_k").

- [ ] **Step 1:** Teste `test_resumo_tem_dois_pontos`: com previsões falsas, o CSV tem as linhas `f1` e `mesmo_k` de cada modelo, e na linha `mesmo_k` o número de alertas é igual ao da fusão.
- [ ] **Step 2:** Rodar o teste. Esperado: FAIL; implementar; esperado: PASS.
- [ ] **Step 3:** Rodar `$PY -m cp3.relatorio_cv`.
- [ ] **Step 4:** Commit.

### Task 11: Avaliação única no teste

**Files:**
- Create: `cp3/teste.py`, `tests/cp3/test_teste.py`

**Interfaces:**
- Consumes: `config_<modelo>.json`, os estados das dobras do σ escolhido, `carregar_tempo`, `suavizar`, `picos`, `top_k`, `avaliar`, `por_jogo`, `linhas_de_base` e `alertas_da_fusao`.
- Produces: `python -m cp3.teste --modelos classico tcn cnn2d`, que grava em `outputs/checkpoint-3/teste/`:
  - `metricas.csv`, com uma linha por (método, ponto de operação "f1" ou "mesmo_k") e as 3 linhas de base;
  - `por_classe.csv`, `por_jogo.csv` e `alertas.csv`;
  - `trava.json`, com o hash SHA-256 de `divisao.json` e das `config_*.json`, mais a data;
  - as figuras `comparacao.png`, `por_classe.png` e `linha_do_tempo_<jogo>.png` (curvas dos 3 modelos, da fusão e as anotações em 1 jogo de teste).

- [ ] **Step 1: Testes que falham:**
  - `test_trava_recusa_config_alterada` (Review Focus 5): com uma trava existente e uma `config_*.json` alterada, sai com erro sem escrever nada.
  - `test_trava_aceita_mesma_config`: com a mesma configuração, regenera as saídas de forma idempotente.
  - `test_media_das_dobras`: com 5 estados falsos de curvas constantes 0,1 a 0,5, a curva final é 0,3.
  - `test_mesmo_k_iguala_fusao`: na linha `mesmo_k`, os alertas de cada modelo são iguais aos da fusão.
- [ ] **Step 2:** Rodar os testes. Esperado: FAIL.
- [ ] **Step 3:** Implementar.
- [ ] **Step 4:** Rodar os testes. Esperado: PASS.
- [ ] **Step 5:** **Só com as 3 configurações congeladas**, rodar `$PY -m cp3.teste --modelos classico tcn cnn2d`. Se algum modelo não terminou, rodar com os prontos e registrar a ausência. Esperado: as tabelas impressas, com 100 jogos.
- [ ] **Step 6:** Commit do código e de `outputs/checkpoint-3/teste/`.

### Task 12 (opcional): Cortes de 30 s

**Files:**
- Create: `cp3/cortes.py`, `tests/cp3/test_cortes.py`

**Interfaces:**
- Consumes: `outputs/checkpoint-3/teste/alertas.csv`.
- Produces: `comandos_corte(video: Path, horarios_s: list[float], saida: Path, antes_s=15, depois_s=15) -> list[list[str]]`, e uma CLI que recebe o caminho de um vídeo local de um jogo de teste e roda os comandos.

- [ ] **Step 1:** Teste `test_comandos`: um horário em 10 s gera início em 0 (não negativo) e duração de 25 s; um horário em 100 s gera `-ss 85 -t 30`, com reencode (`-c:v libx264 -c:a aac`) e arquivo `saida/corte_001_0100s.mp4`.
- [ ] **Step 2:** Rodar o teste. Esperado: FAIL; implementar; esperado: PASS.
- [ ] **Step 3:** Testar com um vídeo 224p de um jogo de teste, baixado à parte e fora do git.
- [ ] **Step 4:** Commit do código.
