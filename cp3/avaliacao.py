"""Avaliação comum do Checkpoint 3, sobre o pareamento de ±15 s do Checkpoint 2."""

import json
from collections import defaultdict
from pathlib import Path

from avaliar_alternativas_checkpoint2 import measure
from avaliar_arousal import FAMILIES
from avaliar_checkpoint2_integral import load_half, predictions, read_manifest, read_source_annotations
from ouvindo_o_gol import ANNOTATIONS_DIR

RAIZ_CP2 = Path.home() / ".cache/pdsi-checkpoint-2-runtime/rodada"
MANIFESTO_CP2 = Path("outputs/checkpoint-2-integral/manifesto.csv")
CONFIG_CP2 = Path("outputs/checkpoint-2-integral/configuracao.json")
PARAMETROS_CP2 = ("percentil_regra", "percentil_ativacao", "peso_ativacao_fusao")
FAMILIAS = tuple(dict.fromkeys(FAMILIES.values()))


def avaliar(previsoes):
    metricas, acertos, anotados, detalhes = measure(list(previsoes))
    por_classe = {familia: (acertos[familia], anotados[familia]) for familia in FAMILIAS}
    return dict(metricas, por_classe=por_classe, detalhes=detalhes)


def por_jogo(previsoes):
    grupos = defaultdict(list)
    for previsao in previsoes:
        grupos[previsao[0]].append(previsao)
    linhas = []
    for jogo, partes in grupos.items():
        metricas, *_ = measure(partes)
        linhas.append(dict(jogo=jogo, **metricas))
    return linhas


def linhas_de_base(jogos, raiz_cp2=RAIZ_CP2, manifesto=MANIFESTO_CP2, config=CONFIG_CP2):
    """Regra, wav2vec e fusão do CP2 nos jogos pedidos, com os parâmetros congelados."""
    origens = read_manifest(Path(manifesto))
    anotacoes = read_source_annotations(ANNOTATIONS_DIR, {origens[j][0] for j in jogos})
    tempos = [load_half(Path(raiz_cp2), anotacoes, jogo, origens[jogo][0], numero)
              for jogo in jogos for numero in (1, 2)]
    configuracao = json.loads(Path(config).read_text(encoding="utf-8"))
    parametros = {chave: configuracao[chave] for chave in PARAMETROS_CP2}
    return {metodo: list(previsoes) for metodo, previsoes in predictions(tempos, parametros).items()}


def alertas_da_fusao(previsoes_fusao):
    return {(jogo, tempo): len(alertas) for jogo, tempo, _, alertas in previsoes_fusao}
