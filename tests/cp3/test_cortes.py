from pathlib import Path

from cp3.cortes import comandos_corte


def valor(comando, opcao):
    return comando[comando.index(opcao) + 1]


def test_comandos():
    saida = Path("clipes")
    primeiro, segundo = comandos_corte(Path("jogo.mkv"), [10.0, 100.0], saida)
    assert (valor(primeiro, "-ss"), valor(primeiro, "-t")) == ("0", "25")
    assert (valor(segundo, "-ss"), valor(segundo, "-t")) == ("85", "30")
    assert valor(segundo, "-c:v") == "libx264" and valor(segundo, "-c:a") == "aac"
    assert valor(segundo, "-i") == "jogo.mkv"
    assert segundo[-1] == str(saida / "corte_002_0100s.mp4")
    assert primeiro[0] == "ffmpeg"
