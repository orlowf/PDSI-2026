import numpy as np
import torch
from torch import nn

from cp3.redes import CNN2D, TCN
from cp3.treino_rede import RedeCNN, RedeTCN
from tests.cp3.test_classico import tempo_com_rajadas


def test_tcn_forma():
    assert TCN()(torch.zeros(2, 128, 300)).shape == (2, 300)


def test_tcn_campo_receptivo():
    rede = TCN().eval()
    x = torch.randn(1, 128, 400, requires_grad=True)
    rede(x)[0, 150].backward()
    influencia = x.grad.abs().sum(dim=1)[0]
    distancia = (torch.arange(400) - 150).abs()
    assert influencia[distancia > 127].sum() == 0
    assert (influencia[distancia <= 127] > 0).all()


def test_cnn_forma():
    assert CNN2D()(torch.zeros(2, 1, 1000, 64)).shape == (2, 40)


class Constante(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 40)


def test_cnn_prever_cobre_tempo():
    t = tempo_com_rajadas(3, n_passos=190, eventos_s=[])
    p = RedeCNN(dispositivo="cpu").prever(Constante(), t)
    assert p.shape == (190,)
    assert np.allclose(p, 0.5)


def test_tcn_aprende_e_reduz_perda():
    treino = [tempo_com_rajadas(s) for s in range(6)]
    rede = RedeTCN(dispositivo="cpu", max_epocas=3, paciencia=10, recortes_por_epoca=48)
    rede.treinar(treino, [tempo_com_rajadas(99)], 4.0)
    assert rede.historico[2] < rede.historico[0]


def test_cnn_aprende_e_reduz_perda():
    treino = [tempo_com_rajadas(s) for s in range(6)]
    rede = RedeCNN(dispositivo="cpu", max_epocas=3, paciencia=10, recortes_por_epoca=96)
    rede.treinar(treino, [tempo_com_rajadas(99)], 4.0)
    assert rede.historico[2] < rede.historico[0]


def test_salvar_e_carregar(tmp_path):
    rede = RedeTCN(dispositivo="cpu", max_epocas=1, recortes_por_epoca=8)
    estado = rede.treinar([tempo_com_rajadas(1)], [tempo_com_rajadas(2)], 4.0)
    rede.salvar(estado, tmp_path / "dobra0")
    t = tempo_com_rajadas(5)
    assert np.allclose(rede.prever(estado, t), rede.prever(rede.carregar(tmp_path / "dobra0"), t), atol=1e-6)
