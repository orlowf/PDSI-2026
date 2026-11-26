"""Arquiteturas dos modelos 2 (TCN 1D) e 3 (CNN 2D) do Checkpoint 3."""

from torch import nn


class _BlocoTemporal(nn.Module):
    def __init__(self, canais, dilatacao, kernel, dropout):
        super().__init__()
        self.conv = nn.Conv1d(canais, canais, kernel, dilation=dilatacao, padding=dilatacao * (kernel - 1) // 2)
        self.norm = nn.BatchNorm1d(canais)
        self.ativacao = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.ativacao(self.norm(self.conv(x))))


class TCN(nn.Module):
    """Convoluções dilatadas não causais; campo receptivo de 255 passos (±64 s a 0,5 s)."""

    def __init__(self, entrada=128, canais=64, blocos=7, kernel=3, dropout=0.1):
        super().__init__()
        self.entrada = nn.Conv1d(entrada, canais, 1)
        self.blocos = nn.Sequential(*[_BlocoTemporal(canais, 2**b, kernel, dropout) for b in range(blocos)])
        self.saida = nn.Conv1d(canais, 1, 1)

    def forward(self, x):  # (B, entrada, T) -> (B, T) em logits
        return self.saida(self.blocos(self.entrada(x))).squeeze(1)


CANAIS_CNN = (16, 32, 64, 64, 128)
POOLS_CNN = ((1, 2), (5, 2), (1, 2), (5, 2), (1, 4))  # (tempo, frequência): 1000×64 -> 40×1


class CNN2D(nn.Module):
    def __init__(self, canais=CANAIS_CNN, pools=POOLS_CNN):
        super().__init__()
        camadas, anterior = [], 1
        for c, pool in zip(canais, pools):
            camadas += [nn.Conv2d(anterior, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(pool)]
            anterior = c
        self.corpo = nn.Sequential(*camadas)
        self.saida = nn.Conv2d(anterior, 1, 1)

    def forward(self, x):  # (B, 1, 1000, 64) -> (B, 40) em logits
        y = self.saida(self.corpo(x))
        return y.reshape(y.shape[0], -1)
