"""Treino e previsão dos modelos neurais do Checkpoint 3 (TCN 1D e CNN 2D)."""

import argparse
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

from cp3.conjunto import PASSO_S, QUADROS_POR_PASSO, reduzido
from cp3.redes import CNN2D, TCN
from cp3.rotulos import alvo_gaussiano

PASSOS_CNN = 40  # 20 s
QUADROS_CNN = PASSOS_CNN * QUADROS_POR_PASSO
SALTO_PREVISAO_CNN = 20  # janelas de 20 s a cada 10 s


def _dispositivo_padrao():
    return "mps" if torch.backends.mps.is_available() else "cpu"


class _RedeBase:
    def __init__(self, dispositivo=None, max_epocas=60, paciencia=5, recortes_por_epoca=512, lote=16, lr=1e-3):
        self.dispositivo = dispositivo or _dispositivo_padrao()
        self.max_epocas = max_epocas
        self.paciencia = paciencia
        self.recortes_por_epoca = recortes_por_epoca
        self.lote = lote
        self.lr = lr
        self.historico = []
        self._cache = {}

    def _preparado(self, t):
        chave = (t.jogo, t.numero)
        if chave not in self._cache:
            self._cache[chave] = self._preparar(t)
        return self._cache[chave]

    def _dados(self, tempos, sigma_s):
        return [(t, self._preparado(t), alvo_gaussiano([e for e, _ in t.eventos], t.n_passos, PASSO_S, sigma_s))
                for t in tempos]

    def _perda(self, logits, alvo, peso):
        return F.binary_cross_entropy_with_logits(logits, alvo, pos_weight=peso)

    def treinar(self, treino, validacao, sigma_s):
        torch.manual_seed(2026)
        rng = np.random.default_rng(2026)
        dados = self._dados(treino, sigma_s)
        dados_val = self._dados(validacao, sigma_s)
        media = float(np.mean(np.concatenate([alvo for *_, alvo in dados])))
        peso = torch.tensor(float(np.clip((1 - media) / max(media, 1e-6), 1, 100)), device=self.dispositivo)
        rede = self.nova().to(self.dispositivo)
        otimizador = torch.optim.AdamW(rede.parameters(), lr=self.lr, weight_decay=1e-4)
        melhor, melhor_estado, sem_melhora = float("inf"), None, 0
        self.historico = []
        for _ in range(self.max_epocas):
            rede.train()
            perdas = []
            for _ in range(max(1, self.recortes_por_epoca // self.lote)):
                x, y = self._lote(dados, rng, treinando=True)
                perda = self._perda(rede(x.to(self.dispositivo)), y.to(self.dispositivo), peso)
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
                perdas.append(perda.item())
            self.historico.append(float(np.mean(perdas)))
            perda_val = self._perda_validacao(rede, dados_val, peso)
            if perda_val < melhor - 1e-4:
                melhor, sem_melhora = perda_val, 0
                melhor_estado = copy.deepcopy({k: v.detach().cpu() for k, v in rede.state_dict().items()})
            else:
                sem_melhora += 1
                if sem_melhora >= self.paciencia:
                    break
        rede.load_state_dict(melhor_estado)
        return rede.eval()

    def salvar(self, estado, base):
        torch.save(estado.state_dict(), base.with_suffix(".pt"))

    def carregar(self, base):
        rede = self.nova()
        rede.load_state_dict(torch.load(base.with_suffix(".pt"), map_location="cpu"))
        return rede.eval()


class RedeTCN(_RedeBase):
    PASSOS_RECORTE = 600  # 5 min

    def nova(self):
        return TCN()

    def _preparar(self, t):
        return reduzido(t)

    def _lote(self, dados, rng, treinando):
        comprimento = min(self.PASSOS_RECORTE, min(len(alvo) for *_, alvo in dados))
        xs, ys = [], []
        for i in rng.integers(0, len(dados), self.lote):
            _, x, alvo = dados[i]
            inicio = int(rng.integers(0, len(alvo) - comprimento + 1))
            xs.append(x[inicio:inicio + comprimento].T)
            ys.append(alvo[inicio:inicio + comprimento])
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    @torch.no_grad()
    def _perda_validacao(self, rede, dados_val, peso):
        rede.eval()
        perdas = [self._perda(rede(torch.from_numpy(x.T[None]).to(self.dispositivo)),
                              torch.from_numpy(alvo[None]).to(self.dispositivo), peso).item()
                  for _, x, alvo in dados_val]
        return float(np.mean(perdas))

    @torch.no_grad()
    def prever(self, estado, t):
        estado = estado.to(self.dispositivo).eval()
        x = torch.from_numpy(self._preparado(t).T[None]).to(self.dispositivo)
        return torch.sigmoid(estado(x))[0].cpu().numpy().astype(np.float32)


class RedeCNN(_RedeBase):
    def __init__(self, recortes_por_epoca=4000, lote=32, max_epocas=30, **kwargs):
        super().__init__(recortes_por_epoca=recortes_por_epoca, lote=lote, max_epocas=max_epocas, **kwargs)

    def nova(self):
        return CNN2D()

    def _preparar(self, t):
        """Mediana e MAD de cada banda no tempo inteiro; o log-mel continua em memmap."""
        mel = np.asarray(t.logmel, dtype=np.float32)
        mediana = np.median(mel, axis=0)
        mad = np.median(np.abs(mel - mediana), axis=0) * 1.4826 + 1e-6
        return mediana.astype(np.float32), mad.astype(np.float32)

    def _janela(self, t, inicio_passo, ganho_db=0.0):
        mediana, mad = self._preparado(t)
        a = inicio_passo * QUADROS_POR_PASSO
        mel = np.asarray(t.logmel[a:a + QUADROS_CNN], dtype=np.float32)
        if len(mel) < QUADROS_CNN:
            mel = np.vstack([mel, np.repeat(mediana[None], QUADROS_CNN - len(mel), axis=0)])
        return (mel + ganho_db - mediana) / mad

    @staticmethod
    def _mascarar(x, rng):
        for _ in range(2):
            largura = int(rng.integers(0, 101))
            inicio = int(rng.integers(0, QUADROS_CNN - largura + 1))
            x[inicio:inicio + largura] = 0
            largura = int(rng.integers(0, 9))
            inicio = int(rng.integers(0, x.shape[1] - largura + 1))
            x[:, inicio:inicio + largura] = 0
        return x

    def _inicio(self, t, alvo, rng, com_evento):
        ultimo = max(0, len(alvo) - PASSOS_CNN)
        passos_eventos = [int(e / PASSO_S) for e, _ in t.eventos if int(e / PASSO_S) < len(alvo)]
        if com_evento and passos_eventos:
            passo = passos_eventos[int(rng.integers(len(passos_eventos)))]
            return int(np.clip(rng.integers(passo - PASSOS_CNN + 1, passo + 1), 0, ultimo))
        return int(rng.integers(0, ultimo + 1))

    def _recorte(self, item, rng, com_evento, treinando):
        t, _, alvo = item
        inicio = self._inicio(t, alvo, rng, com_evento)
        x = self._janela(t, inicio, rng.uniform(-6, 6) if treinando else 0.0)
        if treinando:
            x = self._mascarar(x, rng)
        y = np.zeros(PASSOS_CNN, dtype=np.float32)
        trecho = alvo[inicio:inicio + PASSOS_CNN]
        y[:len(trecho)] = trecho
        return x, y

    def _lote(self, dados, rng, treinando):
        pares = [self._recorte(dados[int(i)], rng, k % 2 == 0, treinando)
                 for k, i in enumerate(rng.integers(0, len(dados), self.lote))]
        xs, ys = zip(*pares)
        return torch.from_numpy(np.stack(xs)[:, None]), torch.from_numpy(np.stack(ys))

    @torch.no_grad()
    def _perda_validacao(self, rede, dados_val, peso):
        rede.eval()
        rng = np.random.default_rng(7)
        perdas = []
        for _ in range(4):
            x, y = self._lote(dados_val, rng, treinando=False)
            perdas.append(self._perda(rede(x.to(self.dispositivo)), y.to(self.dispositivo), peso).item())
        return float(np.mean(perdas))

    @torch.no_grad()
    def prever(self, estado, t):
        estado = estado.to(self.dispositivo).eval()
        n = t.n_passos
        inicios = list(range(0, max(1, n - PASSOS_CNN + 1), SALTO_PREVISAO_CNN))
        if inicios[-1] != max(0, n - PASSOS_CNN):
            inicios.append(max(0, n - PASSOS_CNN))
        soma, contagem = np.zeros(n, np.float64), np.zeros(n, np.float64)
        for k in range(0, len(inicios), 64):
            grupo = inicios[k:k + 64]
            x = torch.from_numpy(np.stack([self._janela(t, i) for i in grupo])[:, None]).to(self.dispositivo)
            p = torch.sigmoid(estado(x)).cpu().numpy()
            for inicio, valores in zip(grupo, p):
                fim = min(n, inicio + PASSOS_CNN)
                soma[inicio:fim] += valores[:fim - inicio]
                contagem[inicio:fim] += 1
        return (soma / np.maximum(contagem, 1)).astype(np.float32)


MODELOS = {"tcn": RedeTCN, "cnn2d": RedeCNN}


def main(argv=None):
    from cp3.conjunto import carregar_tempo, tempos_prontos
    from cp3.cv import escolher_operacao, rodar_cv
    from cp3.divisao import ler_divisao

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", choices=sorted(MODELOS), required=True)
    parser.add_argument("--max-epocas", type=int)
    args = parser.parse_args(argv)
    divisao = ler_divisao()
    tempos = [carregar_tempo(j, n) for j, n in tempos_prontos(divisao["treino"])]
    print(f"{len(tempos)} tempos de treino prontos", flush=True)
    opcoes = {"max_epocas": args.max_epocas} if args.max_epocas else {}
    rodar_cv(args.modelo, MODELOS[args.modelo](**opcoes), tempos, divisao["dobras"])
    print(json.dumps(escolher_operacao(args.modelo, tempos)), flush=True)


if __name__ == "__main__":
    main()
