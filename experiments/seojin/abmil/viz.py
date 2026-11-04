"""ABMIL attention 시각화: 신호(표준화) + 떨림 대역 주파수 + attention을 붉은 배경으로.

행 1: 가속도 크기(표준화, z-score). 배경 빨강 농도 = 그 2초 윈도우의 attention.
행 2: 스펙트로그램(0~15Hz). 떨림 대역 3~12Hz를 점선으로 표시, 흰 점 = 윈도우별 피크 Hz.
모든 bag은 train 통계로 z-score 표준화 -> 같은 축에서 비교 가능.

python3 tremor/viz.py march            # hold-out 피험자. 결과: tremor/data/viz_march.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
from prep import load_split, FS, WIN, HOP, T
from abmil import fit, predict

ROOT = Path(__file__).resolve().parent


def stitch(bag):
    """(19, 100, 6) 중첩 윈도우 -> (T, 6) 연속 신호 (중첩 구간은 첫 윈도우 값 사용)."""
    x = np.zeros((T, bag.shape[-1]), np.float32)
    for k, w in enumerate(bag):
        x[k * HOP:k * HOP + WIN] = w
    return x


def spectrogram(sig):
    """윈도우별 FFT -> (freq, n_win) 파워, 윈도우별 피크 Hz(2~15Hz)."""
    f = np.fft.rfftfreq(WIN, 1 / FS)
    P = (np.abs(np.fft.rfft(sig, axis=1)) ** 2).sum(-1)            # (n_win, freq) 3축 합
    m = (f >= 2) & (f < 15)
    return f, P.T, f[m][P[:, m].argmax(1)]


def plot(holdout, n_show=8):
    Xtr, ytr, Xte, yte = load_split(holdout)
    p, A = predict(fit(Xtr, ytr), Xte)
    order = np.concatenate([np.where(yte == 1)[0][:n_show // 2], np.where(yte == 0)[0][:n_show // 2]])
    fig, axes = plt.subplots(len(order) * 2, 1, figsize=(14, 2.2 * len(order)), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.3, 1] * len(order), hspace=0.08))
    t = np.arange(T) / FS
    for r, i in enumerate(order):
        bag, a = Xte[i], A[i]
        sig = stitch(bag[:, :, :3])                               # 표준화된 가속도 3축
        mag = np.linalg.norm(sig, axis=1)
        ax, ax2 = axes[2 * r], axes[2 * r + 1]
        # attention -> 붉은 배경. 균등(1/K)이면 투명, 0.3 이상이면 최대 농도. bag 간 절대 비교 가능.
        shade = np.clip((a - 1 / len(a)) / (0.3 - 1 / len(a)), 0, 1)
        for k, sk in enumerate(shade):
            ax.axvspan(k * HOP / FS, (k * HOP + WIN) / FS, color="red", alpha=float(sk) * 0.5, lw=0)
            ax2.axvspan(k * HOP / FS, (k * HOP + WIN) / FS, color="red", alpha=float(sk) * 0.3, lw=0)
        ax.plot(t, mag, color="k", lw=0.6)
        ax.set_ylabel("|acc| z", fontsize=8); ax.set_ylim(0, 8)
        ax.text(0.01, 0.95, f"{'TREMOR' if yte[i] else 'non'}  p={p[i]:.2f}  att max={a.max():.2f}",
                transform=ax.transAxes, va="top", fontsize=9, fontweight="bold", color="darkred" if yte[i] else "navy")
        f, P, pk = spectrogram(bag[:, :, :3])
        ax2.pcolormesh((np.arange(len(a)) + 1) * HOP / FS, f, np.log10(P + 1e-6), cmap="Greys", shading="auto")
        ax2.scatter((np.arange(len(a)) + 1) * HOP / FS, pk, s=10, color="white", edgecolor="k", lw=0.4, zorder=3)
        for hz in (3, 12): ax2.axhline(hz, color="orange", ls=":", lw=0.8)
        ax2.set_ylim(0, 15); ax2.set_ylabel("Hz", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"ABMIL attention · hold-out {holdout} | red shade = attention above uniform (white = evenly spread), dotted = 3-12 Hz tremor band, white dot = peak Hz", fontsize=11)
    out = ROOT / f"data/viz_{holdout}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    print(plot(sys.argv[1] if len(sys.argv) > 1 else "march"))
