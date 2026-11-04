"""ABMIL (Attention-based Deep MIL, Ilse et al. 2018) 떨림 분류기.

bags.npz 입력: X (bag, inst, 100샘플, 6채널), y (bag,), subject (bag,)
  bag  = 20초 녹음 1개  -> 라벨 1개 (tremor=1 / non=0)
  inst = 2초 윈도우 19개 -> 라벨 없음. 어느 윈도우가 떨림인지 모델이 attention으로 찾는다.

python3 tremor/abmil.py            # 6명 LOSO + 자체검증
python3 tremor/abmil.py seojin     # 한 명만 hold-out, attention 출력
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from prep import load_split, OUT as BAGS

torch.manual_seed(0); np.random.seed(0)
EPOCHS, LR = 40, 1e-3


class ABMIL(nn.Module):
    def __init__(self, ch=6, h=64, att=32):
        super().__init__()
        # 1) instance encoder: 윈도우 하나(100샘플 x 6채널) -> 벡터 h개.
        #    Conv1d가 시간축을 훑으며 진동 패턴을 잡고, AvgPool로 길이를 1로 접는다.
        self.enc = nn.Sequential(
            nn.Conv1d(ch, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, h, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        # 2) gated attention (논문 식 9): a_k = softmax_k( w^T [tanh(V h_k) * sigmoid(U h_k)] )
        #    tanh 가지 = "무슨 특징인가", sigmoid 가지 = "얼마나 통과시킬까"(게이트)
        self.V, self.U, self.w = nn.Linear(h, att), nn.Linear(h, att), nn.Linear(att, 1)
        # 3) bag classifier: attention 가중합 벡터 -> 로짓 1개
        self.cls = nn.Linear(h, 1)

    def forward(self, x):                       # x: (B, K, T, C)
        B, K, T, C = x.shape
        # 인코더는 윈도우 단위로 동작하므로 (B*K) 개의 독립 시퀀스로 펼친다. Conv1d는 (N, C, T) 순서.
        h = self.enc(x.reshape(B * K, T, C).transpose(1, 2)).reshape(B, K, -1)   # (B, K, h)
        a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))), dim=1)  # (B, K, 1), K축 합=1
        z = (a * h).sum(1)                      # (B, h)  attention 가중 평균 = bag 표현
        return self.cls(z).squeeze(-1), a.squeeze(-1)   # 로짓 (B,), attention (B, K)


def fit(Xtr, ytr):
    m = ABMIL(Xtr.shape[-1])
    opt = torch.optim.Adam(m.parameters(), LR)
    X, y = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    # 라벨은 bag 단위 하나뿐 -> 손실도 bag 로짓에만 걸린다. instance 라벨 없이 학습되는 게 MIL의 핵심.
    for _ in range(EPOCHS):
        for i in torch.randperm(len(X)).split(16):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(m(X[i])[0], y[i])
            loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def predict(m, X):
    logit, a = m(torch.tensor(X))
    return torch.sigmoid(logit).numpy(), a.numpy()


def balanced_acc(y, p):
    pred = p > 0.5
    return 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())


def loso(verbose=True):
    subjects = sorted(set(np.load(BAGS)["subject"]))
    res = {}
    for s in subjects:
        Xtr, ytr, Xte, yte = load_split(s)
        p, _ = predict(fit(Xtr, ytr), Xte)
        res[s] = balanced_acc(yte, p)
        if verbose: print(f"  hold-out {s:8s} bal.acc {res[s]:.3f}  (n={len(yte)})")
    return res


def demo():
    # 1) 모양 검사: 랜덤 입력 -> 로짓 (B,), attention (B,K) 합이 1
    m = ABMIL()
    lg, a = m(torch.randn(3, 19, 100, 6))
    assert lg.shape == (3,) and a.shape == (3, 19) and torch.allclose(a.sum(1), torch.ones(3))
    # 2) LOSO
    print("LOSO (6 folds):")
    res = loso()
    mean = np.mean(list(res.values()))
    print(f"  mean bal.acc {mean:.3f}   (RF 0.992 / kNN 0.950)")
    assert mean > 0.85, "ABMIL 학습 실패 -> lr/epoch 확인"
    # 3) attention이 쓸모 있는지: 떨림 bag에서 attention 최대 윈도우의 3~12Hz 파워가 평균보다 커야 한다
    from prep import bandpower
    Xtr, ytr, Xte, yte = load_split("seojin")
    p, a = predict(fit(Xtr, ytr), Xte)
    bp = bandpower(Xte, 3, 12)                                   # (bag, inst)
    top = bp[np.arange(len(a)), a.argmax(1)]                     # attention 1등 윈도우의 파워
    ratio = np.median(top[yte == 1] / bp[yte == 1].mean(1))
    print(f"\nseojin hold-out: bal.acc {balanced_acc(yte, p):.3f}")
    print(f"떨림 bag에서 attention 1등 윈도우 파워 / bag 평균 파워 = {ratio:.2f}x  (>1이면 떨림 구간을 가리킴)")
    print("attention 예시(떨림 bag 1개, 19 윈도우):", np.round(a[yte == 1][0], 2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    if len(sys.argv) > 1:
        Xtr, ytr, Xte, yte = load_split(sys.argv[1])
        p, a = predict(fit(Xtr, ytr), Xte)
        print(f"hold-out {sys.argv[1]}: bal.acc {balanced_acc(yte, p):.3f}")
        for yi, pi, ai in zip(yte, p, a):
            print(f"  y={yi} p={pi:.2f}  att={np.round(ai, 2)}")
    else:
        demo()
