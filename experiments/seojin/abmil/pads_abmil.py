"""PADS(실제 환자) ABMIL: bag = 피험자 1명, instance = 모든 과제·양손목 녹음의 2초 윈도우.

라벨(ada-iot-2ne1/scripts/prepare_pads.py와 동일):
  Healthy -> 0, Essential Tremor 또는 코멘트에 tremor/mixed type -> 1, 그 외(무진전형 PD, MS...) 제외
전처리: 첫 0.5초(워치 진동 큐) 제거 -> 100Hz를 50Hz로 다운샘플 -> prep.highpass -> 2초 윈도우, 1초 hop
결과: 피험자 단위 5-fold CV balanced acc + AUC, attention이 어느 과제에 몰리는지.

python3 tremor/pads_abmil.py
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from prep import highpass
from abmil import fit, predict, balanced_acc

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
import os
PADS = Path(os.environ.get("PADS_RAW", ROOT / "data_sources/seojin/pads/raw")) / "physionet.org/files/parkinsons-disease-smartwatch/1.0.0"
OUT = HERE / "data/pads_bags.npz"
DROP, WIN, HOP = 50, 100, 50          # 100Hz 기준 0.5초 제거; 50Hz 기준 2초 윈도우, 1초 hop
TASKS = ["Relaxed", "RelaxedTask", "StretchHold", "HoldWeight", "LiftHold", "CrossArms",
         "DrinkGlas", "PointFinger", "TouchNose", "TouchIndex", "Entrainment"]
RAD2DEG = 180 / np.pi                 # prep.py 채널 단위(gyro dps)에 맞춤


def label(p):
    if p["condition"] == "Healthy": return 0
    import re
    if p["condition"] == "Essential Tremor" or re.search("tremor|mixed type", p["disease_comment"], re.I): return 1
    return None


def load_recording(path):
    """txt -> (n_win, 100, 6) 50Hz 윈도우."""
    x = np.loadtxt(path, delimiter=",", usecols=range(1, 7), dtype=np.float32)[DROP:]
    x[:, 3:] *= RAD2DEG
    x = x[: len(x) // 2 * 2].reshape(-1, 2, 6).mean(1)           # 100Hz -> 50Hz (쌍 평균)
    x = highpass(x)
    return np.stack([x[i:i + WIN] for i in range(0, len(x) - WIN + 1, HOP)])


def build():
    X, y, subj, inst_task = [], [], [], None
    for pf in sorted((PADS / "patients").glob("patient_*.json")):
        p = json.loads(pf.read_text()); lab = label(p)
        if lab is None: continue
        sid, bag, tags = p["id"], [], []
        for t in TASKS:
            for side in ("LeftWrist", "RightWrist"):
                f = PADS / f"movement/timeseries/{sid}_{t}_{side}.txt"
                if not f.exists(): continue
                w = load_recording(f); bag.append(w); tags += [f"{t}/{side[0]}"] * len(w)
        X.append(np.concatenate(bag)); y.append(lab); subj.append(sid)
        if inst_task is None: inst_task = np.array(tags)
    K = min(len(b) for b in X)
    assert all(len(b) == K for b in X), "피험자별 instance 수가 다름 -> 패딩 필요"
    X = np.stack(X).astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, y=np.array(y), subject=np.array(subj), inst_task=inst_task)
    return X, np.array(y), np.array(subj), inst_task


def cv(X, y, folds=5, seed=0):
    """피험자=bag 이므로 bag 단위 분할이 곧 subject-independent. train 통계로만 정규화."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    p, A = np.zeros(len(y)), np.zeros(X.shape[:2], np.float32)
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        flat = X[tr].reshape(-1, 6); mu, sd = flat.mean(0), flat.std(0) + 1e-8
        m = fit((X[tr] - mu) / sd, y[tr])
        p[te], A[te] = predict(m, (X[te] - mu) / sd)
        print(f"  fold {k+1}/{folds} done", flush=True)
    return p, A


def auc(y, p):
    r = p.argsort().argsort(); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def demo():
    X, y, subj, inst_task = build() if not OUT.exists() else (lambda d: (d["X"], d["y"], d["subject"], d["inst_task"]))(np.load(OUT))
    print(f"bags {X.shape}  tremor {y.sum()} / healthy {(y==0).sum()}   instance/bag {X.shape[1]} ({len(set(inst_task))} 녹음 x {X.shape[1]//len(set(inst_task))} 윈도우)")
    import abmil; abmil.EPOCHS = 25
    CV = OUT.with_name("pads_cv.npz")
    if CV.exists(): d = np.load(CV); p, A = d["p"], d["A"]
    else: p, A = cv(X, y); np.savez_compressed(CV, p=p, A=A)
    print(f"\n5-fold subject CV: bal.acc {balanced_acc(y, p):.3f}   AUC {auc(y, p):.3f}   (RF B2 subject: 0.79 / 0.91)")
    # attention 해석: 전체로 학습해 떨림군의 attention을 과제별로 합산
    flat = X.reshape(-1, 6); mu, sd = flat.mean(0), flat.std(0) + 1e-8
    _, a = predict(fit((X - mu) / sd, y), (X - mu) / sd)
    print("\n떨림군 attention 과제별 합 (균등이면 각 1/11 = 9.1%):")
    for t in TASKS:
        m = np.char.startswith(inst_task, t + "/")
        print(f"  {t:12s} {a[y==1][:, m].sum(1).mean()*100:5.1f}%   (healthy {a[y==0][:, m].sum(1).mean()*100:5.1f}%)")
    assert np.isfinite(p).all()


if __name__ == "__main__":
    demo()
