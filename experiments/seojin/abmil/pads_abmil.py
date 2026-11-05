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


def best_thr(y, p):
    ths = np.linspace(0.05, 0.95, 181)
    return max(ths, key=lambda t: balanced_acc(y, (p > t).astype(float) * 0.5 + 0.25))   # p>t 를 0.75/0.25로 바꿔 bal.acc 계산


def cv_nested(X, y, folds=5, inner=3, seed=0):
    """바깥 fold마다: train을 inner-fold로 돌려 out-of-fold p -> 임계값 선택 -> train 전체로 학습 -> test에 적용.
    클래스 가중 손실(balance=True). 반환: p, A, thr(fold별 임계값, bag마다 자기 fold 값)."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    p, A, thr = np.zeros(len(y)), np.zeros(X.shape[:2], np.float32), np.zeros(len(y))
    for k in range(folds):
        te = idx[k::folds]; tr = np.setdiff1d(idx, te)
        flat = X[tr].reshape(-1, 6); mu, sd = flat.mean(0), flat.std(0) + 1e-8
        norm = lambda Z: (Z - mu) / sd
        pin = np.zeros(len(tr))
        for j in range(inner):                                        # 안쪽: train만으로 out-of-fold p
            ite = np.arange(len(tr))[j::inner]; itr = np.setdiff1d(np.arange(len(tr)), ite)
            pin[ite], _ = predict(fit(norm(X[tr][itr]), y[tr][itr], balance=True), norm(X[tr][ite]))
        t = best_thr(y[tr], pin)
        m = fit(norm(X[tr]), y[tr], balance=True)
        p[te], A[te] = predict(m, norm(X[te])); thr[te] = t
        print(f"  outer fold {k+1}/{folds}: thr={t:.2f}  (inner bal.acc {balanced_acc(y[tr], (pin > t).astype(float)*0.5+0.25):.3f})", flush=True)
    return p, A, thr


def report(y, p, thr, band=0.05, folds=5, seed=0):
    """fold별 precision/sensitivity/specificity/F1/bal.acc 평균±표준편차. band: 임계값 ±band는 uncertain으로 보류."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y))
    rows = []
    for k in range(folds):
        te = idx[k::folds]; yy, pp, tt = y[te], p[te], thr[te]
        sure = np.abs(pp - tt) > band; pred = pp > tt
        TP = (pred & (yy==1) & sure).sum(); FN = (~pred & (yy==1) & sure).sum()
        FP = (pred & (yy==0) & sure).sum(); TN = (~pred & (yy==0) & sure).sum()
        sens, spec = TP/max(TP+FN,1), TN/max(TN+FP,1)
        rows.append([TP/max(TP+FP,1), sens, spec, 2*TP/max(2*TP+FP+FN,1), (sens+spec)/2, 1-sure.mean()])
    r = np.array(rows); m, s = r.mean(0), r.std(0)
    names = ["precision", "sensitivity", "specificity", "F1", "bal.acc", "uncertain율"]
    print(f"  (uncertain = |p - thr| <= {band}, 보류 제외하고 계산)")
    for n, a, b in zip(names, m, s): print(f"  {n:12s} {a:.3f} ± {b:.3f}")
    return dict(zip(names, m))


def auc(y, p):
    r = p.argsort().argsort(); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def demo():
    X, y, subj, inst_task = build() if not OUT.exists() else (lambda d: (d["X"], d["y"], d["subject"], d["inst_task"]))(np.load(OUT))
    print(f"bags {X.shape}  tremor {y.sum()} / healthy {(y==0).sum()}   instance/bag {X.shape[1]} ({len(set(inst_task))} 녹음 x {X.shape[1]//len(set(inst_task))} 윈도우)")
    import abmil; abmil.EPOCHS = 25
    CV = OUT.with_name("pads_cv_nested.npz")
    if CV.exists(): d = np.load(CV); p, A, thr = d["p"], d["A"], d["thr"]
    else: p, A, thr = cv_nested(X, y); np.savez_compressed(CV, p=p, A=A, thr=thr)
    print(f"\n5-fold nested CV (클래스 가중 + 학습 fold 내 임계값): AUC {auc(y, p):.3f}   fold별 thr {np.unique(thr).round(2)}")
    print("보류 없음 (band=0):"); report(y, p, thr, band=0)
    print("보류 ±0.05:"); r = report(y, p, thr, band=0.05)
    noev = (np.abs(p - 0.5) < 0.03)
    print(f"\n근거 없는 사람(p 0.47~0.53) {noev.sum()}명 -> 판정: tremor {((p>thr)&noev&(np.abs(p-thr)>0.05)).sum()}, non {((p<=thr)&noev&(np.abs(p-thr)>0.05)).sum()}, uncertain {(noev&(np.abs(p-thr)<=0.05)).sum()}")
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
