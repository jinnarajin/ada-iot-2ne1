"""Tremor / non-tremor IMU 전처리: 두 repo -> MIL(ABMIL)용 npz 하나.

bag  = 20초 녹음 1개 (label: tremor=1 / non-tremor=0)
inst = 2초 윈도우, 1초 hop -> bag당 19개
chan = acc xyz(g), gyro xyz(dps), 중력/드리프트 제거됨

python3 tremor/prep.py        # 빌드 + 자체검증
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]          # 레포 루트
RAW, OUT = ROOT / "data/raw", HERE / "data/bags.npz"

FS, DUR = 50.0, 20.0                 # ds2가 ~50Hz(지터)라 50Hz가 정직한 공통 그리드. 진전 3~12Hz엔 충분
T = int(FS * DUR)                    # 1000
WIN, HOP = int(2 * FS), int(1 * FS)  # 100 / 50
MA = 51                              # ~1s 이동평균 = 약 0.4Hz 고역통과 (중력·자세 제거)
CH = ["acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]
# angle_*_deg는 버린다: 절대 자세라 드리프트/피험자 자세가 그대로 라벨 누설이 됨


def parse(folder: str):
    """'tremor_seojin' / 'nontremor_BIBI' / 'mu_still' -> (label, subject)."""
    toks = folder.split("_")
    for i, tk in enumerate(toks):
        if tk in ("tremor", "nontremor", "still"):
            return int(tk == "tremor"), "_".join(toks[:i] + toks[i + 1:])
    raise ValueError(folder)


def load(path: Path):
    """CSV -> 50Hz 균일 (T, 6). 20초를 못 채우면 None."""
    df = pd.read_csv(path)
    t = df["elapsed_ms"].to_numpy(float) / 1000.0
    t, keep = np.unique(t, return_index=True)      # np.interp는 단조증가 요구 (ds2에 중복 타임스탬프 있음)
    if t[-1] < DUR - 0.05:
        return None
    grid = np.arange(T) / FS
    return np.stack([np.interp(grid, t, df[c].to_numpy(float)[keep]) for c in CH], 1)


def highpass(x):
    """x - 이동평균. 중력 성분과 저주파 드리프트 제거 (zero-phase, scipy 불필요)."""
    pad = np.pad(x, ((MA // 2, MA // 2), (0, 0)), mode="edge")
    k = np.ones(MA) / MA
    ma = np.stack([np.convolve(pad[:, c], k, "valid") for c in range(x.shape[1])], 1)
    return x - ma


def windows(x):
    """(T,6) -> (n_inst, WIN, 6)"""
    idx = range(0, T - WIN + 1, HOP)
    return np.stack([x[i:i + WIN] for i in idx])


def build():
    X, y, subj, src, files = [], [], [], [], []
    for ds in sorted(p for p in RAW.iterdir() if p.is_dir()):
        for folder in sorted(p for p in ds.iterdir() if p.is_dir()):
            lab, who = parse(folder.name)
            for f in sorted(folder.glob("*.csv")):
                sig = load(f)
                if sig is None:
                    print(f"  skip (짧음): {f.relative_to(RAW)}", file=sys.stderr)
                    continue
                X.append(windows(highpass(sig)).astype(np.float32))
                y.append(lab); subj.append(who); src.append(ds.name)
                files.append(str(f.relative_to(RAW)))
    X = np.stack(X)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, y=np.array(y, np.int64), subject=np.array(subj),
                        source=np.array(src), files=np.array(files), channels=np.array(CH), fs=FS)
    return X, np.array(y), np.array(subj), np.array(src)


def load_split(holdout, path=OUT):
    """피험자 1명 hold-out. 정규화 통계는 train에서만 -> 누설 없음."""
    d = np.load(path, allow_pickle=False)
    X, y, s = d["X"], d["y"], d["subject"]
    te = s == holdout
    assert te.any(), f"unknown subject {holdout}: {sorted(set(s))}"
    flat = X[~te].reshape(-1, X.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    return (X[~te] - mu) / sd, y[~te], (X[te] - mu) / sd, y[te]


def bandpower(X, lo, hi):
    """윈도우별 lo~hi Hz 대역 파워 합 (전 채널)."""
    f = np.fft.rfftfreq(WIN, 1 / FS)
    p = np.abs(np.fft.rfft(X, axis=-2)) ** 2
    m = (f >= lo) & (f < hi)
    return p[..., m, :].sum((-1, -2))


def demo():
    X, y, subj, src = build()
    n_inst = (T - WIN) // HOP + 1
    assert X.shape == (len(y), n_inst, WIN, len(CH)), X.shape
    assert np.isfinite(X).all()
    assert len(set(subj)) == 6 and set(np.unique(y)) == {0, 1}

    tp = bandpower(X, 3, 12).mean(1)              # bag당 평균 진전대역 파워
    sep = np.median(tp[y == 1]) / np.median(tp[y == 0])
    assert sep > 3, f"진전 대역 분리 실패 (ratio={sep:.2f}) -> 필터/리샘플 확인"

    print(f"\nX {X.shape}  (bags, inst, samples, ch)  {X.nbytes/1e6:.1f} MB -> {OUT.relative_to(ROOT)}")
    print(f"tremor {int(y.sum())} / non {int((y==0).sum())}")
    for s in sorted(set(subj)):
        m = subj == s
        print(f"  {s:8s} {src[m][0]}  n={m.sum():3d}  tremor={y[m].sum():3d}  "
              f"3-12Hz power t/n={np.median(tp[m&(y==1)])/np.median(tp[m&(y==0)]):5.1f}x")
    print(f"\n전체 3-12Hz 파워비 tremor/non = {sep:.1f}x  (자체검증 통과)")
    Xtr, ytr, Xte, yte = load_split("seojin")
    print(f"load_split('seojin'): train {Xtr.shape[0]} bags / test {Xte.shape[0]} bags")


if __name__ == "__main__":
    demo()
