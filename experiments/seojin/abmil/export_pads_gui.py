"""PADS ABMIL 5-fold CV 결과를 GUI JSON으로. bag = 피험자, 녹음 22개(11과제 x 양손목) 각각 8윈도우.
python3 tremor/export_pads_gui.py -> tremor/gui_pads/data.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from prep import FS, WIN, HOP
from pads_abmil import OUT as BAGS, cv, TASKS, PADS
from viz import spectrogram
import abmil; abmil.EPOCHS = 25

torch.manual_seed(0)
d = np.load(BAGS); X, y, subj, inst_task = d["X"], d["y"], d["subject"], d["inst_task"]
CV = BAGS.with_name("pads_cv.npz")
if CV.exists(): c = np.load(CV); p, A = c["p"], c["A"]
else: p, A = cv(X, y); np.savez_compressed(CV, p=p, A=A)
recs = list(dict.fromkeys(inst_task)); span = {r: np.where(inst_task == r)[0] for r in recs}   # 녹음별 instance 인덱스(길이 다를 수 있음)
cond = {}
for s in subj:
    pt = json.loads((PADS / f"patients/patient_{s}.json").read_text()); cond[s] = pt["condition"] + ("" if pt["disease_comment"] == "-" else " · " + pt["disease_comment"])
f = None; out = {"fs": FS, "win": WIN, "hop": HOP, "recs": recs, "subjects": []}
for i in range(len(y)):
    bag = {"id": str(subj[i]), "cond": cond[subj[i]], "y": int(y[i]), "p": round(float(p[i]), 3), "recs": []}
    for r, name in enumerate(recs):
        ix = span[name]; w = X[i, ix]                       # (n_win, 100, 6)
        sig = np.concatenate([w[0]] + [wk[HOP:] for wk in w[1:]])[:, :3]
        mag = np.linalg.norm(sig, axis=1)
        fr, P, pk = spectrogram(w[:, :, :3])
        band = ((fr >= 3) & (fr < 12)); tot = ((fr >= 0.5) & (fr < 15))
        tr = P[band].sum(0) / (P[tot].sum(0) + 1e-9)                      # 3~12Hz 파워 / 0.5~15Hz 파워
        m = fr < 15; f = fr[m].round(2).tolist()
        bag["recs"].append({"name": name, "att": A[i, ix].round(4).tolist(), "mag": mag[::8].round(1).tolist(),
                            "spec": (lambda L: ((L - L.min()) / (np.ptp(L) + 1e-9) * 99).astype(int).T.tolist())(np.log10(P[m] + 1e-6)), "peak": pk.round(1).tolist(), "tr": tr.round(2).tolist()})
    out["subjects"].append(bag)
out["freqs"] = f
(Path(__file__).parent / "gui_pads").mkdir(exist_ok=True)
json.dump(out, open(Path(__file__).parent / "gui_pads/data.json", "w"), separators=(",", ":"))
print("->", (Path(__file__).parent / "gui_pads/data.json").stat().st_size // 1024, "KB,", len(y), "subjects")
