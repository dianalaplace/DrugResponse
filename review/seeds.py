"""Outer-CV seed sensitivity of notebook models A/B + TNBC subgroup CI (notebook code unchanged)."""
import json, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
nb = json.load(open(sys.argv[1]))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
ns = {}
for i in (0, 2, 3, 4, 5):
    exec(code[i], ns)
X, y = ns["X"], ns["y"]
res = {"A_clinical_LR": [], "B_clinical_genes_LR": []}
oofs = {}
for seed in (0, 1, 2, 3, 42):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    for name in res:
        pipe, grid = ns["MODELS"][name]
        p = pd.Series(np.nan, index=y.index)
        for tr, te in outer.split(X, y):
            s = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1).fit(X.iloc[tr], y.iloc[tr])
            p.iloc[te] = s.predict_proba(X.iloc[te])[:, 1]
        res[name].append(roc_auc_score(y, p)); oofs[(name, seed)] = p
for name, v in res.items():
    print(f"{name}: AUC per seed {np.round(v,3)} mean {np.mean(v):.3f} ± {np.std(v):.3f}")
d = np.array(res["B_clinical_genes_LR"]) - np.array(res["A_clinical_LR"])
print("Delta B-A per seed:", np.round(d, 3))
c = ns["clinical"]
tn = (c["her2"] == 0) & (c["er"] == 0) & (c["pr"] == 0)
p = oofs[("B_clinical_genes_LR", 42)][tn]; yy = y[tn].values; p = p.values
rng = np.random.default_rng(42); a = []
for _ in range(1000):
    i = rng.integers(0, len(yy), len(yy))
    if yy[i].min() != yy[i].max(): a.append(roc_auc_score(yy[i], p[i]))
print(f"TNBC model B AUC {roc_auc_score(yy, p):.3f}, 95% CI {np.percentile(a,2.5):.3f}-{np.percentile(a,97.5):.3f}, n={len(yy)}, pCR={yy.sum()}")
h = c["her2"] == 1
print("HER2+ by cohort:", c.loc[h, "cohort"].value_counts().to_dict())
