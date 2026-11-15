"""Leakage audit: executes the notebook's own code cells (unchanged) and probes them.

Usage (cwd = notebook run dir, so data/raw resolves): python -I audit.py <notebook.ipynb>
"""

import json
import sys

import numpy as np
import pandas as pd

nb = json.load(open(sys.argv[1]))
code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
ns: dict = {}
# cells: 0 imports, 1 wget (skip), 2 parse, 3 harmonise, 4 pool, 5 models, 6 nested CV (skip)
for i in (0, 2, 3, 4, 5):
    exec(code[i], ns)

X, y, CLIN, GENES = ns["X"], ns["y"], ns["CLINICAL_COLS"], ns["GENE_COLS"]
print("Feature columns used (clinical):", CLIN)
leaky = [
    "dlda30_prediction",
    "rcb_0_i_prediction",
    "chemosensitivity_prediction",
    "pathologic_response_rcb_class",
    "pam50_class",
    "set_class",
    "ggi_class",
]
print("Any leaky column in X?:", [c for c in X.columns if c in leaky] or "none")
print("X columns == clinical + probes only:", list(X.columns) == CLIN + GENES)
print("pCR rate by cohort:", y.groupby(X["cohort"]).mean().round(3).to_dict())

# ---- parse bug: GSE20194 characteristics rows are shifted for some samples ----
c20 = ns["clin_20194"]
raw: dict[str, dict[str, str]] = {}
import gzip

with gzip.open(
    "data/raw/GSE20194_series_matrix.txt.gz", "rt", encoding="latin-1"
) as fh:
    for line in fh:
        if line.startswith("!Sample_geo_accession"):
            ids = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            raw = {i: {} for i in ids}
        elif line.startswith("!Sample_characteristics_ch1"):
            for i, cell in zip(ids, line.rstrip("\n").split("\t")[1:]):
                cell = cell.strip('"')
                if ": " in cell:
                    k, v = cell.split(": ", 1)
                    raw[i][k] = v
        elif line.startswith("!series_matrix_table_begin"):
            break
truth = pd.DataFrame(raw).T
cols = [
    "age",
    "er_status",
    "pr_status",
    "her2 status",
    "tbefore",
    "nbefore",
    "bmngrd",
    "pcr_vs_rd",
]
mis = c20[cols].astype(str) != truth.reindex(c20.index)[cols].astype(str)
bad = mis.any(axis=1)
print(f"\nGSE20194 samples with >=1 misparsed field: {bad.sum()} / {len(c20)}")
print("misparsed count per field:", mis.sum().to_dict())
ex = c20.loc[bad, cols].head(4)
print("example notebook-parsed rows:\n", ex.to_string())
print("same rows, correct:\n", truth.loc[ex.index, cols].to_string())
Xb = ns["X_clin_b"]
print("notebook grade values in GSE20194:", sorted(Xb["grade"].dropna().unique()))
print(
    "notebook t_stage values GSE20194:",
    sorted(Xb["t_stage"].dropna().unique()),
    "| GSE25065:",
    sorted(ns["X_clin_a"]["t_stage"].dropna().unique()),
)
true_lab = truth["pcr_vs_rd"].map({"pCR": 1, "RD": 0})
dropped = ns["y_b"].index[ns["y_b"].isna()]
print(
    f"samples dropped as 'unlabeled' in GSE20194: {len(dropped)}; true labels of those:",
    true_lab.loc[dropped].value_counts().to_dict(),
)

# ---- hyperparameters chosen (from the executed notebook outputs not available here: rerun cheap) ----
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

outer = StratifiedKFold(5, shuffle=True, random_state=42)
inner = StratifiedKFold(3, shuffle=True, random_state=42)


def nested(name, yy, Xmat=X):
    pipe, grid = ns["MODELS"][name]
    p = pd.Series(np.nan, index=yy.index)
    chosen = []
    for tr, te in outer.split(Xmat, yy):
        s = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1).fit(
            Xmat.iloc[tr], yy.iloc[tr]
        )
        p.iloc[te] = s.predict_proba(Xmat.iloc[te])[:, 1]
        chosen.append(s.best_params_)
    return roc_auc_score(yy, p), chosen


auc, chosen = nested("B_clinical_genes_LR", y)
print(f"\nModel B AUC (rerun) = {auc:.3f}; chosen params per outer fold: {chosen}")
auc, chosen = nested("A_clinical_LR", y)
print(f"Model A AUC (rerun) = {auc:.3f}; chosen params: {chosen}")

# ---- shuffled-label test on the notebook's real model B pipeline ----
rng = np.random.default_rng(42)
perm = []
for _ in range(3):
    ys = pd.Series(rng.permutation(y.values), index=y.index)
    perm.append(nested("B_clinical_genes_LR", ys)[0])
print("Model B nested-CV AUC with shuffled y (3 perms):", np.round(perm, 3))

# ---- sensitivity: same notebook models on correctly parsed GSE20194 ----
c20_fixed = truth.reindex(c20.index).replace({"NA": np.nan, "": np.nan})
Xb_fix, yb_fix = ns["harmonise_20194"](c20_fixed)
clin_fix = pd.concat([ns["X_clin_a"].assign(cohort=0), Xb_fix.assign(cohort=1)])
y_fix = pd.concat([ns["y_a"], yb_fix])
expr_full = pd.concat(
    [
        ns["zscore_within_cohort"](ns["expr_25065"][GENES]),
        ns["zscore_within_cohort"](ns["expr_20194"][GENES]),
    ]
)
k = y_fix.notna()
X_fix = pd.concat([clin_fix[k], expr_full[k]], axis=1)
y_fix = y_fix[k].astype(int)
print(f"\nCorrect parse: n={len(y_fix)}, pCR rate={y_fix.mean():.3f}")
for name in ("A_clinical_LR", "B_clinical_genes_LR"):
    print(
        f"  {name} on correctly parsed data: AUC = {nested(name, y_fix, X_fix)[0]:.3f}"
    )
