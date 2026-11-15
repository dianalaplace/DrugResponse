"""Independent re-implementation (does not import or copy notebook functions).

Usage: python -I independent_check.py <dir with GSE25065/GSE20194 series matrices>
"""

import csv
import gzip
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
DATA = Path(sys.argv[1])


def read_matrix(path: Path):
    """Return sample ids, dict key->list of values, probe ids, expression (probes x samples)."""
    meta: dict[str, list[str]] = {}
    ids: list[str] = []
    probes: list[str] = []
    rows: list[list[float]] = []
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        reader = csv.reader(fh, delimiter="\t")
        table = False
        header_seen = False
        for rec in reader:
            if not rec:
                continue
            tag = rec[0]
            if tag == "!series_matrix_table_begin":
                table = True
                continue
            if tag == "!series_matrix_table_end":
                break
            if table:
                if not header_seen:
                    header_seen = True
                    assert rec[1:] == ids, "column order != sample order"
                    continue
                probes.append(tag)
                rows.append(
                    [
                        float(v) if v not in ("", "null", "NA") else np.nan
                        for v in rec[1:]
                    ]
                )
                continue
            if tag == "!Sample_geo_accession":
                ids = rec[1:]
            elif tag == "!Sample_characteristics_ch1":
                for col, cell in enumerate(rec[1:]):
                    if ": " not in cell:
                        continue
                    k, v = cell.split(": ", 1)
                    meta.setdefault(k.strip().lower(), [""] * len(ids))[col] = v.strip()
    return ids, meta, probes, np.array(rows)


def num(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return np.nan


def digit(v: str) -> float:
    d = [c for c in v if c.isdigit()]
    return float(d[0]) if d else np.nan


def pn(v: str) -> float:
    return {"P": 1.0, "N": 0.0}.get(v, np.nan)


KEYS = {  # feature -> (key in GSE25065, key in GSE20194, converter)
    "age": ("age_years", "age", num),
    "er": ("er_status_ihc", "er_status", pn),
    "pr": ("pr_status_ihc", "pr_status", pn),
    "her2": ("her2_status", "her2 status", pn),
    "t": ("clinical_t_stage", "tbefore", digit),
    "n": ("clinical_nodal_status", "nbefore", digit),
    "grade": ("grade", "bmngrd", num),
}
LABEL = ("pathologic_response_pcr_rd", "pcr_vs_rd")

cohorts = []
for ci, gse in enumerate(["GSE25065", "GSE20194"]):
    ids, meta, probes, ex = read_matrix(DATA / f"{gse}_series_matrix.txt.gz")
    lab = np.array([{"pCR": 1.0, "RD": 0.0}.get(v, np.nan) for v in meta[LABEL[ci]]])
    clin = np.column_stack(
        [[KEYS[c][2](v) for v in meta[KEYS[c][ci]]] for c in KEYS] + [np.full(len(ids), ci)]
    )
    print(
        f"{gse}: n={len(ids)}, unlabeled={np.isnan(lab).sum()}, probes={len(probes)}, "
        f"expr range {np.nanmin(ex):.2f}..{np.nanmax(ex):.2f}, median {np.nanmedian(ex):.2f}"
    )
    cohorts.append({"ids": ids, "probes": probes, "ex": ex, "y": lab, "clin": clin, "meta": meta})

# --- duplicate-array check across cohorts (raw values, log2 if needed) ---
pa, pb = cohorts[0]["probes"], cohorts[1]["probes"]
common = sorted(set(pa) & set(pb) - {p for p in pa if p.startswith("AFFX")})
ia = [pa.index(p) for p in common]
ib = [pb.index(p) for p in common]
A = cohorts[0]["ex"][ia]
B = cohorts[1]["ex"][ib]
logA = np.log2(np.clip(A, 1, None)) if np.nanmax(A) > 100 else A
logB = np.log2(np.clip(B, 1, None)) if np.nanmax(B) > 100 else B


def colz(M):
    M = np.nan_to_num(M - np.nanmean(M, 0))
    return M / np.linalg.norm(M, axis=0)


corr_ab = colz(logA).T @ colz(logB)
corr_aa = colz(logA).T @ colz(logA)
np.fill_diagonal(corr_aa, 0)
print(
    f"cross-cohort max sample corr: {corr_ab.max():.4f}; pairs >0.98: {(corr_ab > 0.98).sum()}"
)
print(
    f"within GSE25065 max off-diag corr: {corr_aa.max():.4f}; 99th pct of cross-cohort max: "
    f"{np.percentile(corr_ab.max(1), 99):.4f}"
)

# --- pooled matrix: per-cohort z-score of probes (unsupervised) ---
Z = []
for M in (A, B):
    M = M.T
    Z.append((M - M.mean(0)) / np.where(M.std(0) == 0, 1, M.std(0)))
G = np.vstack(Z)
C = np.vstack([c["clin"] for c in cohorts])
y = np.concatenate([c["y"] for c in cohorts])
keep = ~np.isnan(y)
G, C, y = G[keep], C[keep], y[keep].astype(int)
print(
    f"pooled: n={len(y)}, pCR rate={y.mean():.3f}, clinical={C.shape[1]}, probes={G.shape[1]}"
)


def prep_clin(tr, te):
    med = np.nanmedian(C[tr], 0)
    Xtr, Xte = (
        np.where(np.isnan(C[tr]), med, C[tr]),
        np.where(np.isnan(C[te]), med, C[te]),
    )
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def run(use_genes: bool, yy: np.ndarray, C_reg: float, k: int = 500, seed: int = SEED):
    oof = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(C, yy):
        Xtr, Xte = prep_clin(tr, te)
        if use_genes:
            top = np.argsort(G[tr].var(0))[::-1][:k]  # selected on training fold only
            gtr, gte = G[tr][:, top], G[te][:, top]
            mu, sd = gtr.mean(0), gtr.std(0)
            Xtr = np.hstack([Xtr, (gtr - mu) / sd])
            Xte = np.hstack([Xte, (gte - mu) / sd])
        m = LogisticRegression(C=C_reg, class_weight="balanced", max_iter=5000).fit(
            Xtr, yy[tr]
        )
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


print("\nPlain 5-fold CV (fixed C, no tuning):")
for label, genes, creg in [
    ("clinical LR, C=1", False, 1.0),
    ("clinical LR, C=0.1", False, 0.1),
    ("clinical+500 probes LR, C=0.01", True, 0.01),
    ("clinical+500 probes LR, C=0.001", True, 0.001),
]:
    aucs = [roc_auc_score(y, run(genes, y, creg, seed=s)) for s in range(5)]
    p = run(genes, y, creg)
    print(
        f"  {label:34s} AUC(seed42)={roc_auc_score(y, p):.3f}  PR-AUC={average_precision_score(y, p):.3f}  "
        f"AUC over 5 CV seeds: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}"
    )

# per-cohort AUC of pooled model and cross-cohort transfer
coh = C[:, -1]
p = run(True, y, 0.01)
for c in (0, 1):
    print(
        f"  clinical+genes OOF AUC within cohort {c}: {roc_auc_score(y[coh == c], p[coh == c]):.3f}"
    )

print(
    "\nCross-cohort (train one, test other), clinical+500 probes, C=0.01 / clinical only C=1:"
)
for a, b in ((0, 1), (1, 0)):
    tr, te = np.where(coh == a)[0], np.where(coh == b)[0]
    for genes, creg in ((False, 1.0), (True, 0.01)):
        Xtr, Xte = prep_clin(tr, te)
        Xtr, Xte = Xtr[:, :-1], Xte[:, :-1]  # cohort column is constant here
        if genes:
            top = np.argsort(G[tr].var(0))[::-1][:500]
            Xtr, Xte = np.hstack([Xtr, G[tr][:, top]]), np.hstack([Xte, G[te][:, top]])
        m = LogisticRegression(C=creg, class_weight="balanced", max_iter=5000).fit(
            Xtr, y[tr]
        )
        print(
            f"  train {a} -> test {b}  {'clin+genes' if genes else 'clinical  '}  "
            f"AUC={roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]):.3f}"
        )

print("\nShuffled-label sanity (clinical+500 probes, C=0.01), 5 permutations:")
rng = np.random.default_rng(SEED)
perm_aucs = []
for _ in range(5):
    ys = rng.permutation(y)
    perm_aucs.append(roc_auc_score(ys, run(True, ys, 0.01)))
print("  AUCs:", np.round(perm_aucs, 3), f"mean={np.mean(perm_aucs):.3f}")

# extra: what the excluded GSE25065 columns would do (shows they are leaky, so excluding them matters)
m0 = cohorts[0]["meta"]
lab0 = cohorts[0]["y"]
ok = ~np.isnan(lab0)
for key in ("dlda30_prediction", "rcb_0_i_prediction", "pathologic_response_rcb_class"):
    v = np.array(m0[key])[ok]
    score = np.isin(v, ["pCR", "RCB-0/I", "RCB-0", "RCB-0/I "]).astype(float)
    print(
        f"  GSE25065 {key}: values={sorted(set(v))[:6]}, AUC as predictor={roc_auc_score(lab0[ok], score):.3f}"
    )

# regimen check in GSE20194 (trastuzumab-containing arms?)
tc = np.array(cohorts[1]["meta"]["treatment code"])
vals, cnt = np.unique(tc, return_counts=True)
print("\nGSE20194 treatment codes:", dict(zip(vals, cnt)))
