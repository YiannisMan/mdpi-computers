# -*- coding: utf-8 -*-
"""
abst_lib.py — Κοινή βιβλιοθήκη ανάλυσης για τα Πειράματα 1–3.

Περιέχει: Wilson CI, ορισμό υποπληθυσμών, C-Abst/F-Abst/Hazard, rates_per_run,
summarize_runs (mean/min–max), majority_vote, exact McNemar, paired bootstrap,
Holm–Bonferroni, pairwise_table, και βοηθητικά για B–C paired agreement.

Σχεδιαστική αρχή (Οδηγός πρωτοκόλλου):
  - abstain = (mode == "B"). Οι invalid έξοδοι ΔΕΝ είναι abstain (μετρώνται ως «απάντηση»).
  - Τα 3 runs ΔΕΝ είναι ανεξάρτητα (ίδιες ερωτήσεις) -> Wilson CI πάνω στο majority vote.

Εξαρτήσεις: numpy, pandas. (sklearn προαιρετικό: cohen_kappa_score, multipletests.)
"""
import math
import numpy as np
import pandas as pd

Z95 = 1.959963984540054


# ─────────────────────────────────────────────────────────────────────────
#  Wilson 95% CI (ίδιος τύπος με το evaluate_full.py για συνέχεια)
# ─────────────────────────────────────────────────────────────────────────
def wilson_ci(correct, total, z=Z95):
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


# ─────────────────────────────────────────────────────────────────────────
#  Υποπληθυσμοί
# ─────────────────────────────────────────────────────────────────────────
def population(subgroups_df, name):
    """Επιστρέφει pandas Index με τα question_id του υποπληθυσμού.
    name ∈ {present, missA, missU, missing, all}."""
    q = subgroups_df
    if name == "missing":
        ids = q[q.subgroup.isin(["missA", "missU"])].question_id
    elif name == "all":
        ids = q.question_id
    else:
        ids = q[q.subgroup == name].question_id
    return pd.Index(ids.tolist())


# ─────────────────────────────────────────────────────────────────────────
#  Rates ανά run
# ─────────────────────────────────────────────────────────────────────────
def _rate(abstain_series, ids):
    s = abstain_series.reindex(ids).dropna()
    if len(s) == 0:
        return float("nan"), 0, 0
    k = int(s.sum()); n = int(len(s))
    return 100.0 * k / n, k, n

def rates_per_run(runs_df, subgroups_df):
    """
    runs_df: στήλες [config, run, question_id, abstain(bool)].
    Επιστρέφει DataFrame: μία γραμμή ανά (config, run) με C-Abst (missing/missA/missU),
    F-Abst (present), και τα πλήθη.
    """
    out = []
    pops = {p: population(subgroups_df, p) for p in ("missing", "missA", "missU", "present")}
    for (cfg, run), g in runs_df.groupby(["config", "run"]):
        ab = g.set_index("question_id").abstain.astype(float)
        row = {"config": cfg, "run": run}
        for p in ("missing", "missA", "missU"):
            r, k, n = _rate(ab, pops[p]); row[f"C_Abst_{p}"] = r; row[f"n_{p}"] = n
        r, k, n = _rate(ab, pops["present"]); row["F_Abst_present"] = r; row["n_present"] = n
        row["Hazard_missing"] = (100.0 - row["C_Abst_missing"]) if not math.isnan(row["C_Abst_missing"]) else float("nan")
        out.append(row)
    return pd.DataFrame(out)

def summarize_runs(per_run_df, metrics=("C_Abst_missing", "C_Abst_missA", "C_Abst_missU",
                                        "F_Abst_present", "Hazard_missing")):
    """mean (min–max) ανά config, στα 3 runs."""
    rows = []
    for cfg, g in per_run_df.groupby("config"):
        row = {"config": cfg, "n_runs": len(g)}
        for m in metrics:
            vals = g[m].dropna().values
            if len(vals):
                row[m] = f"{vals.mean():.1f} ({vals.min():.1f}-{vals.max():.1f})"
            else:
                row[m] = "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
#  Majority vote ανά incident + Wilson στο majority vote
# ─────────────────────────────────────────────────────────────────────────
def majority_vote(runs_df, config, ids):
    """Επιστρέφει pandas Series (index=question_id) με majority-vote abstain (bool)
    των runs ενός config, περιορισμένο στα ids."""
    g = runs_df[runs_df.config == config]
    piv = g.pivot_table(index="question_id", columns="run", values="abstain", aggfunc="first")
    piv = piv.reindex(ids)
    mv = (piv.mean(axis=1) >= 0.5)
    return mv

def wilson_on_majority(runs_df, config, subgroups_df, pop_name):
    ids = population(subgroups_df, pop_name)
    mv = majority_vote(runs_df, config, ids).dropna()
    k = int(mv.sum()); n = int(len(mv))
    p, lo, hi = wilson_ci(k, n)
    return {"config": config, "population": pop_name, "k": k, "n": n,
            "pct": 100 * p, "ci_lo": 100 * lo, "ci_hi": 100 * hi}


# ─────────────────────────────────────────────────────────────────────────
#  Exact McNemar
# ─────────────────────────────────────────────────────────────────────────
def mcnemar_exact(a, b):
    """
    a, b: δυαδικά arrays ίδιου μήκους (π.χ. abstain αποφάσεις δύο configs στις ίδιες
    ερωτήσεις). Exact McNemar (binomial) στα discordant ζεύγη.
    Επιστρέφει dict με b01, b10, p (two-sided exact), delta_pp.
    """
    a = np.asarray(a).astype(bool); b = np.asarray(b).astype(bool)
    b01 = int(np.sum(a & ~b))   # A=1, B=0
    b10 = int(np.sum(~a & b))   # A=0, B=1
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        # two-sided exact binomial, p=0.5
        from math import comb
        tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
        p = min(1.0, 2 * tail)
    delta_pp = 100.0 * (a.mean() - b.mean())
    return {"b01": b01, "b10": b10, "n_discordant": n, "p": p, "delta_pp": delta_pp}


# ─────────────────────────────────────────────────────────────────────────
#  Paired bootstrap στη διαφορά ποσοστών (incident-level)
# ─────────────────────────────────────────────────────────────────────────
def paired_bootstrap_diff(a_mean, b_mean, B=10000, seed=42):
    """
    a_mean, b_mean: arrays με το ΜΕΣΟ abstain-rate ανά incident στα 3 runs (0..1),
    ευθυγραμμισμένα στα ίδια incidents. Resample σε επίπεδο incident.
    Επιστρέφει (delta_pp, lo_pp, hi_pp) 95% percentile CI.
    """
    a_mean = np.asarray(a_mean, float); b_mean = np.asarray(b_mean, float)
    n = len(a_mean)
    rng = np.random.default_rng(seed)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        diffs[i] = a_mean[idx].mean() - b_mean[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return 100 * (a_mean.mean() - b_mean.mean()), 100 * lo, 100 * hi


# ─────────────────────────────────────────────────────────────────────────
#  Holm–Bonferroni
# ─────────────────────────────────────────────────────────────────────────
def holm(pvals):
    """Επιστρέφει list με τα Holm-adjusted p-values (ίδια σειρά με την είσοδο)."""
    try:
        from statsmodels.stats.multitest import multipletests
        return list(multipletests(pvals, method="holm")[1])
    except Exception:
        m = len(pvals)
        order = sorted(range(m), key=lambda i: pvals[i])
        adj = [0.0] * m
        running = 0.0
        for rank, i in enumerate(order):
            val = (m - rank) * pvals[i]
            running = max(running, val)
            adj[i] = min(1.0, running)
        return adj


# ─────────────────────────────────────────────────────────────────────────
#  Pairwise πίνακας (McNemar + bootstrap + Holm) για μια οικογένεια
# ─────────────────────────────────────────────────────────────────────────
def _incident_mean_rate(runs_df, config, ids):
    """Μέσο abstain-rate ανά incident στα 3 runs, ευθυγραμμισμένο στα ids."""
    g = runs_df[runs_df.config == config]
    piv = g.pivot_table(index="question_id", columns="run", values="abstain", aggfunc="first")
    return piv.reindex(ids).mean(axis=1)

def pairwise_table(runs_df, subgroups_df, pairs, metric_pop, B=10000, seed=42):
    """
    pairs: list από (A, B) config names.
    metric_pop: ("C-Abst","missing") ή ("F-Abst","present") — ορίζει τον πληθυσμό.
    Επιστρέφει DataFrame με A,B,delta_pp,boot_lo,boot_hi,mcnemar_p,holm_p.
    """
    _, pop = metric_pop
    ids = population(subgroups_df, pop)
    rows, pvals = [], []
    for A, Bc in pairs:
        a_mv = majority_vote(runs_df, A, ids).dropna()
        b_mv = majority_vote(runs_df, Bc, ids).reindex(a_mv.index).dropna()
        common = a_mv.index.intersection(b_mv.index)
        mc = mcnemar_exact(a_mv.loc[common].values, b_mv.loc[common].values)
        a_rate = _incident_mean_rate(runs_df, A, common)
        b_rate = _incident_mean_rate(runs_df, Bc, common)
        d, lo, hi = paired_bootstrap_diff(a_rate.values, b_rate.values, B=B, seed=seed)
        rows.append({"A": A, "B": Bc, "population": pop, "metric": metric_pop[0],
                     "delta_pp": round(d, 2), "boot_lo": round(lo, 2), "boot_hi": round(hi, 2),
                     "mcnemar_p": mc["p"], "b01": mc["b01"], "b10": mc["b10"]})
        pvals.append(mc["p"])
    for r, hp in zip(rows, holm(pvals)):
        r["holm_p"] = hp
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
#  B–C paired agreement (Πείραμα 1Ε)
# ─────────────────────────────────────────────────────────────────────────
def cohen_kappa(a, b):
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(a, b))
    except Exception:
        a = np.asarray(a).astype(int); b = np.asarray(b).astype(int)
        n = len(a)
        po = np.mean(a == b)
        pa1 = a.mean(); pb1 = b.mean()
        pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
        return float((po - pe) / (1 - pe)) if (1 - pe) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────
#  Gate threshold selection (χρησιμοποιείται στο Πείραμα 3Α — εδώ για πληρότητα)
# ─────────────────────────────────────────────────────────────────────────
def curve(score, present, grid=None):
    """Για κάθε κατώφλι T: C-Abst (στο missing) και F-Abst (στο present),
    όπου abstain <=> score < T. present: bool array (True=GT-present)."""
    score = np.asarray(score, float); present = np.asarray(present, bool)
    if grid is None:
        grid = np.quantile(score, np.linspace(0, 1, 101))
    rows = []
    miss = ~present
    for T in grid:
        ab = score < T
        c = 100 * (ab & miss).sum() / max(1, miss.sum())
        f = 100 * (ab & present).sum() / max(1, present.sum())
        rows.append({"T": float(T), "C_Abst": c, "F_Abst": f})
    return pd.DataFrame(rows)

def select_threshold(score, present, max_f=25.0):
    """Επιλέγει T που μεγιστοποιεί το C-Abst υπό F-Abst <= max_f. Επιστρέφει (T, C, F)."""
    cv = curve(score, present)
    ok = cv[cv.F_Abst <= max_f]
    if len(ok) == 0:
        best = cv.iloc[cv.F_Abst.argmin()]
    else:
        best = ok.iloc[ok.C_Abst.argmax()]
    return float(best["T"]), float(best["C_Abst"]), float(best["F_Abst"])
