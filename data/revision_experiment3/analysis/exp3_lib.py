# -*- coding: utf-8 -*-
"""
exp3_lib.py — επεκτάσεις για το Πείραμα 3 (baselines, καμπύλες, iso-F-Abst, AUROC).
Χτίζει πάνω στο abst_lib (Wilson, McNemar, bootstrap, select_threshold, majority_vote).

Ορισμοί (ίδιοι με τα Πειράματα 1-2):
  present = answerable & gold στο παγωμένο top-5 ·  missing = ¬present.
  abstain(baseline) = (score < T) Ή abstain(Base)  — «φίλτρο πριν τον Base».
  C-Abst = % abstain στο missing ·  F-Abst = % abstain στο present.
"""
import os, sys, re, math
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import abst_lib as A

# Σταθερό feature set — ΔΙΑΘΕΣΙΜΟ ΚΑΙ στο OOF ΚΑΙ στο παγωμένο TEST (retrieval_test.jsonl).
# (overlap_max, rrf_top1, bm25_max, dense_max ΔΕΝ αποθηκεύονται στο frozen TEST -> εξαιρούνται
#  για συνέπεια TRAIN↔TEST· μπορούν να προστεθούν αν επεκταθεί το frozen schema.)
FEATURES = ["max_logit", "logit2", "mean_logit", "gap12", "softmax_entropy", "q_len", "n_tech_tokens"]


# ── n_tech_tokens: ΠΙΣΤΟ αντίγραφο του extract_technical_tokens (main.py / exp1b_oof_scores) ──
def n_tech_tokens(text):
    if not text:
        return 0
    t = set()
    t.update(re.findall(r'\b[vV]?\d+(?:\.\d+){1,4}\b', text))
    t.update(m.lower() for m in re.findall(r'\b(?:swg|nas|isg|ibm)\d{5,}[A-Z]?\b', text, re.I))
    t.update(m.lower() for m in re.findall(r'\b[A-Z]{2,}[0-9]{4,}[A-Z]?\b', text))
    t.update(m.lower() for m in re.findall(r'\bEJP[A-Z0-9]+\b', text))
    t.update(m.lower() for m in re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text))
    t.update(m.lower() for m in re.findall(r'\b[A-Z]{3,5}\b', text))
    t.update(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text.lower()))
    return len([x for x in t if len(x) >= 3])


def softmax_entropy(logits):
    if not logits:
        return 0.0
    mx = max(logits)
    exps = [math.exp(l - mx) for l in logits]
    s = sum(exps); ps = [e / s for e in exps]
    return -sum(p * math.log(p + 1e-12) for p in ps)


def features_from_logits(logits, qtext):
    s = sorted(logits, reverse=True)
    return {"max_logit": max(logits) if logits else 0.0,
            "logit2": s[1] if len(s) > 1 else 0.0,
            "mean_logit": (sum(logits) / len(logits)) if logits else 0.0,
            "gap12": (s[0] - s[1]) if len(s) > 1 else 0.0,
            "softmax_entropy": softmax_entropy(logits),
            "q_len": len((qtext or "").split()),
            "n_tech_tokens": n_tech_tokens(qtext or "")}


def build_features(df):
    """df με έτοιμες στήλες FEATURES (OOF) -> τις επιστρέφει· αλλιώς σφάλμα.
    Για TEST χρησιμοποιήστε features_from_logits ανά γραμμή (βλ. exp3a_prepare)."""
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise KeyError(f"λείπουν features: {missing}")
    return df[FEATURES].astype(float).values


# ── B1: answerability classifier (logistic regression + 5-fold CV) ──
def fit_answerability(X, present, seed=42):
    """Logistic regression με StandardScaler· επιλογή C με 5-fold CV (AUROC).
    Επιστρέφει (pipeline εκπαιδευμένο σε όλα, p_cv = out-of-fold P(present))."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
    y = np.asarray(present).astype(int)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, {"lr__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}, scoring="roc_auc", cv=cv)
    gs.fit(X, y)
    best = gs.best_estimator_
    p_cv = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]
    best.fit(X, y)
    return best, p_cv, gs.best_params_["lr__C"]


# ── AUROC (score διακρίνει present vs missing· υψηλό score = present) ──
def auroc_missing(score, present):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(present).astype(int)
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, float)))


def auroc_ci(score, present, B=2000, seed=42):
    from sklearn.metrics import roc_auc_score
    score = np.asarray(score, float); y = np.asarray(present).astype(int)
    n = len(y); rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ── Καμπύλη C-Abst vs F-Abst (score ως φίλτρο, + Base abstain) ──
def curve_tradeoff(score, present, base_abstain=None, grid=None):
    score = np.asarray(score, float); present = np.asarray(present, bool)
    ba = (np.zeros(len(score), bool) if base_abstain is None else np.asarray(base_abstain, bool))
    miss = ~present
    if grid is None:
        lo, hi = np.min(score), np.max(score)
        grid = np.linspace(lo - 1e-6, hi + 1e-6, 201)
    rows = []
    for T in grid:
        ab = (score < T) | ba
        c = 100 * (ab & miss).sum() / max(1, miss.sum())
        f = 100 * (ab & present).sum() / max(1, present.sum())
        rows.append({"T": float(T), "C_abst": c, "F_abst": f})
    return pd.DataFrame(rows)


def iso_f_abst(score, present, base_abstain, target_f):
    """Το ΜΕΓΑΛΥΤΕΡΟ T με F-Abst ≤ target_f. Επιστρέφει (T*, C_abst, F_abst)."""
    cv = curve_tradeoff(score, present, base_abstain)
    ok = cv[cv.F_abst <= target_f + 1e-9]
    best = (ok.iloc[ok["T"].argmax()] if len(ok) else cv.iloc[cv.F_abst.argmin()])
    return float(best["T"]), float(best["C_abst"]), float(best["F_abst"])


def iso_f_abst_ci(score, present, base_abstain, qlora_incident_rate, B=2000, seed=42):
    """Bootstrap σε επίπεδο incident· σε κάθε resample επανεπιλέγεται το T* στο ίδιο
    F-Abst-στόχο (= το μέσο F-Abst της QLoRA στο resample). Επιστρέφει (lo, med, hi) του C-Abst."""
    score = np.asarray(score, float); present = np.asarray(present, bool)
    ba = np.asarray(base_abstain, bool); qa = np.asarray(qlora_incident_rate, float)
    n = len(score); rng = np.random.default_rng(seed); cs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        tf = 100 * qa[idx][present[idx]].mean() if present[idx].any() else 25.0
        _, c, _ = iso_f_abst(score[idx], present[idx], ba[idx], tf)
        cs.append(c)
    return (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 50)), float(np.percentile(cs, 97.5)))


def select_theta_hi(score, present, prec=0.90):
    """Το ΜΙΚΡΟΤΕΡΟ theta ώστε precision(present | score ≥ theta) ≥ prec (CRAG 'Correct')."""
    score = np.asarray(score, float); present = np.asarray(present, bool)
    for T in np.unique(score):
        sel = score >= T
        if sel.sum() == 0:
            continue
        if present[sel].mean() >= prec:
            return float(T)
    return float(np.max(score))


def per_incident_rate(runs, config):
    """Μέσο abstain-rate ανά incident στα runs ενός config (Series, index=question_id, [0,1])."""
    g = runs[runs.config == config]
    return g.pivot_table(index="question_id", columns="run", values="abstain", aggfunc="first").mean(axis=1)
