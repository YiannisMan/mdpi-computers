#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 3Α — Προετοιμασία στο TRAIN (ΧΩΡΙΣ LLM): Gate + B1 answerability classifier.

Είσοδος: retrieval_train_oof.jsonl (600, Πείραμα 1Β) με present + features.
  present = answerable & gold στο OOF top-5 ·  missing = ¬present.

Παράγει:
  data/b1_classifier.pkl        — logistic regression (StandardScaler+LR) εκπαιδευμένο σε όλο το TRAIN
  data/thresholds_partial.json  — T_dep (gate), p_star (B1), θέση του T=5.0, AUROC (gate, B1)
  (το B2/CRAG προστίθεται από το exp3a_b2_crag.py· το exp3_freeze.py τα ενοποιεί)

Κριτήριο κατωφλίου (ΟΛΑ): max C-Abst υπό F-Abst ≤ 25% (προ-καταχωρημένο, στο TRAIN OOF).

ΕΚΤΕΛΕΣΗ (pandas/numpy/scikit-learn· ΧΩΡΙΣ LLM):
  python exp3a_prepare.py
"""
import json, os, sys, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, "analysis")
import abst_lib as A
import exp3_lib as E3

OOF   = os.getenv("OOF_PATH", "../revision_experiment1/data/retrieval_train_oof.jsonl")
MAX_F = float(os.getenv("MAX_F", "25.0"))
T5    = float(os.getenv("T5", "5.0"))
OUT   = "data"


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(OOF):
        print(f"[ERROR] λείπει {OOF} (Πείραμα 1Β)"); sys.exit(1)
    tr = pd.read_json(OOF, lines=True)
    present = tr["present"].astype(bool).values
    n_pres = int(present.sum()); n_miss = int((~present).sum())
    print(f"TRAIN OOF: {len(tr)} ερωτήσεις | present={n_pres} missing={n_miss}")

    # ── Gate: score = max reranker logit top-5 ──
    max_logit = (tr["max_logit"].values if "max_logit" in tr.columns
                 else tr["rr"].apply(max).values)
    T_dep, gc, gf = A.select_threshold(max_logit, present, max_f=MAX_F)
    gate_auroc = E3.auroc_missing(max_logit, present)
    gate_auroc_ci = E3.auroc_ci(max_logit, present)
    cv = A.curve(max_logit, present)
    at5 = cv.iloc[(cv["T"] - T5).abs().argmin()].to_dict()   # θέση του T=5.0 στην καμπύλη
    print(f"Gate: T_dep={T_dep:.4f} (C-Abst={gc:.1f}%, F-Abst={gf:.1f}%)  AUROC={gate_auroc:.3f}")
    print(f"      T=5.0 στην TRAIN καμπύλη: C-Abst={at5['C_Abst']:.1f}%, F-Abst={at5['F_Abst']:.1f}%")

    # ── B1: answerability classifier ──
    X = E3.build_features(tr)
    clf, p_cv, bestC = E3.fit_answerability(X, present)
    p_star, c1, f1 = A.select_threshold(p_cv, present, max_f=MAX_F)
    b1_auroc = E3.auroc_missing(p_cv, present)         # out-of-fold AUROC
    b1_auroc_ci = E3.auroc_ci(p_cv, present)
    print(f"B1 classifier: C={bestC}  p_star={p_star:.4f} (C-Abst={c1:.1f}%, F-Abst={f1:.1f}%)  "
          f"AUROC(oof)={b1_auroc:.3f}")
    if b1_auroc < 0.55:
        print("  [ΠΡΟΣΟΧΗ] AUROC ≈ 0.5 — ελέγξτε labels/σειρά features πριν συμπεράνετε.")

    pickle.dump({"clf": clf, "features": E3.FEATURES}, open(f"{OUT}/b1_classifier.pkl", "wb"))
    thr = {
        "criterion": f"max C-Abst s.t. F-Abst <= {MAX_F}% (TRAIN out-of-fold)",
        "n_present": n_pres, "n_missing": n_miss,
        "gate": {"T_dep": float(T_dep), "C_abst_train": round(gc, 2), "F_abst_train": round(gf, 2),
                 "auroc": round(gate_auroc, 4), "auroc_ci": [round(x, 4) for x in gate_auroc_ci],
                 "T5_on_train_curve": {"C_abst": round(at5["C_Abst"], 2), "F_abst": round(at5["F_Abst"], 2)}},
        "B1_clf": {"p_star": float(p_star), "C": bestC, "C_abst_train": round(c1, 2),
                   "F_abst_train": round(f1, 2), "auroc_oof": round(b1_auroc, 4),
                   "auroc_ci": [round(x, 4) for x in b1_auroc_ci], "features": E3.FEATURES},
    }
    json.dump(thr, open(f"{OUT}/thresholds_partial.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[OK] data/b1_classifier.pkl, data/thresholds_partial.json")
    print("  -> μετά: exp3a_b2_crag.py (monoT5, προαιρετικό) & exp3_freeze.py, μετά exp3b_report.py")


if __name__ == "__main__":
    main()
