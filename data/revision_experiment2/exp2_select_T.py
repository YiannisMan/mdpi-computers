#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 2 — προαπαιτούμενο (= 3Α-gate): επιλογή κατωφλίου T_dep του Gate ΣΤΟ TRAIN.

Διαβάζει τα 5-fold out-of-fold scores (retrieval_train_oof.jsonl, Πείραμα 1Β) και επιλέγει
το T που ΜΕΓΙΣΤΟΠΟΙΕΙ το C-Abst (αποχή στο missing) ΥΠΟ F-Abst ≤ 25% — κριτήριο
προ-καταχωρημένο (≈ το F-Abst 24% της QLoRA στο άρθρο). ΚΑΜΙΑ ματιά στο TEST.

score = max reranker logit του top-5 (OOF)·  present = answerable & gold στο top-5·
abstain(Gate) ⇔ score < T.

ΕΞΟΔΟΣ: data/T_dep.json

ΕΚΤΕΛΕΣΗ:
  python exp2_select_T.py
"""
import json, os, sys
sys.path.insert(0, "analysis")
import abst_lib as A

OOF   = os.getenv("OOF_PATH", "../revision_experiment1/data/retrieval_train_oof.jsonl")
OUT   = os.getenv("TDEP_OUT", "data/T_dep.json")
MAX_F = float(os.getenv("MAX_F", "25.0"))


def main():
    if not os.path.exists(OOF):
        print(f"[ERROR] λείπει {OOF} (τρέξτε 1Β --score-fold στο Πείραμα 1)"); sys.exit(1)
    recs = [json.loads(l) for l in open(OOF, encoding="utf-8") if l.strip()]
    score = [float(r.get("max_logit", 0.0)) for r in recs]
    present = [bool(r.get("present")) for r in recs]
    T, C, F = A.select_threshold(score, present, max_f=MAX_F)
    n_pres = int(sum(present)); n_miss = len(present) - n_pres
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    out = {"T_dep": T, "C_Abst_train_pct": round(C, 2), "F_Abst_train_pct": round(F, 2),
           "max_f_constraint_pct": MAX_F, "n_present": n_pres, "n_missing": n_miss,
           "n_oof": len(recs), "score": "max_rerank_logit_top5", "source": OOF}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[OK] {OUT}")
    print(f"  T_dep = {T:.4f}   (TRAIN-OOF: C-Abst={C:.1f}%, F-Abst={F:.1f}% ≤ {MAX_F}%)")
    print(f"  n_present={n_pres}  n_missing={n_miss}  (OOF n={len(recs)})")
    print("  -> το exp2_report.py το διαβάζει αυτόματα από data/T_dep.json")


if __name__ == "__main__":
    main()
