#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 3Α — Βήμα 4: πάγωμα κατωφλίων -> thresholds.json (commit ΠΡΙΝ το 3Β).
Ενοποιεί data/thresholds_partial.json (Gate+B1) + data/crag_thresholds.json (B2, αν υπάρχει).

ΕΚΤΕΛΕΣΗ:  python exp3_freeze.py
"""
import json, os, sys
OUT = "data/thresholds.json"


def main():
    if not os.path.exists("data/thresholds_partial.json"):
        print("[ERROR] λείπει data/thresholds_partial.json — τρέξτε exp3a_prepare.py"); sys.exit(1)
    thr = json.load(open("data/thresholds_partial.json", encoding="utf-8"))
    if os.path.exists("data/crag_thresholds.json"):
        thr["B2_crag"] = json.load(open("data/crag_thresholds.json", encoding="utf-8"))
    else:
        thr["B2_crag"] = None
        print("[ΣΗΜΕΙΩΣΗ] B2/CRAG δεν βρέθηκε — το 3Β θα τρέξει με Gate + B1 μόνο.")
    json.dump(thr, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[OK] {OUT} (commit το πριν το 3Β):")
    print(f"  gate.T_dep={thr['gate']['T_dep']:.4f}  AUROC={thr['gate']['auroc']}")
    print(f"  B1.p_star={thr['B1_clf']['p_star']:.4f}  AUROC(oof)={thr['B1_clf']['auroc_oof']}")
    if thr["B2_crag"]:
        print(f"  B2.theta_lo={thr['B2_crag']['theta_lo']:.3f} theta_hi={thr['B2_crag']['theta_hi']:.3f} "
              f"AUROC={thr['B2_crag']['auroc']}")


if __name__ == "__main__":
    main()
