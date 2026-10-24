#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Δ — Βήμα 0 (παραλλαγή CSV): REGRESSION χωρίς αναγέννηση.

Τα παλιά παγωμένα top-5 (ΕΙΣΟΔΟΙ) δεν σώθηκαν ξεχωριστά, οπότε αντί να ξανατρέξουμε το
παλιό pipeline, ΕΠΑΛΗΘΕΥΟΥΜΕ ότι τα ΑΠΟΘΗΚΕΥΜΕΝΑ αποτελέσματα της Φάσης 2
(eval_base.csv / eval_finetuned.csv) αναπαράγουν τους αριθμούς του άρθρου — ΜΕ ΤΟΥΣ
ΔΙΚΟΥΣ ΜΑΣ ορισμούς υποπληθυσμών/μετρικών:

  present = answerable & gt_in_context==1        (GT-present)
  missing = answerable & gt_in_context==0        (GT-missing)
  C-Abst  = 100 * mean(abstained | missing)
  F-Abst  = 100 * mean(abstained | present)

Στόχοι άρθρου (n: present=104 / missing=56):
  Base  C-Abst 7.1% (4/56),   F-Abst 6.7%
  QLoRA C-Abst 48.2% (27/56), F-Abst 24.0%

ΤΙ ΕΛΕΓΧΕΙ: αφού είναι ΤΑ ΙΔΙΑ δεδομένα, η συμφωνία πρέπει να είναι ~ακριβής. Αυτό
επικυρώνει ότι οι ΟΡΙΣΜΟΙ μας (present/missing, abstain) ταυτίζονται με του άρθρου —
το πιο επιρρεπές σε λάθη κομμάτι — ΠΡΙΝ τρέξουμε τα νέα runs στις 310.
ΔΕΝ επικυρώνει τη διαδρομή Ollama (prompt/template/num_ctx/temperature): αυτό θα απαιτούσε
τα παλιά frozen docs ως εισόδους, που δεν υπάρχουν. Το stage_bc_hooks είναι ήδη πιστό
αντίγραφο του evaluate_generation.py.

ΕΚΤΕΛΕΣΗ (stdlib μόνο· host ή container):
  python exp1d_regression_from_csv.py --base /data/eval_base.csv --qlora /data/eval_finetuned.csv
"""
import csv, argparse

TARGETS = {"Base":  {"C": 7.1,  "F": 6.7,  "n_miss": 56, "n_pres": 104},
           "QLoRA": {"C": 48.2, "F": 24.0, "n_miss": 56, "n_pres": 104}}
TOL = 5.0


def is_answerable(row):
    g = (row.get("gt_id") or "").strip()
    return g not in ("", "-", "None", "nan", "NaN")


def to_int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def analyze(path):
    present, missing = [], []
    n_total = n_ans = n_skip = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            n_total += 1
            if not is_answerable(row):
                continue
            n_ans += 1
            gic = to_int(row.get("gt_in_context"))
            ab = to_int(row.get("abstained"))
            if ab is None or gic is None:
                n_skip += 1
                continue
            (present if gic == 1 else missing).append(ab)
    c = 100 * sum(missing) / len(missing) if missing else 0.0
    fa = 100 * sum(present) / len(present) if present else 0.0
    return {"n_total": n_total, "n_ans": n_ans, "n_skip": n_skip,
            "n_present": len(present), "n_missing": len(missing),
            "C_Abst": c, "F_Abst": fa,
            "ab_miss": sum(missing), "ab_pres": sum(present)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",  default="/data/eval_base.csv")
    ap.add_argument("--qlora", default="/data/eval_finetuned.csv")
    a = ap.parse_args()

    ok_all = True
    counts_ok = True
    for name, path in [("Base", a.base), ("QLoRA", a.qlora)]:
        r = analyze(path)
        t = TARGETS[name]
        okc = abs(r["C_Abst"] - t["C"]) <= TOL
        okf = abs(r["F_Abst"] - t["F"]) <= TOL
        okn = (r["n_present"] == t["n_pres"] and r["n_missing"] == t["n_miss"])
        ok_all = ok_all and okc and okf
        counts_ok = counts_ok and okn
        print(f"\n{name}  ({path})")
        print(f"  rows={r['n_total']}  answerable={r['n_ans']}  skip={r['n_skip']}  "
              f"present={r['n_present']} missing={r['n_missing']}  "
              f"{'OK (104/56)' if okn else 'ΠΡΟΣΟΧΗ: counts != 104/56'}")
        print(f"  C-Abst = {r['C_Abst']:5.1f}%  (άρθρο {t['C']}%, Δ={r['C_Abst']-t['C']:+.1f})  "
              f"[{r['ab_miss']}/{r['n_missing']}]  -> {'OK' if okc else 'FAIL'}")
        print(f"  F-Abst = {r['F_Abst']:5.1f}%  (άρθρο {t['F']}%, Δ={r['F_Abst']-t['F']:+.1f})  "
              f"[{r['ab_pres']}/{r['n_present']}]  -> {'OK' if okf else 'FAIL'}")

    print("\n" + "=" * 62)
    if not counts_ok:
        print("[ΠΡΟΣΟΧΗ] Τα πλήθη present/missing δεν είναι 104/56 — ίσως τα CSV έχουν άλλο")
        print("  σύνολο (π.χ. περιλαμβάνουν unanswerable ή άλλο split). Ελέγξτε πριν συμπεράνετε.")
    if ok_all:
        print("REGRESSION (CSV): PASS — τα αποθηκευμένα Φάσης 2 αναπαράγουν το άρθρο.")
        print("  Οι ορισμοί present/missing + abstain συμφωνούν -> προχωράτε στα νέα runs (310).")
    else:
        print("REGRESSION (CSV): FAIL — ελέγξτε ορισμούς/αρχεία ΠΡΙΝ τα νέα runs.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
