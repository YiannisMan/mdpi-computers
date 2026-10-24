#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Ε — Ανάλυση (δουλεύει πλήρως πάνω στα εξαγόμενα JSONL).

(α) Paired B–C agreement: ανά run, agreement + Cohen's kappa + exact McNemar,
    και ΘΟΡΥΒΟΣ ΒΑΣΗΣ (συμφωνία Stage B με τον εαυτό του ανά ζεύγος runs).
    Ερμηνεία: αφού retrieval/adapter/prompt είναι ίδια, η συμφωνία B–C πρέπει να είναι
    τουλάχιστον όσο η B–B. Αν είναι σημαντικά χαμηλότερη -> ο worker αλλοιώνει την είσοδο.

(β) Πίνακας Stage C ανά υποπληθυσμό (present/missA/missU/σύνολο): n, % auto-resolved,
    % escalated, ως mean (min–max) των 3 runs. Η γραμμή missU απαντά στο R1-M4.

ΚΥΡΙΟ: όλες οι 310 ερωτήσεις (report-only για διπλότυπες/κοινά gold — ANALYSIS_PLAN §1.1).
ΕΥΑΙΣΘΗΣΙΑ (προαιρετικά): με --exclude <qids.txt> υπολογίζεται ΕΠΙΠΛΕΟΝ ο ίδιος πίνακας
χωρίς τις 23 διπλότυπες (287), post-hoc, από τα ΙΔΙΑ logs — ΚΑΝΕΝΑ δεύτερο πέρασμα LLM.

ΕΚΤΕΛΕΣΗ:
  python exp1e_analysis.py                                    # κύριο (310)
  python exp1e_analysis.py --exclude data/test_exclude_qids.txt   # + ευαισθησία 287
"""
import glob, json, os, sys, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, "analysis")
import abst_lib as A


def load_stage(stage_dir, config_filter=None):
    rows = []
    for p in glob.glob(f"{stage_dir}/*.jsonl"):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if config_filter and r.get("config") != config_filter:
                continue
            rows.append(r)
    return pd.DataFrame(rows)


def paired_bc(B, C, runs, label):
    print("=" * 64)
    print(f"(α) PAIRED B–C AGREEMENT — {label}")
    print("=" * 64)

    def series(df, run):
        return df[df.run == run].set_index("question_id").abstain.astype(int)

    for k in runs:
        b, c = series(B, k), series(C, k)
        common = b.index.intersection(c.index)
        b2, c2 = b.loc[common], c.loc[common]
        agr = (b2 == c2).mean()
        kappa = A.cohen_kappa(b2.values, c2.values)
        mc = A.mcnemar_exact(b2.values, c2.values)
        print(f"  run {k}: agreement={agr:.3f}  kappa={kappa:.3f}  "
              f"McNemar p={mc['p']:.3g} (b01={mc['b01']}, b10={mc['b10']})  n={len(common)}")
    print("\n  ΘΟΡΥΒΟΣ ΒΑΣΗΣ (Stage B vs εαυτός του):")
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        if i in runs and j in runs:
            bi, bj = series(B, i), series(B, j)
            common = bi.index.intersection(bj.index)
            agr = (bi.loc[common] == bj.loc[common]).mean()
            print(f"    B run{i} vs run{j}: agreement={agr:.3f}")


def stageC_table(C, subs, runs, label, out_csv):
    print("\n" + "=" * 64)
    print(f"(β) STAGE C — auto-resolved / escalated ανά υποπληθυσμό  [mean (min–max)] — {label}")
    print("=" * 64)
    C = C.merge(subs, on="question_id", how="left")
    pops = {"present": ["present"], "missA": ["missA"], "missU": ["missU"],
            "missing": ["missA", "missU"], "all": ["present", "missA", "missU"]}
    rows = []
    for pop, sgs in pops.items():
        sub = C[C.subgroup.isin(sgs)]
        per_run_esc = []
        for k in runs:
            rk = sub[sub.run == k]
            if len(rk) == 0:
                continue
            per_run_esc.append(100 * rk.abstain.mean())
        if not per_run_esc:
            continue
        esc_arr = np.array(per_run_esc)
        n = int(sub[sub.run == runs[0]].shape[0]) if runs else 0
        rows.append({
            "subgroup": pop, "n": n,
            "escalated_%": f"{esc_arr.mean():.1f} ({esc_arr.min():.1f}-{esc_arr.max():.1f})",
            "auto_resolved_%": f"{100-esc_arr.mean():.1f} ({100-esc_arr.max():.1f}-{100-esc_arr.min():.1f})",
        })
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    os.makedirs("analysis/tables", exist_ok=True)
    tab.to_csv(out_csv, index=False)
    print(f"\n  -> {out_csv}")
    mu = tab[tab.subgroup == "missU"]
    if len(mu):
        print(f"  [R1-M4] Από τις γνωστά unanswerable (missU), auto-resolved: "
              f"{mu.iloc[0]['auto_resolved_%']}")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default=None,
                    help="αρχείο qids προς αποκλεισμό (ευαισθησία 287) — π.χ. data/test_exclude_qids.txt")
    args = ap.parse_args()

    subs = pd.read_csv("data/test_subgroups.csv")
    B = load_stage("logs/stageB", config_filter="QLoRA")   # Stage B QLoRA = σύγκριση με C
    C = load_stage("logs/stageC")
    if B.empty or C.empty:
        print("[WARN] λείπουν logs Stage B (QLoRA) ή Stage C. Τρέξτε 1Δ/1Ε πρώτα.")
        return
    for df in (B, C):
        if "abstain" not in df.columns:
            df["abstain"] = df["mode"].eq("B")
    runs = sorted(set(B.run) & set(C.run))

    # ── ΚΥΡΙΟ: όλες οι 310 ──
    paired_bc(B, C, runs, label="ΚΥΡΙΟ (310)")
    stageC_table(C, subs, runs, "ΚΥΡΙΟ (310)", "analysis/tables/stageC_by_subgroup.csv")

    # ── ΕΥΑΙΣΘΗΣΙΑ: χωρίς τις διπλότυπες (287), post-hoc ──
    if args.exclude:
        if not os.path.exists(args.exclude):
            print(f"\n[WARN] δεν βρέθηκε {args.exclude} — παραλείπεται η ευαισθησία 287.")
            return
        excl = {l.strip() for l in open(args.exclude, encoding="utf-8") if l.strip()}
        Bc = B[~B.question_id.isin(excl)]
        Cc = C[~C.question_id.isin(excl)]
        print("\n\n" + "#" * 64)
        print(f"# ΑΝΑΛΥΣΗ ΕΥΑΙΣΘΗΣΙΑΣ — χωρίς {len(excl)} διπλότυπες (ίδια logs, χωρίς νέο LLM)")
        print("#" * 64)
        paired_bc(Bc, Cc, runs, label="ΕΥΑΙΣΘΗΣΙΑ (287)")
        stageC_table(Cc, subs, runs, "ΕΥΑΙΣΘΗΣΙΑ (287)",
                     "analysis/tables/stageC_by_subgroup_clean287.csv")


if __name__ == "__main__":
    main()
