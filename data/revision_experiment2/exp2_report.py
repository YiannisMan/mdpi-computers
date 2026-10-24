#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 2 — 3 επαναλήψεις + στατιστικοί έλεγχοι (ΧΩΡΙΣ νέο inference· μόνο ανάλυση logs 1Δ).

CONFIGS: Base, Gate (T_dep), Gate_T5 (T=5.0), QLoRA.
  abstain(Gate) = (gate_max_logit < T) Ή abstain(Base)  — κανένα νέο inference.

Παράγει (για reviewers/καθηγητή):
  analysis/tables/exp2_table4.csv               — Table 4: C-Abst/F-Abst/Hazard/citation/ROUGE-L ανά config
  analysis/tables/exp2_table4_wilson.csv         — Wilson 95% CI (majority) ανά config × πληθυσμό
  analysis/tables/exp2_pairwise_main.csv         — 6 γραμμές (3 ζεύγη × 2 μετρικές): McNemar+bootstrap+Holm
  analysis/tables/exp2_pairwise_sensitivity.csv  — per-run McNemar (ευαισθησία στο majority vote)
  analysis/tables/exp2_per_run.csv               — rates ανά (config, run)
  logs/exp2_report.json / logs/exp2_report.md

ΕΚΤΕΛΕΣΗ (pandas/numpy· sklearn/statsmodels προαιρετικά — υπάρχουν fallbacks):
  python exp2_report.py     # T_dep από data/T_dep.json (ή env TDEP). ROUGE-L αν βρεθεί το TAR.
"""
import glob, json, os, sys, datetime, tarfile, re
import numpy as np
import pandas as pd
sys.path.insert(0, "analysis")
import abst_lib as A

STAGEB_DIR = os.getenv("STAGEB_DIR", "../revision_experiment1/logs/stageB")
SUBS_CSV   = os.getenv("SUBS_CSV", "../revision_experiment1/data/test_subgroups.csv")
TAR_PATH   = os.getenv("TECHQA_TAR", "/data/TechQA.tar.gz")
TDEP_JSON  = os.getenv("TDEP_JSON", "data/T_dep.json")
T5         = float(os.getenv("T5", "5.0"))
NBOOT      = int(os.getenv("NBOOT", "10000"))
OUT_TAB, OUT_LOG = "analysis/tables", "logs"
COLS = ["config", "run", "question_id", "abstain"]


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)

def df_to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def get_Tdep():
    v = os.getenv("TDEP")
    if v:
        return float(v)
    if os.path.exists(TDEP_JSON):
        return float(json.load(open(TDEP_JSON, encoding="utf-8"))["T_dep"])
    print(f"[ERROR] δεν βρέθηκε T_dep ({TDEP_JSON}) — τρέξτε exp2_select_T.py ή δώστε env TDEP")
    sys.exit(1)


def load_stageB():
    rows = []
    for p in sorted(glob.glob(f"{STAGEB_DIR}/*.jsonl")):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[ERROR] κανένα Stage B log στο {STAGEB_DIR}"); sys.exit(1)
    if "abstain" not in df.columns:
        df["abstain"] = df["mode"].eq("B")
    df["abstain"] = df["abstain"].astype(bool)
    return df


def derive_gate(b, T, name):
    g = b[b.config == "Base"].copy()
    g["abstain"] = (g["gate_max_logit"].astype(float) < T) | g["abstain"].astype(bool)
    g["config"] = name
    return g


# ── ROUGE-L (LCS F1), self-contained ──
def _lcs(a, b):
    lb = len(b)
    if not a or not b:
        return 0
    dp = [0] * (lb + 1)
    for i in range(len(a)):
        prev = 0
        ai = a[i]
        for j in range(1, lb + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ai == b[j - 1] else (dp[j] if dp[j] >= dp[j - 1] else dp[j - 1])
            prev = tmp
    return dp[lb]

def rouge_l(cand, ref):
    c = re.findall(r"\w+", (cand or "").lower())
    r = re.findall(r"\w+", (ref or "").lower())
    if not c or not r:
        return 0.0
    l = _lcs(c, r); p = l / len(c); rec = l / len(r)
    return 0.0 if (p + rec) == 0 else 2 * p * rec / (p + rec)


def load_gold():
    ans = {}
    if not os.path.exists(TAR_PATH):
        return ans
    try:
        with tarfile.open(TAR_PATH, "r:*") as tar:
            m = next((x for x in tar.getmembers() if x.name.endswith("dev_Q_A.json")), None)
            if not m:
                return ans
            for r in json.load(tar.extractfile(m)):
                a = r.get("ANSWER")
                if isinstance(a, list):
                    a = " ".join(map(str, a))
                ans[r.get("QUESTION_ID")] = a or ""
    except Exception as e:
        print(f"[WARN] gold answers (ROUGE-L n/a): {e}")
    return ans


def mm(vals):
    v = [x for x in vals if x == x]  # drop nan
    if not v:
        return "n/a"
    v = np.array(v, float)
    return f"{v.mean():.1f} ({v.min():.1f}-{v.max():.1f})"


def quality_per_run(under, subs, gold, gate_T=None):
    """citation validity (%) + ROUGE-L στις present ΠΟΥ ΑΠΑΝΤΗΘΗΚΑΝ, ανά run.
    under: rows του υποκείμενου generator (Base ή QLoRA). gate_T: κόβει όσες gate_max_logit<T."""
    m = under.merge(subs, on="question_id", how="left")
    pres = m[m.subgroup == "present"].copy()
    answered = (pres["mode"] == "A")
    if gate_T is not None:
        answered = answered & (pres["gate_max_logit"].astype(float) >= gate_T)
    pres = pres[answered].copy()
    pres["rl"] = pres.apply(lambda r: rouge_l(r.get("raw_output", ""), gold.get(r["question_id"], "")),
                            axis=1) if gold else np.nan
    rows = []
    for run, g in pres.groupby("run"):
        rows.append({"run": run,
                     "citation_valid": 100 * g["citation_valid"].astype(float).mean() if len(g) else np.nan,
                     "rouge_l": g["rl"].mean() if (gold and len(g)) else np.nan,
                     "n_ans_present": len(g)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_TAB, exist_ok=True); os.makedirs(OUT_LOG, exist_ok=True)
    T_dep = get_Tdep()
    b = load_stageB()
    subs = pd.read_csv(SUBS_CSV)
    gold = load_gold()

    base = b[b.config == "Base"]
    qlora = b[b.config == "QLoRA"]
    gate = derive_gate(b, T_dep, "Gate")
    gate5 = derive_gate(b, T5, "Gate_T5")
    runs = pd.concat([base[COLS], gate[COLS], gate5[COLS], qlora[COLS]], ignore_index=True)
    CONFIGS = ["Base", "Gate", "Gate_T5", "QLoRA"]

    # ── Table 4: abstain μετρικές ──
    per = A.rates_per_run(runs, subs)
    per.to_csv(f"{OUT_TAB}/exp2_per_run.csv", index=False)
    summary = A.summarize_runs(per)   # mean (min-max) C-Abst missing/missA/missU, F-Abst present, Hazard

    # ποιότητα (citation, ROUGE-L) ανά config
    qual = {}
    qual["Base"]    = quality_per_run(base, subs, gold, None)
    qual["QLoRA"]   = quality_per_run(qlora, subs, gold, None)
    qual["Gate"]    = quality_per_run(base, subs, gold, T_dep)
    qual["Gate_T5"] = quality_per_run(base, subs, gold, T5)
    qual_rows = []
    for cfg in CONFIGS:
        q = qual[cfg]
        qual_rows.append({"config": cfg,
                          "citation_valid": mm(q["citation_valid"].tolist()),
                          "rouge_l": mm((100 * q["rouge_l"]).tolist()) if gold else "n/a",
                          "n_ans_present": int(q["n_ans_present"].max()) if len(q) else 0})
    table4 = summary.merge(pd.DataFrame(qual_rows), on="config")
    table4 = table4[["config", "C_Abst_missing", "C_Abst_missA", "C_Abst_missU",
                     "Hazard_missing", "F_Abst_present", "citation_valid", "rouge_l", "n_ans_present"]]
    table4.to_csv(f"{OUT_TAB}/exp2_table4.csv", index=False)

    # Wilson CI (majority) ανά config × πληθυσμό
    wrows = []
    for cfg in CONFIGS:
        for pop in ("missing", "missA", "missU", "present"):
            wrows.append(A.wilson_on_majority(runs, cfg, subs, pop))
    wilson = pd.DataFrame(wrows)
    wilson.to_csv(f"{OUT_TAB}/exp2_table4_wilson.csv", index=False)

    # ── 2Β: pairwise main family ──
    pairs = [("Gate", "Base"), ("QLoRA", "Base"), ("QLoRA", "Gate")]
    pm_c = A.pairwise_table(runs, subs, pairs, ("C-Abst", "missing"), B=NBOOT)
    pm_f = A.pairwise_table(runs, subs, pairs, ("F-Abst", "present"), B=NBOOT)
    pairwise = pd.concat([pm_c, pm_f], ignore_index=True)
    pairwise.to_csv(f"{OUT_TAB}/exp2_pairwise_main.csv", index=False)

    # ευαισθησία: per-run McNemar
    srows = []
    for k in sorted(runs.run.unique()):
        rk = runs[runs.run == k]
        for pop in ("missing", "present"):
            ids = A.population(subs, pop).tolist()
            for Ac, Bc in pairs:
                a = rk[rk.config == Ac].set_index("question_id").abstain.reindex(ids).dropna()
                bb = rk[rk.config == Bc].set_index("question_id").abstain.reindex(ids).dropna()
                common = a.index.intersection(bb.index)
                mc = A.mcnemar_exact(a.loc[common].astype(int).values, bb.loc[common].astype(int).values)
                srows.append({"run": k, "population": pop, "A": Ac, "B": Bc,
                              "p": round(mc["p"], 4), "delta_pp": round(mc["delta_pp"], 2)})
    sens = pd.DataFrame(srows)
    sens.to_csv(f"{OUT_TAB}/exp2_pairwise_sensitivity.csv", index=False)
    sens_summary = (sens.groupby(["population", "A", "B"]).p.agg(["min", "max"]).reset_index()
                    .rename(columns={"min": "p_min", "max": "p_max"}))

    # ── verdict ανά pairwise γραμμή (κανόνες 2Γ) ──
    def verdict(row):
        sig = (row["holm_p"] < 0.05) and not (row["boot_lo"] <= 0 <= row["boot_hi"])
        if sig:
            direction = "higher" if row["delta_pp"] > 0 else "lower"
            return f"significantly {direction}"
        return "no detectable difference"
    pairwise["verdict"] = pairwise.apply(verdict, axis=1)
    pairwise.to_csv(f"{OUT_TAB}/exp2_pairwise_main.csv", index=False)

    # ── JSON ──
    report = {
        "experiment": "2_Stats",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "T_dep": T_dep, "T5": T5, "n_bootstrap": NBOOT,
        "test_set": {"present": int((subs.subgroup == "present").sum()),
                     "missA": int((subs.subgroup == "missA").sum()),
                     "missU": int((subs.subgroup == "missU").sum())},
        "table4": table4.to_dict(orient="records"),
        "wilson_majority": wilson.to_dict(orient="records"),
        "pairwise_main": pairwise.to_dict(orient="records"),
        "pairwise_sensitivity": sens_summary.to_dict(orient="records"),
    }
    json.dump(report, open(f"{OUT_LOG}/exp2_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ── Markdown ──
    md = ["# Πείραμα 2 — 3 επαναλήψεις & στατιστικοί έλεγχοι\n",
          f"_Παράχθηκε: {report['generated_utc']}_\n",
          f"**T_dep = {T_dep:.3f}** (TRAIN out-of-fold)· Gate_T5 με T=5,0· bootstrap {NBOOT} resamples. "
          f"TEST: present={report['test_set']['present']}, missA={report['test_set']['missA']}, "
          f"missU={report['test_set']['missU']}.\n",
          "## Table 4 — mean (min–max) των 3 runs\n", df_to_md(table4),
          "\n## Wilson 95% CI στο majority vote\n"]
    wt = wilson.copy()
    wt["pct (95% CI)"] = wt.apply(lambda r: f"{r['pct']:.1f}% ({r['ci_lo']:.1f}-{r['ci_hi']:.1f}) [{r['k']}/{r['n']}]", axis=1)
    md.append(df_to_md(wt[["config", "population", "pct (95% CI)"]]))
    md += ["\n## Pairwise (McNemar + paired bootstrap + Holm)\n",
           df_to_md(pairwise[["metric", "A", "B", "delta_pp", "boot_lo", "boot_hi",
                              "mcnemar_p", "holm_p", "verdict"]]),
           "\n## Ευαισθησία — per-run McNemar p (min–max στα 3 runs)\n",
           df_to_md(sens_summary),
           "\n## Ερμηνεία / κανόνες διατύπωσης\n",
           "- `p_holm < 0,05` ΚΑΙ bootstrap CI χωρίς το 0 → «significantly higher/lower (Δ=x pp, "
           "95% CI [a,b], Holm p=…)».\n"
           "- `p_holm ≥ 0,05` → «no detectable difference» (όχι «equivalent»).\n"
           "- Effect size (Δ pp) αναφέρεται ΠΑΝΤΑ μαζί με το p.\n"
           "- Ισχύς: με ~200 incidents στο missing, διαφορές < ~10 pp συνήθως δεν ανιχνεύονται — "
           "δηλώνεται ως όριο, όχι ισοδυναμία.\n"
           "- temperature 0: η διακύμανση run-to-run αποδίδεται σε μη-ντετερμινισμό GPU, όχι sampling.\n"]
    open(f"{OUT_LOG}/exp2_report.md", "w", encoding="utf-8").write("\n".join(md))

    print("ΠΑΡΑΔΟΤΕΑ (αρχεία):")
    for p in (sorted(glob.glob(f"{OUT_TAB}/exp2_*.csv")) +
              [f"{OUT_LOG}/exp2_report.json", f"{OUT_LOG}/exp2_report.md"]):
        print("  " + p)
    print(f"\nT_dep={T_dep:.3f}")
    print("\nTable 4:"); print(table4.to_string(index=False))
    print("\nPairwise (main):")
    print(pairwise[["metric", "A", "B", "delta_pp", "mcnemar_p", "holm_p", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
