#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Δ — Αναφορά αποτελεσμάτων Stage B ΣΕ ΑΡΧΕΙΑ (για reviewers/καθηγητή).

Παράγει (ΟΧΙ απλώς στο cmd):
  analysis/tables/exp1d_integrity.csv        — οι 4 έλεγχοι ακεραιότητας ανά (config,run)
  analysis/tables/exp1d_stageB_per_run.csv    — C-Abst/F-Abst/Hazard ανά (config,run)
  analysis/tables/exp1d_stageB_summary.csv    — mean (min–max) 3 runs + Wilson CI (majority) ανά config
  analysis/tables/exp1d_perquestion.csv       — ανά ερώτηση: runs, majority, subgroup, citation
  logs/exp1d_report.json                      — ΟΛΑ (machine-readable) + verdicts + sha256
  logs/exp1d_report.md                        — ανθρώπινη σύνοψη (αυτό δίνεται στον καθηγητή)

ΟΡΙΣΜΟΙ (ANALYSIS_PLAN):
  abstain = (mode == "B")·  invalid ΔΕΝ είναι abstain (μετρά ως «απάντηση»).
  C-Abst = % abstain στο missing (σωστή αποχή· ΥΨΗΛΟ=καλό, Hazard=100−C-Abst).
  F-Abst = % abstain στο present (εσφαλμένη αποχή· ΧΑΜΗΛΟ=καλό).
  Τα 3 runs ΔΕΝ είναι ανεξάρτητα -> Wilson 95% CI πάνω στο majority vote.

ΕΚΤΕΛΕΣΗ (host venv με pandas/numpy, ή in-container με pip install pandas numpy):
  python exp1d_report.py
"""
import glob, json, os, sys, datetime
import numpy as np
import pandas as pd
sys.path.insert(0, "analysis")
import abst_lib as A

def df_to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


LOG_DIR   = os.getenv("STAGEB_DIR", "logs/stageB")
SUBS_CSV  = os.getenv("SUBS_CSV", "data/test_subgroups.csv")
FROZEN    = os.getenv("FROZEN_PATH", "data/retrieval_test.jsonl")
ENV_JSON  = os.getenv("ENV_JSON", "env.json")
EXPECT_N  = int(os.getenv("EXPECT_N", "310"))
OUT_TAB   = "analysis/tables"
OUT_LOG   = "logs"


def load_logs():
    rows = []
    for p in sorted(glob.glob(f"{LOG_DIR}/*_run*.jsonl")):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    if "abstain" not in df.columns and "mode" in df.columns:
        df["abstain"] = df["mode"].eq("B")
    return df


def load_frozen_ids():
    ids = {}
    if not os.path.exists(FROZEN):
        return ids
    for l in open(FROZEN, encoding="utf-8"):
        l = l.strip()
        if l:
            r = json.loads(l); ids[r["question_id"]] = list(r.get("ids") or [])
    return ids


def integrity(df, frozen):
    """Οι 4 έλεγχοι ανά (config,run). Επιστρέφει (DataFrame, prompt_hash_info, all_ok)."""
    rec = []
    all_ok = True
    for (cfg, run), g in df.groupby(["config", "run"]):
        qids = g["question_id"].tolist()
        uniq = len(set(qids)); dup = len(qids) - uniq
        mism = missing = 0
        if frozen:
            for _, r in g.iterrows():
                fid = frozen.get(r["question_id"])
                if fid is None:
                    missing += 1
                elif list(r.get("retrieved_ids") or []) != fid:
                    mism += 1
        n = len(g)
        n_inv = int(g["invalid"].fillna(False).astype(bool).sum()) if "invalid" in g else 0
        n_tr = int(g["truncated"].fillna(False).astype(bool).sum()) if "truncated" in g else 0
        ok = (n == EXPECT_N) and (dup == 0) and (mism == 0 and missing == 0)
        all_ok = all_ok and ok
        rec.append({"config": cfg, "run": run, "lines": n, "unique_qids": uniq, "duplicates": dup,
                    "ids_match_frozen": "OK" if (frozen and mism == 0 and missing == 0) else
                                        ("n/a" if not frozen else f"{mism}mism/{missing}missing"),
                    "invalid": n_inv, "invalid_pct": round(100 * n_inv / n, 2),
                    "truncated": n_tr, "truncated_pct": round(100 * n_tr / n, 2),
                    "pass": bool(ok)})
    # έλεγχος 4: prompt template hash vs env.json
    ph = {"computed": None, "env_json": None, "match": None}
    try:
        os.environ.setdefault("MAIN_PY_PATH", "/app/main.py")
        import stage_bc_hooks as H
        ph["computed"] = H.prompt_template_sha256() if H.RAG_SYSTEM_PROMPT else None
    except Exception as e:
        ph["computed"] = f"(n/a: {e})"
    if os.path.exists(ENV_JSON):
        env = json.load(open(ENV_JSON, encoding="utf-8"))
        ph["env_json"] = (env.get("sha256") or {}).get("prompt_template")
    ph["match"] = (ph["computed"] is not None and ph["computed"] == ph["env_json"])
    all_ok = all_ok and bool(ph["match"])
    return pd.DataFrame(rec), ph, all_ok


def stageB_metrics(df, subs):
    runs_df = df[["config", "run", "question_id", "abstain"]].copy()
    per_run = A.rates_per_run(runs_df, subs)
    summary = A.summarize_runs(per_run)

    # Wilson CI στο majority vote ανά config × πληθυσμό
    wilson_rows = []
    for cfg in sorted(df["config"].unique()):
        for pop in ("missing", "missA", "missU", "present"):
            wilson_rows.append(A.wilson_on_majority(runs_df, cfg, subs, pop))
    wilson = pd.DataFrame(wilson_rows)

    # citation validity (answered present) + invalid/truncated ανά config (pooled 3 runs)
    m = df.merge(subs, on="question_id", how="left")
    extra = []
    for cfg, g in m.groupby("config"):
        pres_ans = g[(g.subgroup == "present") & (g["mode"] == "A")]
        cv = 100 * pres_ans["citation_valid"].astype(float).mean() if len(pres_ans) else float("nan")
        extra.append({"config": cfg,
                      "citation_valid_present_answered_pct": round(cv, 1) if cv == cv else "n/a",
                      "n_present_answered": int(len(pres_ans)),
                      "invalid_pct_pooled": round(100 * g["invalid"].astype(bool).mean(), 2),
                      "truncated_pct_pooled": round(100 * g["truncated"].astype(bool).mean(), 2)})
    extra = pd.DataFrame(extra)
    return per_run, summary, wilson, extra


def per_question(df, subs):
    runs = sorted(df["run"].unique())
    piv = df.pivot_table(index=["config", "question_id"], columns="run",
                         values="abstain", aggfunc="first")
    piv.columns = [f"abstain_run{r}" for r in piv.columns]
    piv = piv.reset_index()
    piv["majority_abstain"] = (df.groupby(["config", "question_id"]).abstain.mean()
                               .reindex(list(zip(piv.config, piv.question_id))).values >= 0.5)
    # citation_valid majority (per qid,config) + any_invalid
    cv = df.groupby(["config", "question_id"]).citation_valid.mean()
    inv = df.groupby(["config", "question_id"]).invalid.max()
    key = list(zip(piv.config, piv.question_id))
    piv["citation_valid_mean"] = cv.reindex(key).values
    piv["any_invalid"] = inv.reindex(key).values
    piv = piv.merge(subs, on="question_id", how="left")
    return piv


def main():
    os.makedirs(OUT_TAB, exist_ok=True); os.makedirs(OUT_LOG, exist_ok=True)
    df = load_logs()
    if df.empty:
        print(f"[ERROR] κανένα log στο {LOG_DIR}"); sys.exit(1)
    if not os.path.exists(SUBS_CSV):
        print(f"[ERROR] λείπει {SUBS_CSV} (τρέξτε 1Γ)"); sys.exit(1)
    subs = pd.read_csv(SUBS_CSV)
    frozen = load_frozen_ids()

    integ, ph, integ_ok = integrity(df, frozen)
    per_run, summary, wilson, extra = stageB_metrics(df, subs)
    pq = per_question(df, subs)

    # ── γράψιμο CSV ──
    integ.to_csv(f"{OUT_TAB}/exp1d_integrity.csv", index=False)
    per_run.to_csv(f"{OUT_TAB}/exp1d_stageB_per_run.csv", index=False)
    summary.merge(extra, on="config").to_csv(f"{OUT_TAB}/exp1d_stageB_summary.csv", index=False)
    wilson.to_csv(f"{OUT_TAB}/exp1d_stageB_wilson_majority.csv", index=False)
    pq.to_csv(f"{OUT_TAB}/exp1d_perquestion.csv", index=False)

    # ── JSON (machine-readable, πλήρες) ──
    report = {
        "experiment": "1D_StageB",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_set": {"n_total": int(subs.shape[0]),
                     "present": int((subs.subgroup == "present").sum()),
                     "missA": int((subs.subgroup == "missA").sum()),
                     "missU": int((subs.subgroup == "missU").sum())},
        "integrity": {"per_config_run": integ.to_dict(orient="records"),
                      "prompt_template": ph, "all_pass": bool(integ_ok)},
        "stageB_per_run": per_run.to_dict(orient="records"),
        "stageB_summary": summary.merge(extra, on="config").to_dict(orient="records"),
        "wilson_majority": wilson.to_dict(orient="records"),
        "env_sha256": (json.load(open(ENV_JSON, encoding="utf-8")).get("sha256")
                       if os.path.exists(ENV_JSON) else None),
    }
    json.dump(report, open(f"{OUT_LOG}/exp1d_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ── Markdown σύνοψη (για τον καθηγητή) ──
    md = []
    md.append("# Πείραμα 1Δ — Stage B: σύνοψη & επαλήθευση\n")
    md.append(f"_Παράχθηκε: {report['generated_utc']}_\n")
    md.append(f"**Held-out TEST:** {report['test_set']['n_total']} ερωτήσεις "
              f"(present={report['test_set']['present']}, missA={report['test_set']['missA']}, "
              f"missU={report['test_set']['missU']}).\n")
    md.append("## 1. Ακεραιότητα (4 έλεγχοι)\n")
    md.append(df_to_md(integ))
    pm = ph["match"]
    md.append(f"\n**Έλεγχος prompt template:** computed=`{ph['computed']}` "
              f"{'== env.json ✓' if pm else '!= env.json ✗'}\n")
    md.append(f"\n**Συμπέρασμα ακεραιότητας:** {'PASS — όλα τα logs ακέραια & συνεπή.' if integ_ok else 'ΥΠΑΡΧΟΥΝ ΑΠΟΚΛΙΣΕΙΣ.'}\n")
    md.append("## 2. Αποτελέσματα Stage B — mean (min–max) των 3 runs\n")
    md.append(df_to_md(summary.merge(extra, on="config")))
    md.append("\n## 3. Wilson 95% CI στο majority vote (3 runs)\n")
    wtab = wilson.copy()
    wtab["pct (95% CI)"] = wtab.apply(lambda r: f"{r['pct']:.1f}% ({r['ci_lo']:.1f}-{r['ci_hi']:.1f}) [{r['k']}/{r['n']}]", axis=1)
    md.append(df_to_md(wtab[["config","population","pct (95% CI)"]]))
    md.append("\n## Ερμηνεία\n")
    md.append("- **C-Abst** (missing): σωστή αποχή — υψηλό = καλό· **Hazard** = 100−C-Abst.\n"
              "- **F-Abst** (present): εσφαλμένη αποχή — χαμηλό = καλό.\n"
              "- invalid = έξοδοι χωρίς έγκυρη δομή/παραπομπή (μετρώνται ως «απάντηση», όχι abstain).\n"
              "- Τα 3 runs δεν είναι ανεξάρτητα (ίδιες ερωτήσεις) → αναφέρεται Wilson CI στο majority vote.\n")
    open(f"{OUT_LOG}/exp1d_report.md", "w", encoding="utf-8").write("\n".join(md))

    # ── σύντομη εκτύπωση ──
    print("ΠΑΡΑΔΟΤΕΑ (αρχεία):")
    for p in [f"{OUT_TAB}/exp1d_integrity.csv", f"{OUT_TAB}/exp1d_stageB_per_run.csv",
              f"{OUT_TAB}/exp1d_stageB_summary.csv", f"{OUT_TAB}/exp1d_stageB_wilson_majority.csv",
              f"{OUT_TAB}/exp1d_perquestion.csv", f"{OUT_LOG}/exp1d_report.json",
              f"{OUT_LOG}/exp1d_report.md"]:
        print("  " + p)
    print(f"\nΑκεραιότητα: {'PASS' if integ_ok else 'ΕΛΕΓΞΕ'}  | prompt template "
          f"{'== env.json' if ph['match'] else '!= / n/a'}")
    print("\nΣύνοψη Stage B:")
    print(summary.merge(extra, on="config").to_string(index=False))


if __name__ == "__main__":
    main()
