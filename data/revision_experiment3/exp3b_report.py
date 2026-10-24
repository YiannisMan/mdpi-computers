#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 3Β — Εφαρμογή baselines στο TEST + καμπύλες + iso-F-Abst + AUROC + report.

Όλα ως «φίλτρο πριν τον Base»: abstain = (score < κατώφλι) Ή abstain(Base). Κανένα νέο
inference (εκτός των CRAG-Ambiguous, που εδώ αντιμετωπίζονται ως score-filter — βλ. README).

Παράγει:
  analysis/figures/abstention_tradeoff.pdf/.png   — καμπύλες Gate/B1/(B2) + σημεία
  analysis/tables/exp3_iso_fabst.csv               — iso-F-Abst ανά μέθοδο (+ CI)
  analysis/tables/exp3_auroc.csv                    — AUROC (TEST) ανά score + CI
  analysis/tables/exp3_table4_supplement.csv        — γραμμές B1/B2/Gate/Gate_T5/Gate∘QLoRA
  analysis/tables/exp3_pairwise_secondary.csv       — QLoRA vs B1/B2 (+ Gate), McNemar/bootstrap/Holm
  analysis/tables/exp3_curve_{gate,b1,b2}.csv
  logs/exp3_report.json / logs/exp3_report.md

ΕΚΤΕΛΕΣΗ (pandas/numpy/scikit-learn/matplotlib):
  python exp3b_report.py
"""
import glob, json, os, sys, datetime, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "analysis")
import abst_lib as A
import exp3_lib as E3

STAGEB_DIR = os.getenv("STAGEB_DIR", "../revision_experiment1/logs/stageB")
SUBS_CSV   = os.getenv("SUBS_CSV", "../revision_experiment1/data/test_subgroups.csv")
FROZEN     = os.getenv("FROZEN_PATH", "../revision_experiment1/data/retrieval_test.jsonl")
DOCS       = os.getenv("TEST_DOCS", "../revision_experiment1/data/retrieval_test_docs.jsonl")
THR        = os.getenv("THRESHOLDS", "data/thresholds.json")
THR_PART   = "data/thresholds_partial.json"
CRAG_TEST  = "data/crag_scores_test.json"
CRAG_THR   = "data/crag_thresholds.json"
CLF_PKL    = "data/b1_classifier.pkl"
T5         = float(os.getenv("T5", "5.0"))
NBOOT      = int(os.getenv("NBOOT", "2000"))
OUT_TAB, OUT_FIG, OUT_LOG = "analysis/tables", "analysis/figures", "logs"


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)

def df_to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(out)


def load_thresholds():
    if os.path.exists(THR):
        return json.load(open(THR, encoding="utf-8"))
    if os.path.exists(THR_PART):
        return json.load(open(THR_PART, encoding="utf-8"))
    print(f"[ERROR] λείπουν κατώφλια ({THR} ή {THR_PART}) — τρέξτε exp3a_prepare.py"); sys.exit(1)


def load_stageB():
    rows = []
    for p in sorted(glob.glob(f"{STAGEB_DIR}/*.jsonl")):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    df = pd.DataFrame(rows)
    if "abstain" not in df.columns:
        df["abstain"] = df["mode"].eq("B")
    df["abstain"] = df["abstain"].astype(bool)
    return df


def main():
    os.makedirs(OUT_TAB, exist_ok=True); os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_LOG, exist_ok=True)
    thr = load_thresholds()
    T_dep = thr["gate"]["T_dep"]; p_star = thr["B1_clf"]["p_star"]
    subs = pd.read_csv(SUBS_CSV)
    b = load_stageB()
    frozen = {json.loads(l)["question_id"]: json.loads(l) for l in open(FROZEN, encoding="utf-8") if l.strip()}
    docs = {json.loads(l)["question_id"]: json.loads(l) for l in open(DOCS, encoding="utf-8") if l.strip()}
    qids = list(frozen.keys())

    # ── TEST scores ──
    gate_score = {q: float(frozen[q].get("gate_max_logit")) for q in qids}
    clf = pickle.load(open(CLF_PKL, "rb"))["clf"]
    featrows = []
    for q in qids:
        logits = frozen[q].get("rerank_logits") or []
        ticket = docs.get(q, {}).get("ticket", "")
        featrows.append([E3.features_from_logits(logits, ticket)[f] for f in E3.FEATURES])
    p_b1 = dict(zip(qids, clf.predict_proba(np.array(featrows))[:, 1]))
    crag = None
    if os.path.exists(CRAG_TEST):
        crag = {q: float(v) for q, v in json.load(open(CRAG_TEST)).items()}

    # ── per-question frame ──
    sub_map = dict(zip(subs.question_id, subs.subgroup))
    present = np.array([sub_map.get(q) == "present" for q in qids])
    base_runs = b[b.config == "Base"]
    base_maj = A.majority_vote(b, "Base", pd.Index(qids)).reindex(qids).fillna(False).astype(bool).values

    # ── runs_ext: configs ως φίλτρα πριν τον Base ──
    COLS = ["config", "run", "question_id", "abstain"]
    def gate_like(T, name):
        g = base_runs.copy()
        g["abstain"] = (g["gate_max_logit"].astype(float) < T) | g["abstain"].astype(bool)
        g["config"] = name; return g[COLS]
    def score_filter(score_map, thr_v, name):
        g = base_runs.copy()
        low = g["question_id"].map(lambda q: score_map.get(q, np.inf) < thr_v)
        g["abstain"] = g["abstain"].astype(bool) | low.values
        g["config"] = name; return g[COLS]

    parts = [b[b.config == "Base"][COLS], b[b.config == "QLoRA"][COLS],
             gate_like(T_dep, "Gate"), gate_like(T5, "Gate_T5"),
             score_filter(p_b1, p_star, "B1_clf")]
    configs = ["Base", "QLoRA", "Gate", "Gate_T5", "B1_clf"]
    if crag is not None and os.path.exists(CRAG_THR):
        theta_lo = json.load(open(CRAG_THR))["theta_lo"]
        parts.append(score_filter(crag, theta_lo, "B2_crag")); configs.append("B2_crag")
    # Gate∘QLoRA: abstain αν Gate Ή QLoRA
    gate_rows = gate_like(T_dep, "GateQLoRA").set_index(["run", "question_id"]).abstain
    ql_rows = b[b.config == "QLoRA"].set_index(["run", "question_id"]).abstain.astype(bool)
    comp = (gate_rows | ql_rows.reindex(gate_rows.index).fillna(False)).reset_index()
    comp["config"] = "Gate∘QLoRA"
    parts.append(comp[COLS]); configs.append("Gate∘QLoRA")
    runs = pd.concat(parts, ignore_index=True)

    # ── Table 4 supplement (mean min-max + Wilson majority) ──
    per = A.rates_per_run(runs, subs)
    summ = A.summarize_runs(per)
    wrows = [A.wilson_on_majority(runs, c, subs, p) for c in configs for p in ("missing", "present")]
    wilson = pd.DataFrame(wrows)
    summ.to_csv(f"{OUT_TAB}/exp3_table4_supplement.csv", index=False)
    wilson.to_csv(f"{OUT_TAB}/exp3_table4_wilson.csv", index=False)

    # ── καμπύλες + σχήμα ──
    scores = {"Gate (reranker logit)": np.array([gate_score[q] for q in qids]),
              "B1 answerability clf": np.array([p_b1[q] for q in qids])}
    if crag is not None:
        scores["B2 CRAG (monoT5)"] = np.array([crag.get(q, np.nan) for q in qids])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, sc in scores.items():
        ok = np.isfinite(sc)
        cv = E3.curve_tradeoff(sc[ok], present[ok], base_maj[ok])
        cv.to_csv(f"{OUT_TAB}/exp3_curve_{name.split()[0].lower()}.csv", index=False)
        ax.plot(cv.F_abst, cv.C_abst, label=name, lw=1.6)
    ax.plot([0, 100], [0, 100], "k:", lw=0.8, label="random")
    # σημεία: Base, QLoRA (majority)
    def pt(cfg, marker, lab):
        mv_m = A.majority_vote(runs, cfg, A.population(subs, "missing")).mean() * 100
        mv_p = A.majority_vote(runs, cfg, A.population(subs, "present")).mean() * 100
        ax.scatter([mv_p], [mv_m], marker=marker, s=60, zorder=5, label=lab)
    for cfg, mk in [("Base", "s"), ("QLoRA", "*"), ("Gate", "o"), ("B1_clf", "^")]:
        if cfg in configs:
            pt(cfg, mk, cfg)
    ax.set_xlabel("False abstention on GT-present (%)")
    ax.set_ylabel("Correct abstention on GT-missing (%)")
    ax.legend(fontsize=7); ax.set_title("Abstention trade-off", fontsize=10)
    fig.savefig(f"{OUT_FIG}/abstention_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_FIG}/abstention_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── iso-F-Abst (target = QLoRA F-Abst) ──
    qa = E3.per_incident_rate(runs, "QLoRA").reindex(qids).fillna(0).values
    target_f = 100 * qa[present].mean()
    iso_rows = []
    for name, sc in scores.items():
        ok = np.isfinite(sc)
        Tstar, c_a, f_a = E3.iso_f_abst(sc[ok], present[ok], base_maj[ok], target_f)
        lo, med, hi = E3.iso_f_abst_ci(sc[ok], present[ok], base_maj[ok], qa[ok], B=NBOOT)
        qlora_c = 100 * qa[~present].mean()
        iso_rows.append({"method": name, "T_star": round(Tstar, 4), "F_abst": round(f_a, 1),
                         "C_abst": round(c_a, 1), "C_abst_CI": f"[{lo:.1f}, {hi:.1f}]",
                         "QLoRA_C_abst": round(qlora_c, 1),
                         "QLoRA_minus_method_pp": round(qlora_c - c_a, 1)})
    iso = pd.DataFrame(iso_rows); iso.to_csv(f"{OUT_TAB}/exp3_iso_fabst.csv", index=False)

    # ── AUROC (TEST) ──
    au_rows = []
    for name, sc in scores.items():
        ok = np.isfinite(sc)
        au = E3.auroc_missing(sc[ok], present[ok]); lo, hi = E3.auroc_ci(sc[ok], present[ok], B=NBOOT)
        au_rows.append({"score": name, "AUROC": round(au, 3), "CI": f"[{lo:.3f}, {hi:.3f}]"})
    auroc = pd.DataFrame(au_rows); auroc.to_csv(f"{OUT_TAB}/exp3_auroc.csv", index=False)

    # ── pairwise δευτερεύουσα οικογένεια ──
    pairs = [("QLoRA", "B1_clf")]
    if "B2_crag" in configs:
        pairs.append(("QLoRA", "B2_crag"))
    pm_c = A.pairwise_table(runs, subs, pairs, ("C-Abst", "missing"), B=NBOOT)
    pm_f = A.pairwise_table(runs, subs, pairs, ("F-Abst", "present"), B=NBOOT)
    pw = pd.concat([pm_c, pm_f], ignore_index=True)
    pw.to_csv(f"{OUT_TAB}/exp3_pairwise_secondary.csv", index=False)

    # ── report ──
    report = {"experiment": "3_Baselines",
              "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "thresholds": thr, "target_F_abst_QLoRA": round(target_f, 1),
              "iso_fabst": iso.to_dict(orient="records"), "auroc": auroc.to_dict(orient="records"),
              "table4_supplement": summ.to_dict(orient="records"),
              "pairwise_secondary": pw.to_dict(orient="records"),
              "configs": configs, "crag_available": crag is not None}
    json.dump(report, open(f"{OUT_LOG}/exp3_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    md = ["# Πείραμα 3 — Baselines & καμπύλη gate\n",
          f"_Παράχθηκε: {report['generated_utc']}_\n",
          f"T_dep={T_dep:.3f}, p*(B1)={p_star:.3f}"
          + (f", CRAG θ_lo={json.load(open(CRAG_THR))['theta_lo']:.3f}" if crag is not None else " (B2/CRAG: δεν τρέχθηκε)")
          + f". iso-F-Abst target = QLoRA F-Abst = {target_f:.1f}%.\n",
          "## AUROC στο TEST (ποιότητα score χωρίς κατώφλι)\n", df_to_md(auroc),
          "\n## iso-F-Abst (ίδιο κόστος F-Abst με QLoRA)\n", df_to_md(iso),
          "\n## Table 4 — συμπλήρωμα (mean min–max 3 runs)\n", df_to_md(summ),
          "\n## Pairwise δευτερεύουσα οικογένεια (QLoRA vs baselines)\n",
          df_to_md(pw[["metric", "A", "B", "delta_pp", "boot_lo", "boot_hi", "mcnemar_p", "holm_p"]]),
          "\n## Σημειώσεις\n",
          "- Σχήμα: `analysis/figures/abstention_tradeoff.pdf/.png` (καμπύλες Gate/B1"
          + ("/B2" if crag is not None else "") + " + σημεία Base/QLoRA/Gate/B1).\n"
          "- iso-F-Abst: αν QLoRA_minus_method_pp>0 με CI χωρίς το 0 → η QLoRA υπερέχει στο ίδιο F-Abst.\n"
          "- Τα κατώφλια επιλέχθηκαν ΜΟΝΟ στο TRAIN· η καμπύλη δείχνει όλα τα T (χωρίς cherry-picking).\n"]
    open(f"{OUT_LOG}/exp3_report.md", "w", encoding="utf-8").write("\n".join(md))

    print("ΠΑΡΑΔΟΤΕΑ:")
    for p in sorted(glob.glob(f"{OUT_TAB}/exp3_*.csv")) + [f"{OUT_FIG}/abstention_tradeoff.png",
             f"{OUT_LOG}/exp3_report.json", f"{OUT_LOG}/exp3_report.md"]:
        print("  " + p)
    print("\nAUROC:"); print(auroc.to_string(index=False))
    print("\niso-F-Abst:"); print(iso.to_string(index=False))
    print("\nTable 4 supplement:"); print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
