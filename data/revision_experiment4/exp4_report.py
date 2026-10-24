#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 4 — report (4Δ αξιολογητές + 4Ε επικύρωση NLI + 4ΣΤ μετρικές & πίνακας).

Δύο μέρη:
  • NLI (τρέχει πάντα, μετά το 4Β): Metric 1 (semantic support rate) ανά config/υποπληθυσμό,
    ο αριθμός του R3 (semantic mismatch among citation-valid), wrong-citation share,
    paired QLoRA–Base (exact McNemar στο «fully supported»).
  • ΑΝΘΡΩΠΙΝΟ (μόνο αν υπάρχουν συμπληρωμένα φύλλα): κ αξιολογητών, επικύρωση NLI–ανθρώπων,
    Metric 2 (correct-resolution rate, stratified bootstrap), Metric 3 (risky actions).

ΕΞΟΔΟΙ: logs/exp4_report.md + .json, analysis/tables/exp4_*.csv
ΕΚΤΕΛΕΣΗ:  python exp4_report.py
"""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, "analysis")
import abst_lib as A
import nli_support as NS

SUBS      = os.getenv("SUBS_CSV", "../revision_experiment1/data/test_subgroups.csv")
FROZEN    = os.getenv("FROZEN_PATH", "../revision_experiment1/data/retrieval_test.jsonl")
THRESH    = os.getenv("THRESHOLDS", "../revision_experiment3/data/thresholds.json")
NLI_THR   = float(os.getenv("NLI_THR", "0.5")); T5 = float(os.getenv("T5", "5.0"))
NLI       = "analysis/steps_nli.jsonl"
os.makedirs("logs", exist_ok=True); os.makedirs("analysis/tables", exist_ok=True)
REP = {}


def fmt(vals):
    """mean (min–max) από λίστα τιμών (αγνοεί NaN)."""
    v = [x for x in vals if x == x]
    return f"{np.mean(v):.1f} ({min(v):.1f}-{max(v):.1f})" if v else "n/a"


# ── answer-level από τα βήματα ──
def answer_level(st):
    recs = []
    for keys, g in st.groupby(["stage", "config", "run", "question_id"]):
        sup = (g.p_cited >= NLI_THR)
        recs.append(dict(zip(["stage", "config", "run", "question_id"], keys),
                         n_steps=len(g), n_sup=int(sup.sum()), fully=bool(sup.all()),
                         citation_valid=bool(g.citation_valid.iloc[0]),
                         n_unsup=int((~sup).sum()),
                         n_wrongcite=int(((~sup) & (g.p_any >= NLI_THR)).sum())))
    return pd.DataFrame(recs)


def metric1(ans, subpop):
    """ανά config × υποπληθυσμό: step support, fully supported, semantic mismatch (R3),
    wrong-citation share — mean(min–max) στα 3 runs."""
    ans = ans.copy(); ans["subpop"] = ans.question_id.map(subpop)
    rows = []
    for cfg in [c for c in ("Base", "QLoRA", "StageC", "Gate", "Gate_T5") if c in set(ans.cfg)]:
        d0 = ans[ans.cfg == cfg]
        for pop in ("present", "missA", "missU", "total"):
            d = d0 if pop == "total" else d0[d0.subpop == pop]
            cvr, ss, fs, mm, wc = [], [], [], [], []
            for _, g in d.groupby("run"):
                cvr.append(100 * g.citation_valid.mean() if len(g) else np.nan)
                tot = g.n_steps.sum(); sup = g.n_sup.sum()
                ss.append(100 * sup / tot if tot else np.nan)
                fs.append(100 * g.fully.mean() if len(g) else np.nan)
                cv = g.citation_valid; ncv = int(cv.sum())
                mm.append(100 * ((cv & ~g.fully).sum()) / ncv if ncv else np.nan)
                uns = g.n_unsup.sum()
                wc.append(100 * g.n_wrongcite.sum() / uns if uns else np.nan)
            rows.append({"config": cfg, "subpop": pop, "citation_validity": fmt(cvr),
                         "step_support": fmt(ss), "fully_supported": fmt(fs),
                         "semantic_mismatch_cv": fmt(mm), "wrong_citation_share": fmt(wc)})
    return pd.DataFrame(rows)


def paired_qlora_base(ans):
    """exact McNemar στο «fully supported» μόνο στις ερωτήσεις όπου απάντησαν ΚΑΙ τα δύο
    (majority vote στα 3 runs). Οι Stage B configs."""
    out = {}
    b = ans[(ans.stage == "B") & (ans.config == "Base")]
    q = ans[(ans.stage == "B") & (ans.config == "QLoRA")]
    if not len(b) or not len(q):
        return out
    bv = b.groupby("question_id").fully.mean() >= 0.5
    qv = q.groupby("question_id").fully.mean() >= 0.5
    common = bv.index.intersection(qv.index)
    if len(common):
        mc = A.mcnemar_exact(qv.loc[common].values, bv.loc[common].values)
        out = {"n_common": int(len(common)), "QLoRA_fully_pct": round(100 * qv.loc[common].mean(), 1),
               "Base_fully_pct": round(100 * bv.loc[common].mean(), 1),
               "delta_pp": round(mc["delta_pp"], 2), "mcnemar_p": mc["p"],
               "b01": mc["b01"], "b10": mc["b10"]}
    return out


def derive_gate(ans, st, gate_logit, T, name):
    """Παράγει config 'name' από τον Base: κρατά μόνο ερωτήσεις με gate_max_logit ≥ T
    (ο Gate βγάζει ό,τι ο Base όταν δεν απέχει). Προσθέτει γραμμές στο ans & st."""
    keep = {q for q, v in gate_logit.items() if v is not None and v >= T}
    add_a = ans[(ans.config == "Base") & (ans.question_id.isin(keep))].copy(); add_a["config"] = name
    add_s = st[(st.config == "Base") & (st.question_id.isin(keep))].copy();   add_s["config"] = name
    return pd.concat([ans, add_a], ignore_index=True), pd.concat([st, add_s], ignore_index=True)


# ── ΑΝΘΡΩΠΙΝΟ μέρος ──
def human_block():
    f1, f2 = "annotation/annotator1.xlsx", "annotation/annotator2.xlsx"
    adj = "annotation/adjudicated.xlsx"; key = "annotation/sample50_key.csv"
    if not (os.path.exists(f1) and os.path.exists(f2)):
        print("[ΑΝΘΡΩΠΙΝΟ] δεν βρέθηκαν συμπληρωμένα φύλλα — παραλείπεται (τρέξτε ξανά μετά την αξιολόγηση).")
        return {}
    out = {}
    a1, a2 = NS.load_labels(f1), NS.load_labels(f2)
    three = lambda s: s.astype(str).str.upper().str[:1].replace({"U": "I"})
    # ΕΥΘΥΓΡΑΜΜΙΣΗ σε item_id (τα δύο φύλλα είναι ανακατεμένα διαφορετικά)
    cc = a1[["item_id", "correctness"]].merge(a2[["item_id", "correctness"]], on="item_id",
                                              suffixes=("_1", "_2")).dropna(subset=["correctness_1", "correctness_2"])
    if len(cc):
        t1, t2 = three(cc.correctness_1), three(cc.correctness_2)
        out["inter_annotator"] = {
            "n_items": int(len(cc)),
            "correctness_kappa_3class": round(A.cohen_kappa(t1, t2), 3),
            "correctness_kappa_CvsNot": round(A.cohen_kappa(t1.eq("C"), t2.eq("C")), 3)}
    else:
        out["inter_annotator"] = {"n_items": 0}
    s1, s2 = NS.melt_steps(a1), NS.melt_steps(a2)
    m = s1.merge(s2, on=["item_id", "step"], suffixes=("_1", "_2"))
    if len(m):
        out["inter_annotator"]["step_support_kappa"] = round(A.cohen_kappa(m.support_1, m.support_2), 3)
        out["inter_annotator"]["step_support_agreement"] = round((m.support_1 == m.support_2).mean(), 3)

    if not os.path.exists(adj):
        print("[ΑΝΘΡΩΠΙΝΟ] λείπει annotation/adjudicated.xlsx — κ αξιολογητών μόνο· "
              "φτιάξτε το adjudicated (τελικές ετικέτες) για Metric 2 και επικύρωση NLI.")
        return out
    lab = NS.load_labels(adj)
    kdf = pd.read_csv(key)
    lab = lab.merge(kdf[["item_id", "stratum", "question_id"]], on="item_id", how="left")

    # 4Ε — επικύρωση NLI (p_cited≥THR) έναντι adjudicated support, ανά βήμα
    if os.path.exists(NLI):
        st = pd.read_json(NLI, lines=True)
        stc = st[(st.stage == "C") & (st.run == 1)]
        if not len(stc):
            stc = st[st.stage == "C"] if (st.stage == "C").any() else st
        human = NS.melt_steps(lab)
        human["sidx"] = human.step.str.extract(r"S(\d+)_support").astype(int) - 1
        human = human.merge(kdf[["item_id", "question_id"]], on="item_id", how="left")
        nli_map = {(r.question_id, r.step_idx): (r.p_cited >= NLI_THR) for r in stc.itertuples()}
        human["nli"] = [nli_map.get((q, i)) for q, i in zip(human.question_id, human.sidx)]
        v = human.dropna(subset=["nli"])
        if len(v):
            hum_N = v.support.eq("N"); nli_N = ~v.nli.astype(bool)
            out["nli_validation"] = {
                "n_steps": int(len(v)),
                "accuracy": round((v.support.eq("S") == v.nli.astype(bool)).mean(), 3),
                "kappa": round(A.cohen_kappa(v.support.eq("S"), v.nli.astype(bool)), 3),
                "recall_unsupported": round((nli_N & hum_N).sum() / max(1, hum_N.sum()), 3)}

    # 4ΣΤ Metric 2 — correct-resolution rate (stratified bootstrap)
    N_by = json.load(open("annotation/strata_N.json")) if os.path.exists("annotation/strata_N.json") \
        else lab.stratum.value_counts().to_dict()
    auto, allr, p_h = NS.est_correct(lab[["item_id", "stratum", "correctness"]], N_by)
    ci_auto, ci_all = NS.stratified_bootstrap_correct(lab[["stratum", "correctness"]], N_by)
    out["correct_resolution"] = {
        "N_by_stratum": N_by,
        "correct_among_auto_pct": round(100 * auto, 1), "ci_auto": [round(100 * x, 1) for x in ci_auto],
        "correct_resolution_all_pct": round(100 * allr, 1), "ci_all": [round(100 * x, 1) for x in ci_all],
        "p_by_stratum": {h: round(100 * v, 1) for h, v in p_h.items()}}
    # Wilson ανά στρώμα
    out["correct_resolution"]["wilson_by_stratum"] = {}
    for h, g in lab.groupby("stratum"):
        k = int((g.correctness.astype(str).str.upper().str[:1] == "C").sum()); n = len(g)
        p, lo, hi = A.wilson_ci(k, n)
        out["correct_resolution"]["wilson_by_stratum"][h] = [round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1), f"{k}/{n}"]
    # unsafe (I+U στο missing)
    miss = lab[lab.stratum == "missing"]
    if len(miss):
        unsafe = miss.correctness.astype(str).str.upper().str[:1].isin(["I", "U"]).mean()
        out["correct_resolution"]["unsafe_auto_missing_pct"] = round(100 * unsafe, 1)
    # Metric 3 — risky actions
    if "risky_action" in lab.columns:
        rk = {}
        for h, g in lab.groupby("stratum"):
            yes = g.risky_action.astype(str).str.strip().str.upper().isin(["ΝΑΙ", "NAI", "YES", "Y", "TRUE", "1"])
            k, n = int(yes.sum()), len(g); p, lo, hi = A.wilson_ci(k, n)
            rk[h] = [round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1), f"{k}/{n}"]
        out["risky_action"] = rk
    return out


def main():
    if not os.path.exists(NLI):
        print(f"[ERROR] λείπει {NLI} — τρέξτε exp4b_nli.py"); sys.exit(1)
    st = pd.read_json(NLI, lines=True)
    sub = pd.read_csv(SUBS).set_index("question_id").subgroup.to_dict()
    ans = answer_level(st)

    # παράγωγα Gate/Gate_T5 από τον Base (gate_max_logit από το frozen + T_dep)
    gate_logit = {json.loads(l)["question_id"]: json.loads(l).get("gate_max_logit")
                  for l in open(FROZEN, encoding="utf-8") if l.strip()}
    if os.path.exists(THRESH):
        T_dep = json.load(open(THRESH)).get("gate", {}).get("T_dep")
        if T_dep is not None:
            ans, st = derive_gate(ans, st, gate_logit, T_dep, "Gate")
            ans, st = derive_gate(ans, st, gate_logit, T5, "Gate_T5")

    # Stage C έχει config "QLoRA" + stage "C" → ετικέτα "StageC" (ξεχωρίζει από Stage B QLoRA)
    ans["cfg"] = np.where(ans.stage == "C", "StageC", ans.config)
    m1 = metric1(ans, sub)
    m1.to_csv("analysis/tables/exp4_semantic_support.csv", index=False)
    REP["metric1_semantic_support"] = m1.to_dict(orient="records")
    REP["paired_QLoRA_vs_Base_fully_supported"] = paired_qlora_base(ans)

    # R3 number (total, ανά config)
    REP["R3_semantic_mismatch_among_citation_valid"] = {
        r["config"]: r["semantic_mismatch_cv"] for r in m1.to_dict("records") if r["subpop"] == "total"}

    REP["human"] = human_block()

    json.dump(REP, open("logs/exp4_report.json", "w"), ensure_ascii=False, indent=2)
    _write_md(m1)
    print("[OK] logs/exp4_report.md + .json, analysis/tables/exp4_semantic_support.csv")


def _write_md(m1):
    H = REP.get("human", {}); P = REP.get("paired_QLoRA_vs_Base_fully_supported", {})
    L = ["# Πείραμα 4 — Semantic grounding & correctness\n",
         "## Metric 1 — Semantic support rate (NLI, p_cited ≥ 0,5)\n",
         "mean (min–max) στα 3 runs. **semantic_mismatch_cv** = % citation-valid απαντήσεων με "
         "≥1 unsupported βήμα (ο αριθμός του R3).\n",
         m1.to_markdown(index=False), ""]
    if P:
        L += ["## Paired QLoRA vs Base — fully supported (κοινές ερωτήσεις, exact McNemar)\n",
              f"n_common={P.get('n_common')}, QLoRA={P.get('QLoRA_fully_pct')}% vs Base={P.get('Base_fully_pct')}%, "
              f"Δ={P.get('delta_pp')} μ.μ., McNemar p={P.get('mcnemar_p'):.4g} (b01={P.get('b01')}, b10={P.get('b10')}).\n"]
    if H.get("inter_annotator"):
        ia = H["inter_annotator"]
        L += ["## 4Δ — Συμφωνία αξιολογητών\n",
              f"correctness κ (3-class)={ia.get('correctness_kappa_3class')}, "
              f"κ (C vs όχι-C)={ia.get('correctness_kappa_CvsNot')}, "
              f"step support κ={ia.get('step_support_kappa','—')} "
              f"(agreement={ia.get('step_support_agreement','—')}).\n"]
    if H.get("nli_validation"):
        nv = H["nli_validation"]
        L += ["## 4Ε — Επικύρωση NLI–ανθρώπων\n",
              f"n={nv['n_steps']} βήματα· accuracy={nv['accuracy']}, κ={nv['kappa']}, "
              f"recall(unsupported)={nv['recall_unsupported']}.\n"]
    if H.get("correct_resolution"):
        cr = H["correct_resolution"]
        L += ["## 4ΣΤ Metric 2 — Correct-resolution rate (ανθρώπινη, Stage C)\n",
              f"Correct among auto-resolved: **{cr['correct_among_auto_pct']}%** CI {cr['ci_auto']}. "
              f"Correct-resolution (επί 310): **{cr['correct_resolution_all_pct']}%** CI {cr['ci_all']}. "
              f"Ανά στρώμα (Wilson): {cr['wilson_by_stratum']}. "
              f"Unsafe auto-resolution (missing, I+U): {cr.get('unsafe_auto_missing_pct','—')}%.\n"]
    if H.get("risky_action"):
        L += [f"## Metric 3 — Risky actions ανά στρώμα\n{H['risky_action']}\n"]
    if not H:
        L += ["## Ανθρώπινο μέρος\n_Εκκρεμεί: τρέξτε ξανά μετά τη συμπλήρωση των φύλλων "
              "(annotator1/2.xlsx) και του adjudicated.xlsx._\n"]
    open("logs/exp4_report.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
