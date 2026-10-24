#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 4Β — NLI entailment ανά βήμα -> analysis/steps_nli.jsonl.

ΒΑΡΥ ΒΗΜΑ: DeBERTa-v3-large NLI (fp16, GPU — T4 αρκεί). Το κείμενο των cited/retrieved
εγγράφων έρχεται από το retrieval_test_docs.jsonl (τα top-5 κάθε TEST ερώτησης έχουν content).
Caches ανά (docs, βήμα): τα runs 1–3 έχουν πολλές πανομοιότυπες απαντήσεις.

ΕΞΟΔΟΙ: analysis/steps_nli.jsonl  (τα βήματα + p_cited, p_any)
ΕΚΤΕΛΕΣΗ:  python exp4b_nli.py
"""
import os, sys, json
import pandas as pd
sys.path.insert(0, "analysis")
import nli_support as NS

TEST_DOCS = os.getenv("TEST_DOCS", "../revision_experiment1/data/retrieval_test_docs.jsonl")
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
CHUNK = int(os.getenv("CHUNK", "400")); STRIDE = int(os.getenv("STRIDE", "200"))
STEPS = "analysis/steps.jsonl"; OUT = "analysis/steps_nli.jsonl"


def main():
    if not os.path.exists(STEPS):
        print(f"[ERROR] λείπει {STEPS} — τρέξτε πρώτα exp4a_prepare_steps.py"); sys.exit(1)
    st = pd.read_json(STEPS, lines=True)
    doc_index = NS.build_doc_index(TEST_DOCS)
    print(f"[docs] {len(doc_index)} μοναδικά technotes (TEST top-5)")
    scorer = NS.NLIScorer(NLI_MODEL, doc_index, chunk=CHUNK, stride=STRIDE)
    print(f"[NLI] {NLI_MODEL}  dev={scorer.dev}  ENT_idx={scorer.ENT}")

    # resume: επαναχρησιμοποίησε ήδη βαθμολογημένα βήματα (ίδιο text) — γλιτώνει το rescore του Stage B
    prev = {}
    if os.path.exists(OUT):
        for r in (json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()):
            prev[(r.get("stage"), r.get("config"), r.get("run"), r.get("question_id"), r.get("step_idx"))] = \
                (r.get("p_cited"), r.get("p_any"), r.get("text"))
        if prev:
            print(f"[resume] {len(prev)} ήδη βαθμολογημένα βήματα στο {OUT} — επαναχρήση των αμετάβλητων")

    # έλεγχος truncation: βήματα με hypothesis > (512-CHUNK) tokens θα ανάγκαζαν κόψιμο premise
    hyp_budget = 512 - CHUNK - 3
    too_long = sum(len(scorer.tok(t, add_special_tokens=False)["input_ids"]) > hyp_budget
                   for t in st["text"].unique())
    print(f"[έλεγχος] μοναδικά βήματα με hypothesis > {hyp_budget} tokens: {too_long} "
          f"(θέλουμε 0 — αλλιώς premise κόβεται)")

    pc, pa, reused = [], [], 0
    for i, r in enumerate(st.itertuples()):
        key = (getattr(r, "stage", None), r.config, r.run, r.question_id, r.step_idx)
        cached = prev.get(key)
        if cached and cached[0] is not None and cached[2] == r.text:
            p_cited, p_any = cached[0], cached[1]; reused += 1
        else:
            p_cited, p_any = scorer.score_step(r.text, getattr(r, "cited_ids", None),
                                               getattr(r, "retrieved_ids", None),
                                               bool(getattr(r, "citation_valid", False)))
        pc.append(p_cited); pa.append(p_any)
        if (i + 1) % 500 == 0:
            print(f"  NLI {i+1}/{len(st)}  (resume {reused}, cache {len(scorer._entail_cache)})")
    st["p_cited"] = pc; st["p_any"] = pa
    if reused:
        print(f"[resume] επαναχρησιμοποιήθηκαν {reused}/{len(st)} βήματα· βαθμολογήθηκαν νέα {len(st)-reused}")
    st.to_json(OUT, orient="records", lines=True, force_ascii=False)
    print(f"[OK] {OUT}: {len(st)} βήματα  (μοναδικά entailment: {len(scorer._entail_cache)})")
    print(f"[σύνοψη] p_cited≥0.5: {100*(st.p_cited>=0.5).mean():.1f}%  |  "
          f"p_any≥0.5: {100*(st.p_any>=0.5).mean():.1f}%")


if __name__ == "__main__":
    main()
