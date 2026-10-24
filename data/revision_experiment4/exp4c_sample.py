#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 4Γ — στρωματοποιημένο δείγμα 50 (25 present + 25 missing) από Stage C run1
(auto-resolved ⟺ mode "A"), + 5 calibration (disjoint), + ΤΥΦΛΑ φύλλα αξιολόγησης.

Τα φύλλα των αξιολογητών ΔΕΝ περιέχουν: stratum, σκορ NLI, config, run, question_id.
Η σειρά ανακατεύεται ανά αξιολογητή. Το ΚΛΕΙΔΙ (sample50_key.csv) ΔΕΝ δίνεται.

ΕΞΟΔΟΙ:
  annotation/sample50_key.csv   (ΚΛΕΙΔΙ — κρατήστε το)
  annotation/annotator1.xlsx, annotator2.xlsx, calibration.xlsx
ΕΚΤΕΛΕΣΗ:  python exp4c_sample.py
"""
import os, sys, json, glob, tarfile
import numpy as np, pandas as pd
sys.path.insert(0, "analysis")
import nli_support as NS

STAGEC_DIR = os.getenv("STAGEC_DIR", "../revision_experiment1/logs/stageC")
SUBS       = os.getenv("SUBS_CSV", "../revision_experiment1/data/test_subgroups.csv")
TEST_DOCS  = os.getenv("TEST_DOCS", "../revision_experiment1/data/retrieval_test_docs.jsonl")
FROZEN     = os.getenv("FROZEN_PATH", "../revision_experiment1/data/retrieval_test.jsonl")
TAR        = os.getenv("TECHQA_TAR", "/data/TechQA.tar.gz")
SEED = int(os.getenv("SAMPLE_SEED", "2026"))
NPS  = int(os.getenv("N_PER_STRATUM", "25")); NCAL = int(os.getenv("N_CALIB", "5"))


def stagec_run1():
    files = sorted(glob.glob(os.path.join(STAGEC_DIR, "*run1*.jsonl"))) \
        or sorted(glob.glob(os.path.join(STAGEC_DIR, "*run_1*.jsonl"))) \
        or sorted(glob.glob(os.path.join(STAGEC_DIR, "*.jsonl")))[:1]
    if not files:
        print(f"[ERROR] δεν βρέθηκε Stage C run1 σε {STAGEC_DIR}"); sys.exit(1)
    print(f"[Stage C run1] {os.path.basename(files[0])}")
    return pd.DataFrame(json.loads(l) for l in open(files[0], encoding="utf-8") if l.strip())


def load_tar_json(member):
    with tarfile.open(TAR, "r:*") as t:
        m = next((x for x in t.getmembers() if x.name.endswith(member)), None)
        return json.load(t.extractfile(m)) if m else []


def main():
    os.makedirs("annotation", exist_ok=True)
    sub = pd.read_csv(SUBS).set_index("question_id").subgroup
    c = stagec_run1()
    # auto-resolved: το Stage C χρησιμοποιεί abstain/escalated (όχι mode)
    if "mode" in c.columns:
        c = c[c["mode"] == "A"].copy()
    else:
        ab = c["abstain"].fillna(False).astype(bool) if "abstain" in c.columns else pd.Series(False, index=c.index)
        es = c["escalated"].fillna(False).astype(bool) if "escalated" in c.columns else pd.Series(False, index=c.index)
        c = c[(~ab) & (~es)].copy()
    if "citation_valid" not in c.columns:
        c["citation_valid"] = [bool(ci) and all(x in set(ri or []) for x in (ci or []))
                               for ci, ri in zip(c.get("cited_ids", [[]] * len(c)), c.get("retrieved_ids", [[]] * len(c)))]
    c["stratum"] = np.where(c.question_id.map(sub).eq("present"), "present", "missing")

    fro = {json.loads(l)["question_id"]: json.loads(l)
           for l in open(FROZEN, encoding="utf-8") if l.strip()}
    c["answerable"] = c.question_id.map(lambda q: int(fro.get(q, {}).get("answerable", 1)))
    c["retrieved_ids"] = c.question_id.map(lambda q: fro.get(q, {}).get("ids", []))

    # δείγμα 25+25 (ή όσα υπάρχουν) + 5 calibration disjoint
    sample = pd.concat([g.sample(n=min(NPS, len(g)), random_state=SEED)
                        for _, g in c.groupby("stratum")])
    rest = c.drop(sample.index)
    calib = rest.sample(n=min(NCAL, len(rest)), random_state=7) if len(rest) else rest.iloc[:0]

    rng = np.random.default_rng(SEED)
    sample = sample.copy(); sample["item_id"] = [f"I{k:03d}" for k in rng.permutation(len(sample))]
    calib = calib.copy();   calib["item_id"] = [f"C{k:03d}" for k in range(len(calib))]

    key_cols = ["item_id", "question_id", "stratum", "config", "run",
                "answerable", "cited_ids", "citation_valid"]
    sample.reindex(columns=[k for k in key_cols if k in sample.columns]) \
          .to_csv("annotation/sample50_key.csv", index=False)
    # N_h = #auto-resolved ανά στρώμα (run1) — για το stratified bootstrap του Metric 2
    json.dump(c.stratum.value_counts().to_dict(),
              open("annotation/strata_N.json", "w", encoding="utf-8"), ensure_ascii=False)

    # υλικό φύλλων
    doc_index = NS.build_doc_index(TEST_DOCS)
    qa_raw = {(r.get("QUESTION_ID") or r.get("question_id")): r for r in load_tar_json("dev_Q_A.json")}

    def qa_map(ids):
        out = {}
        for qid in ids:
            r = qa_raw.get(qid, {})
            out[qid] = {"title": r.get("QUESTION_TITLE", ""), "text": r.get("QUESTION_TEXT", ""),
                        "answer": (r.get("ANSWER") or r.get("answer") or ""),
                        "gold_id": (r.get("DOCUMENT_ID") or "")}
        return out

    for a, seed in (("annotator1", 11), ("annotator2", 22)):
        sh = sample.sample(frac=1, random_state=seed)
        NS.build_sheet(sh, doc_index, qa_map(sh.question_id)).to_excel(f"annotation/{a}.xlsx", index=False)
    if len(calib):
        NS.build_sheet(calib, doc_index, qa_map(calib.question_id)).to_excel("annotation/calibration.xlsx", index=False)

    vc = sample.stratum.value_counts().to_dict()
    print(f"[OK] δείγμα {len(sample)} {vc} + {len(calib)} calibration")
    print("     annotation/sample50_key.csv (ΚΛΕΙΔΙ), annotator1.xlsx, annotator2.xlsx, calibration.xlsx")
    print(f"[έλεγχος] επικάλυψη δείγματος/calibration: "
          f"{len(set(sample.question_id) & set(calib.question_id))} (πρέπει 0)")
    print("[υπενθύμιση] τα φύλλα δεν περιέχουν stratum/NLI/config/question_id (Γενικός κανόνας τυφλότητας).")


if __name__ == "__main__":
    main()
