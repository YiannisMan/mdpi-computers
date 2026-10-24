#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 4Β (έλεγχος ορθότητας NLI) — θετικός & αρνητικός έλεγχος, ~100 ζεύγη το καθένα.

  Θετικός : hypothesis = πρόταση από το ΙΔΙΟ έγγραφο   -> αναμ. supported ≥ 90%
  Αρνητικός: ίδια πρόταση vs ΤΥΧΑΙΟ άλλο έγγραφο        -> αναμ. supported ≤ 10%

ΜΗΧΑΝΙΚΟΣ έλεγχος: επιβεβαιώνει chunking / label index / truncation. ΔΕΝ ρυθμίζει κατώφλι,
άρα η πηγή εγγράφων δεν επηρεάζει το συμπέρασμα — χρησιμοποιεί τα ήδη διαθέσιμα top-5 έγγραφα
(retrieval_test_docs.jsonl), ΧΩΡΙΣ corpus/ijson/streaming (γρήγορο & σταθερό).

ΕΞΟΔΟΣ: analysis/tables/exp4_nli_sanity.json
ΕΚΤΕΛΕΣΗ:  python exp4b_nli_sanity.py
"""
import os, sys, re, json, random
sys.path.insert(0, "analysis")
import nli_support as NS

TEST_DOCS = os.getenv("TEST_DOCS", "../revision_experiment1/data/retrieval_test_docs.jsonl")
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
THR = float(os.getenv("NLI_THR", "0.5")); N = int(os.getenv("SANITY_N", "100")); SEED = 2026
SENT = re.compile(r"(?<=[.!?])\s+")


def main():
    rng = random.Random(SEED)
    doc_index = NS.build_doc_index(TEST_DOCS)
    docs = [(i, NS._clean_html(t)) for i, t in doc_index.items()]
    docs = [(i, t) for i, t in docs if len(t.split()) > 40]
    rng.shuffle(docs)
    print(f"[docs] {len(docs)} κατάλληλα έγγραφα (retrieval_test_docs.jsonl)", flush=True)
    if len(docs) < 10:
        print("[ERROR] πολύ λίγα έγγραφα — ελέγξτε TEST_DOCS."); sys.exit(1)

    scorer = NS.NLIScorer(NLI_MODEL, dict(doc_index))
    print(f"[NLI] dev={scorer.dev}  ENT_idx={scorer.ENT}", flush=True)

    pos, neg = [], []
    m = min(N, len(docs))
    for k in range(m):
        did, txt = docs[k]
        sents = [s for s in SENT.split(txt) if 6 <= len(s.split()) <= 40]
        if not sents:
            continue
        hyp = rng.choice(sents)
        pos.append(scorer.p_entail(scorer.chunks(did), hyp) >= THR)        # ίδιο έγγραφο
        other = docs[(k + 7) % len(docs)][0]                              # άλλο έγγραφο
        neg.append(scorer.p_entail(scorer.chunks(other), hyp) >= THR)
        if (k + 1) % 25 == 0:
            print(f"  sanity {k+1}/{m}", flush=True)

    pr = 100 * sum(pos) / max(1, len(pos)); nr = 100 * sum(neg) / max(1, len(neg))
    res = {"positive_supported_pct": round(pr, 1), "negative_supported_pct": round(nr, 1),
           "n_pos": len(pos), "n_neg": len(neg), "threshold": THR,
           "doc_source": "retrieval_test_docs (top-5)", "pass": bool(pr >= 90 and nr <= 10)}
    os.makedirs("analysis/tables", exist_ok=True)
    json.dump(res, open("analysis/tables/exp4_nli_sanity.json", "w"), ensure_ascii=False, indent=2)
    print(res, flush=True)
    print("[OK] NLI sanity PASS" if res["pass"] else
          "[ΠΡΟΣΟΧΗ] εκτός ορίων — ελέγξτε chunking / label index / truncation")


if __name__ == "__main__":
    main()
