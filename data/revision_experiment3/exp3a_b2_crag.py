#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 3 — B2: CRAG-style evaluator (monoT5), scoring του top-5 σε TRAIN & TEST.

ΒΑΡΥ ΒΗΜΑ — εξαρτήσεις: transformers + torch + GPU, κατέβασμα του μοντέλου
`castorini/monot5-base-msmarco` (~850MB), και πρόσβαση στο corpus για τα κείμενα των
TRAIN top-5 εγγράφων. Είναι ΔΙΑΦΟΡΕΤΙΚΟ μοντέλο από τον reranker μας (όχι δεύτερο gate).

score(doc) = log P("true") του monoT5 για "Query: … Document: … Relevant:" (υψηλό = σχετικό).
crag_max(ερώτηση) = max των 5 doc scores.

ΕΙΣΟΔΟΙ:
  TEST  top-5 κείμενα: ../revision_experiment1/data/retrieval_test_docs.jsonl (έχει content) ✓
  TRAIN top-5 κείμενα: από το CORPUS, ανά id των OOF top-5. ΡΥΘΜΙΣΤΕ CORPUS_PATH + load_corpus().
  TRAIN queries: training_Q_A.json (από το TAR).  TEST queries: το 'ticket' των docs.

ΕΞΟΔΟΙ:
  data/crag_scores_train.json   {qid: crag_max}
  data/crag_scores_test.json    {qid: crag_max}
  data/crag_thresholds.json      {theta_lo, theta_hi, auroc, auroc_ci}

ΕΚΤΕΛΕΣΗ (in-container με GPU):
  python exp3a_b2_crag.py            # TRAIN + TEST scoring + κατώφλια
Αν το corpus δεν φορτώνει, τρέξτε με --test-only (μόνο TEST) και δώστε αργότερα TRAIN.
"""
import json, os, sys, bz2, re, argparse, tarfile
import numpy as np
import pandas as pd
sys.path.insert(0, "analysis")
import abst_lib as A
import exp3_lib as E3

OOF        = os.getenv("OOF_PATH", "../revision_experiment1/data/retrieval_train_oof.jsonl")
TEST_DOCS  = os.getenv("TEST_DOCS", "../revision_experiment1/data/retrieval_test_docs.jsonl")
TAR_PATH   = os.getenv("TECHQA_TAR", "/data/TechQA.tar.gz")
CORPUS_PATH = os.getenv("CORPUS_PATH", "/data/full_technote_collection.txt.bz2")
MODEL_NAME = os.getenv("CRAG_MODEL", "castorini/monot5-base-msmarco")
MAX_F      = float(os.getenv("MAX_F", "25.0"))
DOC_WORDS  = int(os.getenv("CRAG_DOC_WORDS", "400"))
OUT        = "data"

F_QID, F_TITLE, F_TEXT = "QUESTION_ID", "QUESTION_TITLE", "QUESTION_TEXT"


# ── Corpus loader — streaming από TAR ή δίσκο. Format TechQA: JSON array [{"id","content"},…] ──
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
_STREAM_REFS = []  # κρατάει ανοιχτά τα file objects του TAR ώστε να μην κλείσει ο GC το stream


def _clean_html(s):
    return WS_RE.sub(" ", TAG_RE.sub(" ", s or "")).strip()


def _open_corpus_stream():
    """Binary stream του (αποσυμπιεσμένου) corpus: από δίσκο αν υπάρχει, αλλιώς member του TECHQA_TAR."""
    p = CORPUS_PATH
    if p and os.path.exists(p):
        raw = open(p, "rb")
    else:
        tar = tarfile.open(TAR_PATH, "r:*")
        _STREAM_REFS.append(tar)
        m = next((x for x in tar.getmembers() if x.name == p or x.name.endswith(p)), None)
        if not m:
            return None
        raw = tar.extractfile(m)
    _STREAM_REFS.append(raw)
    if (p or "").endswith(".bz2"):
        bz = bz2.BZ2File(raw)
        _STREAM_REFS.append(bz)
        return bz
    return raw


def load_corpus_filtered(needed):
    """Streaming parse του JSON array· κρατά ΜΟΝΟ τα ζητούμενα ids -> dict id -> καθαρό κείμενο.
    Δεν φορτώνει όλο το collection στη μνήμη (σαρώνει ~800k, κρατά μόνο τα needed)."""
    need = set(map(str, needed))
    if not need:
        return {}
    stream = _open_corpus_stream()
    if stream is None:
        print(f"[WARN] corpus member δεν βρέθηκε: {CORPUS_PATH} (TAR={TAR_PATH})", file=sys.stderr)
        return {}
    try:
        import ijson
    except ImportError:
        print("[ERROR] χρειάζεται ijson — τρέξτε: pip install ijson", file=sys.stderr)
        return {}
    out, seen = {}, 0
    # multiple_values=True: το corpus είναι ΠΟΛΛΑ συνενωμένα JSON arrays ([{...}][{...}]), όχι ένα
    for obj in ijson.items(stream, "item", multiple_values=True):
        seen += 1
        if seen % 100000 == 0:
            print(f"  [corpus] σάρωση {seen} technotes… ({len(out)}/{len(need)})")
        idv = obj.get("id")
        if idv is None:
            continue
        idv = str(idv)
        if idv in need:
            out[idv] = _clean_html(obj.get("content") or obj.get("text") or "")
            if len(out) == len(need):
                break
    print(f"[corpus] {len(out)}/{len(need)} technotes βρέθηκαν (σαρώθηκαν {seen}).")
    if not out:
        print("[WARN] 0 technotes — έλεγχος CORPUS_PATH/format.", file=sys.stderr)
    return out


def load_tar_qa(member):
    with tarfile.open(TAR_PATH, "r:*") as tar:
        m = next((x for x in tar.getmembers() if x.name.endswith(member)), None)
        if not m:
            return {}
        return {r[F_QID]: r for r in json.load(tar.extractfile(m))}


def load_monot5():
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(MODEL_NAME)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    m = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dt).to(dev).eval()
    TRUE, FALSE = tok.encode("true")[0], tok.encode("false")[0]

    @torch.no_grad()
    def relevance(query, doc):
        x = tok(f"Query: {query} Document: {doc} Relevant:", return_tensors="pt",
                truncation=True, max_length=512).to(dev)
        out = m.generate(**x, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
        lp = out.scores[0][0, [FALSE, TRUE]].float().log_softmax(-1)
        return float(lp[1].item())
    return relevance


def crag_max_for(relevance, query, docs):
    if not docs:
        return float("-inf")
    return max(relevance(query, (d or "")[:DOC_WORDS * 6]) for d in docs)


def score_test(relevance):
    docs = {json.loads(l)["question_id"]: json.loads(l)
            for l in open(TEST_DOCS, encoding="utf-8") if l.strip()}
    out = {}
    for i, (qid, d) in enumerate(docs.items()):
        q = d.get("ticket", "")
        texts = [s.get("content") or s.get("title") or "" for s in d.get("solutions", [])]
        out[qid] = crag_max_for(relevance, q, texts)
        if (i + 1) % 25 == 0:
            print(f"  TEST crag {i+1}/{len(docs)}")
    json.dump(out, open(f"{OUT}/crag_scores_test.json", "w"), ensure_ascii=False)
    print(f"[OK] data/crag_scores_test.json ({len(out)})")
    return out


def score_train(relevance):
    tr = pd.read_json(OOF, lines=True)
    needed = {str(did) for ids in tr["ids"] for did in (ids or [])}
    corpus = load_corpus_filtered(needed)
    if not corpus:
        print("[SKIP] TRAIN scoring — ρυθμίστε CORPUS_PATH/load_corpus().")
        return {}
    qa = load_tar_qa("training_Q_A.json")
    if not qa:
        print("[WARN] δεν βρέθηκαν TRAIN queries (training_Q_A.json) — συνεχίζω με κενό query.", file=sys.stderr)
    out = {}
    for i, row in tr.iterrows():
        qid = row["question_id"]
        rq = qa.get(qid, {})
        query = str(rq.get(F_TEXT) or rq.get(F_TITLE) or "")
        texts = [corpus.get(str(did), "") for did in (row.get("ids") or [])]
        out[qid] = crag_max_for(relevance, query, [t for t in texts if t])
        if (i + 1) % 50 == 0:
            print(f"  TRAIN crag {i+1}/{len(tr)}")
    json.dump(out, open(f"{OUT}/crag_scores_train.json", "w"), ensure_ascii=False)
    print(f"[OK] data/crag_scores_train.json ({len(out)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-only", action="store_true", help="μόνο TEST scoring (χωρίς corpus)")
    ap.add_argument("--skip-test", action="store_true", help="παράλειψη TEST (υπάρχει ήδη crag_scores_test.json)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rel = load_monot5()
    if args.skip_test and os.path.exists(f"{OUT}/crag_scores_test.json"):
        print("[skip-test] χρήση υπάρχοντος data/crag_scores_test.json")
    else:
        score_test(rel)
    if args.test_only:
        print("  (--test-only: παραλείφθηκε TRAIN· τρέξτε ξανά με corpus για τα κατώφλια)")
        return
    train_scores = score_train(rel)

    # ── κατώφλια στο TRAIN ──
    if train_scores:
        tr = pd.read_json(OOF, lines=True)
        present = tr["present"].astype(bool).values
        cm = np.array([train_scores.get(q, float("-inf")) for q in tr["question_id"]])
        valid = np.isfinite(cm)
        theta_lo, c2, f2 = A.select_threshold(cm[valid], present[valid], max_f=MAX_F)
        theta_hi = E3.select_theta_hi(cm[valid], present[valid], prec=0.90)
        auroc = E3.auroc_missing(cm[valid], present[valid])
        auroc_ci = E3.auroc_ci(cm[valid], present[valid])
        json.dump({"theta_lo": float(theta_lo), "theta_hi": float(theta_hi),
                   "C_abst_train": round(c2, 2), "F_abst_train": round(f2, 2),
                   "auroc": round(auroc, 4), "auroc_ci": [round(x, 4) for x in auroc_ci]},
                  open(f"{OUT}/crag_thresholds.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[OK] data/crag_thresholds.json: theta_lo={theta_lo:.3f} theta_hi={theta_hi:.3f} "
              f"AUROC={auroc:.3f}")


if __name__ == "__main__":
    main()
