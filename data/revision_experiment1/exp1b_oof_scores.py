#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Β (μέρος 2) — 5-fold out-of-fold scores στο TRAIN.

ΓΙΑΤΙ: τα scores του TRAIN από reranker που εκπαιδεύτηκε σε αυτό είναι μεροληπτικά
(υπερβολικά υψηλά). Για να επιλεγεί το T του gate και τα baselines (Πείραμα 3) ΧΩΡΙΣ
να αγγίξουμε το TEST, χρειαζόμαστε out-of-fold scores: το fold k βαθμολογείται από
μοντέλο που ΔΕΝ το είδε.

ΡΟΗ (λόγω του service-based retrieval, το cross-fit γίνεται ανά fold):
  1) python exp1b_oof_scores.py --make-folds           # φτιάχνει data/folds.json (seed 42)
  2) για k=0..4:
       - εκπαιδεύστε μοντέλο στα answerable των ΑΛΛΩΝ 4 folds:
             python exp1b_oof_scores.py --train-fold k
         (παράγει models/reranker_fold{k}/)
       - δείξτε το service σε αυτό (FINETUNED_RERANKER_PATH=models/reranker_fold{k}) & restart
       - python exp1b_oof_scores.py --score-fold k     # βαθμολογεί ΜΟΝΟ το fold k
  3) python exp1b_oof_scores.py --ks                   # έλεγχος μετατόπισης κλίμακας (KS)

Παράγει: data/retrieval_train_oof.jsonl  (ένα record ανά TRAIN ερώτηση, με features για Πείραμα 3)
"""
import tarfile, json, os, sys, random, math, argparse, urllib.request, re

TAR_PATH   = os.getenv("TECHQA_TAR", "/data/TechQA.tar")
SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8080/search")
OOF_PATH   = "data/retrieval_train_oof.jsonl"
FOLDS_PATH = "data/folds.json"
SEED, K_FOLDS = 42, 5

F_QID, F_TITLE, F_TEXT, F_DOC, F_ANS = \
    "QUESTION_ID", "QUESTION_TITLE", "QUESTION_TEXT", "DOCUMENT", "ANSWERABLE"


def load_qa(member):
    with tarfile.open(TAR_PATH, "r:*") as tar:
        m = next((x for x in tar.getmembers() if x.name.endswith(member)), None)
        if not m: raise FileNotFoundError(member)
        return json.load(tar.extractfile(m))


def search(q, n):
    payload = json.dumps({"issue_description": q, "n_results": n}).encode()
    req = urllib.request.Request(SEARCH_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


# ── n_tech_tokens: ΠΙΣΤΟ αντίγραφο του extract_technical_tokens από main.py (συνοπτικό) ──
def n_tech_tokens(text):
    if not text: return 0
    t = set()
    t.update(re.findall(r'\b[vV]?\d+(?:\.\d+){1,4}\b', text))
    t.update(m.lower() for m in re.findall(r'\b(?:swg|nas|isg|ibm)\d{5,}[A-Z]?\b', text, re.I))
    t.update(m.lower() for m in re.findall(r'\b[A-Z]{2,}[0-9]{4,}[A-Z]?\b', text))
    t.update(m.lower() for m in re.findall(r'\bEJP[A-Z0-9]+\b', text))
    t.update(m.lower() for m in re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text))
    t.update(m.lower() for m in re.findall(r'\b[A-Z]{3,5}\b', text))
    t.update(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text.lower()))
    return len([x for x in t if len(x) >= 3])


def softmax_entropy(logits):
    if not logits: return 0.0
    mx = max(logits)
    exps = [math.exp(l - mx) for l in logits]
    s = sum(exps); ps = [e / s for e in exps]
    return -sum(p * math.log(p + 1e-12) for p in ps)


def make_folds():
    train = load_qa("training_Q_A.json")
    qids = [r[F_QID] for r in train]
    rng = random.Random(SEED); rng.shuffle(qids)
    folds = {str(k): [] for k in range(K_FOLDS)}
    for i, q in enumerate(qids):
        folds[str(i % K_FOLDS)].append(q)
    os.makedirs("data", exist_ok=True)
    json.dump(folds, open(FOLDS_PATH, "w"))
    print(f"[OK] {FOLDS_PATH}: 5 folds από {len(qids)} TRAIN ερωτήσεις (seed {SEED})")
    for k in folds: print(f"  fold {k}: {len(folds[k])}")


def train_fold(k):
    """Εκπαιδεύει μοντέλο στα answerable των ΑΛΛΩΝ 4 folds -> models/reranker_fold{k}."""
    folds = json.load(open(FOLDS_PATH))
    train = {r[F_QID]: r for r in load_qa("training_Q_A.json")}
    out_q = set(folds[str(k)])
    in_qids = [q for kk in folds if kk != str(k) for q in folds[kk]]
    ans = [train[q] for q in in_qids if train[q].get(F_ANS) == "Y" and train[q].get(F_DOC)]
    print(f"fold {k}: εκπαίδευση σε {len(ans)} answerable (εκτός fold {k})")

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    import torch
    examples = []
    for i, r in enumerate(ans):
        q = str(r.get(F_TEXT) or r.get(F_TITLE) or ""); gid = r.get(F_DOC)
        try:
            sols = search(q, 20).get("suggested_solutions", [])
        except Exception:
            continue
        gold = next((s for s in sols if s.get("technote_id") == gid), None)
        if not gold: continue
        pos = (gold.get("content") or gold.get("title") or "")[:2000]
        negs = [(s.get("content") or s.get("title") or "")[:2000]
                for s in sols if s.get("technote_id") != gid][:4]
        if not negs: continue
        examples.append(InputExample(texts=[q, pos], label=1.0))
        for nt in negs: examples.append(InputExample(texts=[q, nt], label=0.0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1,
                         max_length=512, device=device)
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    warmup = int(0.10 * len(loader) * 3)
    out = f"models/reranker_fold{k}"
    model.fit(train_dataloader=loader, epochs=3, warmup_steps=warmup, output_path=out,
              show_progress_bar=True)
    model.save(out)
    print(f"[OK] -> {out}  (τώρα δείξτε το service εκεί & restart, μετά --score-fold {k})")


def score_fold(k):
    """Βαθμολογεί τις ερωτήσεις του fold k μέσω του service (που ΠΡΕΠΕΙ να τρέχει με
    τον reranker_fold{k}). Γράφει OOF records (answerable ΚΑΙ unanswerable)."""
    folds = json.load(open(FOLDS_PATH))
    train = {r[F_QID]: r for r in load_qa("training_Q_A.json")}
    qids = folds[str(k)]
    done = set()
    if os.path.exists(OOF_PATH):
        done = {json.loads(l)["question_id"] for l in open(OOF_PATH)}
    fout = open(OOF_PATH, "a", encoding="utf-8")
    n = 0
    for q in qids:
        if q in done: continue
        r = train[q]
        text = str(r.get(F_TEXT) or r.get(F_TITLE) or "")
        gid = r.get(F_DOC); is_ans = (r.get(F_ANS) == "Y" and bool(gid))
        try:
            sols = search(text, 5).get("suggested_solutions", [])
        except Exception as e:
            print(f"  [{q}] search failed: {e}", file=sys.stderr); continue
        ids = [s.get("technote_id") for s in sols]
        logits = [float(s.get("rerank_score", 0.0)) for s in sols]
        overlaps = [float(s.get("overlap_score", 0.0)) for s in sols]
        rrfs = [float(s.get("rrf_score", 0.0)) for s in sols]
        present = bool(is_ans and gid in ids)
        logits_sorted = sorted(logits, reverse=True)
        rec = {
            "question_id": q, "answerable": bool(is_ans), "fold": k,
            "present": present, "ids": ids, "rr": logits,
            # features για Πείραμα 3 (ΔΙΑΘΕΣΙΜΑ από το /search):
            "max_logit": max(logits) if logits else 0.0,
            "logit2": logits_sorted[1] if len(logits_sorted) > 1 else 0.0,
            "mean_logit": (sum(logits) / len(logits)) if logits else 0.0,
            "gap12": (logits_sorted[0] - logits_sorted[1]) if len(logits_sorted) > 1 else 0.0,
            "softmax_entropy": softmax_entropy(logits),
            "overlap_max": max(overlaps) if overlaps else 0.0,
            "rrf_top1": rrfs[0] if rrfs else 0.0,
            "q_len": len(text.split()),
            "n_tech_tokens": n_tech_tokens(text),
            # ΔΕΝ εκτίθενται από το /search -> null (προαιρετικό patch· βλ. README 1Β):
            "bm25_max": None, "dense_max": None,
        }
        fout.write(json.dumps(rec) + "\n"); fout.flush(); n += 1
        if n % 25 == 0: print(f"  fold {k}: {n} scored")
    fout.close()
    print(f"[OK] fold {k}: γράφτηκαν {n} νέα OOF records -> {OOF_PATH}")


def ks_check():
    """Σύγκριση διαμέσου max-logit των unanswerable TRAIN (OOF) vs TEST (train_only)."""
    recs = [json.loads(l) for l in open(OOF_PATH)]
    tr_unans = [r["max_logit"] for r in recs if not r["answerable"]]
    test_path = "data/retrieval_test.jsonl"
    if not os.path.exists(test_path):
        print("[WARN] λείπει data/retrieval_test.jsonl (τρέξτε 1Γ πρώτα) — KS παραλείπεται.")
        return
    test = [json.loads(l) for l in open(test_path)]
    te_unans = [r["gate_max_logit"] for r in test if not r["answerable"]]
    if not tr_unans or not te_unans:
        print("[WARN] δεν υπάρχουν unanswerable και στα δύο σύνολα."); return
    med_tr = sorted(tr_unans)[len(tr_unans)//2]
    med_te = sorted(te_unans)[len(te_unans)//2]
    diff = abs(med_tr - med_te)
    try:
        from scipy.stats import ks_2samp
        ks = ks_2samp(tr_unans, te_unans)
        ksline = f"KS stat={ks.statistic:.3f} p={ks.pvalue:.3g}"
    except Exception:
        ksline = "(scipy μη διαθέσιμο — μόνο διάμεσοι)"
    print(f"median max-logit unanswerable: TRAIN-OOF={med_tr:.3f}  TEST={med_te:.3f}  "
          f"|Δ|={diff:.3f}")
    print(f"  {ksline}")
    if diff > 0.5:
        print("  [ΠΡΟΣΟΧΗ] |Δ| > 0.5 logit -> εφαρμόστε στο TEST το ΜΕΣΟ logit των 5 fold "
              "μοντέλων ως ανάλυση ευαισθησίας και αναφέρετέ το (βλ. ANALYSIS_PLAN).")
    else:
        print("  [OK] μετατόπιση κλίμακας εντός ανοχής (<=0.5 logit).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-folds", action="store_true")
    ap.add_argument("--train-fold", type=int)
    ap.add_argument("--score-fold", type=int)
    ap.add_argument("--ks", action="store_true")
    a = ap.parse_args()
    if a.make_folds: make_folds()
    elif a.train_fold is not None: train_fold(a.train_fold)
    elif a.score_fold is not None: score_fold(a.score_fold)
    elif a.ks: ks_check()
    else: ap.print_help()
