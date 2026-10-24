#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Β (μέρος 1) — Εκπαίδευση reranker ΜΟΝΟ στο TRAIN.

ΣΚΟΠΟΣ: cross-encoder που ΔΕΝ έχει δει το TEST (dev). Ίδιες παράμετροι με το άρθρο,
εκτός από το split (μόνο answerable TRAIN).

ΠΑΡΑΜΕΤΡΟΙ (από το πρωτόκολλο):
  Μοντέλο:    cross-encoder/ms-marco-MiniLM-L-6-v2  (ή ό,τι ακριβώς χρησιμοποιήσατε)
  Δεδομένα:   ΜΟΝΟ answerable TRAIN· 1 positive + 4 negatives ανά ερώτηση
  Epochs/batch/maxlen/warmup: 3 / 16 / 512 / 10% των βημάτων
  Validation: 10% ΑΝΑ ΕΡΩΤΗΣΗ (όχι ανά ζεύγος)
  Seed:       42

ΣΗΜΑΝΤΙΚΟ: Η εξόρυξη negatives πρέπει να γίνει με το ΙΔΙΟ script/λογική με το αρχικό
σας reranker notebook. Εδώ γίνεται με hard negatives από το /search (top-k retrieved
που ΔΕΝ είναι το gold). Αν το αρχικό σας notebook μινάρει αλλιώς, αντικαταστήστε το
mine_negatives() ώστε να ταιριάζει ακριβώς.

ΕΚΤΕΛΕΣΗ (GPU — host container ή Kaggle· χρειάζεται sentence-transformers):
  python exp1b_reranker_train_only.py
Παράγει: models/reranker_train_only/
"""
import tarfile, json, os, sys, random, urllib.request

TAR_PATH   = os.getenv("TECHQA_TAR", "/data/TechQA.tar.gz")
SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8080/search")
OUT_MODEL  = os.getenv("OUT_MODEL", "models/reranker_train_only")
BASE_MODEL = os.getenv("RERANKER_BASE", "cross-encoder/ms-marco-MiniLM-L-6-v2")
N_NEG      = 4
EPOCHS     = 3
BATCH      = 16
MAXLEN     = 512
SEED       = 42
NEG_POOL   = 20     # πόσα retrieved κοιτάμε για να βρούμε negatives

F_QID, F_TITLE, F_TEXT, F_DOC, F_ANS = \
    "QUESTION_ID", "QUESTION_TITLE", "QUESTION_TEXT", "DOCUMENT", "ANSWERABLE"


def load_qa(member_endswith):
    with tarfile.open(TAR_PATH, "r:*") as tar:
        m = next((x for x in tar.getmembers() if x.name.endswith(member_endswith)), None)
        if not m:
            raise FileNotFoundError(member_endswith)
        return json.load(tar.extractfile(m))


def search(q, n):
    payload = json.dumps({"issue_description": q, "n_results": n}).encode()
    req = urllib.request.Request(SEARCH_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def doc_text_of(sol):
    return (sol.get("content") or sol.get("title") or "")[:2000]


def mine_negatives(question, gold_id, k=NEG_POOL):
    """Hard negatives: retrieved docs που δεν είναι το gold. Επιστρέφει [(text, id), ...]."""
    try:
        sols = search(question, k).get("suggested_solutions", [])
    except Exception:
        return []
    return [(doc_text_of(s), s.get("technote_id")) for s in sols
            if s.get("technote_id") != gold_id]


def gold_text(gold_id):
    """Παίρνει το κείμενο του gold technote μέσω ενός στοχευμένου search."""
    # Το gold μπορεί να μην είναι #1· ζητάμε βαθύ pool και το ψάχνουμε.
    return None  # βλ. σημείωση: χρησιμοποιούμε το gold doc από το pool παρακάτω


def main():
    random.seed(SEED)
    train = load_qa("training_Q_A.json")
    dev   = load_qa("dev_Q_A.json")
    train_qids = {r[F_QID] for r in train}
    test_qids  = {r[F_QID] for r in dev}

    # ── Κανόνας ασφαλείας του πρωτοκόλλου ──
    assert not (train_qids & test_qids), "ΔΙΑΡΡΟΗ: train_qids ∩ test_qids != ∅"
    print(f"[OK] disjoint: TRAIN {len(train_qids)} ∩ TEST {len(test_qids)} = 0")

    ans = [r for r in train if r.get(F_ANS) == "Y" and r.get(F_DOC)]
    print(f"Answerable TRAIN: {len(ans)}")

    try:
        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader
        import torch
    except Exception as e:
        print(f"[ERROR] χρειάζεται sentence-transformers + torch: {e}", file=sys.stderr)
        sys.exit(1)

    # Train/val split ΑΝΑ ΕΡΩΤΗΣΗ (10%)
    random.shuffle(ans)
    n_val = max(1, int(0.10 * len(ans)))
    val_q, train_q = ans[:n_val], ans[n_val:]
    print(f"Split ανά ερώτηση: train {len(train_q)} / val {len(val_q)}")

    def build_examples(questions):
        examples = []
        for i, r in enumerate(questions):
            q = str(r.get(F_TEXT) or r.get(F_TITLE) or "")
            gid = r.get(F_DOC)
            # pool από /search· βρίσκουμε το gold text + negatives από το ίδιο pool
            try:
                sols = search(q, NEG_POOL).get("suggested_solutions", [])
            except Exception:
                continue
            gold_sol = next((s for s in sols if s.get("technote_id") == gid), None)
            if gold_sol is None:
                # αν το gold δεν είναι στο pool, το θετικό δείγμα λείπει -> skip ερώτηση
                continue
            pos_text = doc_text_of(gold_sol)
            negs = [doc_text_of(s) for s in sols if s.get("technote_id") != gid][:N_NEG]
            if len(negs) < 1:
                continue
            examples.append(InputExample(texts=[q, pos_text], label=1.0))
            for nt in negs:
                examples.append(InputExample(texts=[q, nt], label=0.0))
            if (i + 1) % 25 == 0:
                print(f"  built {i+1}/{len(questions)} ερωτήσεις...")
        return examples

    print("Εξόρυξη θετικών/αρνητικών (μέσω /search)...")
    train_examples = build_examples(train_q)
    print(f"Training examples (pairs): {len(train_examples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAXLEN, device=device)
    loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH)
    warmup = int(0.10 * len(loader) * EPOCHS)

    os.makedirs(os.path.dirname(OUT_MODEL) or ".", exist_ok=True)
    print(f"Εκπαίδευση: epochs={EPOCHS} batch={BATCH} maxlen={MAXLEN} warmup={warmup} device={device}")
    model.fit(train_dataloader=loader, epochs=EPOCHS, warmup_steps=warmup,
              output_path=OUT_MODEL, show_progress_bar=True)
    model.save(OUT_MODEL)
    print(f"[OK] Αποθηκεύτηκε -> {OUT_MODEL}")
    print("\n>>> ΕΠΟΜΕΝΟ: αντιγράψτε το στο /app/reranker_model του service για το 1Γ,")
    print("    ή δείξτε το FINETUNED_RERANKER_PATH εκεί, και τρέξτε exp1c_stageA_freeze.py.")
    # Αποθήκευση των val qids για αναφορά disjointness
    with open("models/reranker_train_only_val_qids.json", "w") as f:
        json.dump([r[F_QID] for r in val_q], f)


if __name__ == "__main__":
    main()
