#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Ε — Stage C × 3 runs, στις ΙΔΙΕΣ 310 ερωτήσεις (paired με Stage B).

Η απόφαση auto-resolve/escalate είναι ο ΑΥΘΕΝΤΙΚΟΣ jira_worker.decide()
(MODE B Ή gate Ή faith), που αναπαράγεται πιστά στο stage_bc_hooks.decide().
Το retrieval είναι ΠΑΓΩΜΕΝΟ (data/retrieval_test*.jsonl) — ΚΑΝΕΝΑ νέο retrieval στον worker.

ΔΥΟ ΛΕΙΤΟΥΡΓΙΕΣ:
  --mode generate  : (πλήρως εκτελέσιμη) frozen top-5 -> Ollama -> decide() -> JSONL.
                     Δίνει τους επιστημονικούς αριθμούς + paired B–C χωρίς Jira.
  --mode jira      : πλήρης βρόχος Jira μέσω των ΔΙΚΩΝ ΣΑΣ modules (pseudo_users_jira +
                     jira_worker), με call_resolve monkeypatched σε frozen_resolve
                     (ώστε ο worker να ΜΗΝ ξανακάνει retrieval). Βλ. σημειώσεις κάτω.

ΕΞΟΔΟΣ: logs/stageC/C_run<k>.jsonl  (stage:"C", abstain=true όταν escalated).

ΕΚΤΕΛΕΣΗ (ένα run τη φορά, restart Ollama ανάμεσα):
  python exp1e_stageC.py --mode generate --model techqa-llama3 --run 1 --temp 0
"""
import json, os, sys, time, random, argparse, pathlib, urllib.request
import stage_bc_hooks as H

OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://ollama:11434/api/chat")  # in-container service name
NUM_CTX     = int(os.getenv("NUM_CTX", "8192"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "1024"))
DOCS_PATH   = os.getenv("DOCS_PATH", "data/retrieval_test_docs.jsonl")
FROZEN_PATH = os.getenv("FROZEN_PATH", "data/retrieval_test.jsonl")
LOG_DIR     = "logs/stageC"
GATE_T      = os.getenv("RAG_GATE_THRESHOLD", "").strip()   # κενό = gate OFF (όπως κύρια runs)
GATE_T      = float(GATE_T) if GATE_T else None


def load_frozen():
    docs = {json.loads(l)["question_id"]: json.loads(l) for l in open(DOCS_PATH, encoding="utf-8")}
    meta = {json.loads(l)["question_id"]: json.loads(l) for l in open(FROZEN_PATH, encoding="utf-8")}
    return docs, meta


def ollama(model, messages, temperature):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return (json.loads(r.read().decode()).get("message", {}) or {}).get("content", "")


def frozen_resolve(qid, docs, meta, model, temperature):
    """Αναπαράγει το dict του /resolve ΑΠΟ παγωμένο retrieval (ΟΧΙ νέο retrieval).
    sources = [{technote_id, title, final_score, rerank_score}] (όπως main.build_rag_context).
    gated = (GATE_T set AND max_rerank < GATE_T)."""
    d, m = docs[qid], meta[qid]
    messages = H.build_messages(d["ticket"], d["solutions"])
    logits = m["rerank_logits"]; max_rerank = max(logits) if logits else None
    gated = bool(GATE_T is not None and max_rerank is not None and max_rerank < GATE_T)
    answer = H.ABSTAIN_PHRASE if gated else ollama(model, messages, temperature)
    sources = [{"technote_id": s["technote_id"], "title": s.get("title", ""),
                "rerank_score": lg} for s, lg in zip(d["solutions"], logits)]
    return {"ticket": d["ticket"], "answer": answer, "sources": sources,
            "model": model, "gated": gated, "max_rerank_score": max_rerank}


def run_generate(args):
    docs, meta = load_frozen()
    qids = list(docs.keys())
    order = random.Random(2000 + args.run).sample(qids, len(qids))
    out = pathlib.Path(LOG_DIR) / f"C_run{args.run}.jsonl"
    done = {json.loads(l)["question_id"] for l in out.open()} if out.exists() else set()
    print(f"[Stage C generate run {args.run}] {len(qids)} tickets | model={args.model} temp={args.temp}")
    with out.open("a", encoding="utf-8") as f:
        for i, qid in enumerate(order):
            if qid in done:
                continue
            t0 = time.time()
            try:
                res = frozen_resolve(qid, docs, meta, args.model, args.temp)
            except Exception as e:
                print(f"  [{qid}] FAILED: {e}", file=sys.stderr); continue
            escalate, reason = H.decide(res)                 # == jira_worker.decide()
            m = meta[qid]
            f.write(json.dumps({
                "stage": "C", "config": "QLoRA", "run": args.run, "question_id": qid,
                "split": "test", "retrieved_ids": m["ids"], "rerank_logits": m["rerank_logits"],
                "gate_max_logit": m["gate_max_logit"], "raw_output": res["answer"],
                "abstain": bool(escalate), "escalated": bool(escalate), "reason": reason,
                "gated": res["gated"], "cited_ids": H.cited_ids_bracketed(res["answer"]),
                "model": res["model"], "latency_s": round(time.time() - t0, 2),
            }, ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(order)}")
    lines = [json.loads(l) for l in out.open()]
    ids = [r["question_id"] for r in lines]
    assert len(ids) == len(set(ids)), "διπλοί qid!"
    nesc = sum(r["escalated"] for r in lines)
    print(f"[OK] {out}: {len(lines)} tickets | escalated={nesc} auto={len(lines)-nesc}")
    print("  (επαναλάβετε για run 2,3 με restart Ollama· μετά: python exp1e_analysis.py)")


def run_jira(args):
    """Πλήρης βρόχος Jira μέσω των ΔΙΚΩΝ ΣΑΣ modules.
    ΑΠΑΙΤΕΙ στο sys.path: pseudo_users_jira.py, jira_worker.py, jira_client.py.
    Τεχνική: monkeypatch jira_worker.call_resolve -> frozen_resolve, ώστε ο worker να
    τρέχει ΑΥΤΟΥΣΙΟΣ (decide/comments/labels/transitions/assign) αλλά ΧΩΡΙΣ νέο retrieval."""
    sys.path.insert(0, os.getenv("APP_DIR", "/app"))
    sys.path.insert(0, os.getenv("DATA_DIR", "/data"))
    docs, meta = load_frozen()
    try:
        import jira_worker
    except Exception as e:
        print(f"[ERROR] δεν βρέθηκε jira_worker/jira_client στο sys.path: {e}", file=sys.stderr)
        print("  Ορίστε APP_DIR/DATA_DIR ώστε να εισάγονται τα modules σας.", file=sys.stderr)
        sys.exit(1)

    # map description -> qid (ο worker περνά το description στο call_resolve)
    desc_to_qid = {d["ticket"].strip(): q for q, d in docs.items()}

    def _frozen_call_resolve(description):
        qid = desc_to_qid.get((description or "").strip())
        if qid is None:  # fallback: ταίριασμα με startswith (truncation ticket)
            qid = next((q for q, d in docs.items()
                        if d["ticket"].strip().startswith((description or "").strip()[:200])), None)
        if qid is None:
            raise RuntimeError("δεν ταιριάζει ticket με frozen qid")
        return frozen_resolve(qid, docs, meta, args.model, args.temp)

    jira_worker.call_resolve = _frozen_call_resolve     # <-- ΠΑΓΩΜΕΝΟ retrieval στον worker

    print(">>> ΟΔΗΓΙΕΣ πλήρους Jira run (ανά run k):")
    print("  1) Δημιουργήστε ΚΑΘΑΡΟ project & 310 tickets με το pseudo_users_jira.py σας")
    print("     (summary=QUESTION_TITLE, description=QUESTION_TEXT· GT map ΕΞΩ από το Jira).")
    print("  2) Τρέξτε το jira_worker.main() — τώρα με frozen call_resolve (patched εδώ).")
    print("  3) Από το jira_worker_log.csv -> C_run{k}.jsonl (abstain=escalated), join ticket->qid.")
    print("  Η συνάρτηση αυτή έχει κάνει το κρίσιμο patch· συμπληρώστε τα βήματα 1&3 με τα")
    print("  δικά σας pseudo_users_jira/jira_client entrypoints (ονόματα συναρτήσεων).")
    # Σκόπιμα δεν καλούμε jira_worker.main() αυτόματα: χρειάζεται Jira creds + το δικό σας
    # project-bootstrap. Το patch είναι το μόνο μη-προφανές κομμάτι και είναι έτοιμο.


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["generate", "jira"], default="generate")
    ap.add_argument("--model", default="techqa-llama3")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)
    (run_generate if args.mode == "generate" else run_jira)(args)
