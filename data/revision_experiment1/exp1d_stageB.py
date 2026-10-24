#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Δ — Stage B: Base και QLoRA × 3 runs, στις 310 ερωτήσεις του TEST.

Διαβάζει το ΠΑΓΩΜΕΝΟ retrieval (data/retrieval_test*.jsonl) — ΚΑΝΕΝΑ νέο retrieval.
Χτίζει το prompt + parser μέσω stage_bc_hooks.py (ο ΑΥΘΕΝΤΙΚΟΣ σας κώδικας).
Γράφει per-incident JSONL (logs/stageB/<config>_run<k>.jsonl), resumable, με truncation flag.

Το Gate ΔΕΝ τρέχει εδώ: υπολογίζεται στο Πείραμα 2/3 από τα logs του Base (gate_max_logit).

ΠΡΩΤΟΚΟΛΛΟ ΑΝΑ RUN:
  - επανεκκίνηση Ollama server + OLLAMA_NUM_PARALLEL=1 (ΕΞΩ από το script — βλ. README)
  - διαφορετική τυχαία σειρά ερωτήσεων (seed = 1000 + run)
  - temperature 0· seed=run καταγράφεται (δεν αλλάζει sampling)

ΕΚΤΕΛΕΣΗ (ένα run τη φορά, με restart του Ollama ανάμεσα):
  python exp1d_stageB.py --config Base  --model llama3:8b-instruct-q4_K_M --run 1
  python exp1d_stageB.py --config QLoRA --model techqa-llama3            --run 1
  ... (runs 2, 3)
"""
import json, os, sys, time, random, argparse, pathlib, urllib.request
import stage_bc_hooks as H

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/chat")  # in-container service name
NUM_CTX     = int(os.getenv("NUM_CTX", "8192"))       # == main.py LLM_NUM_CTX
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "1024"))   # == main.py LLM_NUM_PREDICT (ΟΧΙ 512)
DOCS_PATH   = os.getenv("DOCS_PATH", "data/retrieval_test_docs.jsonl")
FROZEN_PATH = os.getenv("FROZEN_PATH", "data/retrieval_test.jsonl")
LOG_DIR     = "logs/stageB"
# ΠΡΟΣΟΧΗ temperature: το main.py έχει default LLM_TEMPERATURE=0.2, αλλά το άρθρο/
# ANALYSIS_PLAN δηλώνει temperature 0. Επιβεβαιώστε τι χρησιμοποίησαν οι ΑΡΧΙΚΟΙ
# αριθμοί και ευθυγραμμίστε (βλ. README §temperature). Default εδώ: 0 (πρωτόκολλο).


def load_frozen():
    docs = {json.loads(l)["question_id"]: json.loads(l) for l in open(DOCS_PATH, encoding="utf-8")}
    meta = {json.loads(l)["question_id"]: json.loads(l) for l in open(FROZEN_PATH, encoding="utf-8")}
    return docs, meta


def call_ollama(model, messages, run, temperature):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "seed": run,
                           "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base | QLoRA")
    ap.add_argument("--model", required=True, help="π.χ. llama3:8b-instruct-q4_K_M ή techqa-llama3")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="temperature (πρωτόκολλο: 0· main.py default ήταν 0.2)")
    args = ap.parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)

    docs, meta = load_frozen()
    qids = list(docs.keys())
    assert set(qids) == set(meta.keys()), "mismatch docs vs frozen"
    print(f"[{args.config} run {args.run}] {len(qids)} ερωτήσεις | model={args.model}")

    out = pathlib.Path(LOG_DIR) / f"{args.config}_run{args.run}.jsonl"
    done = {json.loads(l)["question_id"] for l in out.open()} if out.exists() else set()
    order = random.Random(1000 + args.run).sample(qids, len(qids))

    n_trunc = n_invalid = 0
    with out.open("a", encoding="utf-8") as f:
        for i, qid in enumerate(order):
            if qid in done:
                continue
            d = docs[qid]; m = meta[qid]
            messages = H.build_messages(d["ticket"], d["solutions"])
            t0 = time.time()
            try:
                resp = call_ollama(args.model, messages, args.run, args.temp)
            except Exception as e:
                print(f"  [{qid}] OLLAMA FAILED: {e}", file=sys.stderr)
                continue
            raw = (resp.get("message", {}) or {}).get("content", "")
            retrieved_ids = m["ids"]
            mode, cited, valid = H.parse_output(raw, retrieved_ids)   # mode ∈ {A,B}
            invalid = H.invalid_diagnostic(raw, retrieved_ids)        # διαγνωστικό (μετρά ως 'απάντηση')
            if invalid:
                n_invalid += 1
            pet = resp.get("prompt_eval_count")
            truncated = bool(pet is not None and pet >= NUM_CTX - 1)
            if truncated:
                n_trunc += 1
            f.write(json.dumps({
                "stage": "B", "config": args.config, "run": args.run, "seed": args.run,
                "question_id": qid, "split": "test",
                "retrieved_ids": retrieved_ids, "rerank_logits": m["rerank_logits"],
                "fused_scores": m["fused_scores"], "gate_max_logit": m["gate_max_logit"],
                "raw_output": raw, "mode": mode, "abstain": (mode == "B"),
                "invalid": invalid, "cited_ids": cited, "citation_valid": valid,
                "latency_s": round(time.time() - t0, 2),
                "n_prompt_tokens": pet, "n_gen_tokens": resp.get("eval_count"),
                "truncated": truncated,
            }, ensure_ascii=False) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(order)} | invalid={n_invalid} trunc={n_trunc}")

    # Έλεγχοι ορθότητας
    lines = [json.loads(l) for l in out.open()]
    ids_seen = [r["question_id"] for r in lines]
    assert len(ids_seen) == len(set(ids_seen)), "διπλοί question_id!"
    print(f"\n[OK] {out}: {len(lines)} γραμμές | invalid={n_invalid} truncated={n_trunc}")
    if n_trunc:
        print(f"  [ΠΡΟΣΟΧΗ] {n_trunc} truncated — αν είναι πολλά, αυξήστε num_ctx και "
              f"επαναλάβετε ΟΛΑ τα configs.")
    print(f"  (επαναλάβετε για run 2, 3 — ΜΕ restart του Ollama ανάμεσα)")


if __name__ == "__main__":
    main()
