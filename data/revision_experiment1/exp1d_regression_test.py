#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Δ — Βήμα 0: REGRESSION TEST (πριν από ΟΠΟΙΟΔΗΠΟΤΕ νέο run).

Το ΠΑΛΙΟ pipeline (ΠΑΛΙΟΣ reranker, ΠΑΛΙΑ παγωμένα top-5, 160 ερωτήσεις) πρέπει να
αναπαράγει τους αριθμούς του άρθρου εντός ±5 μονάδων:
    Base  C-Abst 7,1% (4/56),   F-Abst 6,7%
    QLoRA C-Abst 48,2% (27/56), F-Abst 24,0%
Αν αποτύχει -> ΔΕΝ ξεκινάμε νέα runs· βρίσκουμε τη διαφορά (prompt, chat template,
quantization, num_ctx, κομμένο prompt).

ΕΙΣΟΔΟΙ (από τα ΠΑΛΙΑ δεδομένα της διπλωματικής — δείξτε τα paths):
  --docs   : old retrieval_test_docs.jsonl ισοδύναμο για τις 160 (ticket + top-5 docs)
  --subs   : old subgroups csv (question_id, subgroup ∈ {present, missing})
             όπου present=104, missing=56 (η παλιά διαμέριση GT-present/GT-missing)
  --base-model / --qlora-model

ΕΚΤΕΛΕΣΗ (Ollama πάνω, ΠΑΛΙΟΣ reranker/top-5):
  python exp1d_regression_test.py --docs OLD_docs.jsonl --subs OLD_subs.csv \
      --base-model llama3:8b-instruct-q4_K_M --qlora-model techqa-llama3
"""
import json, csv, os, argparse, urllib.request
import stage_bc_hooks as H

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/chat")  # in-container service name
NUM_CTX, NUM_PREDICT = int(os.getenv("NUM_CTX", "8192")), int(os.getenv("NUM_PREDICT", "1024"))
# temperature: βάλτε ΑΥΤΟ που χρησιμοποίησαν οι ΑΡΧΙΚΟΙ αριθμοί του άρθρου (main.py
# default = 0.2). Αν το regression αποκλίνει, δοκιμάστε και 0.2 και 0.

TARGETS = {  # άρθρο
    "Base":  {"C_Abst": 7.1,  "F_Abst": 6.7},
    "QLoRA": {"C_Abst": 48.2, "F_Abst": 24.0},
}
TOL = 5.0


def call(model, messages, temperature):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return (json.loads(r.read().decode()).get("message", {}) or {}).get("content", "")


def run_config(name, model, docs, subs, temperature):
    present = [q for q, g in subs.items() if g == "present"]
    missing = [q for q, g in subs.items() if g == "missing"]
    abst = {}
    for qid, d in docs.items():
        msg = H.build_messages(d["ticket"], d["solutions"])
        raw = call(model, msg, temperature)
        ids = [s.get("technote_id") for s in d["solutions"]]
        mode, _, _ = H.parse_output(raw, ids)
        abst[qid] = (mode == "B")
    c = 100 * sum(abst[q] for q in missing) / max(1, len(missing))
    f = 100 * sum(abst[q] for q in present) / max(1, len(present))
    return c, f, len(missing), len(present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--subs", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--qlora-model", required=True)
    ap.add_argument("--temp", type=float, default=0.2,
                    help="temperature των ΑΡΧΙΚΩΝ runs (main.py default=0.2)")
    a = ap.parse_args()

    docs = {json.loads(l)["question_id"]: json.loads(l) for l in open(a.docs, encoding="utf-8")}
    subs = {}
    with open(a.subs, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            subs[row["question_id"]] = row["subgroup"]
    print(f"Regression set: {len(docs)} docs | present={sum(v=='present' for v in subs.values())} "
          f"missing={sum(v=='missing' for v in subs.values())}")

    ok_all = True
    for name, model in [("Base", a.base_model), ("QLoRA", a.qlora_model)]:
        c, f, nm, npz = run_config(name, model, docs, subs, a.temp)
        tc, tf = TARGETS[name]["C_Abst"], TARGETS[name]["F_Abst"]
        okc, okf = abs(c - tc) <= TOL, abs(f - tf) <= TOL
        ok_all = ok_all and okc and okf
        print(f"\n{name}:")
        print(f"  C-Abst = {c:.1f}%  (άρθρο {tc}%, Δ={c-tc:+.1f})  -> {'OK' if okc else 'FAIL'}  [{nm} missing]")
        print(f"  F-Abst = {f:.1f}%  (άρθρο {tf}%, Δ={f-tf:+.1f})  -> {'OK' if okf else 'FAIL'}  [{npz} present]")

    print("\n" + "=" * 60)
    if ok_all:
        print("REGRESSION TEST: PASS — εντός ±5pp. Μπορείτε να προχωρήσετε στα νέα runs.")
    else:
        print("REGRESSION TEST: FAIL — ΜΗΝ ξεκινήσετε νέα runs. Βρείτε τη διαφορά πρώτα")
        print("  (prompt / chat template / quantization / num_ctx / truncation).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
