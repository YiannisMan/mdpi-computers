#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Δ — Synopsis / επαλήθευση ακεραιότητας των Stage B logs.

ΕΛΕΓΧΟΙ:
 1) 310 γραμμές ανά (config, run), χωρίς διπλούς question_id.
 2) retrieved_ids κάθε γραμμής == data/retrieval_test.jsonl  (assert ανά γραμμή).
 3) % invalid και % truncated ανά config/run.
 4) Hash του prompt template (build_messages κενό) == env.json (sha256.prompt_template).

ΕΚΤΕΛΕΣΗ (in-container, όπου υπάρχει /app/main.py):
  docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
    python exp1d_verify.py"

Env (προαιρετικά): STAGEB_DIR=logs/stageB  FROZEN_PATH=data/retrieval_test.jsonl  ENV_JSON=env.json  EXPECT_N=310
"""
import json, glob, os, sys
import stage_bc_hooks as H

LOG_DIR  = os.getenv("STAGEB_DIR", "logs/stageB")
FROZEN   = os.getenv("FROZEN_PATH", "data/retrieval_test.jsonl")
ENV_JSON = os.getenv("ENV_JSON", "env.json")
EXPECT_N = int(os.getenv("EXPECT_N", "310"))


def load_frozen_ids():
    ids = {}
    with open(FROZEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids[r["question_id"]] = list(r.get("ids") or [])
    return ids


def main():
    frozen = load_frozen_ids()
    print(f"frozen: {len(frozen)} ερωτήσεις ({FROZEN})")

    files = sorted(glob.glob(f"{LOG_DIR}/*_run*.jsonl"))
    if not files:
        print(f"[ERROR] κανένα *_run*.jsonl στο {LOG_DIR}")
        sys.exit(1)

    all_ok = True
    print("\n" + "=" * 92)
    print(f"{'file':22s} {'lines':>5} {'uniq':>5} {'dup':>3} {'ids==frozen':>13} "
          f"{'invalid':>13} {'truncated':>12} {'absB':>6}")
    print("-" * 92)
    for p in files:
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        qids = [r["question_id"] for r in rows]
        uniq = len(set(qids)); dup = len(qids) - uniq
        mism = missing = 0
        for r in rows:
            fid = frozen.get(r["question_id"])
            if fid is None:
                missing += 1
            elif list(r.get("retrieved_ids") or []) != fid:
                mism += 1
        n_inv = sum(1 for r in rows if r.get("invalid"))
        n_tr = sum(1 for r in rows if r.get("truncated"))
        n_absB = sum(1 for r in rows if r.get("mode") == "B" or r.get("abstain"))
        ok_lines = (len(rows) == EXPECT_N)
        ok_dup = (dup == 0)
        ok_ids = (mism == 0 and missing == 0)
        all_ok = all_ok and ok_lines and ok_dup and ok_ids
        name = os.path.basename(p).replace(".jsonl", "")
        ids_s = "OK" if ok_ids else f"{mism}mis/{missing}abs"
        flag = "" if (ok_lines and ok_dup and ok_ids) else "  <-- ΕΛΕΓΞΕ"
        print(f"{name:22s} {len(rows):5d} {uniq:5d} {dup:3d} {ids_s:>13} "
              f"{n_inv:3d} ({100*n_inv/len(rows):4.1f}%) {n_tr:3d} ({100*n_tr/len(rows):4.1f}%) "
              f"{n_absB:6d}{flag}")
    print("-" * 92)
    print("Έλεγχοι 1-3:", "PASS" if all_ok else "ΥΠΑΡΧΟΥΝ ΑΠΟΚΛΙΣΕΙΣ (δες παραπάνω)")

    # ── Έλεγχος 4: prompt template hash vs env.json ──
    print("\nΈλεγχος 4 — prompt template hash vs env.json:")
    if not H.RAG_SYSTEM_PROMPT:
        print("  [WARN] RAG_SYSTEM_PROMPT κενό — το main.py δεν φορτώθηκε (δες MAIN_PY_PATH).")
    th = H.prompt_template_sha256()
    print(f"  computed = {th}")
    env_hash = None
    if os.path.exists(ENV_JSON):
        env = json.load(open(ENV_JSON, encoding="utf-8"))
        env_hash = (env.get("sha256") or {}).get("prompt_template")
    if env_hash is None:
        print(f"  [WARN] δεν υπάρχει sha256.prompt_template στο {ENV_JSON} — ξανατρέξτε το")
        print("         capture_env.py (ενημερωμένο) ΜΕ ΙΔΙΟ main.py, και επαναλάβετε.")
        all_ok = False
    elif env_hash == th:
        print("  == env.json  -> OK")
    else:
        print(f"  != env.json ({env_hash})  -> FAIL (το prompt template άλλαξε μετά το capture_env)")
        all_ok = False

    print("\n" + "=" * 92)
    print("1Δ VERIFY: PASS — όλα τα logs ακέραια και συνεπή." if all_ok
          else "1Δ VERIFY: έλεγξε τα FAIL/WARN παραπάνω.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
