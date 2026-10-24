#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Γ — Stage A (retrieval) στο TEST + ΠΑΓΩΜΑ του top-5.

ΠΡΟΫΠΟΘΕΣΗ: το /search service τρέχει με τον reranker ΜΟΝΟ-TRAIN του 1Β
(FINETUNED_RERANKER_PATH -> models/reranker_train_only). Για τη "leaky" γραμμή
του Supplementary, ξανατρέξτε το ίδιο με τον ΠΑΛΙΟ reranker και --tag leaky.

ΤΙ ΚΑΝΕΙ (πάνω στις 310 ερωτήσεις του dev = TEST):
  1. Καλεί /search (n_results=DIAG_POOL) για κάθε ερώτηση.
  2. Stage A metrics στις answerable: Top-1/3/5/10 + MRR + Wilson 95% CI.
  3. ΠΑΓΩΝΕΙ το top-5 κάθε ερώτησης -> data/retrieval_test.jsonl
     (question_id, ids[5], rerank_logits[5], fused_scores[5], gate_max_logit).
  4. Φτιάχνει data/test_subgroups.csv (present / missA / missU).

ΣΗΜΕΙΩΣΗ: το /search επιστρέφει ανά doc: technote_id, rerank_score (= raw cross-encoder
logit), final_score (= fused). Αυτά γράφονται ως rerank_logits / fused_scores.

ΕΚΤΕΛΕΣΗ:
  python exp1c_stageA_freeze.py --tag main                  # ΚΥΡΙΟ: όλες οι 310 (report-only)
  python exp1c_stageA_freeze.py --tag main --exclude-dups   # ευαισθησία: 287 (ξεχωριστά _clean287)
  python exp1c_stageA_freeze.py --tag leaky                 # με τον ΠΑΛΙΟ reranker στο service
"""
import tarfile, json, csv, math, time, os, sys, argparse, urllib.request

API_URL   = os.getenv("SEARCH_URL", "http://localhost:8080/search")
TAR_PATH  = os.getenv("TECHQA_TAR", "/data/TechQA.tar")
DIAG_POOL = int(os.getenv("DIAG_POOL", "100"))   # για MRR/θέση· freeze top-5 από το ίδιο
OUT_DIR   = "data"
LOG_DIR   = "logs/stageA"

F_QID, F_TITLE, F_TEXT, F_DOC, F_ANS = \
    "QUESTION_ID", "QUESTION_TITLE", "QUESTION_TEXT", "DOCUMENT", "ANSWERABLE"


def wilson_ci(correct, total, z=1.959963984540054):
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def load_dev():
    with tarfile.open(TAR_PATH, "r:*") as tar:
        m = next((x for x in tar.getmembers() if x.name.endswith("dev_Q_A.json")), None)
        if not m:
            raise FileNotFoundError("dev_Q_A.json δεν βρέθηκε στο tar")
        return json.load(tar.extractfile(m))


def load_exclude():
    """Dev qids προς αποκλεισμό (διπλότυπες ερωτήσεις TRAIN↔TEST — βλ. 1Α inspect).
    Σειρά προτεραιότητας: EXCLUDE_QIDS env -> data/test_exclude_qids.txt -> data/dup_questions.csv.
    Επιστρέφει set (κενό αν δεν υπάρχει τίποτα)."""
    p = os.getenv("EXCLUDE_QIDS", f"{OUT_DIR}/test_exclude_qids.txt")
    if os.path.exists(p):
        return {l.strip() for l in open(p, encoding="utf-8") if l.strip()}
    dup = f"{OUT_DIR}/dup_questions.csv"
    if os.path.exists(dup):
        with open(dup, newline="", encoding="utf-8") as f:
            return {row["dev_qid"] for row in csv.DictReader(f) if row.get("dev_qid")}
    return set()


def search(question, n):
    payload = json.dumps({"issue_description": question, "n_results": n}).encode()
    req = urllib.request.Request(API_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", help="main | leaky")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude-dups", action="store_true",
                    help="ΑΝΑΛΥΣΗ ΕΥΑΙΣΘΗΣΙΑΣ: αποκλεισμός των 23 διπλοτύπων (TEST 287). "
                         "Γράφει σε ΞΕΧΩΡΙΣΤΑ αρχεία _clean287, ΔΕΝ αγγίζει το ΚΥΡΙΟ παγωμένο 310. "
                         "Προεπιλογή (χωρίς το flag): κύριο TEST = όλες οι 310 — report-only.")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

    dev = load_dev()
    n_before = len(dev)
    # ΠΡΟΕΠΙΛΟΓΗ: όλες οι 310 (οι διπλότυπες ΔΗΛΩΝΟΝΤΑΙ, δεν αφαιρούνται — οδηγία reviewer).
    # Ο αποκλεισμός είναι opt-in (--exclude-dups) ΜΟΝΟ για την ανάλυση ευαισθησίας στις 287.
    exclude = load_exclude() if args.exclude_dups else set()
    if exclude:
        dev = [q for q in dev if q.get(F_QID) not in exclude]
        print(f"[{args.tag}] Ανάλυση ευαισθησίας: αποκλείστηκαν {n_before - len(dev)} "
              f"διπλότυπες — από {n_before} -> {len(dev)}")
    elif args.exclude_dups:
        print(f"[{args.tag}] [ΠΡΟΣΟΧΗ] ζητήθηκε --exclude-dups αλλά δεν βρέθηκε λίστα "
              f"(data/test_exclude_qids.txt ή dup_questions.csv) — τρέχει στις {len(dev)}.")
    if args.limit:
        dev = dev[:args.limit]
    answerable = [q for q in dev if q.get(F_ANS) == "Y" and q.get(F_DOC)]
    print(f"[{args.tag}] TEST: {len(dev)} ερωτήσεις ({len(answerable)} answerable)")

    # ΚΥΡΙΟ (310) -> canonical frozen. Με --exclude-dups (287) -> ΞΕΧΩΡΙΣΤΑ αρχεία _clean287.
    main_run = not args.exclude_dups
    sfx = args.tag if main_run else f"{args.tag}_clean287"
    if main_run and args.tag == "main":
        freeze_path = f"{OUT_DIR}/retrieval_test.jsonl"
        docs_path   = f"{OUT_DIR}/retrieval_test_docs.jsonl"
    else:
        freeze_path = f"{LOG_DIR}/retrieval_test_{sfx}.jsonl"
        docs_path   = f"{LOG_DIR}/retrieval_test_docs_{sfx}.jsonl"
    sub_path    = f"{OUT_DIR}/test_subgroups.csv"
    per_path    = f"{LOG_DIR}/stageA_perq_{sfx}.csv"

    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rr_list = []
    subgroups = []            # (question_id, subgroup)
    perq = []                 # per-question Stage A rows
    fz = open(freeze_path, "w", encoding="utf-8")
    fd = open(docs_path, "w", encoding="utf-8")   # companion: ticket + top-5 docs (για Stage B/C)
    start = time.time()
    n_ans_done = 0

    for idx, item in enumerate(dev):
        qid = item.get(F_QID, "?")
        question = str(item.get(F_TEXT) or item.get(F_TITLE) or "No description provided")
        is_ans = (item.get(F_ANS) == "Y" and bool(item.get(F_DOC)))
        gt_id = item.get(F_DOC)

        try:
            res = search(question, DIAG_POOL)
            sols = res.get("suggested_solutions", [])
        except Exception as e:
            print(f"  [{qid}] SEARCH FAILED: {e}", file=sys.stderr)
            sols = []

        ids_all = [s.get("technote_id") for s in sols]
        top5 = sols[:5]
        top5_ids = [s.get("technote_id") for s in top5]
        rerank_logits = [float(s.get("rerank_score", 0.0)) for s in top5]
        fused_scores  = [float(s.get("final_score", 0.0)) for s in top5]
        gate_max = max(rerank_logits) if rerank_logits else float("-inf")

        # ΠΑΓΩΜΑ top-5 (schema πρωτοκόλλου: ids / logits / fused)
        fz.write(json.dumps({
            "question_id": qid, "answerable": bool(is_ans), "gold_id": gt_id,
            "ids": top5_ids, "rerank_logits": rerank_logits,
            "fused_scores": fused_scores, "gate_max_logit": gate_max,
        }) + "\n"); fz.flush()

        # Companion: ticket + ΠΑΓΩΜΕΝΑ έγγραφα (ώστε Stage B/C να χτίζουν prompt ΧΩΡΙΣ νέο retrieval)
        fd.write(json.dumps({
            "question_id": qid, "ticket": question,
            "solutions": [{"technote_id": s.get("technote_id"),
                           "title": s.get("title", ""),
                           "content": s.get("content", "")} for s in top5],
        }, ensure_ascii=False) + "\n"); fd.flush()

        # Υποπληθυσμός
        if not is_ans:
            subgroups.append((qid, "missU"))
        else:
            subgroups.append((qid, "present" if gt_id in top5_ids else "missA"))

        # Stage A metrics (μόνο answerable)
        if is_ans:
            n_ans_done += 1
            pos = (ids_all.index(gt_id) + 1) if gt_id in ids_all else None
            for k in (1, 3, 5, 10):
                if pos is not None and pos <= k:
                    hits[k] += 1
            rr_list.append(1.0 / pos if pos else 0.0)
            perq.append({"qid": qid, "gt_id": gt_id, "position": pos if pos else -1,
                         "in_top5": int(gt_id in top5_ids)})

        if (idx + 1) % 25 == 0:
            el = time.time() - start
            eta = (len(dev) - idx - 1) / ((idx + 1) / el) if el > 0 else 0
            print(f"  [{idx+1}/{len(dev)}] ans so far {n_ans_done} | ETA {eta/60:.1f} min")

    fz.close(); fd.close()

    # subgroups.csv (γράφεται ΜΟΝΟ στο ΚΥΡΙΟ main 310 — είναι το παγωμένο)
    if main_run and args.tag == "main":
        with open(sub_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["question_id", "subgroup"]); w.writerows(subgroups)

    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "gt_id", "position", "in_top5"])
        w.writeheader(); w.writerows(perq)

    # ── Report ──────────────────────────────────────────────────────────
    total = n_ans_done
    print("\n" + "=" * 70)
    print(f"STAGE A — tag={args.tag} — {total} answerable questions")
    print("=" * 70)
    lines = []
    for k in (1, 3, 5, 10):
        p, lo, hi = wilson_ci(hits[k], total)
        line = (f"  Top-{k:<2} = {p*100:5.2f}%  (95% CI {lo*100:.1f}-{hi*100:.1f})  "
                f"[{hits[k]}/{total}]")
        print(line); lines.append(line)
    mrr = sum(rr_list) / total if total else 0.0
    print(f"  MRR = {mrr:.4f}")
    recall5 = hits[5] / total if total else 0.0
    n_present = sum(1 for _, g in subgroups if g == "present")
    n_missA   = sum(1 for _, g in subgroups if g == "missA")
    n_missU   = sum(1 for _, g in subgroups if g == "missU")
    print(f"\n  Υποπληθυσμοί TEST: present={n_present}  missA={n_missA}  missU={n_missU}  "
          f"(missing={n_missA+n_missU})")
    print(f"  recall@5 = {recall5*100:.1f}%  "
          f"({'αναμενόμενη πτώση <65% — OK' if recall5 < 0.65 else 'σταθερό'})")
    if recall5 < 0.40:
        print("  [ΠΡΟΣΟΧΗ] recall@5 < 40% — ελέγξτε index/πεδίο gold id ΠΡΙΝ αναφέρετε.")

    # Stage A summary file
    with open(f"{LOG_DIR}/stageA_summary_{sfx}.txt", "w", encoding="utf-8") as f:
        f.write(f"STAGE A tag={sfx}  (n_answerable={total}, n_total={len(dev)})\n")
        f.write("\n".join(lines) + f"\n  MRR = {mrr:.4f}\n")
        f.write(f"  present={n_present} missA={n_missA} missU={n_missU}\n")

    print("\nΠαραδοτέα:")
    if main_run and args.tag == "main":
        print(f"  {freeze_path}   (ΚΥΡΙΟ παγωμένο top-5 στις 310 — κάντε git add + commit!)")
        print(f"  {sub_path}")
    elif args.exclude_dups:
        print(f"  {freeze_path}   (clean 287 — μόνο για ανάλυση ευαισθησίας)")
    else:
        print(f"  {freeze_path}   (leaky reranker — μόνο για Supplementary)")
    print(f"  {per_path}")
    print(f"  {LOG_DIR}/stageA_summary_{sfx}.txt")
    if main_run and args.tag == "main":
        print("\n>>> ΕΠΟΜΕΝΟ: git add data/retrieval_test.jsonl data/test_subgroups.csv && git commit")


if __name__ == "__main__":
    main()
