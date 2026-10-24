import tarfile
import json
import csv
import math
import time
import urllib.request

"""
ΡΟΛΟΣ:
Πλήρης αξιολόγηση του συστήματος αναζήτησης σε ΟΛΟ το dev set του TechQA.

ΣΕ ΣΧΕΣΗ ΜΕ ΤΗΝ ΠΑΛΙΑ ΕΚΔΟΣΗ:
- Τρέχει σε όλα τα answerable questions (όχι μόνο 30) -> σταθερό αποτέλεσμα
- Υπολογίζει 95% confidence interval (Wilson score)
- Εξάγει CSV με per-ticket αποτελέσματα για γραφήματα στη διπλωματική
- Μετράει Top-1, Top-3, Top-5, Top-10 accuracy ταυτόχρονα
- Mean Reciprocal Rank (MRR) -- standard IR metric
- F1@K -- αρμονικός μέσος Precision@K και Recall@K

ΣΗΜΕΙΩΣΗ για το F1@K (single ground truth ανά query):
  Με ένα μόνο σχετικό έγγραφο ανά ερώτημα:
    Recall@K    = 1 αν το GT είναι στα top-K, αλλιώς 0  (== accuracy@K)
    Precision@K = 1/K αν το GT είναι στα top-K, αλλιώς 0
    F1@K        = 2*P*R / (P+R)
  Το macro-F1@K είναι ο μέσος όρος του F1@K σε όλα τα ερωτήματα.

ΓΙΑΤΙ: Με 30 samples το 95% CI είναι ±14% -- ένα run 57% κι ένα 80% είναι
στατιστικά το ίδιο σύστημα. Με ~600 samples το CI πέφτει σε ±3-4%.
"""

# ── Config ────────────────────────────────────
API_URL    = "http://localhost:8080/search"
TAR_PATH   = "/data/TechQA.tar.gz"
DIAG_POOL  = 100          # Πόσα αποτελέσματα ζητάμε ανά query
LIMIT      = None         # None = όλο το dev set. Βάλε π.χ. 50 για γρήγορο test.
CSV_OUT    = "/data/eval_results.csv"


def wilson_ci(correct: int, total: int, z: float = 1.96):
    """
    95% Wilson score confidence interval για binomial proportion.
    Πιο αξιόπιστο από το κανονικό CI για μικρά/ακραία ποσοστά.
    """
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def run_evaluation(limit=None):
    print("Loading dev_Q_A.json...")
    try:
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            dev_member = next(
                (m for m in tar.getmembers() if m.name.endswith("dev_Q_A.json")),
                None
            )
            if not dev_member:
                print("Error: dev_Q_A.json not found")
                return
            qa_data = json.load(tar.extractfile(dev_member))
    except Exception as e:
        print(f"Reading error: {e}")
        return

    answerable = [q for q in qa_data if q.get("ANSWERABLE") == "Y" and q.get("DOCUMENT")]
    if limit:
        answerable = answerable[:limit]

    print(f"Evaluating {len(answerable)} answerable questions...")
    print("=" * 70)

    # Μετρητές
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    f1_sums = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}  # άθροισμα F1@K ανά query
    reciprocal_ranks = []
    rows = []  # για CSV

    start = time.time()

    for idx, item in enumerate(answerable):
        raw_text  = item.get("QUESTION_TEXT")
        raw_title = item.get("QUESTION_TITLE")
        question  = str(raw_text or raw_title or "No description provided")
        gt_id     = item.get("DOCUMENT")
        qid       = item.get("QUESTION_ID", "?")

        payload = json.dumps({
            "issue_description": question,
            "n_results":         DIAG_POOL,
        }).encode("utf-8")
        req = urllib.request.Request(
            API_URL, data=payload, headers={"Content-Type": "application/json"}
        )

        try:
            response = urllib.request.urlopen(req)
            result   = json.loads(response.read().decode("utf-8"))
            suggested = [s.get("technote_id") for s in result.get("suggested_solutions", [])]
        except Exception as e:
            print(f"  [{qid}] REQUEST FAILED: {e}")
            # Καταγράφουμε ως αποτυχία ώστε να μην αλλοιωθεί το denominator
            rows.append({"qid": qid, "gt_id": gt_id, "position": -1,
                         "top1": 0, "top3": 0, "top5": 0, "top10": 0})
            reciprocal_ranks.append(0.0)
            continue

        # Θέση του GT
        pos = suggested.index(gt_id) + 1 if gt_id in suggested else None

        for k in (1, 3, 5, 10):
            if pos is not None and pos <= k:
                hits[k] += 1
                # Single GT: Precision@K = 1/K, Recall@K = 1 (GT βρέθηκε)
                precision = 1.0 / k
                recall    = 1.0
                f1_sums[k] += 2 * precision * recall / (precision + recall)
            # αν δεν βρέθηκε στα top-K: P=R=0 -> F1=0 (δεν προσθέτουμε τίποτα)

        rr = 1.0 / pos if pos is not None else 0.0
        reciprocal_ranks.append(rr)

        rows.append({
            "qid":      qid,
            "gt_id":    gt_id,
            "position": pos if pos is not None else -1,
            "top1":     1 if (pos and pos <= 1) else 0,
            "top3":     1 if (pos and pos <= 3) else 0,
            "top5":     1 if (pos and pos <= 5) else 0,
            "top10":    1 if (pos and pos <= 10) else 0,
            "f1_at_3":  round(2 * (1/3) / ((1/3) + 1), 4) if (pos and pos <= 3) else 0.0,
            "f1_at_10": round(2 * (1/10) / ((1/10) + 1), 4) if (pos and pos <= 10) else 0.0,
        })

        # Progress κάθε 25 tickets
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            eta  = (len(answerable) - idx - 1) / rate
            cur_acc = hits[3] / (idx + 1) * 100
            print(f"  [{idx+1}/{len(answerable)}] "
                  f"Top-3 so far: {cur_acc:.1f}% | "
                  f"ETA: {eta/60:.1f} min")

    total = len(answerable)
    elapsed = time.time() - start

    # ── Αποτελέσματα ──────────────────────────
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS — {total} questions, {elapsed/60:.1f} min")
    print("=" * 70)

    for k in (1, 3, 5, 10):
        p, lo, hi = wilson_ci(hits[k], total)
        macro_f1 = f1_sums[k] / total if total else 0.0
        print(f"  Top-{k:<2} Accuracy: {p*100:5.2f}%  "
              f"(95% CI: {lo*100:.1f}% - {hi*100:.1f}%)  "
              f"[{hits[k]}/{total}]  |  F1@{k}: {macro_f1:.4f}")

    mrr = sum(reciprocal_ranks) / total if total else 0.0
    print(f"  MRR (Mean Reciprocal Rank): {mrr:.4f}")

    # ── F1 επεξήγηση ──────────────────────────
    print("\n" + "=" * 70)
    print("F1@K BREAKDOWN (macro-averaged, single GT ανά query)")
    print("=" * 70)
    print("  Precision@K = 1/K (αν GT στα top-K), Recall@K = 1 (αν GT στα top-K)")
    for k in (1, 3, 5, 10):
        macro_f1 = f1_sums[k] / total if total else 0.0
        # Theoretical max F1@K αν ΟΛΑ τα queries πετύχαιναν:
        max_f1 = 2 * (1/k) / ((1/k) + 1)
        print(f"  F1@{k:<2}: {macro_f1:.4f}  (max δυνατό: {max_f1:.4f}, "
              f"επιτεύχθηκε {macro_f1/max_f1*100:.0f}% του max)")

    # ── Position histogram ────────────────────
    print("\n" + "=" * 70)
    print("POSITION HISTOGRAM")
    print("=" * 70)
    buckets = {"1-3": 0, "4-10": 0, "11-25": 0, "26-50": 0,
               "51-100": 0, "missing": 0}
    for r in rows:
        pos = r["position"]
        if pos == -1:
            buckets["missing"] += 1
        elif pos <= 3:
            buckets["1-3"] += 1
        elif pos <= 10:
            buckets["4-10"] += 1
        elif pos <= 25:
            buckets["11-25"] += 1
        elif pos <= 50:
            buckets["26-50"] += 1
        else:
            buckets["51-100"] += 1

    for label, count in buckets.items():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:<10} | {count:4d} ({pct:5.1f}%) {bar}")

    # ── CSV export ────────────────────────────
    try:
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["qid", "gt_id", "position",
                               "top1", "top3", "top5", "top10",
                               "f1_at_3", "f1_at_10"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-ticket results saved to: {CSV_OUT}")
        print("(Χρησιμοποίησέ το για γραφήματα στη διπλωματική)")
    except Exception as e:
        print(f"\nCSV export failed: {e}")

    print("=" * 70)


if __name__ == "__main__":
    run_evaluation(limit=LIMIT)