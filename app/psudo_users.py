import tarfile
import json
import urllib.request
import urllib.parse

"""
ΡΟΛΟΣ:
Προσομοιώνει πραγματικούς χρήστες που υποβάλλουν IT tickets για να ελέγξει 
την ακρίβεια (Accuracy) του συστήματος αναζήτησης.

ΛΕΙΤΟΥΡΓΙΕΣ:
1. Αντλεί πραγματικά ερωτήματα χρηστών από το dev_Q_A.json.
2. Στέλνει τα αιτήματα στο FastAPI API (/search).
3. Συγκρίνει την απάντηση του συστήματος με τη "σωστή" απάντηση (Ground Truth).
4. Υπολογίζει το ποσοστό επιτυχίας (Top-N Accuracy).

DIAGNOSTIC MODE:
Όταν είναι True, για κάθε ticket ζητάμε top-100 αποτελέσματα και καταγράφουμε
την ακριβή θέση του ground truth. Αυτό μας δείχνει αν ένα FAILURE είναι
"παραλίγο επιτυχία" (θέση 4-10) ή βαθύτερο πρόβλημα (θέση >50 ή missing).
"""

# -- Config ----------------------------------
DIAGNOSTIC_MODE = True   # True = ζητάμε 100 για να βρούμε θέση GT
TOP_N           = 3       # Αν το GT είναι στις Ν πρώτες, θεωρείται επιτυχία
DIAG_POOL       = 100     # Πόσα αποτελέσματα ζητάμε σε diagnostic mode
API_URL         = "http://localhost:8080/search"
TAR_PATH        = "/data/TechQA.tar.gz"


def classify_position(pos: int | None) -> str:
    """Κατηγοριοποιεί τη θέση του ground truth."""
    if pos is None:
        return "MISSING (not in top-100)"
    if pos == 0:
        return "PERFECT (#1)"
    if pos < TOP_N:
        return f"SUCCESS (#{pos + 1})"
    if pos < 10:
        return f"NEAR-MISS (#{pos + 1}) — θα σωζόταν με top-10"
    if pos < 25:
        return f"CLOSE (#{pos + 1}) — reranker tuning"
    if pos < 50:
        return f"FAR (#{pos + 1}) — reranker αποτυγχάνει"
    return f"VERY-FAR (#{pos + 1}) — retrieval ή reranker σοβαρό πρόβλημα"


def run_simulation(limit: int):
    print("Reading actual IT tickets from dev_Q_A.json...")

    try:
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            dev_member = next(
                (m for m in tar.getmembers() if m.name.endswith("dev_Q_A.json")),
                None
            )
            if not dev_member:
                print("Error: dev_Q_A.json was not found in the tar.gz file")
                return
            f = tar.extractfile(dev_member)
            qa_data = json.load(f)
    except Exception as e:
        print(f"Reading error: {e}")
        return

    correct_predictions = 0
    total_tested        = 0

    # Για το diagnostic summary
    position_histogram = {
        "top3":       0,   # 0-2
        "top4_10":    0,   # 3-9
        "top11_25":   0,
        "top26_50":   0,
        "top51_100":  0,
        "missing":    0,
    }
    positions_detailed = []  # list of (qid, gt_id, position)

    mode_label = "DIAGNOSTIC" if DIAGNOSTIC_MODE else "NORMAL"
    request_n  = DIAG_POOL if DIAGNOSTIC_MODE else TOP_N

    print(f"\n[{mode_label} MODE] Simulation for {limit} tickets | "
          f"requesting top-{request_n}, scoring top-{TOP_N}")
    print("=" * 70)

    for item in qa_data:
        if item.get('ANSWERABLE') != 'Y':
            continue
        if total_tested >= limit:
            break

        raw_text      = item.get('QUESTION_TEXT')
        raw_title     = item.get('QUESTION_TITLE')
        question_text = str(raw_text or raw_title or "No description provided")
        gt_doc_id     = item.get('DOCUMENT')
        qid           = item.get('QUESTION_ID', '?')

        payload = json.dumps({
            "issue_description": question_text,
            "n_results":         request_n,
        }).encode('utf-8')

        req = urllib.request.Request(
            API_URL,
            data=payload,
            headers={'Content-Type': 'application/json'}
        )

        try:
            response = urllib.request.urlopen(req)
            result   = json.loads(response.read().decode('utf-8'))

            suggested_ids = [
                sol.get('technote_id')
                for sol in result.get('suggested_solutions', [])
            ]

            # Top-N accuracy (πρώτες TOP_N θέσεις)
            top_n_ids  = suggested_ids[:TOP_N]
            is_correct = gt_doc_id in top_n_ids

            # Ακριβής θέση του GT στο συνολικό pool
            try:
                gt_position = suggested_ids.index(gt_doc_id)
            except ValueError:
                gt_position = None

            if is_correct:
                correct_predictions += 1
                status = f"SUCCESS (found at #{gt_position + 1})"
            else:
                status = f"FAILURE - {classify_position(gt_position)}"

            # Ενημέρωση histogram
            if gt_position is None:
                position_histogram["missing"] += 1
            elif gt_position < 3:
                position_histogram["top3"] += 1
            elif gt_position < 10:
                position_histogram["top4_10"] += 1
            elif gt_position < 25:
                position_histogram["top11_25"] += 1
            elif gt_position < 50:
                position_histogram["top26_50"] += 1
            else:
                position_histogram["top51_100"] += 1

            positions_detailed.append((qid, gt_doc_id, gt_position))

            # Εκτύπωση
            short_q = question_text.strip().replace('\n', ' ')[:1200]
            print(f"\n[{qid}] {short_q}...")
            print(f"  Ground Truth: {gt_doc_id}")
            print(f"  Top-{TOP_N} from AI: {', '.join(top_n_ids)}")
            if DIAGNOSTIC_MODE and gt_position is not None and gt_position >= TOP_N:
                # Δείξε τη θέση του GT και τα "νικητές" που το έκρυψαν
                above = suggested_ids[:gt_position]
                print(f"  GT position in top-{DIAG_POOL}: #{gt_position + 1}")
                print(f"  Beaten by: {', '.join(above[:5])}"
                      + ("..." if len(above) > 5 else ""))
            print(f"  Result: {status}")
            print("-" * 70)

            total_tested += 1

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"API HTTP Error {e.code}: {error_body}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    # -- Τελικά stats ----------------------------------
    if total_tested == 0:
        return

    accuracy = (correct_predictions / total_tested) * 100
    print("\n" + "=" * 70)
    print(f"TOP-{TOP_N} ACCURACY RESULTS")
    print("=" * 70)
    print(f"  Tickets tested:      {total_tested}")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Accuracy:            {accuracy:.2f}%")

    if DIAGNOSTIC_MODE:
        print("\n" + "=" * 70)
        print(f"GROUND TRUTH POSITION HISTOGRAM (in top-{DIAG_POOL})")
        print("=" * 70)
        labels = [
            ("top3", f"Position 1-3   (SUCCESS)          "),
            ("top4_10", f"Position 4-10  (near-miss)        "),
            ("top11_25", f"Position 11-25 (reranker tunable) "),
            ("top26_50", f"Position 26-50 (reranker bad)     "),
            ("top51_100", f"Position 51-100(retrieval weak)   "),
            ("missing", f"Not in top-{DIAG_POOL} (retrieval broken) "),
        ]
        for key, label in labels:
            count = position_histogram[key]
            pct = (count / total_tested) * 100
            print(f"  {label} | {count:3d} ({pct:5.1f}%)")

        print("\n" + "=" * 70)

if __name__ == "__main__":
    limit = 30
    run_simulation(limit)