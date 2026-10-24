import tarfile
import json
import csv
import math
import time
import re
import urllib.request

"""
ΡΟΛΟΣ (Φάση 2):
Αντικειμενική αξιολόγηση του RAG layer (/resolve) + συλλογή δεδομένων για
retrieval-confidence GATING. Το αντίστοιχο του evaluate_full.py για το
ΣΤΑΔΙΟ ΠΑΡΑΓΩΓΗΣ. Sweep του K και μετράει:

  1. Retrieval recall@K  : μπήκε το Ground Truth στα K sources;
  2. Abstention          : σωστό abstain (GT missing) vs false abstain (GT present)
                           vs answer-anyway-when-GT-missing (το DEV_Q000 hazard).
  3. Faithfulness        : cited technote IDs ⊆ sources (καμία εφεύρεση ID).
  4. Correctness (ROUGE-L): vs gold ANSWER, ΧΩΡΙΣΤΑ GT-in vs GT-not-in context.
  5. Retrieval confidence: το max ΩΜΟ rerank logit ανά ticket — το σήμα για το gate.

ΓΙΑΤΙ ΩΜΟ rerank logit (κι όχι final_score): το final_score κανονικοποιείται
minmax ΑΝΑ query, άρα το top doc παίρνει ~1.0 ανεξάρτητα απ' το αν το retrieval
είναι όντως καλό -> ΑΧΡΗΣΤΟ για cross-query threshold. Το ωμό CrossEncoder logit
έχει συνεπή κλίμακα μεταξύ queries -> κατάλληλο για gate (abstain αν max_rerank < T).

Η ιδέα του gate: αντί να εμπιστευόμαστε την (θορυβώδη) κρίση του LLM για το πότε
να κάνει abstain, κόβουμε ΠΡΙΝ το generation με βάση το retrieval confidence που
ΗΔΗ παράγει η Φάση 1. Αυτό το script συλλέγει τα scores ώστε να βρεθεί το threshold.

ΣΗΜΕΙΩΣΗ ROUGE-L: απλό LCS-F1 (ντετερμινιστικό, μηδέν εξαρτήσεις). Διάβασέ το ως
ΣΧΕΤΙΚΗ μετρική. Faithfulness/abstention είναι ακριβή.

ΕΚΤΕΛΕΣΗ (μέσα στο container):
  docker exec -it knowledge_api python /app/evaluate_generation.py
"""

# ── Config ────────────────────────────────────────────────────────────
API_RESOLVE_URL   = "http://localhost:8080/resolve"
TAR_PATH          = "/data/TechQA.tar.gz"
K_VALUES          = [5]   # FULL THESIS RUN: βάλε [5] (ή [3, 5]) για ταχύτητα
LIMIT             = None           # FULL THESIS RUN: βάλε None (= ολο το dev set, 160)
GATE_K            = 5            # ποιο K να αναλυθεί για το gating threshold
CSV_OUT           = "/data/generation_eval_results.csv"
REQUEST_TIMEOUT   = 360

# Πρεπει να ταιριαζει με το ABSTAIN_PHRASE του main.py (substring match).
ABSTAIN_SUBSTRING = "does not contain a confident solution"

# Technote-ID shape (swg21410143, nas8N1022458...). Χωρις \b — lookarounds.
ID_PATTERN   = re.compile("(?<![A-Za-z0-9])[A-Za-z]{2,4}[0-9][A-Za-z0-9]{4,}(?![A-Za-z0-9])")
WORD_PATTERN = re.compile("[A-Za-z0-9]+")


def wilson_ci(correct, total, z=1.96):
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def lcs_length(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if x == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[len(b)]


def rouge_l_f1(candidate, reference):
    cand = WORD_PATTERN.findall((candidate or "").lower())
    ref  = WORD_PATTERN.findall((reference or "").lower())
    if not cand or not ref:
        return 0.0
    l = lcs_length(cand, ref)
    if l == 0:
        return 0.0
    prec = l / len(cand)
    rec  = l / len(ref)
    return 2 * prec * rec / (prec + rec)


def extract_technote_ids(text):
    return {m.lower() for m in ID_PATTERN.findall(text or "")}


def is_abstention(text):
    return ABSTAIN_SUBSTRING in (text or "").lower()


def pct(x):
    return f"{x * 100:.1f}%"


def score_stats(values):
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return {"n": n, "mean": sum(xs) / n, "median": med, "min": xs[0], "max": xs[-1]}


def resolve(ticket, k):
    payload = json.dumps({"issue_description": ticket, "n_docs": k}).encode("utf-8")
    req = urllib.request.Request(
        API_RESOLVE_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def load_answerable(limit):
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.endswith("dev_Q_A.json")), None
        )
        if not member:
            raise FileNotFoundError("dev_Q_A.json not found in tar")
        qa = json.load(tar.extractfile(member))
    items = [q for q in qa if q.get("ANSWERABLE") == "Y" and q.get("DOCUMENT")]
    if limit:
        items = items[:limit]
    return items


def evaluate_for_k(answerable, k):
    rows = []
    n = len(answerable)

    gt_in = abstain = answered = 0
    n_pos = n_neg = 0
    pos_abstain = neg_abstain = 0
    faithful = invented = cited_any = 0
    rouge_all, rouge_pos, rouge_neg = [], [], []
    gate_pairs = []   # (max_rerank, is_in) — για separation + threshold sweep

    start = time.time()
    for idx, item in enumerate(answerable):
        ticket = str(item.get("QUESTION_TEXT") or item.get("QUESTION_TITLE") or "")
        gt_id  = item.get("DOCUMENT")
        gold   = item.get("ANSWER") or ""
        qid    = item.get("QUESTION_ID", "?")

        try:
            res = resolve(ticket, k)
        except Exception as e:
            print(f"  [{qid}] REQUEST FAILED: {e}", flush=True)
            rows.append({"qid": qid, "k": k, "gt_id": gt_id, "gt_in_context": 0,
                         "gt_position": -1, "abstained": "", "faithful": "",
                         "cited_valid": 0, "n_invented": 0, "rouge_l": "",
                         "top_final_score": "", "top_rerank_score": "",
                         "max_rerank_score": "", "gt_rerank_score": "",
                         "gt_final_score": ""})
            continue

        answer     = res.get("answer", "") or ""
        sources    = res.get("sources", []) or []
        source_ids = [s.get("technote_id") for s in sources]
        source_set = {str(s).lower() for s in source_ids if s}

        is_in  = gt_id in source_ids
        gt_pos = (source_ids.index(gt_id) + 1) if is_in else -1
        ab     = is_abstention(answer)

        cited         = extract_technote_ids(answer)
        cited_valid   = cited & source_set
        cited_invalid = cited - source_set
        no_invented   = (len(cited_invalid) == 0)
        rl            = None if ab else rouge_l_f1(answer, gold)

        # ── Retrieval-confidence scores ──
        # Υπολογιζονται ΑΠΕΥΘΕΙΑΣ απο το sources array, ωστε να δουλευουν
        # ανεξαρτητα απο το αν το deployed API εκθετει top-level πεδια.
        rerank_vals = [s.get("rerank_score") for s in sources
                       if s.get("rerank_score") is not None]
        max_rerank = max(rerank_vals) if rerank_vals else res.get("max_rerank_score")
        top_rerank = sources[0].get("rerank_score") if sources else None
        top_final  = sources[0].get("final_score") if sources else res.get("top_final_score")
        gt_rerank = gt_final = None
        if is_in:
            for s in sources:
                if s.get("technote_id") == gt_id:
                    gt_rerank = s.get("rerank_score")
                    gt_final  = s.get("final_score")
                    break
        if max_rerank is not None:
            gate_pairs.append((max_rerank, is_in))

        if is_in:
            n_pos += 1
            gt_in += 1
        else:
            n_neg += 1

        if ab:
            abstain += 1
            if is_in:
                pos_abstain += 1
            else:
                neg_abstain += 1
        else:
            answered += 1
            if no_invented:
                faithful += 1
            if len(cited_invalid) > 0:
                invented += 1
            if len(cited_valid) >= 1:
                cited_any += 1
            rouge_all.append(rl)
            (rouge_pos if is_in else rouge_neg).append(rl)

        def r4(x):
            return "" if x is None else round(x, 4)

        rows.append({
            "qid": qid, "k": k, "gt_id": gt_id,
            "gt_in_context": 1 if is_in else 0,
            "gt_position": gt_pos,
            "abstained": 1 if ab else 0,
            "faithful": ("" if ab else (1 if no_invented else 0)),
            "cited_valid": len(cited_valid),
            "n_invented": len(cited_invalid),
            "rouge_l": ("" if rl is None else round(rl, 4)),
            "top_final_score":  r4(top_final),
            "top_rerank_score": r4(top_rerank),
            "max_rerank_score": r4(max_rerank),
            "gt_rerank_score":  r4(gt_rerank),
            "gt_final_score":   r4(gt_final),
        })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n - idx - 1) / rate if rate > 0 else 0
            print(f"  [K={k}] {idx + 1}/{n} | GT-in-ctx: {gt_in}/{idx + 1} | "
                  f"ETA {eta / 60:.1f} min", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "k": k, "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "gt_in": gt_in, "abstain": abstain, "answered": answered,
        "faithful": faithful, "invented": invented, "cited_any": cited_any,
        "pos_abstain": pos_abstain, "neg_abstain": neg_abstain,
        "rouge_all": mean(rouge_all), "rouge_pos": mean(rouge_pos),
        "rouge_neg": mean(rouge_neg), "gate_pairs": gate_pairs,
    }
    return summary, rows


def print_k_summary(s):
    k, n = s["k"], s["n"]
    print()
    print(f"RESULTS for K={k}  (n={n}, GT-in-context={s['n_pos']}, GT-missing={s['n_neg']})")
    p, lo, hi = wilson_ci(s["gt_in"], n)
    print(f"  Retrieval recall@{k} (GT in context): {pct(p)}  (95% CI {pct(lo)}-{pct(hi)})")
    print(f"  Abstained overall: {s['abstain']}/{n} = {pct(s['abstain'] / n if n else 0)}")
    if s["n_neg"]:
        cab = s["neg_abstain"] / s["n_neg"]
        print(f"  Correct abstention (abstained when GT MISSING): "
              f"{s['neg_abstain']}/{s['n_neg']} = {pct(cab)}")
        print(f"    -> answered anyway when GT missing (DEV_Q000 hazard): "
              f"{s['n_neg'] - s['neg_abstain']}/{s['n_neg']} = {pct(1 - cab)}")
    if s["n_pos"]:
        print(f"  False abstention (abstained when GT PRESENT): "
              f"{s['pos_abstain']}/{s['n_pos']} = {pct(s['pos_abstain'] / s['n_pos'])}")
    if s["answered"]:
        pf, lof, hif = wilson_ci(s["faithful"], s["answered"])
        print(f"  Faithfulness — no invented IDs (answered only): "
              f"{s['faithful']}/{s['answered']} = {pct(pf)}  (95% CI {pct(lof)}-{pct(hif)})")
        print(f"  Cited >=1 real source technote: "
              f"{s['cited_any']}/{s['answered']} = {pct(s['cited_any'] / s['answered'])}")
        print(f"  Invented technote IDs (>=1 fake): "
              f"{s['invented']}/{s['answered']} = {pct(s['invented'] / s['answered'])}")
    print(f"  ROUGE-L vs gold (answered): {s['rouge_all']:.4f}  "
          f"| GT-in: {s['rouge_pos']:.4f}  | GT-out: {s['rouge_neg']:.4f}")

    # Retrieval-confidence separation (το σημα για το gate)
    pairs = s.get("gate_pairs", [])
    sp = score_stats([r for r, g in pairs if g])
    sn = score_stats([r for r, g in pairs if not g])
    print("  Retrieval confidence (max raw rerank logit) — GT-in vs GT-missing:")
    if sp:
        print(f"    GT-in    : n={sp['n']:>2} mean={sp['mean']:.2f} median={sp['median']:.2f} "
              f"range=[{sp['min']:.2f}, {sp['max']:.2f}]")
    if sn:
        print(f"    GT-miss  : n={sn['n']:>2} mean={sn['mean']:.2f} median={sn['median']:.2f} "
              f"range=[{sn['min']:.2f}, {sn['max']:.2f}]")


def print_comparison(summaries):
    print()
    print("=" * 70)
    print("COMPARISON ACROSS K")
    print("=" * 70)
    header = (f"{'K':>3} | {'GT@K':>6} | {'Abst':>6} | {'C-Abst':>7} | {'F-Abst':>7} | "
              f"{'Faith':>6} | {'Inv':>5} | {'ROUGE':>6} | {'R(in)':>6} | {'R(out)':>6}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        n = s["n"]
        gtk = s["gt_in"] / n if n else 0
        ab  = s["abstain"] / n if n else 0
        cab = s["neg_abstain"] / s["n_neg"] if s["n_neg"] else 0
        fab = s["pos_abstain"] / s["n_pos"] if s["n_pos"] else 0
        fa  = s["faithful"] / s["answered"] if s["answered"] else 0
        inv = s["invented"] / s["answered"] if s["answered"] else 0
        print(f"{s['k']:>3} | {pct(gtk):>6} | {pct(ab):>6} | {pct(cab):>7} | {pct(fab):>7} | "
              f"{pct(fa):>6} | {pct(inv):>5} | {s['rouge_all']:.3f} | "
              f"{s['rouge_pos']:.3f} | {s['rouge_neg']:.3f}")
    print()
    print("Legend:")
    print("  GT@K=recall@K | Abst=abstain rate | C-Abst=abstain|GT-missing (higher better)")
    print("  F-Abst=abstain|GT-present (lower better) | Faith/Inv over answered")
    print("  ROUGE / R(in) / R(out) = ROUGE-L vs gold (answered): all / GT-in / GT-out")


def print_gate_analysis(summaries):
    s = next((x for x in summaries if x["k"] == GATE_K), None)
    if not s:
        return
    pairs = [(r, g) for r, g in s.get("gate_pairs", []) if r is not None]
    pos = [r for r, g in pairs if g]        # GT-in    -> θελουμε ABOVE threshold
    neg = [r for r, g in pairs if not g]    # GT-miss  -> θελουμε BELOW threshold
    print()
    print("=" * 78)
    print(f"RETRIEVAL-CONFIDENCE GATE ANALYSIS  (K={GATE_K}, signal = max raw rerank logit)")
    print("=" * 78)
    if not pos or not neg:
        print("Not enough GT-in / GT-missing samples for a threshold sweep.")
        return
    print("Idea: abstain BEFORE generation when max_rerank < T.")
    print("Goal: T που στελνει τα GT-missing κατω (correct gate) χωρις να κοβει GT-in.")
    print()
    print("If GT-missing scores sit clearly below GT-in scores, the gate works.")
    print()
    allv = sorted(r for r, _ in pairs)
    lo, hi = allv[0], allv[-1]
    steps = 14
    header = (f"{'T':>7} | {'gate GT-miss':>14} | {'gate GT-in':>13} | "
              f"{'C-Abst':>7} | {'F-Abst':>7}")
    print(header)
    print("-" * len(header))
    for i in range(steps + 1):
        T = lo + (hi - lo) * i / steps
        cg = sum(1 for r in neg if r < T)   # correctly gated (abstain on GT-missing)
        wg = sum(1 for r in pos if r < T)   # wrongly gated (abstain on GT-present)
        print(f"{T:>7.2f} | {(str(cg) + '/' + str(len(neg))):>14} | "
              f"{(str(wg) + '/' + str(len(pos))):>13} | "
              f"{pct(cg / len(neg)):>7} | {pct(wg / len(pos)):>7}")
    print()
    print("Διαλεξε T οπου C-Abst ψηλο ΚΑΙ F-Abst χαμηλο. Αυτο γινεται το gate threshold.")
    print("(Με n=30 ειναι ενδεικτικο· τρεξε full run για σφιχτη εκτιμηση.)")


def write_csv(rows):
    fields = ["qid", "k", "gt_id", "gt_in_context", "gt_position",
              "abstained", "faithful", "cited_valid", "n_invented", "rouge_l",
              "top_final_score", "top_rerank_score", "max_rerank_score",
              "gt_rerank_score", "gt_final_score"]
    try:
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print()
        print(f"Per-ticket CSV written: {CSV_OUT}")
    except Exception as e:
        print(f"CSV write failed: {e}")


def main():
    print("=" * 70)
    print("GENERATION EVALUATION + GATE DATA — RAG layer (Phase 2 baseline)")
    print(f"Resolve endpoint: {API_RESOLVE_URL}")
    print(f"K sweep: {K_VALUES}   |   limit: {LIMIT if LIMIT else 'ALL (160)'}   |   GATE_K: {GATE_K}")
    print("=" * 70)

    answerable = load_answerable(LIMIT)
    total_gen = len(answerable) * len(K_VALUES)
    print(f"Loaded {len(answerable)} answerable tickets.")
    print(f"NOTE: {len(answerable)} tickets x {len(K_VALUES)} K-values = {total_gen} generations.")
    print(" Set LIMIT=None for full run.")

    summaries, all_rows = [], []
    for k in K_VALUES:
        print()
        print("-" * 70)
        print(f"Evaluating K={k} ...")
        print("-" * 70)
        s, rows = evaluate_for_k(answerable, k)
        summaries.append(s)
        all_rows.extend(rows)
        print_k_summary(s)

    print_comparison(summaries)
    print_gate_analysis(summaries)
    write_csv(all_rows)


if __name__ == "__main__":
    main()