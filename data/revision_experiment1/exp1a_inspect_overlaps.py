#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Α (συμπλήρωμα) — Επιθεώρηση των επικαλύψεων TRAIN↔TEST.

Το exp1a_provenance_audit.py ανέφερε ΜΟΝΟ πλήθη:
  - shared gold docs TRAIN<->TEST : 40
  - duplicate questions TRAIN<->TEST: 22
Εδώ τα ΑΠΑΡΙΘΜΟΥΜΕ ρητά ώστε να κριθεί αν είναι πραγματική διαρροή ή φυσικό χαρακτηριστικό
του TechQA — και να γραφτεί τεκμηριωμένα στην απάντηση προς τον R1.

ΠΑΡΑΓΕΙ:
  data/dup_questions.csv      — ζεύγη (train_qid, dev_qid) με ΠΑΝΟΜΟΙΟΤΥΠΟ κανονικοποιημένο
                                κείμενο ερώτησης (τίτλος+σώμα). Αυτά ΧΡΗΖΟΥΝ απόφασης.
  data/shared_gold_docs.csv   — gold technotes που εμφανίζονται ΚΑΙ σε TRAIN ΚΑΙ σε TEST,
                                με ποιες ερωτήσεις τα δείχνουν (φυσικό στο TechQA, δηλώνεται).

ΕΚΤΕΛΕΣΗ (ίδιο περιβάλλον με το 1Α):
  python exp1a_inspect_overlaps.py
"""
import tarfile, json, csv, re, hashlib, os, sys

TAR_PATH = os.getenv("TECHQA_TAR", "/data/TechQA.tar")
OUT_DIR  = "data"
F_QID, F_TITLE, F_TEXT, F_DOC, F_ANS = \
    "QUESTION_ID", "QUESTION_TITLE", "QUESTION_TEXT", "DOCUMENT", "ANSWERABLE"


def load_qa(member):
    with tarfile.open(TAR_PATH, "r:*") as tar:
        m = next((x for x in tar.getmembers() if x.name.endswith(member)), None)
        if not m:
            raise FileNotFoundError(member)
        return json.load(tar.extractfile(m))


def norm(r):
    s = re.sub(r"\W+", " ", ((r.get(F_TITLE) or "") + " " + (r.get(F_TEXT) or "")).lower()).strip()
    return hashlib.sha1(s.encode()).hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = load_qa("training_Q_A.json")
    dev   = load_qa("dev_Q_A.json")

    # ── 1) Διπλότυπες ερωτήσεις (πανομοιότυπο κανονικοποιημένο κείμενο) ──
    train_by_hash = {}
    for r in train:
        train_by_hash.setdefault(norm(r), []).append(r)

    dup_rows = []
    for r in dev:
        h = norm(r)
        if h in train_by_hash:
            for tr in train_by_hash[h]:
                dup_rows.append({
                    "dev_qid": r.get(F_QID), "dev_ans": r.get(F_ANS),
                    "train_qid": tr.get(F_QID), "train_ans": tr.get(F_ANS),
                    "same_gold": int((r.get(F_DOC) or "") == (tr.get(F_DOC) or "")),
                    "dev_gold": r.get(F_DOC), "train_gold": tr.get(F_DOC),
                    "title": (r.get(F_TITLE) or "")[:120],
                    "text_preview": re.sub(r"\s+", " ", (r.get(F_TEXT) or ""))[:200],
                })

    with open(f"{OUT_DIR}/dup_questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dev_qid", "dev_ans", "train_qid", "train_ans",
                                          "same_gold", "dev_gold", "train_gold",
                                          "title", "text_preview"])
        w.writeheader(); w.writerows(dup_rows)

    n_dev_dup = len({r["dev_qid"] for r in dup_rows})
    n_dev_dup_ans = len({r["dev_qid"] for r in dup_rows if r["dev_ans"] == "Y"})
    print(f"Διπλότυπες ερωτήσεις: {len(dup_rows)} ζεύγη | {n_dev_dup} μοναδικές ΤEST ερωτήσεις "
          f"(εκ των οποίων answerable: {n_dev_dup_ans})")
    print(f"  -> {OUT_DIR}/dup_questions.csv")

    # Λίστα αποκλεισμού για το 1Γ (ένα dev_qid ανά γραμμή) — το exp1c_stageA_freeze τη διαβάζει.
    dup_dev_ids_sorted = sorted({r["dev_qid"] for r in dup_rows})
    with open(f"{OUT_DIR}/test_exclude_qids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dup_dev_ids_sorted) + ("\n" if dup_dev_ids_sorted else ""))
    print(f"  -> {OUT_DIR}/test_exclude_qids.txt ({n_dev_dup} qids — ΜΟΝΟ για report + "
          f"προαιρετική ευαισθησία· το κύριο 1Γ τρέχει στις 310, με --exclude-dups στις 287)")

    # ── 2) Κοινά gold technotes ──
    def gold_map(data):
        m = {}
        for r in data:
            g = r.get(F_DOC)
            if g and g != "-":
                m.setdefault(g, []).append(r.get(F_QID))
        return m
    gt_tr, gt_dev = gold_map(train), gold_map(dev)
    shared = sorted(set(gt_tr) & set(gt_dev))

    with open(f"{OUT_DIR}/shared_gold_docs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gold_id", "n_train_q", "n_dev_q", "train_qids", "dev_qids"])
        for g in shared:
            w.writerow([g, len(gt_tr[g]), len(gt_dev[g]),
                        ";".join(gt_tr[g]), ";".join(gt_dev[g])])
    print(f"Κοινά gold technotes: {len(shared)}")
    print(f"  -> {OUT_DIR}/shared_gold_docs.csv")

    # ── Καθαρό (leakage-free) TEST χωρίς τις διπλότυπες — μέγεθος για ευαισθησία ──
    dup_dev_ids = {r["dev_qid"] for r in dup_rows}
    dev_ans = [r for r in dev if r.get(F_ANS) == "Y" and r.get(F_DOC)]
    clean_total = len(dev) - len(dup_dev_ids)
    clean_ans = len([r for r in dev_ans if r.get(F_QID) not in dup_dev_ids])
    print("\n── Μεγέθη TEST ──")
    print(f"  ΚΥΡΙΟ TEST (όλες): {len(dev)} (answerable {len(dev_ans)})")
    print(f"  Ευαισθησία χωρίς διπλότυπες: {clean_total} (answerable {clean_ans})")
    print("\nΑΠΟΦΑΣΗ (οδηγία reviewer — ANALYSIS_PLAN §1.1):")
    print("  - Report-only: οι 23 διπλότυπες (και τα 40 κοινά gold) ΔΗΛΩΝΟΝΤΑΙ, ΔΕΝ αφαιρούνται.")
    print("  - Κύρια αποτελέσματα στις 310· ευαισθησία στις 287 με exp1c --exclude-dups.")
    with open(f"{OUT_DIR}/dup_questions_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(f"dup pairs={len(dup_rows)}  unique dev dups={n_dev_dup} (answerable {n_dev_dup_ans})\n")
        f.write(f"shared gold technotes={len(shared)}\n")
        f.write(f"TEST full=310 (ans {len(dev_ans)}) | TEST χωρίς διπλότυπες={clean_total} (ans {clean_ans})\n")


if __name__ == "__main__":
    main()
