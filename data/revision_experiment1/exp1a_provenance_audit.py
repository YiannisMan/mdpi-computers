#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 1Α — Data provenance audit.

ΤΙ ΠΑΡΑΓΕΙ:
  data/provenance.csv                 — Σύνολο | Αρχείο | n | in_train | in_dev | ans/unans | Χρήση | Τομή με TEST
  data/provenance_intersections.csv   — τομές ανά ζεύγος συνόλων
  data/provenance_report.txt          — κοινά gold docs, διπλότυπα, κανόνες απόφασης

ΕΚΤΕΛΕΣΗ:
  python exp1a_provenance_audit.py
"""
import json, itertools, hashlib, re, csv, os, sys, tarfile, random

# ─────────────────────────────────────────────────────────────────────────
#  ΡΥΘΜΙΣΕΙΣ — προσαρμόστε ΜΟΝΟ τα paths στο δικό σας σύστημα
# ─────────────────────────────────────────────────────────────────────────
TAR_PATH   = os.getenv("TECHQA_TAR", "../TechQA.tar.gz")   # .tar ή .tar.gz (auto-detect)
OUT_DIR    = "data"

# Πεδία TechQA (επιβεβαιωμένα από ingest.py / build_sft_dataset.py)
F_QID   = "QUESTION_ID"
F_TITLE = "QUESTION_TITLE"
F_TEXT  = "QUESTION_TEXT"
F_DOC   = "DOCUMENT"
F_ANS   = "ANSWERABLE"

SETS = {
    # SFT: από το sft_dataset_meta.csv (όλα τα 450, και το 90/10 split με seed=42 από το QLoRA notebook)
    "sft_all":        ("sft_meta",       os.getenv("SFT_META", "../sft_dataset_meta.csv")),
    "sft_train":      ("sft_split",      (os.getenv("SFT_META", "../sft_dataset_meta.csv"), "train")),
    "sft_val":        ("sft_split",      (os.getenv("SFT_META", "../sft_dataset_meta.csv"), "val")),
    # Reranker train/val ids — αναπαραγωγή του pair-level 90/10 split (608 ερωτήσεις × 5 ζεύγη = 3040)
    "reranker_train": ("reranker_split", "train"),
    "reranker_val":   ("reranker_split", "val"),
    # Stage A/B/C ids από τα ΠΑΛΙΑ logs/maps:
    "stageA":         ("csv_field",      (os.getenv("STAGEA_CSV",  "../eval_results.csv"),                   "qid")),
    "stageB":         ("csv_field",      (os.getenv("STAGEB_CSV",  "../generation_eval_results.csv"),        "qid")),
    "stageC_ans":     ("csv_field",      (os.getenv("STAGEC_MAP",  "../jira_ticket_map.csv"),                "qid")),
    "stageC_unans":   ("csv_field",      (os.getenv("STAGEC_MAP",  "../jira_ticket_map.csv"),                "qid")),
    # 30-query engineering subset:
    "subset30":       ("csv_field",      (os.getenv("SUBSET30_CSV", "../generation_eval_results_limit30.csv"), "qid")),
}

STAGEC_MAP_ANS_COL = "answerable"


# ─────────────────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────────────────
def _open_tar(path):
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    return tarfile.open(path, "r:*")

def load_qa_list(member_endswith):
    """Επιστρέφει τη λίστα records (με την αρχική σειρά) για ένα μέλος του tar."""
    with _open_tar(TAR_PATH) as tar:
        m = next((x for x in tar.getmembers() if x.name.endswith(member_endswith)), None)
        if not m:
            raise FileNotFoundError(f"{member_endswith} δεν βρέθηκε στο {TAR_PATH}")
        return json.load(tar.extractfile(m))

def load_qa_file(member_endswith):
    """Επιστρέφει dict qid -> record για ένα μέλος του tar."""
    data = load_qa_list(member_endswith)
    return {r[F_QID]: r for r in data}

def _sft_split_ids(meta_path, which):
    """Αναπαράγει το HuggingFace Dataset.train_test_split(test_size=0.10, seed=42) του QLoRA notebook."""
    if not meta_path or not os.path.exists(meta_path):
        return None
    with open(meta_path, newline="", encoding="utf-8") as f:
        qids = [row["qid"] for row in csv.DictReader(f) if row.get("qid")]
    n = len(qids)
    n_test = int(n * 0.10)
    n_train = n - n_test
    try:
        import numpy as np
        indices = np.random.default_rng(42).permutation(n).tolist()
    except ImportError:
        rng = random.Random(42)
        indices = list(range(n))
        rng.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]
    chosen = train_idx if which == "train" else val_idx
    return {qids[i] for i in chosen}

def _reranker_split_ids(which, train_list, dev_list):
    """
    Αναπαράγει τη λογική του techqa-reranker-v2-finetune.ipynb:
    - Ενώνει train_data + dev_data
    - Κρατάει answerable=='Y' με valid DOCUMENT και μη-κενό ANSWER
    - Φτιάχνει 5 ζεύγη ανά ερώτηση (1 pos + 4 neg) -> 3040 ζεύγη
    - Κάνει random.shuffle και κόβει 90/10 ανά ζεύγος (2736 train / 304 val).
    """
    all_qa = [
        item for item in (train_list + dev_list)
        if item.get(F_ANS) == "Y"
        and item.get(F_DOC) and item.get(F_DOC) != "-"
        and item.get("ANSWER") and str(item.get("ANSWER")).strip()
        and str(item.get(F_TEXT) or item.get(F_TITLE) or "").strip()
    ]
    # Προσομοίωση του random.seed(42) + 4 random.sample κλήσεις ανά ερώτηση + random.shuffle
    rng = random.Random(42)
    pair_qids = []
    dummy_pool = list(range(1000))
    for item in all_qa:
        qid = item[F_QID]
        pair_qids.append(qid)          # 1 positive pair
        rng.sample(dummy_pool, 4)      # καταναλώνει την ίδια εντροπία στο PRNG
        for _ in range(4):
            pair_qids.append(qid)      # 4 negative pairs
    rng.shuffle(pair_qids)
    split_idx = int(len(pair_qids) * 0.9)
    if which == "train":
        return set(pair_qids[:split_idx])
    return set(pair_qids[split_idx:])

def ids_from(spec, train_list=None, dev_list=None):
    """Επιστρέφει set από question-ids για ένα SETS entry. '' / None -> None (n/a)."""
    kind, arg = spec
    try:
        if kind == "sft_meta":
            if not arg or not os.path.exists(arg): return None
            with open(arg, newline="", encoding="utf-8") as f:
                return {row["qid"] for row in csv.DictReader(f) if row.get("qid")}
        if kind == "sft_split":
            meta_path, which = arg
            return _sft_split_ids(meta_path, which)
        if kind == "reranker_split":
            return _reranker_split_ids(arg, train_list, dev_list)
        if kind == "jsonl_field":
            path, field = arg
            if not path or not os.path.exists(path): return None
            out = set()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    rec = json.loads(line)
                    v = rec.get(field)
                    if v: out.add(v)
            return out
        if kind == "csv_field":
            path, col = arg
            if not path or not os.path.exists(path): return None
            with open(path, newline="", encoding="utf-8") as f:
                return {row[col] for row in csv.DictReader(f) if row.get(col)}
        if kind == "json_ids":
            path, field = arg
            if not path or not os.path.exists(path): return None
            data = json.load(open(path, encoding="utf-8"))
            if field is None:
                return set(data)
            return {r[field] for r in data if r.get(field)}
        if kind == "txt_lines":
            if not arg or not os.path.exists(arg): return None
            return {l.strip() for l in open(arg, encoding="utf-8") if l.strip()}
        if kind == "tar_qa":
            member_endswith, ans = arg
            d = load_qa_file(member_endswith)
            if ans is None:
                return set(d.keys())
            return {q for q, r in d.items() if r.get(F_ANS) == ans}
    except Exception as e:
        print(f"  [WARN] loader {kind} απέτυχε ({arg}): {e}", file=sys.stderr)
        return None
    return None

def load_stageC_split(map_csv):
    """Επιστρέφει (ans_ids, unans_ids) από το jira map, με στήλη answerable Y/N."""
    if not map_csv or not os.path.exists(map_csv):
        return None, None
    ans, unans = set(), set()
    with open(map_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row.get("qid")
            if not qid: continue
            if str(row.get(STAGEC_MAP_ANS_COL, "")).strip().upper().startswith("Y"):
                ans.add(qid)
            else:
                unans.add(qid)
    return ans, unans


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Φόρτωση training_Q_A.json και dev_Q_A.json από το tar...")
    train_list = load_qa_list("training_Q_A.json")
    dev_list   = load_qa_list("dev_Q_A.json")
    train = {r[F_QID]: r for r in train_list}
    dev   = {r[F_QID]: r for r in dev_list}
    train_ids, dev_ids = set(train), set(dev)
    print(f"  TRAIN: {len(train_ids)} ερωτήσεις | DEV(TEST): {len(dev_ids)} ερωτήσεις")

    def ans_counts(ids_in_file, src):
        ans = sum(1 for q in ids_in_file if src.get(q, {}).get(F_ANS) == "Y")
        return ans, len(ids_in_file) - ans

    ta, tu = ans_counts(train_ids, train)
    da, du = ans_counts(dev_ids, dev)
    print(f"  TRAIN answerable/unanswerable: {ta}/{tu}")
    print(f"  DEV   answerable/unanswerable: {da}/{du}")

    # Υλοποίηση των συνόλων
    sets = {}
    for name, spec in SETS.items():
        if name in ("stageC_ans", "stageC_unans"):
            continue
        sets[name] = ids_from(spec, train_list=train_list, dev_list=dev_list)

    # Stage C answerable/unanswerable από το jira map
    scmap = SETS["stageC_ans"][1][0] if isinstance(SETS["stageC_ans"][1], tuple) else None
    sc_ans, sc_unans = load_stageC_split(scmap)
    sets["stageC_ans"]   = sc_ans
    sets["stageC_unans"] = sc_unans

    # ── provenance.csv ──────────────────────────────────────────────────
    USE = {
        "sft_all":        "SFT all examples (TRAIN only)",
        "sft_train":      "SFT train split (90%)",
        "sft_val":        "SFT validation split (10%)",
        "reranker_train": "Reranker training (pair-level 90% split over train+dev)",
        "reranker_val":   "Reranker validation (pair-level 10% split over train+dev)",
        "stageA":         "Stage A retrieval eval",
        "stageB":         "Stage B generation eval",
        "stageC_ans":     "Stage C (answerable)",
        "stageC_unans":   "Stage C (unanswerable)",
        "subset30":       "30-query engineering subset",
    }
    rows = []
    for name in SETS:
        s = sets.get(name)
        if s is None:
            rows.append({
                "set": name,
                "source_file": "n/a (δώστε path)",
                "n": 0,
                "in_train": "",
                "in_dev": "",
                "answerable": "",
                "unanswerable": "",
                "use": USE.get(name, ""),
                "intersection_with_TEST": ""
            })
            continue
        in_tr = len(s & train_ids)
        in_dev = len(s & dev_ids)
        in_neither = len(s - train_ids - dev_ids)
        src_file = ("training_Q_A.json" if in_dev == 0 and in_tr > 0
                    else "dev_Q_A.json" if in_tr == 0 and in_dev > 0
                    else "training_Q_A.json + dev_Q_A.json" if in_tr > 0 and in_dev > 0
                    else "mixed/other")
        a = sum(1 for q in s if train.get(q, dev.get(q, {})).get(F_ANS) == "Y")
        rows.append({
            "set": name,
            "source_file": src_file,
            "n": len(s),
            "in_train": in_tr,
            "in_dev": in_dev,
            "answerable": a,
            "unanswerable": len(s) - a,
            "use": USE.get(name, ""),
            "intersection_with_TEST": in_dev,
        })
        extra = f" (in_train={in_tr}, in_dev={in_dev}, in_neither={in_neither})"
        print(f"  {name:16s} n={len(s):4d}{extra}")

    with open(f"{OUT_DIR}/provenance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "set", "source_file", "n", "in_train", "in_dev",
            "answerable", "unanswerable", "use", "intersection_with_TEST"
        ])
        w.writeheader()
        w.writerows(rows)

    # ── intersections ανά ζεύγος ────────────────────────────────────────
    present = {k: v for k, v in sets.items() if v is not None}
    inter_rows = [(a, b, len(present[a] & present[b]))
                  for a, b in itertools.combinations(present, 2)]
    with open(f"{OUT_DIR}/provenance_intersections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["A", "B", "intersection"]); w.writerows(inter_rows)

    # ── report-only: κοινά gold docs & διπλότυπα κείμενα TRAIN↔TEST ──────
    def norm(r):
        s = re.sub(r"\W+", " ", ((r.get(F_TITLE) or "") + (r.get(F_TEXT) or "")).lower())
        return hashlib.sha1(s.encode()).hexdigest()
    gold = lambda d: {r[F_DOC] for r in d.values() if r.get(F_DOC) and r[F_DOC] != "-"}
    shared_gold = gold(train) & gold(dev)
    dup_q = {norm(r) for r in train.values()} & {norm(r) for r in dev.values()}

    # Κανόνες απόφασης
    def nz(x): return x is not None
    rk_tr = sets.get("reranker_train"); rk_va = sets.get("reranker_val")
    sft_tr = sets.get("sft_train"); sft_va = sets.get("sft_val")
    sft_all = sets.get("sft_all")
    decisions = []
    if nz(rk_tr):
        decisions.append(("reranker_train ∩ TEST",
                          len(rk_tr & dev_ids),
                          "≠0 → το 1Β είναι ΥΠΟΧΡΕΩΤΙΚΟ (αναμενόμενο, pair-level split σε train+dev)"))
    if nz(rk_tr) and nz(rk_va):
        decisions.append(("reranker_train ∩ reranker_val",
                          len(rk_tr & rk_va),
                          "≠0 → το 90/10 split(2736/304) έγινε ανά ζεύγος (pair-level), όχι ανά ερώτηση"))
    sft_check = sft_all if (sft_tr is None and sft_va is None) else ((sft_tr or set()) | (sft_va or set()))
    if nz(sft_check):
        n = len(sft_check & dev_ids)
        decisions.append(("SFT ∩ TEST", n,
                          "=0 → ο QLoRA adapter επαναχρησιμοποιείται αυτούσιος"
                          if n == 0 else
                          "≠0 → ΕΠΑΝΕΚΠΑΙΔΕΥΣΗ QLoRA μόνο σε TRAIN (seed 42), νέο sha256"))
    if nz(sc_unans) and nz(sft_all):
        decisions.append(("stageC_unans ∩ SFT", len(sc_unans & sft_all),
                          "=0 αν δεν επικαλύπτονται, αλλιώς δηλώνεται (το νέο Stage C τρέχει στο TEST)"))

    with open(f"{OUT_DIR}/provenance_report.txt", "w", encoding="utf-8") as f:
        f.write("PROVENANCE REPORT (Πείραμα 1Α)\n" + "=" * 60 + "\n")
        f.write(f"TRAIN n={len(train_ids)} (ans {ta}/unans {tu})\n")
        f.write(f"TEST  n={len(dev_ids)} (ans {da}/unans {du})\n\n")
        f.write(f"shared gold docs TRAIN<->TEST : {len(shared_gold)} "
                f"(φυσικό στο TechQA, ΔΕΝ αφαιρούνται, δηλώνονται στο κείμενο)\n")
        f.write(f"duplicate questions TRAIN<->TEST: {len(dup_q)}\n\n")
        f.write("ΚΑΝΟΝΕΣ ΑΠΟΦΑΣΗΣ:\n")
        for label, n, action in decisions:
            f.write(f"  {label:28s} = {n:4d}  -> {action}\n")
        f.write("\nΠΡΟΣΟΧΗ: στο νέο άρθρο η στήλη intersection_with_TEST στο "
                "provenance.csv ΠΡΕΠΕΙ να είναι 0 σε κάθε γραμμή που μπαίνει στην "
                "αξιολόγηση (reranker_train είναι η μόνη αναμενόμενη εξαίρεση, γι' "
                "αυτό γίνεται το 1Β).\n")

    print("\n" + "=" * 60)
    print("ΟΛΟΚΛΗΡΩΘΗΚΕ 1Α. Παραδοτέα:")
    print(f"  {OUT_DIR}/provenance.csv")
    print(f"  {OUT_DIR}/provenance_intersections.csv")
    print(f"  {OUT_DIR}/provenance_report.txt")
    print(f"\n  shared gold docs TRAIN<->TEST: {len(shared_gold)} | duplicate questions: {len(dup_q)}")
    for label, n, action in decisions:
        print(f"  {label} = {n} -> {action}")


if __name__ == "__main__":
    main()