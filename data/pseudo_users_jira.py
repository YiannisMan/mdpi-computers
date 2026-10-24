# -*- coding: utf-8 -*-
"""
ΦΑΣΗ 3 - Pseudo-users: δημιουργουν Jira tickets απο πραγματικα ερωτηματα TechQA.

Τρεχει ΜΕΣΑ στο container (εχει προσβαση στο /data/TechQA.tar.gz):
  docker exec -e JIRA_URL=... -e JIRA_EMAIL=... -e JIRA_TOKEN=... -e JIRA_PROJECT=...
      knowledge_api python /data/pseudo_users_jira.py

ΑΡΧΗ ΣΧΕΔΙΑΣΗΣ: το ground truth (GT document) ΔΕΝ γραφεται ΠΟΤΕ στο Jira.
Αποθηκευεται τοπικα στο /data/jira_ticket_map.csv, αποκλειστικα για post-hoc
βαθμολογηση απο το jira_report.py. Ο worker δεν το βλεπει ποτε.
"""
import csv
import json
import os
import random
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jira_client import JiraClient

# -- Config (env-overridable) -----------------------------------------------------
JIRA_URL     = os.getenv("JIRA_URL", "")
JIRA_EMAIL   = os.getenv("JIRA_EMAIL", "")
JIRA_TOKEN   = os.getenv("JIRA_TOKEN", "")
JIRA_PROJECT = os.getenv("JIRA_PROJECT", "")
ISSUE_TYPE   = os.getenv("JIRA_ISSUE_TYPE", "Task")

N_ANSWERABLE   = int(os.getenv("N_ANSWERABLE", "8"))    # με GT στη γνωσιακη βαση
N_UNANSWERABLE = int(os.getenv("N_UNANSWERABLE", "4"))  # χωρις GT -> αναμενεται αποχη
SEED           = int(os.getenv("SEED", "42"))

# testing = ενας pseudo-user (X1) | prod = πολλοί (X1, X2, ...) round-robin
PSEUDO_MODE  = os.getenv("PSEUDO_MODE", "testing").strip().lower()
REPORTER_IDS = [x.strip() for x in os.getenv("REPORTER_IDS", "").split(",") if x.strip()]
ASSIGNEE_ID  = os.getenv("ASSIGNEE_ID", "")             # ο μηχανικός (Υ) σε ολα τα tickets

TAR_PATH = "/data/TechQA.tar.gz"
MAP_CSV  = "/data/jira_ticket_map.csv"
TAG      = "techqa-auto"     # label που παρακολουθεί ο worker


def load_dev_questions():
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        member = next((m for m in tar.getmembers()
                       if m.name.endswith("dev_Q_A.json")), None)
        if member is None:
            raise SystemExit("dev_Q_A.json δεν βρεθηκε στο tar.")
        return json.load(tar.extractfile(member))


def pick(items, answerable_flag, n, rng):
    pool = [it for it in items if it.get("ANSWERABLE") == answerable_flag]
    rng.shuffle(pool)
    return pool[:n]


def pick_reporter(idx):
    """testing -> πάντα ο πρώτος pseudo-user· prod -> round-robin σε όλους"""
    if not REPORTER_IDS:
        return None
    if PSEUDO_MODE == "prod":
        return REPORTER_IDS[idx % len(REPORTER_IDS)]
    return REPORTER_IDS[0]


def main():
    missing = [k for k, v in [("JIRA_URL", JIRA_URL), ("JIRA_EMAIL", JIRA_EMAIL),
                              ("JIRA_TOKEN", JIRA_TOKEN),
                              ("JIRA_PROJECT", JIRA_PROJECT)] if not v]
    if missing:
        raise SystemExit("Λείπουν μεταβλητές περιβάλλοντος: " + ", ".join(missing))

    jira = JiraClient(JIRA_URL, JIRA_EMAIL, JIRA_TOKEN)
    rng = random.Random(SEED)
    data = load_dev_questions()

    selected = (pick(data, "Y", N_ANSWERABLE, rng) +
                pick(data, "N", N_UNANSWERABLE, rng))
    rng.shuffle(selected)

    n_reporters = len(REPORTER_IDS) if PSEUDO_MODE == "prod" else min(1, len(REPORTER_IDS))
    print(f"Δημιουργία {len(selected)} tickets στο project {JIRA_PROJECT} "
          f"({N_ANSWERABLE} answerable / {N_UNANSWERABLE} unanswerable)")
    print(f"Mode: {PSEUDO_MODE} | reporters: {n_reporters or '(default)'} | "
          f"assignee: {'ναι' if ASSIGNEE_ID else 'όχι'}")
    print("=" * 70)

    rows, created = [], 0
    for idx, item in enumerate(selected):
        qid   = item.get("QUESTION_ID", "?")
        title = item.get("QUESTION_TITLE") or "IT support request"
        text  = item.get("QUESTION_TEXT") or title
        gt    = item.get("DOCUMENT") or ""
        answerable = item.get("ANSWERABLE", "N")

        try:
            issue = jira.create_issue(
                JIRA_PROJECT, title, text,
                labels=[TAG, f"qid-{qid}"], issue_type=ISSUE_TYPE,
                reporter_id=pick_reporter(idx),
                assignee_id=ASSIGNEE_ID or None)
            key = issue.get("key", "?")
            created += 1
            print(f"  [{answerable}] {qid:12s} -> {key}  {title[:52]}")
            rows.append({"qid": qid, "issue_key": key,
                         "answerable": answerable, "gt_document": gt})
        except Exception as e:
            print(f"  ΣΦΑΛΜΑ στο {qid}: {e}")

    # Ο πίνακας αντιστοίχισης μένει ΤΟΠΙΚΑ (δεν φεύγει προς Jira)
    try:
        with open(MAP_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["qid", "issue_key",
                                              "answerable", "gt_document"])
            w.writeheader()
            w.writerows(rows)
        print("=" * 70)
        print(f"Δημιουργήθηκαν {created}/{len(selected)} tickets")
        print(f"Πίνακας αντιστοίχισης -> {MAP_CSV}")
    except PermissionError:
        print(f"\nΠροέκυψε σφάλμα στο {MAP_CSV} !")


if __name__ == "__main__":
    main()
