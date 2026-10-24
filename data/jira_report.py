# -*- coding: utf-8 -*-
"""
ΦΑΣΗ 3 — Post-hoc αξιολογηση του βροχου Jira, χαρτογραφημενη στο πλαισιο της Φασης 2.

Συνδυαζει:
  /data/jira_ticket_map.csv  (qid, issue_key, answerable, gt_document)  <- pseudo-users
  /data/jira_worker_log.csv  (issue_key, escalated, reason, cited_ids, retrieved_ids)

Καθε ticket ταξινομειται σε GT-present ή GT-missing (οπως στη Φαση 2):
  GT-present  = answerable ΚΑΙ το gt_document ανακτηθηκε (ειναι στα retrieved)
  GT-missing  = unanswerable Η το gt_document ΔΕΝ ανακτηθηκε

Απο εκει προκυπτουν τα αναλογα της Φασης 2:
  hazard              = auto-resolved | GT-missing   (απαντησε ενω δεν επρεπε)
  ορθη κλιμακωση      = escalated     | GT-missing
  εσφαλμενη κλιμακωση = escalated     | GT-present   (απεσχε ενω το GT ηταν εκει)
  ορθη auto (cited)   = auto ΚΑΙ gt στα cited | GT-present

ΠΡΟΣΟΧΗ: το ground truth χρησιμοποιειται ΜΟΝΟ εδω, εκ των υστερων.
  docker exec knowledge_api python /data/jira_report.py
"""
import csv
import os

MAP_CSV = "/data/jira_ticket_map.csv"
LOG_CSV = "/data/jira_worker_log.csv"


def read(path):
    if not os.path.exists(path):
        raise SystemExit(f"Δεν βρεθηκε: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(n, d):
    return f"{100.0 * n / d:5.1f}%" if d else "  n/a"


def main():
    mapping = {r["issue_key"]: r for r in read(MAP_CSV)}
    latest = {r["issue_key"]: r for r in read(LOG_CSV)}   # τελευταια εγγραφη ανα ticket

    rows = []
    for key, lg in latest.items():
        mp = mapping.get(key)
        if not mp:
            continue
        gt        = mp["gt_document"].strip()
        retrieved = [c for c in lg["retrieved_ids"].split("|") if c]
        cited     = [c for c in lg["cited_ids"].split("|") if c]
        answerable = mp["answerable"] == "Y"
        escalated  = lg["escalated"] == "1"
        gt_present = answerable and bool(gt) and (gt in retrieved)
        rows.append(dict(key=key, answerable=answerable, gt=gt, escalated=escalated,
                         cited=cited, retrieved=retrieved, gt_present=gt_present))

    n = len(rows)
    if n == 0:
        raise SystemExit("Καμια αντιστοιχιση map<->log.")

    auto = [r for r in rows if not r["escalated"]]
    esc  = [r for r in rows if r["escalated"]]
    present = [r for r in rows if r["gt_present"]]      # GT ανακτηθηκε
    missing = [r for r in rows if not r["gt_present"]]  # GT απον/μη ανακτηθεν

    print("=" * 70)
    print(f"ΦΑΣΗ 3 - ΑΠΟΤΕΛΕΣΜΑΤΑ ΒΡΟΧΟΥ JIRA  (n={n} tickets)")
    print("=" * 70)
    print(f"  Αυτόματη επίλυση : {len(auto):3d}  ({pct(len(auto), n)})")
    print(f"  Κλιμάκωση        : {len(esc):3d}  ({pct(len(esc), n)})")
    print(f"  GT-present (ανακτήθηκε) : {len(present):3d}   |   "
          f"GT-missing : {len(missing):3d}")

    # -- Αναλογα Φασης 2 --
    hazard      = [r for r in missing if not r["escalated"]]   # απαντησε ενω GT-missing
    corr_esc    = [r for r in missing if r["escalated"]]       # C-Abst αναλογο
    false_esc   = [r for r in present if r["escalated"]]       # F-Abst αναλογο
    corr_auto   = [r for r in present if (not r["escalated"]) and r["gt"] in r["cited"]]

    print("\n--- Χαρτογράφηση στη Φάση 2 ---")
    print(f"  Ορθή κλιμάκωση (C-Abst)      : {len(corr_esc):3d}/{len(missing)}  "
          f"({pct(len(corr_esc), len(missing))})")
    print(f"  Hazard (auto ενώ GT-missing) : {len(hazard):3d}/{len(missing)}  "
          f"({pct(len(hazard), len(missing))})")
    print(f"  Εσφαλμένη κλιμάκωση (F-Abst) : {len(false_esc):3d}/{len(present)}  "
          f"({pct(len(false_esc), len(present))})")
    print(f"  Ορθή auto (παραπομπή σε GT)  : {len(corr_auto):3d}/{len(present)}  "
          f"({pct(len(corr_auto), len(present))})")

    # -- Αιτιες κλιμακωσης --
    print("\n--- Αιτίες κλιμάκωσης ---")
    causes = {}
    for r in esc:
        # ανακτηση αιτιας απο το log
        pass
    log_by_key = {r["issue_key"]: r for r in read(LOG_CSV)}
    for r in esc:
        tag = log_by_key[r["key"]]["reason"].split("—")[0].strip()
        causes[tag] = causes.get(tag, 0) + 1
    for tag, c in sorted(causes.items(), key=lambda x: -x[1]):
        print(f"  {tag:8s} : {c:3d}  ({pct(c, len(esc))})")

    # -- Ανα τυπο ticket --
    print("\n--- Ανά τύπο ticket ---")
    for flag, label in ((True, "Answerable  "), (False, "Unanswerable")):
        grp = [r for r in rows if r["answerable"] == flag]
        if not grp:
            continue
        e = sum(1 for r in grp if r["escalated"])
        print(f"  {label} (n={len(grp):3d}) : κλιμάκωση {pct(e, len(grp))} | "
              f"αυτόματη {pct(len(grp) - e, len(grp))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
