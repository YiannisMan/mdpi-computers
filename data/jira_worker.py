# -*- coding: utf-8 -*-
"""
ΦΑΣΗ 3 — Jira worker: ο αυτονομος πρακτορας επιλυσης περιστατικων.

Ροη ανα ticket:
  Jira (νεα tickets) -> /resolve -> πολιτικη κλιμακωσης -> σχολιο + ετικετα + μεταβαση

ΠΟΛΙΤΙΚΗ ΚΛΙΜΑΚΩΣΗΣ (ολοι οι ελεγχοι ειναι επαληθευσιμοι ΧΩΡΙΣ ground truth):
  1. MODE B  — το μοντελο δηλωνει αποχη (learned abstention)
  2. GATE    — η εμπιστοσυνη ανακτησης κατω απο το κατωφλι (gated=true)
  3. FAITH   — η απαντηση δεν παραπεμπει σε εγκυρο technote, ή επικαλειται
               αναγνωριστικο που ΔΕΝ υπαρχει στα ανακτηθεντα (επινοημενο)
Οποιοσδηποτε απο τους τρεις -> αναθεση σε ανθρωπο. Αλλιως -> αυτοματη επιλυση.

Τρεχει ΜΕΣΑ στο container:
  docker exec -e JIRA_URL=... -e JIRA_EMAIL=... -e JIRA_TOKEN=... -e JIRA_PROJECT=...
      knowledge_api python /data/jira_worker.py
"""
import csv
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jira_client import JiraClient, adf_to_text

# ── Config ────────────────────────────────────────────────────────────
JIRA_URL     = os.getenv("JIRA_URL", "")
JIRA_EMAIL   = os.getenv("JIRA_EMAIL", "")
JIRA_TOKEN   = os.getenv("JIRA_TOKEN", "")
JIRA_PROJECT = os.getenv("JIRA_PROJECT", "")
HUMAN_ID     = os.getenv("JIRA_HUMAN_ACCOUNT_ID", "")   # κενο -> ο ιδιος ο λογαριασμος
INTERNAL_ROLE = os.getenv("INTERNAL_ROLE", "Administrators")  # role που βλεπει τα διαγνωστικα

RESOLVE_URL  = os.getenv("RESOLVE_URL", "http://localhost:8080/resolve")
POLL_ONCE    = os.getenv("POLL_ONCE", "1") == "1"       # 1 = ενα περασμα, 0 = συνεχης
POLL_SECS    = int(os.getenv("POLL_SECS", "20"))
MAX_TICKETS  = int(os.getenv("MAX_TICKETS", "50"))

DONE_NAMES      = [n.strip() for n in os.getenv("DONE_TRANSITIONS", "Done,Resolve,Resolved,Complete").split(",")]
ESCALATE_NAMES  = [n.strip() for n in os.getenv("ESCALATE_TRANSITIONS", "In Progress,In Review").split(",")]

TAG        = "techqa-auto"
DONE_TAG   = "ai-processed"
LOG_CSV    = "/data/jira_worker_log.csv"

ABSTAIN_MARK = "does not contain a confident solution"
CITE_RE      = re.compile(r"\[([A-Za-z]{2,4}[0-9A-Za-z]{4,})\]")
ID_PREFIXES  = ("swg", "nas", "isg", "ibm")


def call_resolve(description: str) -> dict:
    payload = json.dumps({"issue_description": description}).encode("utf-8")
    req = urllib.request.Request(RESOLVE_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def cited_ids(answer: str):
    """Αναγνωριστικα technote που επικαλειται η απαντηση (prefix-anchored, χωρις διπλα)."""
    out = []
    for m in CITE_RE.findall(answer or ""):
        if m.lower().startswith(ID_PREFIXES) and m not in out:
            out.append(m)
    return out


def decide(result: dict):
    """Επιστρεφει (escalate: bool, reason: str)."""
    answer  = result.get("answer") or ""
    sources = {s.get("technote_id") for s in (result.get("sources") or [])}

    if ABSTAIN_MARK in answer:
        return True, "MODE B - το μοντέλο δήλωσε αποχή"
    if result.get("gated"):
        return True, "GATE - χαμηλή εμπιστοσύνη ανάκτησης"

    cites = cited_ids(answer)
    if not cites:
        return True, "FAITH - η απάντηση δεν παραπέμπει σε κανένα technote"
    invented = [c for c in cites if c not in sources]
    if invented:
        return True, f"FAITH - επικαλείται ανύπαρκτο αναγνωριστικό: {', '.join(invented[:3])}"
    return False, "AUTO - τεκμηριωμένη λύση"


def _sources_block(result):
    return "\n".join(f"- {s.get('technote_id')}: {s.get('title', '')[:90]}"
                     for s in (result.get("sources") or [])[:5]) or "- (κανένα)"


# -- Δημοσια σχολια (τα βλεπει ο χρηστης): καθαρα, χωρις μηχανισμο ----------
def public_solved(result):
    return "Προτεινόμενη λύση\n\n" + (result.get("answer", "") or "")


def public_escalated(result):
    return ("Το αίτημά σας ανατέθηκε σε μηχανικό υποστήριξης, "
            "ο οποίος θα επικοινωνήσει σύντομα.")


# -- Εσωτερικα σχολια (role-restricted): πληρη διαγνωστικα για τον μηχανικο --
def internal_solved(result, cites):
    return (
        "[Εσωτερικό - διαγνωστικά AI]\n"
        "Αυτόματη επίλυση με τεκμηριωμένη λύση.\n"
        f"Μοντέλο: {result.get('model', '?')} | Παραπομπές: {', '.join(cites)}\n"
        "Ανακτηθέντα technotes:\n"
        f"{_sources_block(result)}"
    )


def internal_escalated(result, reason):
    return (
        "[Εσωτερικό - κλιμάκωση AI]\n"
        f"Αιτία κλιμάκωσης: {reason}\n"
        f"Μοντέλο: {result.get('model', '?')}\n"
        "Πλησιέστερα ανακτηθέντα technotes (προς διευκόλυνση του μηχανικού):\n"
        f"{_sources_block(result)}"
    )


def post_internal(jira, key, text):
    """Ποσταρει εσωτερικο σχολιο role-restricted· αν ο ρολος δεν υπαρχει, το κραταει στο log."""
    if not INTERNAL_ROLE:
        return False
    try:
        jira.add_comment(key, text, visibility_role=INTERNAL_ROLE)
        return True
    except Exception as e:
        print(f"  (προσοχή) εσωτερικό σχόλιο χωρίς περιορισμό ρόλου '{INTERNAL_ROLE}': {e}")
        return False


def process(jira, issue, human_id, log_rows):
    key   = issue["key"]
    flds  = issue.get("fields", {})
    title = flds.get("summary", "")
    desc  = adf_to_text(flds.get("description")).strip() or title

    print(f"\n[{key}] {title[:60]}")
    try:
        result = call_resolve(desc)
    except Exception as e:
        print(f"  ΣΦΑΛΜΑ /resolve: {e}")
        return

    escalate, reason = decide(result)
    cites = cited_ids(result.get("answer") or "")

    # Αναθεση στον μηχανικο (Υ) σε ΟΛΑ τα tickets (το dashboard φιλτραρει με το label 'escalated')
    if human_id:
        try:
            jira.assign(key, human_id)
        except Exception as e:
            print(f"  (προσοχή) αδυναμία ανάθεσης: {e}")

    if escalate:
        jira.add_comment(key, public_escalated(result))          # δημοσιο, καθαρο
        post_internal(jira, key, internal_escalated(result, reason))  # role-restricted
        jira.add_labels(key, [DONE_TAG, "escalated"])
        applied = jira.transition_by_name(key, ESCALATE_NAMES)
        print(f"  -> ΚΛΙΜΑΚΩΣΗ | {reason}")
    else:
        jira.add_comment(key, public_solved(result))             # δημοσιο, καθαρο
        post_internal(jira, key, internal_solved(result, cites))  # role-restricted
        jira.add_labels(key, [DONE_TAG, "auto-resolved"])
        applied = jira.transition_by_name(key, DONE_NAMES)
        print(f"  -> ΑΥΤΟΜΑΤΗ ΕΠΙΛΥΣΗ | παραπομπές: {', '.join(cites)}")

    if applied is None:
        names = [t["name"] for t in jira.get_transitions(key)]
        print(f"  (προσοχή) καμία μετάβαση δεν ταίριαξε. Διαθέσιμες: {names}")

    log_rows.append({
        "issue_key": key,
        "escalated": int(escalate),
        "reason": reason,
        "gated": int(bool(result.get("gated"))),
        "model": result.get("model", ""),
        "cited_ids": "|".join(cites),
        "retrieved_ids": "|".join(s.get("technote_id", "")
                                  for s in (result.get("sources") or [])),
        "answer": " ".join((result.get("answer") or "").split())[:500],
    })


def write_log(rows):
    if not rows:
        return
    try:
        exists = os.path.exists(LOG_CSV)
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not exists:
                w.writeheader()
            w.writerows(rows)
        print(f"\nΚαταγραφή -> {LOG_CSV}")
    except PermissionError:
        print(f"\nΤο {LOG_CSV} είναι κλειδωμένο (ανοιχτό σε Excel;).")


def main():
    missing = [k for k, v in [("JIRA_URL", JIRA_URL), ("JIRA_EMAIL", JIRA_EMAIL),
                              ("JIRA_TOKEN", JIRA_TOKEN),
                              ("JIRA_PROJECT", JIRA_PROJECT)] if not v]
    if missing:
        raise SystemExit("Λειπουν μεταβλητες περιβαλλοντος: " + ", ".join(missing))

    jira = JiraClient(JIRA_URL, JIRA_EMAIL, JIRA_TOKEN)
    human_id = HUMAN_ID or jira.myself().get("accountId", "")

    jql = (f'project = "{JIRA_PROJECT}" AND labels = "{TAG}" '
           f'AND labels NOT IN ("{DONE_TAG}") ORDER BY created ASC')

    print("=" * 70)
    print("JIRA WORKER - αυτόνομη επίλυση περιστατικών")
    print(f"  project : {JIRA_PROJECT}")
    print(f"  resolve : {RESOLVE_URL}")
    print(f"  mode    : {'ένα πέρασμα' if POLL_ONCE else f'συνεχής ({POLL_SECS}s)'}")
    print("=" * 70)

    while True:
        issues = jira.search(jql, max_total=MAX_TICKETS)
        if issues:
            print(f"\nΒρέθηκαν {len(issues)} νέα tickets.")
            rows = []
            for issue in issues:
                process(jira, issue, human_id, rows)
            write_log(rows)
            n_esc = sum(r["escalated"] for r in rows)
            print(f"\nΣΥΝΟΨΗ: {len(rows) - n_esc} αυτόματες επιλύσεις | "
                  f"{n_esc} κλιμακώσεις")
        else:
            print("Δεν βρέθηκαν νέα tickets.")

        if POLL_ONCE:
            break
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
