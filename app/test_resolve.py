import sys
import os
import json
import tarfile
import urllib.request

"""
ΡΟΛΟΣ:
Δοκιμή του RAG layer (Φάση 2). Στέλνει tickets στο /resolve και εμφανίζει
τη λύση που παρήγαγε το Llama-3 μαζί με τις πηγές (technotes) που του δόθηκαν.

ΔΥΟ ΤΡΟΠΟΙ ΧΡΗΣΗΣ:
  1) Ελεύθερο ticket από τη γραμμή εντολών:
       python test_resolve.py "WAS fails to start with OutOfMemoryError in the JVM logs"

  2) Χωρίς όρισμα -> τραβάει N πραγματικά answerable tickets από το dev_Q_A.json
     (όπως το psudo_users.py) και δείχνει λύση + αν το Ground Truth μπήκε στις πηγές:
       python test_resolve.py

ENV overrides:
  RESOLVE_URL  (default http://localhost:8080/resolve)
  TECHQA_TAR   (default /data/TechQA.tar.gz)
  N_TICKETS    (default 3)   -- πόσα tickets από το dataset
  RAG_TOP_K    (default 5)   -- πόσα technotes να ζητηθούν
"""

API_URL   = os.getenv("RESOLVE_URL", "http://localhost:8080/resolve")
TAR_PATH  = os.getenv("TECHQA_TAR", "/data/TechQA.tar.gz")
N_TICKETS = int(os.getenv("N_TICKETS", "3"))
N_DOCS    = int(os.getenv("RAG_TOP_K", "5"))


def resolve(ticket: str, n_docs: int = N_DOCS) -> dict:
    payload = json.dumps({"issue_description": ticket, "n_docs": n_docs}).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=360) as r:
        return json.loads(r.read().decode("utf-8"))


def _fmt_score(s) -> str:
    return f"{s:.3f}" if isinstance(s, (int, float)) else "n/a"


def print_result(res: dict, gt_id: str | None = None):
    print("=" * 72)
    print("ANSWER:\n")
    print(res.get("answer", "").strip())
    print("\n" + "-" * 72)
    print(f"SOURCES (technotes fed to {res.get('model', 'LLM')}):")
    for s in res.get("sources", []):
        mark = ""
        if gt_id is not None and s.get("technote_id") == gt_id:
            mark = "   <== GROUND TRUTH"
        print(f"  #{s.get('rank')}  {s.get('technote_id')}  "
              f"(final_score={_fmt_score(s.get('final_score'))}){mark}")
    if gt_id is not None:
        in_src = any(s.get("technote_id") == gt_id for s in res.get("sources", []))
        print(f"\n  Ground truth in retrieved sources: "
              f"{'YES' if in_src else 'NO'}   (GT = {gt_id})")
    print("=" * 72 + "\n")


def from_dataset(limit: int):
    print(f"Loading {limit} answerable tickets from {TAR_PATH}...\n")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.name.endswith("dev_Q_A.json")), None
        )
        if not member:
            print("Error: dev_Q_A.json not found in the tar.gz")
            return
        qa_data = json.load(tar.extractfile(member))

    answerable = [
        q for q in qa_data if q.get("ANSWERABLE") == "Y" and q.get("DOCUMENT")
    ][:limit]

    for item in answerable:
        ticket = str(item.get("QUESTION_TEXT") or item.get("QUESTION_TITLE") or "")
        gt_id  = item.get("DOCUMENT")
        qid    = item.get("QUESTION_ID", "?")
        print(f"\n[{qid}] {ticket.strip().replace(chr(10), ' ')[:120]}...")
        try:
            res = resolve(ticket)
            print_result(res, gt_id=gt_id)
        except Exception as e:
            print(f"  REQUEST FAILED: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Mode 1: ελεύθερο ticket από CLI
        ticket = " ".join(sys.argv[1:])
        print(f"\nTicket: {ticket}\n")
        try:
            print_result(resolve(ticket))
        except Exception as e:
            print(f"REQUEST FAILED: {e}")
    else:
        # Mode 2: πραγματικά tickets από το dataset
        from_dataset(limit=N_TICKETS)
