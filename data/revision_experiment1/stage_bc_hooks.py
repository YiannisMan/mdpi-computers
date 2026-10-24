# -*- coding: utf-8 -*-
"""
stage_bc_hooks.py — ΠΙΣΤΑ αντίγραφα του ΑΥΘΕΝΤΙΚΟΥ κώδικα (prompt/parser/decide).

Ευθυγραμμισμένο με τα αρχεία σας:
  - main.py (/resolve):            RAG_SYSTEM_PROMPT, ABSTAIN_PHRASE, build_rag_context,
                                   build_rag_messages, LLM_* παράμετροι.
  - evaluate_generation.py:        is_abstention (abstain detection), ID_PATTERN,
                                   faithfulness = cited ⊆ sources. <-- ΑΥΤΟΣ ορίζει C/F-Abst.
  - jira_worker.py:                decide() = MODE B Ή gate Ή faith. <-- ΑΥΤΟΣ ορίζει Stage C.

ΣΗΜΕΙΩΣΗ: το /resolve ΞΑΝΑΚΑΝΕΙ retrieval, οπότε για ΠΑΓΩΜΕΝΟ top-5 (πρωτόκολλο) η
γενιά γίνεται ΑΠΕΥΘΕΙΑΣ (build_messages + Ollama), ΟΧΙ μέσω /resolve. Τα frozen docs
(data/retrieval_test_docs.jsonl) έχουν το content (τα sources του /resolve δεν το έχουν).
"""
import os, re, json, hashlib

MAIN_PY_PATH      = os.getenv("MAIN_PY_PATH", "/app/main.py")
LLM_DOC_CHAR_LIMIT    = int(os.getenv("LLM_DOC_CHAR_LIMIT", "1600"))     # == main.py
LLM_TICKET_CHAR_LIMIT = int(os.getenv("LLM_TICKET_CHAR_LIMIT", "3000"))  # == main.py
NL = chr(10)

# ── Διαβάζει τα ΠΡΑΓΜΑΤΙΚΑ constants από το main.py (όπως το build_sft_dataset.py) ──
def load_main_constants(path=MAIN_PY_PATH):
    s = open(path, "r", encoding="utf-8", errors="replace").read().replace(chr(13) + NL, NL)
    ns = {}
    a = s.index("ABSTAIN_PHRASE = ("); b = s.index(")", a) + 1
    exec(s[a:b], ns)
    marker = 'RAG_SYSTEM_PROMPT = """'
    i = s.index(marker) + len(marker); j = s.index('"""', i)
    return ns["ABSTAIN_PHRASE"], s[i:j]

try:
    ABSTAIN_PHRASE, RAG_SYSTEM_PROMPT = load_main_constants()
except Exception as e:
    print(f"[hooks][WARN] constants από {MAIN_PY_PATH}: {e}")
    ABSTAIN_PHRASE = ("The knowledge base does not contain a confident solution for this "
                      "issue. Escalate to a human engineer.")
    RAG_SYSTEM_PROMPT = ""

# ── Σταθερές parser — ΑΚΡΙΒΩΣ όπως evaluate_generation.py & jira_worker.py ──
ABSTAIN_SUBSTRING = "does not contain a confident solution"              # evaluate_generation
ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,4}[0-9][A-Za-z0-9]{4,}(?![A-Za-z0-9])")  # evaluate_generation
CITE_RE    = re.compile(r"\[([A-Za-z]{2,4}[0-9A-Za-z]{4,})\]")           # jira_worker (bracketed)
ID_PREFIXES = ("swg", "nas", "isg", "ibm")                              # jira_worker


# ── build_rag_context / build_rag_messages — ΠΙΣΤΑ στο main.py ──
def build_context(solutions):
    blocks = []
    for i, sol in enumerate(solutions, start=1):
        tid = sol.get("technote_id", "doc_" + str(i))
        title = sol.get("title", "") or ""
        content = sol.get("content", "") or ""
        if content.startswith("Title:"):
            parts = content.split(NL + NL, 1)
            content = parts[1] if len(parts) == 2 else content
        content = content[:LLM_DOC_CHAR_LIMIT]
        blocks.append("[Technote " + str(i) + "] ID: " + tid + NL +
                      "Title: " + title + NL + "Content:" + NL + content + NL + "---")
    return NL.join(blocks)

def build_messages(ticket, solutions):
    """== build_rag_messages(main.py): system=RAG_SYSTEM_PROMPT + user(ticket+context)."""
    ticket = (ticket or "").strip()[:LLM_TICKET_CHAR_LIMIT]
    ctx = build_context(solutions)
    user = ("## Support Ticket" + NL + ticket + NL + NL +
            "## Retrieved Technotes (knowledge base context)" + NL + ctx + NL + NL +
            "## Task" + NL +
            "Provide a resolution for the ticket above, strictly following the rules.")
    return [{"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def prompt_template_sha256():
    """Σταθερό αποτύπωμα του prompt template: build_messages ΧΩΡΙΣ μεταβλητό περιεχόμενο
    (κενό ticket, καμία λύση) -> περιλαμβάνει RAG_SYSTEM_PROMPT + το σκελετό user. Ίδιος
    κώδικας -> ίδιο hash. Το χρησιμοποιούν ΚΑΙ το capture_env.py ΚΑΙ το exp1d_verify.py."""
    canon = json.dumps(build_messages("", []), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ══════════════════════════════════════════════════════════════════════════
#  PARSER — ΑΚΡΙΒΩΣ όπως evaluate_generation.py (ορίζει abstain + citation validity)
# ══════════════════════════════════════════════════════════════════════════
def is_abstention(text):
    return ABSTAIN_SUBSTRING in (text or "").lower()

def extract_technote_ids(text):
    return {m.lower() for m in ID_PATTERN.findall(text or "")}

def parse_output(raw_text, retrieved_ids):
    """
    Επιστρέφει (mode, cited_ids, citation_valid):
      mode = "B" αν is_abstention αλλιώς "A"  (όπως το evaluate_generation: abstain vs answered)
      cited_ids = extract_technote_ids(raw)   (lowercased)
      citation_valid = (cited - sources == ∅) == "faithful" (καμία εφεύρεση ID)
    """
    raw = raw_text or ""
    source_set = {str(r).lower() for r in retrieved_ids if r}
    cited = extract_technote_ids(raw)
    if is_abstention(raw):
        return "B", sorted(cited), True
    citation_valid = (len(cited - source_set) == 0)   # faithful: κανένα invented
    return "A", sorted(cited), bool(citation_valid)

def invalid_diagnostic(raw_text, retrieved_ids):
    """Διαγνωστικό (Οδηγός): απάντηση που δεν είναι έγκυρο Mode A (καμία έγκυρη παραπομπή
    και καμία δομή) ούτε Mode B. Μετριέται ΞΕΧΩΡΙΣΤΑ· στις μετρικές μετρά ως 'απάντηση'."""
    raw = raw_text or ""
    if is_abstention(raw):
        return False
    source_set = {str(r).lower() for r in retrieved_ids if r}
    cited_valid = extract_technote_ids(raw) & source_set
    has_struct = ("root cause" in raw.lower()) or ("technote:" in raw.lower())
    return (len(cited_valid) == 0) and (not has_struct)


# ══════════════════════════════════════════════════════════════════════════
#  DECIDE — ΑΚΡΙΒΩΣ όπως jira_worker.py (ορίζει Stage C escalate/auto-resolve)
#  escalate <=> MODE B Ή gated Ή faith(καμία ή επινοημένη παραπομπή)
# ══════════════════════════════════════════════════════════════════════════
def cited_ids_bracketed(answer):
    out = []
    for m in CITE_RE.findall(answer or ""):
        if m.lower().startswith(ID_PREFIXES) and m not in out:
            out.append(m)
    return out

def decide(result):
    """result: dict με answer, sources([{technote_id,...}]), gated. -> (escalate, reason)."""
    answer = result.get("answer") or ""
    sources = {s.get("technote_id") for s in (result.get("sources") or [])}
    if ABSTAIN_SUBSTRING in answer:                       # jira_worker: case-sensitive substring
        return True, "MODE B - το μοντέλο δήλωσε αποχή"
    if result.get("gated"):
        return True, "GATE - χαμηλή εμπιστοσύνη ανάκτησης"
    cites = cited_ids_bracketed(answer)
    if not cites:
        return True, "FAITH - η απάντηση δεν παραπέμπει σε κανένα technote"
    invented = [c for c in cites if c not in sources]
    if invented:
        return True, f"FAITH - επικαλείται ανύπαρκτο αναγνωριστικό: {', '.join(invented[:3])}"
    return False, "AUTO - τεκμηριωμένη λύση"
