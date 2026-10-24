import tarfile
import json
import csv
import time
import re
import urllib.request

"""
ΡΟΛΟΣ (Φάση Β — data prep για QLoRA):
Χτιζει το SFT dataset απο το TechQA training set, σε μορφη IDENTIKH με το
inference (ιδιο system prompt, ιδιο context format), ωστε το fine-tuning να
μαθαινει ακριβως τη συμπεριφορα που μετραει το evaluate_generation.py.

Για καθε answerable training question:
  1. retrieval top-K (ιδιο /search pipeline) -> πραγματικα top-5 technotes
  2. αν το GT ειναι μεσα -> MODE A target: αναδομηση του gold ANSWER σε
     root-cause -> numbered steps -> [gt_id], ΜΟΝΟ απο gold + GT technote content,
     με faithfulness filter (σωστο citation, καμια εφευρεση id, content grounded).
     Οσα κοπουν -> deterministic fallback (δεν χανεται δειγμα).
  3. αν το GT ΛΕΙΠΕΙ -> MODE B target: ABSTAIN_PHRASE, με τα ΠΡΑΓΜΑΤΙΚΑ
     distractors στο context (ετσι μαθαινει να παραιτειται μπροστα στα ιδια
     confident-looking λαθη που ξεγελασαν το gate).

ΕΞΟΔΟΣ:
  /data/sft_dataset.jsonl       -> {"messages":[system,user,assistant]} ανα γραμμη
  /data/sft_dataset_meta.csv    -> qid, gt_in, gt_id, mode, synth_status, target_len

ΕΚΤΕΛΕΣΗ (μεσα στο container):
  docker exec -it knowledge_api python /app/build_sft_dataset.py
"""

# ── Config ────────────────────────────────────────────────────────────
import os
SEARCH_URL       = os.getenv("SEARCH_URL", "http://localhost:8080/search")
OLLAMA_URL       = os.getenv("OLLAMA_HOST", "http://ollama:11434") + "/api/chat"
SYNTH_MODEL      = os.getenv("SYNTH_MODEL", "llama3:8b-instruct-q4_K_M")
MAIN_PY_PATH     = os.getenv("MAIN_PY_PATH", "/app/main.py")
TAR_PATH         = "/data/TechQA.tar.gz"
TRAIN_MEMBER     = os.getenv("TRAIN_MEMBER", "")     # κενο -> auto-detect training_Q_A.json
OUT_JSONL        = "/data/sft_dataset.jsonl"
OUT_META         = "/data/sft_dataset_meta.csv"

TRAIN_K          = 5
LIMIT            = None      # π.χ. 10 για γρηγορο test· None = ολο το training set
DOC_CHAR_LIMIT   = int(os.getenv("LLM_DOC_CHAR_LIMIT", "1600"))
TICKET_CHAR_LIMIT= int(os.getenv("LLM_TICKET_CHAR_LIMIT", "3000"))
SYNTH_CTX_LIMIT  = 4000      # truncation του GT content που τρωει ο synthesizer
SYNTH_TIMEOUT    = 180
SEARCH_TIMEOUT   = 120

# Faithfulness filter thresholds
GROUNDING_MIN    = 0.35      # min ποσοστο prose-tokens του target που υπαρχουν στο source
                             # (lenient: πιανει χονδρο hallucination, αφηνει paraphrase)

# Technote-id citation pattern: prefix-anchored ωστε να ΜΗΝ μπερδευεται με
# JVM flags / versions (π.χ. -Xmx2048m) που εχουν παρομοιο σχημα.
CITE_ID     = re.compile("(?<![A-Za-z0-9])(?:swg|nas|isg|ibm)[0-9A-Za-z]{4,}(?![A-Za-z0-9])",
                         re.IGNORECASE)
WORD_PATTERN= re.compile("[A-Za-z0-9]+")
STOPWORDS = set("the a an to of and or in on at for with is are be by from this that it as your you "
                "if then will can may should must not no into out over under up down off then than "
                "we they he she his her its their our use used using see also more most".split())

NL = chr(10)


# ── Φερε τα EXACT constants απο το main.py (χωρις import — μηδεν side effects) ──
def load_main_constants(path):
    s = open(path, "r", encoding="utf-8", errors="replace").read().replace(chr(13) + NL, NL)
    ns = {}
    a = s.index("ABSTAIN_PHRASE = (")
    b = s.index(")", a) + 1            # τα strings δεν εχουν ')' μεσα -> ασφαλες
    exec(s[a:b], ns)
    marker = 'RAG_SYSTEM_PROMPT = """'
    i = s.index(marker) + len(marker)
    j = s.index('"""', i)             # το prompt δεν εχει '"""' μεσα
    rag_prompt = s[i:j]
    return ns["ABSTAIN_PHRASE"], rag_prompt


# ── Context building — ΑΚΡΙΒΩΣ οπως το build_rag_context / build_rag_messages ──
def build_context(solutions):
    blocks = []
    for i, sol in enumerate(solutions, start=1):
        tid     = sol.get("technote_id", "doc_" + str(i))
        title   = sol.get("title", "") or ""
        content = sol.get("content", "") or ""
        if content.startswith("Title:"):
            parts = content.split(NL + NL, 1)
            content = parts[1] if len(parts) == 2 else content
        content = content[:DOC_CHAR_LIMIT]
        blocks.append("[Technote " + str(i) + "] ID: " + tid + NL +
                      "Title: " + title + NL +
                      "Content:" + NL + content + NL + "---")
    return NL.join(blocks)


def build_user_message(ticket, context):
    ticket = (ticket or "").strip()[:TICKET_CHAR_LIMIT]
    return ("## Support Ticket" + NL + ticket + NL + NL +
            "## Retrieved Technotes (knowledge base context)" + NL + context + NL + NL +
            "## Task" + NL +
            "Provide a resolution for the ticket above, strictly following the rules.")


# ── HTTP helpers ──────────────────────────────────────────────────────
def http_post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def search(ticket, k):
    return http_post_json(SEARCH_URL, {"issue_description": ticket, "n_results": k}, SEARCH_TIMEOUT)


def synth_structured(ticket, gt_id, gt_title, gt_content, gold_answer):
    """Αναδομει το gold answer στη MODE-A μορφη, grounded ΜΟΝΟ σε gold + GT technote."""
    sys_p = ("You reformat a known-correct IT support answer into a fixed structure. "
             "Use ONLY information present in the provided reference answer and technote. "
             "Do NOT add facts, steps, commands, versions, paths or IDs that are not in them. "
             "Do NOT mention any technote ID other than the one given. "
             "Output ONLY the structured answer: no preamble, no markdown, no bold.")
    fmt = ("Root cause: <one sentence on the underlying cause>." + NL +
           "Resolution steps:" + NL +
           "1. <first concrete step> [" + gt_id + "]" + NL +
           "2. <next step, only if supported by the sources>" + NL +
           "Technote: [" + gt_id + "]")
    user_p = ("## Ticket" + NL + (ticket or "").strip()[:1500] + NL + NL +
              "## Technote (source of truth) ID: " + gt_id + NL +
              (gt_title + NL if gt_title else "") + (gt_content or "")[:SYNTH_CTX_LIMIT] + NL + NL +
              "## Reference answer (correct)" + NL + (gold_answer or "").strip() + NL + NL +
              "## Task" + NL +
              "Rewrite the reference answer in EXACTLY this format:" + NL + NL + fmt + NL + NL +
              "Rules: Begin immediately with 'Root cause:' — output NOTHING before it (no "
              "'Here is...' preamble). No markdown, no bold, only the literal labels shown. "
              "Cite ONLY [" + gt_id + "]. Use ONLY facts from the reference answer and technote; "
              "do not invent steps, and do not repeat the same step.")
    payload = {
        "model": SYNTH_MODEL,
        "messages": [{"role": "system", "content": sys_p},
                     {"role": "user", "content": user_p}],
        "stream": False,
        "options": {"num_ctx": 8192, "temperature": 0.1, "num_predict": 512, "seed": 42},
    }
    data = http_post_json(OLLAMA_URL, payload, SYNTH_TIMEOUT)
    return (data.get("message", {}) or {}).get("content", "").strip()


def clean_target(text):
    """Καθαρίζει το synth output: preamble, markdown bold, διπλες γραμμες."""
    t = (text or "").replace("**", "").replace("##", "")
    lines = t.split(NL)
    # κοψε ο,τι προηγειται της πρωτης γραμμης "root cause" (preamble)
    start = 0
    for i, ln in enumerate(lines):
        if "root cause" in ln.lower() or "root-cause" in ln.lower():
            start = i
            break
    lines = lines[start:]
    # dedupe ιδιες γραμμες (πιανει την επαναληψη) + collapse blanks
    seen, out = set(), []
    for ln in lines:
        s = ln.rstrip()
        key = s.strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(s)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return NL.join(out)


# ── Faithfulness filter ───────────────────────────────────────────────
def content_tokens(text):
    return [w for w in WORD_PATTERN.findall((text or "").lower()) if w not in STOPWORDS]


def is_id_token(w):
    return w.startswith(("swg", "nas", "isg", "ibm")) and any(c.isdigit() for c in w)


def passes_filter(target, gt_id, gold_answer, gt_content):
    ids = {m.lower() for m in CITE_ID.findall(target)}
    gtl = gt_id.lower()
    if gtl not in ids:
        return False, "missing_gt_cite"
    if ids - {gtl}:
        return False, "invented_id"
    body = NL.join(ln for ln in target.split(NL) if "technote:" not in ln.lower())
    tgt = [w for w in content_tokens(body) if not w.isdigit() and not is_id_token(w)]
    if len(tgt) < 8:
        return False, "too_short"
    src = set(content_tokens(gold_answer)) | set(content_tokens(gt_content))
    grounded = sum(1 for w in tgt if w in src) / len(tgt)
    if grounded < GROUNDING_MIN:
        return False, "ungrounded_%.2f" % grounded
    return True, "ok"


def deterministic_fallback(gold_answer, gt_id):
    ans = " ".join((gold_answer or "").split()).strip()
    first = ans.split(". ")[0].strip()
    if first and not first.endswith("."):
        first += "."
    return ("Root cause: " + (first if first else "See the referenced technote.") + NL +
            "Resolution steps:" + NL +
            "1. " + ans + " [" + gt_id + "]" + NL +
            "Technote: [" + gt_id + "]")


def ensure_technote_line(target, gt_id):
    """Επιβαλλει κλεισιμη γραμμη 'Technote: [id]' για ομοιομορφια των MODE-A targets."""
    if any(ln.strip().lower().startswith("technote:") for ln in target.split(NL)):
        return target
    return target.rstrip() + NL + "Technote: [" + gt_id + "]"


# ── Data loading ──────────────────────────────────────────────────────
def load_training():
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        members = tar.getmembers()
        if TRAIN_MEMBER:
            member = next((m for m in members if m.name.endswith(TRAIN_MEMBER)), None)
        else:
            member = next((m for m in members if m.name.endswith("training_Q_A.json")), None)
            if not member:
                member = next((m for m in members
                               if "train" in m.name.lower() and m.name.endswith(".json")), None)
        if not member:
            raise FileNotFoundError("training Q_A json not found in tar (set TRAIN_MEMBER)")
        print("Using training file: " + member.name, flush=True)
        qa = json.load(tar.extractfile(member))
    items = [q for q in qa if q.get("ANSWERABLE") == "Y" and q.get("DOCUMENT")]
    if LIMIT:
        items = items[:LIMIT]
    return items


# ── Main ──────────────────────────────────────────────────────────────
def main():
    abstain_phrase, rag_system_prompt = load_main_constants(MAIN_PY_PATH)
    print("Loaded constants from main.py (prompt %d chars)." % len(rag_system_prompt), flush=True)

    items = load_training()
    n = len(items)
    print("Answerable training tickets: %d" % n, flush=True)

    fout = open(OUT_JSONL, "w", encoding="utf-8")
    fmeta = open(OUT_META, "w", newline="", encoding="utf-8")
    meta = csv.DictWriter(fmeta, fieldnames=["qid", "gt_in", "gt_id", "mode",
                                             "synth_status", "target_len"])
    meta.writeheader()

    stats = {"mode_a": 0, "mode_b": 0, "synth_ok": 0, "fallback": 0, "failed": 0}
    start = time.time()

    for idx, item in enumerate(items):
        ticket = str(item.get("QUESTION_TEXT") or item.get("QUESTION_TITLE") or "")
        gt_id  = item.get("DOCUMENT")
        gold   = item.get("ANSWER") or ""
        qid    = item.get("QUESTION_ID", "?")

        try:
            res = search(ticket, TRAIN_K)
        except Exception as e:
            print("  [%s] SEARCH FAILED: %s" % (qid, e), flush=True)
            stats["failed"] += 1
            continue

        solutions  = res.get("suggested_solutions", []) or []
        source_ids = [s.get("technote_id") for s in solutions]
        context    = build_context(solutions)
        user_msg   = build_user_message(ticket, context)

        if gt_id in source_ids:                     # ── MODE A
            stats["mode_a"] += 1
            gt_sol = next((s for s in solutions if s.get("technote_id") == gt_id), {})
            synth_status = "fallback"
            target = None
            try:
                cand = synth_structured(ticket, gt_id, gt_sol.get("title", ""),
                                        gt_sol.get("content", ""), gold)
                cand = clean_target(cand)
                ok, why = passes_filter(cand, gt_id, gold, gt_sol.get("content", ""))
                if ok:
                    target = cand
                    synth_status = "ok"
                    stats["synth_ok"] += 1
                else:
                    synth_status = "fb_" + why
            except Exception as e:
                synth_status = "fb_exc"
            if target is None:
                target = deterministic_fallback(gold, gt_id)
                stats["fallback"] += 1
            target = ensure_technote_line(target, gt_id)
            mode = "A"
        else:                                       # ── MODE B
            stats["mode_b"] += 1
            target = abstain_phrase
            synth_status = "abstain"
            mode = "B"

        rec = {"messages": [
            {"role": "system",    "content": rag_system_prompt},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": target},
        ]}
        fout.write(json.dumps(rec, ensure_ascii=False) + NL)
        meta.writerow({"qid": qid, "gt_in": int(gt_id in source_ids), "gt_id": gt_id,
                       "mode": mode, "synth_status": synth_status, "target_len": len(target)})

        if (idx + 1) % 10 == 0:
            el = time.time() - start
            rate = (idx + 1) / el if el > 0 else 0
            eta = (n - idx - 1) / rate if rate > 0 else 0
            print("  %d/%d | A:%d B:%d synth_ok:%d fallback:%d | ETA %.1f min"
                  % (idx + 1, n, stats["mode_a"], stats["mode_b"],
                     stats["synth_ok"], stats["fallback"], eta / 60), flush=True)

    fout.close()
    fmeta.close()

    print("", flush=True)
    print("=" * 60, flush=True)
    print("SFT DATASET BUILT", flush=True)
    print("  total written : %d" % (stats["mode_a"] + stats["mode_b"]), flush=True)
    print("  MODE A (solve): %d  (synth_ok %d, fallback %d)"
          % (stats["mode_a"], stats["synth_ok"], stats["fallback"]), flush=True)
    print("  MODE B (abst) : %d" % stats["mode_b"], flush=True)
    print("  search failed : %d" % stats["failed"], flush=True)
    if stats["mode_a"]:
        print("  synth pass-rate: %.1f%%" % (100.0 * stats["synth_ok"] / stats["mode_a"]), flush=True)
    print("  -> %s" % OUT_JSONL, flush=True)
    print("  -> %s" % OUT_META, flush=True)


if __name__ == "__main__":
    main()