# -*- coding: utf-8 -*-
"""
nli_support.py — Βιβλιοθήκη Πειράματος 4 (semantic grounding + correctness).

Περιέχει:
  4Α  parse_steps                       — διάσπαση Mode-A απάντησης σε διάγνωση + βήματα
  4Β  build_doc_index, NLIScorer         — κείμενο εγγράφων (TEST) + entailment ανά βήμα
      load_corpus_filtered               — streaming corpus (TRAIN, για τον έλεγχο NLI)
  4Γ  build_sheet                        — τυφλό φύλλο αξιολόγησης
  4Δ  load_labels, melt_steps            — ανάγνωση φύλλων + μία γραμμή ανά (item, βήμα)
  4ΣΤ est_correct, stratified_bootstrap_correct — σταθμισμένες εκτιμήσεις + CI

torch/transformers φορτώνονται ΜΟΝΟ μέσα στον NLIScorer (lazy), ώστε τα 4Α/4Γ/report
να τρέχουν χωρίς GPU.
"""
import re, os, json, bz2, tarfile
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────
#  4Α — parse βημάτων
# ─────────────────────────────────────────────────────────────────────────
CIT     = re.compile(r"\[?\b(swg[0-9A-Za-z]+|[A-Z]{2}\d{5,})\b\]?")        # παραπομπές technote
ITEM    = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.*)")                       # αριθμημένο/κουκκιδωτό βήμα
GENERIC = re.compile(r"contact (ibm )?support|latest fix ?pack|if the problem persists|"
                     r"open a (pmr|case)|apply the latest|restart (the|if)", re.I)


def parse_steps(raw):
    """Διάσπαση σε διάγνωση (μη-αριθμημένες γραμμές) + βήματα. Αφαιρεί παραπομπές,
    σημαίνει generic, κρατά μόνο ισχυρισμούς ≥ 3 λέξεων."""
    steps, diag = [], []
    for line in (raw or "").splitlines():
        m = ITEM.match(line)
        (steps if m else diag).append((m.group(1) if m else line).strip())
    out = [{"kind": "diagnosis", "text": " ".join(d for d in diag if d)}] if any(diag) else []
    out += [{"kind": "step", "text": s} for s in steps if s]
    for s in out:
        s["text"] = CIT.sub("", s["text"]).strip(" .;:")
        s["generic"] = bool(GENERIC.search(s["text"]))
    return [s for s in out if len(s["text"].split()) >= 3]


# ─────────────────────────────────────────────────────────────────────────
#  4Β — κείμενο εγγράφων (TEST top-5) + corpus (TRAIN) + NLI scorer
# ─────────────────────────────────────────────────────────────────────────
def build_doc_index(test_docs_path):
    """id -> πλήρες κείμενο (title + content) από τα top-5 κάθε TEST ερώτησης
    (retrieval_test_docs.jsonl). Καλύπτει όλα τα cited/retrieved των TEST απαντήσεων."""
    idx = {}
    for line in open(test_docs_path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        for s in d.get("solutions", []):
            tid = str(s.get("technote_id") or s.get("id") or "")
            if not tid:
                continue
            txt = ((s.get("title") or "") + "\n" + (s.get("content") or "")).strip()
            if tid not in idx or len(txt) > len(idx[tid]):
                idx[tid] = txt
    return idx


_REFS = []
_TAG, _WS = re.compile(r"<[^>]+>"), re.compile(r"\s+")


def _clean_html(s):
    return _WS.sub(" ", _TAG.sub(" ", s or "")).strip()


def _open_stream(corpus_path, tar_path):
    p = corpus_path
    if p and os.path.exists(p):
        raw = open(p, "rb")
    else:
        tar = tarfile.open(tar_path, "r:*"); _REFS.append(tar)
        m = next((x for x in tar.getmembers() if x.name == p or x.name.endswith(p)), None)
        if not m:
            return None
        raw = tar.extractfile(m)
    _REFS.append(raw)
    if (p or "").endswith(".bz2"):
        bz = bz2.BZ2File(raw); _REFS.append(bz); return bz
    return raw


def load_corpus_filtered(needed, corpus_path, tar_path):
    """Streaming parse του JSON array corpus, κρατά ΜΟΝΟ τα ζητούμενα ids (TRAIN docs
    για τον έλεγχο NLI). Ίδιος μηχανισμός με το Πείραμα 3."""
    need = set(map(str, needed))
    if not need:
        return {}
    stream = _open_stream(corpus_path, tar_path)
    if stream is None:
        print(f"[WARN] corpus member δεν βρέθηκε: {corpus_path}")
        return {}
    try:
        import ijson
    except ImportError:
        print("[ERROR] χρειάζεται ijson — pip install ijson"); return {}
    out, seen = {}, 0
    # multiple_values=True: το corpus είναι ΠΟΛΛΑ συνενωμένα JSON arrays ([{...}][{...}]), όχι ένα
    for obj in ijson.items(stream, "item", multiple_values=True):
        seen += 1
        i = obj.get("id")
        if i is None:
            continue
        i = str(i)
        if i in need:
            out[i] = _clean_html(obj.get("content") or obj.get("text") or "")
            if len(out) == len(need):
                break
    print(f"[corpus] {len(out)}/{len(need)} technotes (σαρώθηκαν {seen})")
    return out


class NLIScorer:
    """DeBERTa-v3-large NLI. p_entail = max P(entailment) πάνω στα chunks.
    chunks(doc_id) 400 tokens / stride 200. Caches ανά doc_id και ανά (docs, βήμα)."""

    def __init__(self, model_name, doc_index, chunk=400, stride=200, device=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if dev == "cuda" else torch.float32
        self.nli = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=dt).to(dev).eval()
        self.dev = dev
        l2i = {str(k).lower(): v for k, v in self.nli.config.label2id.items()}
        self.ENT = l2i.get("entailment", next((v for k, v in l2i.items() if "entail" in k), 2))
        self.doc_index = doc_index
        self.chunk, self.stride = chunk, stride
        self._chunk_cache, self._entail_cache = {}, {}

    def doc_text(self, doc_id):
        return self.doc_index.get(str(doc_id), "")

    def chunks(self, doc_id):
        key = str(doc_id)
        if key in self._chunk_cache:
            return self._chunk_cache[key]
        txt = self.doc_text(doc_id)
        ids = self.tok(txt, add_special_tokens=False)["input_ids"] if txt else []
        size, stride = self.chunk, self.stride
        ch = tuple(self.tok.decode(ids[i:i + size]) for i in range(0, max(len(ids) - stride, 1), stride)) if ids else ()
        self._chunk_cache[key] = ch
        return ch

    def p_entail(self, premises, hypothesis, bs=32):
        if not premises:
            return 0.0
        torch = self.torch
        best = 0.0
        with torch.no_grad():
            for i in range(0, len(premises), bs):
                batch = list(premises[i:i + bs])
                x = self.tok(batch, [hypothesis] * len(batch), truncation="only_first",
                             max_length=512, padding=True, return_tensors="pt").to(self.dev)
                p = self.nli(**x).logits.float().softmax(-1)[:, self.ENT].max().item()
                best = max(best, p)
        return best

    def score_step(self, text, cited_ids, retrieved_ids, citation_valid):
        """Επιστρέφει (p_cited, p_any). p_cited: max entailment στα cited (∩ retrieved).
        p_any: max στα 5 retrieved — ξεχωρίζει «λάθος citation» από «καθόλου υποστήριξη»."""
        retr = [str(d) for d in (retrieved_ids or [])]
        cited = [str(d) for d in (cited_ids or []) if str(d) in set(retr)]
        if citation_valid and cited:
            k = (tuple(sorted(cited)), text)
            if k not in self._entail_cache:
                self._entail_cache[k] = self.p_entail(sum((self.chunks(d) for d in cited), ()), text)
            p_cited = self._entail_cache[k]
        else:
            p_cited = 0.0
        ka = (tuple(sorted(retr)), text, "any")
        if ka not in self._entail_cache:
            self._entail_cache[ka] = self.p_entail(sum((self.chunks(d) for d in retr), ()), text)
        return p_cited, self._entail_cache[ka]


# ─────────────────────────────────────────────────────────────────────────
#  4Γ — τυφλό φύλλο αξιολόγησης (χωρίς stratum, NLI, config, question_id)
# ─────────────────────────────────────────────────────────────────────────
def build_sheet(sample_df, doc_index, qa_map, max_doc_chars=3000):
    """sample_df: γραμμές με item_id, question_id, raw_output, cited_ids, retrieved_ids, answerable.
    qa_map: question_id -> {title, text, answer, gold_id}."""
    rows = []
    for r in sample_df.itertuples():
        qid = r.question_id
        qa = qa_map.get(qid, {})
        steps = parse_steps(getattr(r, "raw_output", ""))
        ans = "\n".join(f"S{i+1}. {s['text']}" for i, s in enumerate(steps))
        cited = [str(d) for d in (getattr(r, "cited_ids", None) or []) if d]
        cited_txt = "\n\n".join(f"[{d}] {doc_index.get(str(d),'')[:max_doc_chars]}" for d in cited) \
            or "(καμία έγκυρη παραπομπή)"
        if int(getattr(r, "answerable", 1)) == 0:
            gold = "TechQA marks this question as unanswerable: no Technote contains the answer."
        else:
            gold = f"{qa.get('answer','')}\n(gold doc: {qa.get('gold_id','')})"
        row = {"item_id": r.item_id,
               "incident": f"{qa.get('title','')}\n\n{qa.get('text','')}".strip(),
               "answer": ans, "cited_docs": cited_txt, "gold": gold}
        for i in range(len(steps)):
            row[f"S{i+1}_support"] = ""      # S / N
        row["correctness"] = ""              # C / P / I / U
        row["risky_action"] = ""             # Ναι / Όχι
        row["comment"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    scols = sorted([c for c in df.columns if re.fullmatch(r"S\d+_support", c)],
                   key=lambda c: int(re.search(r"\d+", c).group()))
    ordered = ["item_id", "incident", "answer", "cited_docs", "gold"] + scols + \
              ["correctness", "risky_action", "comment"]
    return df.reindex(columns=ordered)


# ─────────────────────────────────────────────────────────────────────────
#  4Δ — ανάγνωση φύλλων + μία γραμμή ανά (item, βήμα)
# ─────────────────────────────────────────────────────────────────────────
def load_labels(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def melt_steps(df):
    """-> DataFrame [item_id, step, support(S/N)] μόνο για συμπληρωμένα κελιά."""
    scol = [c for c in df.columns if re.fullmatch(r"S\d+_support", c)]
    m = df.melt(id_vars=["item_id"], value_vars=scol, var_name="step", value_name="support")
    m = m.dropna(subset=["support"])
    m["support"] = m.support.astype(str).str.strip().str.upper().str[:1]
    return m[m.support.isin(["S", "N"])].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
#  4ΣΤ — correct-resolution rate + stratified bootstrap
# ─────────────────────────────────────────────────────────────────────────
def _is_C(series):
    return series.astype(str).str.strip().str.upper().str[:1].eq("C")


def est_correct(df_labels, N_by_stratum, n_all=310):
    """df_labels: item_id, stratum, correctness. N_by_stratum: dict stratum -> #auto-resolved (run1).
    Επιστρέφει (correct_among_auto, correct_resolution_all, p_by_stratum dict)."""
    p = {h: float(_is_C(g.correctness).mean()) for h, g in df_labels.groupby("stratum")}
    num = sum(N_by_stratum.get(h, 0) * p.get(h, 0.0) for h in N_by_stratum)
    denom = sum(N_by_stratum.values())
    return (num / denom if denom else float("nan")), (num / n_all), p


def stratified_bootstrap_correct(df_labels, N_by_stratum, n_all=310, B=10000, seed=2026):
    """95% CI με επαναδειγματοληψία με επανάθεση ΜΕΣΑ σε κάθε στρώμα (τα N_h είναι γνωστά)."""
    rng = np.random.default_rng(seed)
    arrs = {h: _is_C(g.correctness).to_numpy().astype(float) for h, g in df_labels.groupby("stratum")}
    denom = sum(N_by_stratum.values()) or 1
    num = np.zeros(B)
    for h, arr in arrs.items():
        n = len(arr)
        if n == 0:
            continue
        idx = rng.integers(0, n, size=(B, n))
        num += N_by_stratum.get(h, 0) * arr[idx].mean(axis=1)
    auto, allr = num / denom, num / n_all
    pct = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return pct(auto), pct(allr)
