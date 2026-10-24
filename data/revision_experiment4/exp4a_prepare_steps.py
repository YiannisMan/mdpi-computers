#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΠΕΙΡΑΜΑ 4Α — parse Mode-A απαντήσεων σε βήματα -> analysis/steps.jsonl.

Διαβάζει Stage B (Base, QLoRA × 3) και Stage C (× 3, auto-resolved ⟺ mode == "A").
ΧΩΡΙΣ LLM/NLI/GPU — καθαρό parsing. Οι Gate/Gate_T5/B1/B2 είναι υποσύνολα των
απαντήσεων του Base, άρα δεν χρειάζονται δικό τους parsing (παράγονται στο report).

ΕΚΤΕΛΕΣΗ:  python exp4a_prepare_steps.py
"""
import os, sys, json, glob
import pandas as pd
sys.path.insert(0, "analysis")
import nli_support as NS

STAGEB_DIR = os.getenv("STAGEB_DIR", "../revision_experiment1/logs/stageB")
STAGEC_DIR = os.getenv("STAGEC_DIR", "../revision_experiment1/logs/stageC")
OUT  = "analysis/steps.jsonl"
KEEP = ("stage", "config", "run", "question_id", "cited_ids", "retrieved_ids")


def is_answered(r):
    """«Απάντηση» = Mode-A (Stage B) ή auto-resolved (Stage C: abstain=False & escalated=False)."""
    if "mode" in r:
        return r.get("mode") == "A"
    return (not r.get("abstain", False)) and (not r.get("escalated", False))


def citation_valid(r):
    """Από το log αν υπάρχει· αλλιώς υπολογίζεται (όλα τα cited ∈ retrieved) — Stage C."""
    if "citation_valid" in r:
        return bool(r["citation_valid"])
    cited = r.get("cited_ids") or []
    retr = set(r.get("retrieved_ids") or [])
    return bool(cited) and all(c in retr for c in cited)


def iter_logs():
    for d in (STAGEB_DIR, STAGEC_DIR):
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            yield p


def main():
    os.makedirs("analysis", exist_ok=True)
    logs = list(iter_logs())
    if not logs:
        print(f"[ERROR] δεν βρέθηκαν logs σε {STAGEB_DIR} / {STAGEC_DIR}"); sys.exit(1)
    print(f"[logs] {len(logs)} αρχεία: {[os.path.basename(p) for p in logs]}")
    rows, n_ans, n_zero = [], 0, 0
    for log in logs:
        for line in open(log, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if not is_answered(r):          # μόνο απαντήσεις (όχι abstain/escalated/invalid)
                continue
            n_ans += 1
            cv = citation_valid(r)
            steps = NS.parse_steps(r.get("raw_output", ""))
            if not steps:
                n_zero += 1
            for k, s in enumerate(steps):
                rows.append({**{c: r.get(c) for c in KEEP}, "citation_valid": cv, "step_idx": k, **s})
    df = pd.DataFrame(rows)
    df.to_json(OUT, orient="records", lines=True, force_ascii=False)
    print(f"[OK] {OUT}: {len(df)} βήματα από {n_ans} Mode-A απαντήσεις")

    # ── Έλεγχος: κατανομή πλήθους βημάτων ανά απάντηση (αναμ. ~3–8) ──
    if len(df):
        per = df.groupby(["stage", "config", "run", "question_id"]).size()
        print(f"[έλεγχος] βήματα/απάντηση: median={per.median():.0f} "
              f"min={per.min()} max={per.max()}  (αναμ. ~3–8)")
        ans_by_stage = df.groupby("stage").apply(
            lambda g: g.groupby(["config", "run", "question_id"]).ngroups, include_groups=False)
        print(f"[ανά stage] απαντήσεις: {ans_by_stage.to_dict()}  (πρέπει να υπάρχει και 'C')")
    if n_zero:
        print(f"[ΠΡΟΣΟΧΗ] {n_zero} Mode-A απαντήσεις χωρίς βήματα — δείτε 5 με το μάτι· "
              f"αν ο parser αποτυγχάνει σε κάποιο format, διορθώστε το nli_support.parse_steps και ξανατρέξτε.")


if __name__ == "__main__":
    main()
