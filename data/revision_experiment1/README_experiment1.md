# Experiment 1 - Leakage-free re-evaluation (provenance, reranker, Stage A/B/C)

Foundational, **question-disjoint** re-evaluation of the incident-resolution
pipeline on the TechQA TEST split (310 questions), with a strict data-leakage
audit. This package produces the **frozen retrieval** and the **Stage B/C
generation logs** that every later experiment (2–4) builds on.

## What it does

- **1A - Provenance / leakage audit.** Verifies that no TEST gold technote
  overlaps with material the reranker was trained on
  (`provenance*.csv`, `shared_gold_docs.csv`): TEST ∩ TRAIN = 0 everywhere
  except the reranker-train set, by construction.
- **1B - Reranker (TRAIN-only) + 5-fold OOF.** Trains the cross-encoder
  reranker **only on TRAIN**, then produces **out-of-fold** scores on TRAIN
  (`retrieval_train_oof.jsonl`) so that any abstention threshold can be chosen
  **without ever looking at TEST**.
- **1C - Stage A retrieval + freeze.** Runs retrieval on the 310 TEST
  questions, **freezes the top-5** (`retrieval_test.jsonl`), assigns each
  question to a subpopulation (`present` / `missA` / `missU`) in
  `test_subgroups.csv`, and reports Top-1/3/5/10 + MRR. A `--tag leaky` line
  reproduces the old (leaky) reranker for the Supplementary comparison.
- **1D - Stage B generation.** Base and QLoRA, **3 runs each**, → per-question
  logs (`logs/stageB/*.jsonl`) + a table of abstention, hazard, citation
  validity and ROUGE-L.
- **1E - Stage C (full pipeline).** QLoRA + confidence gate + escalation,
  3 runs, with **paired Stage-B-vs-Stage-C** tests and per-subgroup analysis.

## Important: decoding temperature

Decoding temperature is **fixed at 0** for *all* generation (Stage B and
Stage C). The **three repeated runs** exist solely to quantify **GPU-level
non-determinism at temperature 0** - CUDA reduction kernels are not
bit-identical across runs — and results are reported as **mean (min–max)**
over the 3 runs. **There is no temperature sweep and no 0.2-vs-0 comparison
anywhere in this work.**

## Subpopulations

Every answerable/unanswerable TEST question falls into exactly one group,
derived from the frozen retrieval:

| Group | Meaning |
|---|---|
| `present` | answerable, gold technote **in** the top-5 |
| `missA`   | answerable, gold technote **not** in the top-5 |
| `missU`   | unanswerable (no gold technote exists) |

## Placement

Place `revision_experiment1` inside the project's `data/` folder so the
retrieval service container sees it at `/data/revision_experiment1`.

## How to run (inside the `knowledge_api` container)

GPU stages need the service running plus the model weights (see the repo's
`DATA_AVAILABILITY.md`). The pure-analysis report scripts need only the
shipped logs.

```bash
# 1A - provenance
python exp1a_provenance_audit.py
python exp1a_inspect_overlaps.py

# 1B - TRAIN-only reranker + 5-fold OOF
python exp1b_reranker_train_only.py
python exp1b_oof_scores.py --make-folds
#   for each fold k = 0..4: --train-fold k  → point service at the fold model → --score-fold k
python exp1b_oof_scores.py --ks            # scale-shift (KS) sanity check

# 1C - Stage A + freeze (service = reranker_train_only)
python exp1c_stageA_freeze.py --tag main           # main: all 310 (freezes retrieval_test.jsonl)
python exp1c_stageA_freeze.py --tag leaky          # Supplementary: old (leaky) reranker

# 1D - Stage B generation (3 runs) + report
python exp1d_stageB.py --config Base  --run {1,2,3}
python exp1d_stageB.py --config QLoRA --run {1,2,3}
python exp1d_report.py

# 1E - Stage C (3 runs) + analysis + report
python exp1e_stageC.py --run {1,2,3}
python exp1e_analysis.py
python exp1e_report.py
```

Models: Base = `llama3:8b-instruct-q4_K_M`, QLoRA = `techqa-llama3` (served via
Ollama). Retrieval goes through the service `/search` endpoint.

## Key outputs

- `data/retrieval_test.jsonl` - **frozen** top-5 per TEST question (IDs, reranker
  logits, fused scores, gate logit; no text).
- `data/test_subgroups.csv`, `data/retrieval_train_oof.jsonl`, `data/folds.json`,
  `data/provenance*.csv`.
- `logs/stageB/{Base,QLoRA}_run{1,2,3}.jsonl`, `logs/stageC/C_run{1,2,3}.jsonl`.
- `logs/exp1d_report.json`, `logs/exp1e_report.json`, and the CSV tables under
  `analysis/tables/`.

## Notes

- Because temp = 0 is not bit-deterministic on GPU, all rates are reported as
  **mean (min–max)** over 3 runs; a **±5 pp regression test** validates that a
  re-built GGUF matches the frozen behaviour.
