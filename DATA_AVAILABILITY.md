# Data Availability

This repository contains **all source code** and the **aggregate experimental
results** (per-question scores, metric tables, statistical outputs, figures) that
back the paper. It does **not** redistribute any raw text from the **IBM TechQA**
dataset (IBM license), nor any model weight that exceeds GitHub's 100 MB limit.
Both categories are documented below with exact pointers.

## 1. IBM TechQA (dataset & corpus)

TechQA is distributed by IBM. Download it from the official source and place
`TechQA.tar.gz` at the project root. Cite:

> Castelli et al., *The TechQA Dataset*, ACL 2020.

### Derived files excluded because they embed raw TechQA text

| Excluded file | Raw IBM text it embeds | Regenerate with |
|---|---|---|
| `**/retrieval_test_docs.jsonl` | ticket (question) + technote `content` | `revision_experiment1/exp1c_stageA_freeze.py` (TechQA + retrieval service) |
| `**/annotation/*.xlsx` | `incident` + `cited_docs` + `gold` | `revision_experiment4/exp4c_sample.py` (from `steps.jsonl` + TechQA) |
| `**/retrieval_train_oof.jsonl` | TRAIN features (precautionary) | `revision_experiment1/exp1b_oof_scores.py` — not needed: `T_dep` is published in `revision_experiment2/data/T_dep.json` |

## 2. External large artifacts (> 100 MB) - Google Drive

These are **our own** trained artifacts (no IBM text); they are too large for GitHub
and are hosted on Google Drive. They are needed only to **re-run** the GPU stages;
all reported numbers are already in the published logs/reports.

| Artifact | What it is | Link |
|---|---|---|
| `adapter/adapter_model.safetensors` | QLoRA adapter (the fine-tune; GGUF is built from this via the conversion script) | <https://drive.google.com/drive/folders/16avgBwv6DsQi9kivvlHzhx_XUHzNZ2Sa?usp=drive_link> |
| `reranker_train_only/` | leakage-free cross-encoder reranker (main) | <https://drive.google.com/drive/folders/16avgBwv6DsQi9kivvlHzhx_XUHzNZ2Sa?usp=drive_link> |
| `reranker_model_leaky/` | original (leaky) reranker - Supplementary comparison only | <https://drive.google.com/drive/folders/16avgBwv6DsQi9kivvlHzhx_XUHzNZ2Sa?usp=drive_link> |

> The served **GGUF** is produced from the adapter above (merge → convert, see the
> conversion script / runbook). The **Chroma** vector store, the **BM25** indexes
> (`bm25_tokens.jsonl`, `bm25_index.pkl`, `bm25s_index/`) and the 5 fold rerankers
> are **not** shared (rebuildable from the scripts). The **TechQA** corpus is **not**
> shared (IBM license - see §1).

## 3. What **is** included (safe to publish)

- All Python code and per-experiment READMEs.
- `retrieval_test.jsonl` - frozen protocol: `question_id`, `answerable`,
  `gold_id`, retrieved **IDs**, reranker logits, fused scores (no text).
- Stage B logs (`Base_run*.jsonl`, `QLoRA_run*.jsonl`) - model outputs, IDs, scores.
- All report JSON/MD, metric CSV tables, and figures.
- `T_dep.json`, `thresholds.json`, `steps_nli.jsonl`, `exp4_semantic_support.csv`, sampling keys.

## 4. Reproducing the analysis

With `TechQA.tar.gz` in place, the retrieval service running, and (for GPU stages)
the weights from §2, each `revision_experiment{1..4}/` package regenerates its
inputs and tables via its own `README.md`. The pure-statistics stages are
deterministic (`seed=42`) and reproduce bit-for-bit from the published logs.
