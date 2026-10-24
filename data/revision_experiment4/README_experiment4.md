# Experiment 4 - Semantic grounding (NLI) & human correctness

Goes beyond *citation validity* (is the cited ID among the retrieved ones?) to
ask two deeper questions the reviewers raised:

1. **Metric 1 - semantic support:** does each generated step actually follow
   from the cited evidence? Scored automatically with an **NLI** model.
2. **Metric 2 - human correctness:** would a human engineer judge the answer a
   **correct resolution**? Scored by human annotators.

## Method

- **NLI scorer:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
  Documents are chunked (400 tokens, stride 200); for each answer step,
  `p_cited` / `p_any` = max P(entailment) over the chunks of the cited / all
  retrieved documents. A step is **supported** iff `p_cited ≥ 0.5`.
- **Human annotation:** a stratified sample of answered items is labelled by
  **two annotators**, reconciled into `adjudicated.xlsx`; agreement is reported
  with **Cohen's κappa**.
- **Uncertainty:** stratified **bootstrap (10 000 resamples)** for the
  correctness estimates.

## Placement

Put `revision_experiment4` next to `revision_experiment1` inside `data/`.
The NLI stage (4B) needs a GPU and the top-5 document text
(`retrieval_test_docs.jsonl`, regenerated from TechQA - see
`DATA_AVAILABILITY.md`); the report stage reuses the shipped `steps_nli.jsonl`.

## Steps

```bash
python exp4a_prepare_steps.py    # split Stage B/C answers into steps → steps.jsonl
python exp4b_nli.py              # NLI scoring → analysis/steps_nli.jsonl  (resumable)
python exp4b_nli_sanity.py       # mechanical positive/negative sanity check
python exp4c_sample.py           # draw the stratified human-annotation sample
python exp4_report.py            # Metric 1 + Metric 2 tables, kappa, bootstrap
```

## Outputs

| File | Contents |
|---|---|
| `logs/exp4_report.md` / `.json` | Metric 1 + Metric 2, citation validity, κappa, NLI-validation block |
| `analysis/steps_nli.jsonl` | per-step NLI scores (model-generated step text + `p_cited` / `p_any`) |
| `analysis/tables/exp4_semantic_support.csv` | semantic-support and citation-validity rates per config |
| `annotation/annotator1.xlsx`, `annotator2.xlsx`, `adjudicated.xlsx`, `calibration.xlsx` | human labels (excluded from the public repo - raw TechQA text) |

## Key findings (honest reading)

- **Support ≠ correctness.** A step can be entailed by a retrieved document yet
  still be the wrong resolution - most visible on `missU`, where answering at
  all is already a hazard.
- **Citation errors are mostly not hallucinations:** ~81 % of invalid citations
  point to a real retrieved document, just not the gold one.
- **Human correct-resolution rate ≈ 14.2 %**, well below the ~62 %
  auto-resolution rate - i.e. the system auto-resolves far more often than it is
  actually right, which is exactly why the gate/escalation and the hazard
  framing matter. Reported honestly rather than inflated.

## Notes

- `steps_nli.jsonl` stores only **model-generated step text + scores** (no raw
  corpus), so it is safe to publish; the annotation spreadsheets embed TechQA
  question/answer/technote text and are therefore **excluded** (regenerate via
  `exp4c_sample.py` + TechQA).
- The bootstrap is seeded, so the report reproduces deterministically from
  `steps_nli.jsonl` + the human labels.
