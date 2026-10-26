# Experiment 3 - Secondary abstention baselines (classifier & CRAG)

Compares the system's **learned abstention** against two external
abstention/selective-prediction baselines, **at a matched operating point**
(iso-F-Abst) so the comparison is fair. This is the **secondary** baseline
family referenced by the reviewers.

Baselines:
- **B1 - answerability classifier.** A logistic model over retrieval features
  from the OOF file (max/mean reranker logit, gap, softmax entropy, RRF, overlap…)
  that decides answer vs abstain.
- **B2 - CRAG-style corrective retrieval** (monoT5 relevance), which abstains /
  escalates when retrieval confidence is low.

All operating thresholds are selected on **TRAIN** (never TEST), and every
method is then evaluated on the same frozen TEST retrieval from Experiment 1.

## Placement

Put `revision_experiment3` next to `revision_experiment1` inside `data/`
(relative paths `../revision_experiment1/...`). B2 needs the TechQA corpus, read
by streaming from `TechQA.tar.gz` — so run it inside the `knowledge_api`
container (or point `TECHQA_TAR` at the archive).

## Steps

```bash
python exp3a_prepare.py          # features / inputs for the baselines
python exp3a_b2_crag.py          # B2 (CRAG / monoT5) scoring over the corpus
python exp3_freeze.py            # select thresholds on TRAIN → data/thresholds.json
python exp3b_report.py           # iso-F-Abst comparison + AUROC + figure
```

## Outputs

| File | Contents |
|---|---|
| `logs/exp3_report.json` | summary: iso-F-Abst table, AUROC, pairwise secondary comparisons |
| `data/thresholds.json` | TRAIN-selected operating thresholds for B1 / B2 |
| `analysis/tables/exp3_iso_fabst.csv` | all methods compared at a matched F-Abst |
| `analysis/tables/exp3_auroc.csv` | ranking quality (AUROC) of the abstention scores |
| `analysis/tables/exp3_pairwise_secondary.csv`, `exp3_curve_*.csv` | pairwise tests + risk–coverage curves |
| `analysis/figures/abstention_tradeoff.png` | abstention / hazard trade-off figure |

## Key finding (honest reading)

At a matched operating point, **QLoRA's learned abstention ties the B1
classifier** (no detectable difference at iso-F-Abst); it does not dominate the
external baselines. This is reported as-is rather than overclaimed.

## Notes

- Thresholds are TRAIN-only; the TEST retrieval is the frozen one from
  Experiment 1, so numbers are directly comparable across experiments.
- The TechQA corpus (`full_technote_collection.txt.bz2`, inside `TechQA.tar.gz`)
  is stored as **multiple concatenated JSON arrays**; the loader streams it with
  `ijson(..., multiple_values=True)`.
