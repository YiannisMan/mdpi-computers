# Experiment 2 - Repetition & statistical testing

**Pure analysis** over the Stage B logs from Experiment 1 - **no new inference,
no service** (no Ollama / reranker needed). Adds the statistical rigor the
reviewers asked for (repetition, confidence intervals, paired significance
tests, sensitivity).

Configs: **Base, Gate (T_dep), Gate_T5 (T = 5.0), QLoRA** - all derived from the
same logs. `Gate` is obtained from the Base logs:
`abstain = (gate_max_logit < T_dep) OR abstain(Base)`.

## Placement

Put `revision_experiment2` **next to** `revision_experiment1`, inside `data/`.
The paths in `env2.sh` point to `../revision_experiment1/...`, so the two must
be siblings. Being pure pandas, it runs fine inside the `knowledge_api`
container (where `/data/TechQA.tar.gz` is available for ROUGE-L) or in any host
venv with `pandas numpy statsmodels`.

## Steps

```bash
# 1 - select the gate threshold T_dep on TRAIN only (no peek at TEST)
python exp2_select_T.py          # reads retrieval_train_oof.jsonl → data/T_dep.json

# 2 - Table 4 + pairwise tests + sensitivity
python exp2_report.py            # reads data/T_dep.json automatically
```

`T_dep` is chosen on the 5-fold **TRAIN out-of-fold** scores as the smallest
threshold keeping **F-Abst ≤ 25 %**. The selection is a deterministic sorted
sweep - no randomness, no TEST.

## Outputs

| File | Contents |
|---|---|
| `logs/exp2_report.md` / `.json` | human + machine summary (Table 4, Wilson CIs, pairwise, sensitivity, `T_dep`) |
| `analysis/tables/exp2_table4.csv` | C-Abst (missing / missA / missU), Hazard, F-Abst, citation, ROUGE-L - mean (min–max) |
| `analysis/tables/exp2_table4_wilson.csv` | Wilson 95 % CI on the majority vote, per config × subpopulation |
| `analysis/tables/exp2_pairwise_main.csv` | 3 pairs × 2 metrics: Δpp, bootstrap CI, McNemar p, Holm-corrected p, verdict |
| `analysis/tables/exp2_pairwise_sensitivity.csv` | per-run McNemar p (robustness of the majority-vote conclusion) |
| `analysis/tables/exp2_per_run.csv` | rates per (config, run) |

## Determinism

The paired bootstrap is **seeded (`seed = 42`, 10 000 resamples)** and all other
statistics are analytic, so the entire experiment reproduces **bit-for-bit**
from the published Stage B logs.

## What to note

- Main comparison is **Gate (T_dep)**. `Gate_T5` (T = 5.0) is kept only for
  continuity with the original paper; because the new reranker is scaled
  differently, it may abstain far more.
- Verdict wording: *significantly higher/lower* only when Holm p < 0.05 **and**
  the bootstrap CI excludes 0; otherwise *no detectable difference* (never
  "equivalent").
- With ~200 incidents in `missing`, differences below ~10 pp are usually not
  detectable - reported as a limit of the evaluation, not as equivalence.
- The B1/B2 baselines (answerability classifier, CRAG) are the **secondary**
  family and live in **Experiment 3**.