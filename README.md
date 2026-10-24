# Autonomous IT Incident Resolution over IBM TechQA - Leakage-free Re-evaluation

Source code and **aggregate results** for the MDPI *Computers* revision of a
retrieval-augmented IT incident-resolution system (hybrid retrieval + LLM
fine-tuning), featuring a question-disjoint, **data-leakage-free** re-evaluation
with full statistical testing.

## Experiments
- **revision_experiment1/** - leakage audit, TRAIN-only reranker + 5-fold OOF, frozen TEST retrieval, Stage B/C logs, reports.
- **revision_experiment2/** - 3-run repetition + statistics (Table 4, Wilson CIs, paired bootstrap, McNemar/Holm).
- **revision_experiment3/** - secondary abstention baselines (answerability classifier, CRAG) at iso-F-Abst + AUROC.
- **revision_experiment4/** - NLI semantic-support scoring + human annotation (correct-resolution).

Each package has its own `README.md` with exact, runnable steps.

## Reproducibility
The pure-statistics stages (Experiments 2–4 reports) are deterministic
(`seed=42` bootstrap) and reproduce **bit-for-bit** from the published logs.
The GPU stages are archived as frozen outputs (see `DATA_AVAILABILITY.md`).

## Data & large files
- **IBM TechQA** (dataset + corpus) is **not** redistributed here - IBM license.
  See `DATA_AVAILABILITY.md` for how to obtain it and regenerate derived files.
- **Model weights** (fine-tuned LLM GGUF, rerankers) exceed GitHub's 100 MB limit
  and are hosted externally; links in `DATA_AVAILABILITY.md`.

## Environment
Windows + WSL2 + Docker (service container `knowledge_api`), Ollama, NVIDIA RTX 3060.

## Citation
> (Trustworthy Agentic AI for Autonomous IT Incident Resolution: 
A Retrieval-Augmented LLM with Learned Abstention and 
Human Escalation / computers-4639708)
