# env4.sh — μεταβλητές για το Πείραμα 4 (source env4.sh πριν τα exp4_*.py).
# Τοποθετήστε το revision_experiment4 ΔΙΠΛΑ στα revision_experiment1/3 (μέσα στο data/).

# ── είσοδοι (logs & δεδομένα Πειράματος 1/3) ──
export STAGEB_DIR=../revision_experiment1/logs/stageB                    # Base, QLoRA × 3 (1Δ)
export STAGEC_DIR=../revision_experiment1/logs/stageC                    # Stage C × 3 (1Ε)
export SUBS_CSV=../revision_experiment1/data/test_subgroups.csv          # present/missA/missU (1Γ)
export FROZEN_PATH=../revision_experiment1/data/retrieval_test.jsonl     # top-5 + gate_max_logit (1Γ)
export TEST_DOCS=../revision_experiment1/data/retrieval_test_docs.jsonl  # κείμενο top-5 (cited/retrieved)
export OOF_PATH=../revision_experiment1/data/retrieval_train_oof.jsonl   # TRAIN ids (έλεγχος NLI)
export THRESHOLDS=../revision_experiment3/data/thresholds.json           # T_dep (παράγωγο Gate)
export TECHQA_TAR=/data/TechQA.tar.gz                                    # dev_Q_A.json (gold/incident)
export CORPUS_PATH=TechQA/technote_corpus/full_technote_collection.txt.bz2  # member μέσα στο TAR (έλεγχος NLI)

# ── παράμετροι NLI (ΠΑΓΩΜΕΝΕΣ στο ANALYSIS_PLAN πριν τρέξει οτιδήποτε) ──
export NLI_MODEL=MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
export NLI_THR=0.5          # supported ⟺ p_cited ≥ 0,5 (ευαισθησία: 0,3 και 0,7)
export CHUNK=400            # μέγεθος chunk (tokens)
export STRIDE=200           # βήμα chunk

# ── δειγματοληψία (4Γ) ──
export SAMPLE_SEED=2026
export N_PER_STRATUM=25
export N_CALIB=5
export T5=5.0              # Gate_T5 (ιστορία κατωφλίου, συνέχεια Πειράματος 3)

echo "[env4] STAGEB=$STAGEB_DIR STAGEC=$STAGEC_DIR"
echo "[env4] NLI=$NLI_MODEL thr=$NLI_THR chunk=$CHUNK/$STRIDE seed=$SAMPLE_SEED"
