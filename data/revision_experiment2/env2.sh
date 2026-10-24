# env2.sh — μεταβλητές για το Πείραμα 2 (source env2.sh πριν τρέξετε τα exp2_*.py).
# Το Πείραμα 2 είναι ΚΑΘΑΡΗ ΑΝΑΛΥΣΗ (κανένα LLM/service) πάνω στα logs του Πειράματος 1.
# Τοποθετήστε το revision_experiment2 ΔΙΠΛΑ στο revision_experiment1 (μέσα στο data/),
# ώστε οι σχετικές διαδρομές ../revision_experiment1/... να ισχύουν.

export STAGEB_DIR=../revision_experiment1/logs/stageB            # logs 1Δ (Base, QLoRA × 3)
export SUBS_CSV=../revision_experiment1/data/test_subgroups.csv  # present/missA/missU (1Γ)
export OOF_PATH=../revision_experiment1/data/retrieval_train_oof.jsonl  # 5-fold OOF (1Β)
export TECHQA_TAR=/data/TechQA.tar.gz                           # για ROUGE-L (gold answers)
export TDEP_JSON=data/T_dep.json                               # γράφεται από exp2_select_T.py
export NBOOT=10000                                             # paired bootstrap resamples
export T5=5.0                                                  # Gate_T5 (συνέχεια με το άρθρο)

echo "[env2] STAGEB_DIR=$STAGEB_DIR"
echo "[env2] OOF_PATH=$OOF_PATH  TECHQA_TAR=$TECHQA_TAR"
echo "[env2] NBOOT=$NBOOT  T5=$T5"
