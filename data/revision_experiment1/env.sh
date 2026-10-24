# env.sh — μεταβλητές για εκτέλεση ΜΕΣΑ στο container knowledge_api.
# Χρήση:   source env.sh      (πριν τρέξετε οποιοδήποτε exp1*.py μέσα στο container)
#
# Αν τρέχετε κάτι στον HOST (WSL) αντί για μέσα στο container, άλλαξε:
#   SEARCH_URL -> http://localhost:8080/search   (αν το port 8080 είναι mapped)
#   OLLAMA_URL -> http://localhost:11434/api/chat (αν το 11434 είναι mapped)
#   TECHQA_TAR -> ./data/TechQA.tar  (ή η host διαδρομή)

# ── Δεδομένα / υπηρεσίες (τιμές ΜΕΣΑ στο container) ──
export TECHQA_TAR=/data/TechQA.tar.gz
export GGUF_PATH="/data/techqa-lora-f16.gguf"
export SEARCH_URL=http://localhost:8080/search          # το api σερβίρει τον εαυτό του
export OLLAMA_URL=http://ollama:11434/api/chat          # service name στο docker network
export MAIN_PY_PATH=/app/main.py                        # RAG_SYSTEM_PROMPT / ABSTAIN_PHRASE

# ── LLM παράμετροι (== main.py) ──
export NUM_CTX=8192
export NUM_PREDICT=1024
export LLM_DOC_CHAR_LIMIT=1600
export LLM_TICKET_CHAR_LIMIT=3000

# ── Πηγές id για το 1Α (απόλυτες /data διαδρομές — χρησιμοποιήστε τα RAW .csv, όχι Excel) ──
export SFT_META=/data/sft_dataset_meta.csv
export STAGEA_CSV=/data/eval_results.csv
export STAGEB_CSV=/data/generation_eval_results.csv
export STAGEC_MAP=/data/jira_ticket_map.csv

# ── Για το 1Ε --mode jira (τα modules σας είναι στο /data) ──
export APP_DIR=/app
export DATA_DIR=/data

# ── Gate OFF στα κύρια runs (ο Gate υπολογίζεται post-hoc) ──
unset RAG_GATE_THRESHOLD

echo "[env] TECHQA_TAR=$TECHQA_TAR"
echo "[env] SEARCH_URL=$SEARCH_URL  OLLAMA_URL=$OLLAMA_URL"
echo "[env] NUM_CTX=$NUM_CTX NUM_PREDICT=$NUM_PREDICT  (gate OFF)"
