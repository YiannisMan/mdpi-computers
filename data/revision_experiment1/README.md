# Πείραμα 1 — Επανάληψη χωρίς διαρροή δεδομένων (Stage A, B, C)

Υλοποίηση του Πειράματος 1 του σχεδίου αναθεώρησης (MDPI Computers 4639708), ρυθμισμένη
για τη **δική σας** δομή project (WSL2 + Docker compose: `chromadb`, `api`/`knowledge_api`,
`ollama`, `jira_worker` + RTX 3060). Κανένας αριθμός του άρθρου δεν γράφεται με το χέρι —
όλα βγαίνουν από τα logs.

> **Σύμβαση αυτού του οδηγού:** κάθε βήμα έχει ετικέτα **πού** τρέχει:
> **[HOST]** = στο WSL shell σας, στη ρίζα του project · **[knowledge_api]** = ΜΕΣΑ στο
> container `knowledge_api` μέσω `docker exec`.

---

## 0. Τοποθέτηση φακέλου (μία φορά)  — [HOST]

Ο φάκελος του πειράματος μπαίνει **μέσα στο `data/`** του project, ώστε να φαίνεται στο
container ως `/data/revision_experiment1` (το `data/` είναι ήδη bind-mounted στο `/data`).

```bash
# από τη ρίζα του project (εκεί που είναι το docker-compose.yml)
unzip revision_experiment1.zip            # αν δεν το έχετε ήδη αποσυμπιέσει
mv revision_experiment1 data/             # -> data/revision_experiment1
ls data/revision_experiment1              # πρέπει να δείτε exp1*.py, env.sh, analysis/, ...

# επιβεβαίωση ότι φαίνεται μέσα στο container:
docker exec knowledge_api ls /data/revision_experiment1
```

Η δομή που προκύπτει (host ⇄ container):

| HOST (ρίζα project)                         | Μέσα στο container      |
|---------------------------------------------|-------------------------|
| `app/`                                      | `/app`                  |
| `app/main.py`, `app/jira_worker.py`         | `/app/main.py` κ.λπ.    |
| `app/reranker_model/`                       | `/app/reranker_model`   |
| `data/`                                     | `/data`                 |
| `data/TechQA.tar`, `data/*.csv`             | `/data/TechQA.tar` κ.λπ.|
| `data/revision_experiment1/`                | `/data/revision_experiment1` |

Οι μεταβλητές που δείχνουν στα δεδομένα/υπηρεσίες είναι στο **`env.sh`** — το κάνετε
`source` μία φορά σε κάθε `docker exec` πριν τρέξετε script.

**Έλεγχος ότι τα paths του `env.sh` ταιριάζουν στα αρχεία σας** (ονόματα όπως τα έχετε):
```bash
docker exec knowledge_api bash -c 'ls -la /data/TechQA.tar /data/sft_dataset_meta.csv \
  /data/eval_results.csv /data/generation_eval_results.csv /data/jira_ticket_map.csv'
```
Αν κάποιο έχει άλλο όνομα, διορθώστε το αντίστοιχο `export` στο `data/revision_experiment1/env.sh`.

---

## Επιβεβαιωμένο έναντι του κώδικά σας (main.py, build_sft_dataset.py, jira_worker.py, evaluate_generation.py)
- **Prompt/context:** `stage_bc_hooks.build_messages` == `build_rag_messages/build_rag_context` (main.py).
- **Parser (ορίζει C/F-Abst):** `parse_output` == `evaluate_generation.is_abstention` + `ID_PATTERN` (faithful = cited ⊆ sources).
- **Stage C decision:** `decide()` == `jira_worker.decide()` (MODE B Ή gate Ή faith, bracketed cites).
- **Παράμετροι:** `num_ctx=8192`, `num_predict=1024` (== main.py). **Gate OFF** στα κύρια runs.
- **Ollama:** από μέσα στο container η διεύθυνση είναι `http://ollama:11434` (service name), **όχι** `localhost`.
- **/search:** από μέσα στο `knowledge_api` είναι `http://localhost:8080/search` (το ίδιο το container).
- **temperature:** main.py default **0.2**, άρθρο/plan **0** → δείτε §temperature (απόφαση ΠΡΙΝ το πάγωμα).
- **1Α:** `build_sft_dataset` φορτώνει **μόνο** `training_Q_A.json` & answerable → SFT ∩ TEST = ∅,
  QLoRA adapter **αυτούσιος**· τα MODE-B δεν προέρχονται από unanswerable.

---

## Μοτίβο εκτέλεσης μέσα στο container

Κάθε script τρέχει έτσι (το `-it` δίνει μπάρες προόδου· το `source env.sh` ορίζει τα paths):

```bash
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && <ΕΝΤΟΛΗ>"
```

Για μεγάλα runs που δεν θέλετε να κοπούν αν κλείσει το terminal, προσθέστε το flag
`-d` **ή** τυλίξτε με `nohup ... &` μέσα στο `bash -c`. Παρακάτω δίνεται η απλή μορφή.

---

## Σειρά εκτέλεσης

### 0) Πάγωμα πλάνου + περιβάλλον (ΠΡΙΝ από οτιδήποτε στο TEST)

**[HOST]** — το `capture_env.py` χρειάζεται `git`, `nvidia-smi`, `ollama`, `docker`, που
είναι στον host:
```bash
# από τη ρίζα του project (host)
cd data/revision_experiment1
PROMPTS_FILE=../../app/main.py \
GGUF_PATH=../../data/techqa-lora-f16.gguf \
ADAPTER_DIR=../../adapter \
NUM_CTX=8192 NUM_PREDICT=1024 \
python3 capture_env.py            # -> data/revision_experiment1/env.json
cd ../..
```
(προαιρετικά και το pip freeze του API container: `docker exec knowledge_api pip freeze > data/revision_experiment1/pip_freeze_api.txt`).

Κάντε commit το πλάνο + το περιβάλλον **πριν** αγγίξετε το TEST:
```bash
git add ANALYSIS_PLAN.md data/revision_experiment1/env.json
git commit -m "Freeze analysis plan + env (Exp1)"
```

### 1Α — Provenance audit (χωρίς GPU/Ollama)  — [knowledge_api]
```bash
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1a_provenance_audit.py"
```
Παράγει: `data/revision_experiment1/data/provenance.csv`, `..._intersections.csv`, `..._report.txt`.
**Έλεγχος:** στο `provenance.csv` η στήλη `intersection_with_TEST` = 0 παντού (εκτός
`reranker_train`, αναμενόμενο → γι' αυτό γίνεται το 1Β). Το SFT βγαίνει ήδη disjoint
(όλα τα qids `TRAIN_Q*`) → ο QLoRA adapter επαναχρησιμοποιείται αυτούσιος.

> Αν δείτε `n/a (δώστε path)` σε κάποια γραμμή, το αντίστοιχο csv λείπει ή έχει άλλο όνομα
> στήλης· διορθώστε το `export ..._CSV` στο `env.sh` ή τη στήλη στο `SETS` του script.

### 1Α (συμπλήρωμα) — Επιθεώρηση & καθαρισμός διπλοτύπων  — [knowledge_api]
```bash
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1a_inspect_overlaps.py"
```
Παράγει: `data/dup_questions.csv` (23 ζεύγη διπλοτύπων dev↔train), `data/shared_gold_docs.csv`
(40 κοινά gold — καλοήθη), και `data/test_exclude_qids.txt` (οι 23 dev_qid).
**Οδηγία reviewer = report-only:** οι διπλότυπες (και τα κοινά gold) **δηλώνονται, δεν
αφαιρούνται**. Το κύριο TEST είναι οι **310**. Το `test_exclude_qids.txt` χρησιμεύει μόνο για
(α) την αναφορά στο κείμενο και (β) την προαιρετική ανάλυση ευαισθησίας στις 287
(`--exclude-dups` στο 1Γ, `--exclude` στο 1Ε). Προ-καταχώριση: `ANALYSIS_PLAN.md §1.1`.

### 1Β — Reranker ΜΟΝΟ-TRAIN + 5-fold OOF  — [knowledge_api] (GPU)

Το `knowledge_api` έχει ήδη `sentence-transformers`+`torch`+πρόσβαση GPU και σερβίρει το
`/search` στον εαυτό του, οπότε η εκπαίδευση γίνεται μέσα του.

```bash
# 1) reranker μόνο σε answerable TRAIN -> models/reranker_train_only/
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1b_reranker_train_only.py"

# 2) 5-fold out-of-fold (για το κατώφλι T του gate + baselines Πειράματος 3)
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1b_oof_scores.py --make-folds"
```

Για **κάθε** fold k=0..4 χρειάζεται ο «χορός» του reranker (το service διαβάζει τον reranker
από `/app/reranker_model` κατά την εκκίνηση):

```bash
# (α) εκπαίδευση fold k  [knowledge_api, GPU]
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1b_oof_scores.py --train-fold ${k}"          # -> models/reranker_fold${k}/

# (β) ΒΑΛΕ τον fold-k reranker στο service  [HOST]
cp -r data/revision_experiment1/models/reranker_fold${k} app/reranker_model_fold${k}
rm -rf app/reranker_model && cp -r app/reranker_model_fold${k} app/reranker_model
docker compose restart api        # ή: docker restart knowledge_api

# (γ) βαθμολόγησε ΜΟΝΟ το fold k  [knowledge_api]
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1b_oof_scores.py --score-fold ${k}"          # -> data/retrieval_train_oof.jsonl
```
(το `--ks` τρέχει **μετά** το 1Γ). **Σημ.:** τα features `bm25_max/dense_max` δεν εκτίθενται
από το `/search` → γράφονται `null` (δεν χρειάζονται για το Πείραμα 1).

### 1Γ — Stage A + πάγωμα retrieval  — [knowledge_api] (service με reranker ΜΟΝΟ-TRAIN)

Πρώτα βάλτε τον **reranker_train_only** στο service (κρατήστε αντίγραφο του παλιού):
```bash
# [HOST] — αντίγραφο ασφαλείας του ΠΑΛΙΟΥ (leaky) reranker + τοποθέτηση του νέου
cp -r app/reranker_model app/reranker_model_leaky
rm -rf app/reranker_model && cp -r data/revision_experiment1/models/reranker_train_only app/reranker_model
docker compose restart api
```
```bash
# [knowledge_api] — Stage A + πάγωμα (ΚΥΡΙΟ: όλες οι 310)
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1c_stageA_freeze.py --tag main"
```
Παράγει & **παγώνει** (στις **310** — report-only): `data/retrieval_test.jsonl`,
`data/retrieval_test_docs.jsonl`, `data/test_subgroups.csv`. Τα 1Δ/1Ε διαβάζουν αυτά τα αρχεία
→ τρέχουν στις 310.
Για την **ανάλυση ευαισθησίας στις 287** (Supplementary, ξεχωριστά αρχεία `_clean287`,
δεν αγγίζει το κύριο παγωμένο): `python exp1c_stageA_freeze.py --tag main --exclude-dups`.
Κάντε commit:
```bash
# [HOST]
git add data/revision_experiment1/data/retrieval_test.jsonl \
        data/revision_experiment1/data/retrieval_test_docs.jsonl \
        data/revision_experiment1/data/test_subgroups.csv
git commit -m "Freeze TEST retrieval (Exp1C)"
```

**Leaky γραμμή για το Supplementary** (service με τον ΠΑΛΙΟ reranker):
```bash
# [HOST] επαναφορά του παλιού reranker
rm -rf app/reranker_model && cp -r app/reranker_model_leaky app/reranker_model
docker compose restart api
# [knowledge_api]
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1c_stageA_freeze.py --tag leaky"
```
> **Για τα Stage B/C χρειάζεται ο reranker_train_only** (το παγωμένο top-5 βγήκε με αυτόν).
> Αφού τελειώσετε το leaky, **ξαναβάλτε** τον train_only και restart (όπως στην αρχή του 1Γ).

Έλεγχος μετατόπισης κλίμακας (μετά το 1Γ):
```bash
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1b_oof_scores.py --ks"
```
**Αναμενόμενο:** recall@5 πιθανότατα < 65% — ΟΚ (περισσότερα GT-missing = ισχυρότερο τεστ
abstention). Αν < 40%, ελέγξτε index/πεδίο gold id πριν αναφέρετε.

### Πείραμα 3Α (παρεμβάλλεται — χωρίς LLM): επιλογή `T_dep`
Από το `data/retrieval_train_oof.jsonl` επιλέγεται το κατώφλι του gate + τα κατώφλια των
baselines **μόνο στο TRAIN** (ξεχωριστό παραδοτέο του Πειράματος 3).

### 1Δ — Stage B (Base + QLoRA × 3 runs)  — [knowledge_api] (Ollama)

Το Stage B χτίζει το prompt **ΑΠΟ το παγωμένο top-5** και καλεί Ollama απευθείας (το
`/resolve` θα ξανάκανε retrieval). Gate OFF. Ο reranker του service πρέπει να είναι ο
**train_only** (το παγωμένο top-5 βγήκε έτσι — δεν χρησιμοποιείται ξανά εδώ, αλλά κρατάμε
συνέπεια).

**ΒΗΜΑ 0 — regression gate** (ΠΑΛΙΟ pipeline, 160 ερωτήσεις): πρέπει να περάσει ±5pp πριν
ξεκινήσετε νέα runs. Δείξτε τα **παλιά** παγωμένα αρχεία των 160 (δικά σας):
```bash
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1d_regression_test.py --docs /data/OLD_docs.jsonl --subs /data/OLD_subs.csv \
    --base-model llama3:8b-instruct-q4_K_M --qlora-model techqa-llama3 --temp 0.2"
```

**Νέα runs — ΕΝΑ τη φορά, με restart του Ollama ΑΝΑΜΕΣΑ** (για ανεξάρτητα runs):
```bash
# [HOST] restart ollama + μονό parallelism (πριν από ΚΑΘΕ run)
docker compose restart ollama
docker exec -d ollama bash -c "OLLAMA_NUM_PARALLEL=1 true"   # ή ορίστε το στο compose env του ollama

# [knowledge_api] ένα config/run τη φορά (επαναλάβετε για run 2,3 και για τα δύο configs)
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1d_stageB.py --config Base  --model llama3:8b-instruct-q4_K_M --run 1 --temp 0"
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1d_stageB.py --config QLoRA --model techqa-llama3            --run 1 --temp 0"
```
Παράγει: `logs/stageB/{Base,QLoRA}_run{1,2,3}.jsonl`. Είναι **resumable** (αν κοπεί, ξανατρέξτε
την ίδια εντολή — συνεχίζει). Έλεγχοι: 310 γραμμές/config/run, χωρίς διπλούς qid, `truncated≈0`.

### 1Ε — Stage C (× 3 runs)  — [knowledge_api] (Ollama)

Δύο λειτουργίες, και οι δύο με τον αυθεντικό `decide()` (MODE B Ή gate Ή faith) και
ΠΑΓΩΜΕΝΟ retrieval:

```bash
# (α) generate — ΠΛΗΡΩΣ εκτελέσιμη τώρα (χωρίς Jira): frozen -> Ollama -> decide()
#     restart ollama πριν από ΚΑΘΕ run (όπως στο 1Δ)
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1e_stageC.py --mode generate --model techqa-llama3 --run 1 --temp 0"
# ... run 2, 3

# (β) jira — πλήρης βρόχος μέσω των ΔΙΚΩΝ ΣΑΣ modules (pseudo_users_jira.py/jira_worker.py),
#     με call_resolve monkeypatched σε frozen_resolve (ο worker ΔΕΝ ξανακάνει retrieval).
#     Τα modules εισάγονται από /app & /data (APP_DIR/DATA_DIR στο env.sh):
docker exec -it knowledge_api bash -c "cd /data/revision_experiment1 && source env.sh && \
  python exp1e_stageC.py --mode jira --model techqa-llama3 --run 1 --temp 0"
```
Παράγει: `logs/stageC/C_run{1,2,3}.jsonl`.

### Ανάλυση 1Ε  — [HOST] (χρειάζεται pandas)

Το `exp1e_analysis.py` θέλει `pandas`/`numpy` (+προαιρετικά `scipy`). Τα logs είναι ήδη
ορατά στον host κάτω από `data/revision_experiment1/logs/`. Τρέξτε με ένα host venv:
```bash
# [HOST] (μία φορά) python3 -m venv .venv_analysis && source .venv_analysis/bin/activate && pip install numpy pandas scipy statsmodels scikit-learn
cd data/revision_experiment1
python exp1e_analysis.py          # ΚΥΡΙΟ (310): paired B–C (agreement/kappa/McNemar + θόρυβος βάσης) + πίνακας Stage C
# + ευαισθησία 287 (post-hoc, χωρίς νέο LLM):
python exp1e_analysis.py --exclude data/test_exclude_qids.txt
cd ../..
```
> Εναλλακτικά μέσα στο container: `docker exec -it knowledge_api bash -c "pip install pandas numpy scipy && cd /data/revision_experiment1 && python exp1e_analysis.py"`.

---

## §temperature — ΣΗΜΑΝΤΙΚΟ (απόφαση ΠΡΙΝ το πάγωμα)
Το `main.py` έχει **`LLM_TEMPERATURE` default = 0.2**, ενώ το άρθρο/`ANALYSIS_PLAN` δηλώνει
**temperature 0**. Επιβεβαιώστε με τι έτρεξαν οι **αρχικοί** αριθμοί:
- Το **regression test** τρέχει με την **αρχική** τιμή → `--temp 0.2` (default).
- Τα **νέα runs** με ό,τι δηλώνει το `ANALYSIS_PLAN` (`--temp 0`). Αν το regression περάσει
  στα 0.2 αλλά όχι στα 0, ευθυγραμμίστε: είτε αλλάξτε το `ANALYSIS_PLAN` σε 0.2 και τρέξτε
  τα νέα runs στα 0.2, είτε κρατήστε 0 και αναφέρετε ρητά τη μικρή διαφορά. (Οι δύο πρέπει να
  συμφωνούν πριν παγώσετε.)

## Κριτήριο ολοκλήρωσης Πειράματος 1
- `data/provenance.csv` με μηδενικές τομές με το TEST.
- `data/retrieval_test.jsonl` παγωμένο **με commit**.
- Stage A με Wilson CI· regression test **περασμένο**.
- logs Stage B (Base, QLoRA × 3) και Stage C (× 3) πλήρη, **χωρίς truncation**.
- Πίνακας συμφωνίας B–C **με θόρυβο βάσης**.

## Τι μπαίνει στο άρθρο (από το πρωτόκολλο)
- Νέος πίνακας «Data provenance and disjointness» (§3.2).
- Νέος Table 2 (leakage-free Stage A)· παλιός → Supplementary ως «development-time diagnostic».
- Νέοι Tables Stage B/C· δήλωση disjointness.
- Αγγλικά (έτοιμα): «All retrieval and generation experiments were re-run on a
  question-disjoint split. The cross-encoder was trained exclusively on TechQA training
  questions, and all reported results use the 310 questions of the TechQA development file
  as a held-out test set.»

## Πού τρέχει τι — σύνοψη

| Script | Πού | Εντολή (συνοπτικά) |
|---|---|---|
| `capture_env.py` | **HOST** | `python3 capture_env.py` (paths `../../app`, `../../data`, `../../adapter`) |
| `exp1a_provenance_audit.py` | **knowledge_api** | `python exp1a_provenance_audit.py` |
| `exp1b_reranker_train_only.py` | **knowledge_api** (GPU) | `python exp1b_reranker_train_only.py` |
| `exp1b_oof_scores.py` | **knowledge_api** (+reranker dance ανά fold) | `--make-folds` / `--train-fold k` / `--score-fold k` / `--ks` |
| `exp1c_stageA_freeze.py` | **knowledge_api** (service=train_only, μετά leaky) | `--tag main` / `--tag leaky` |
| `exp1d_regression_test.py` | **knowledge_api** (Ollama) | `--docs ... --subs ... --temp 0.2` |
| `exp1d_stageB.py` | **knowledge_api** (Ollama) | `--config Base/QLoRA --model ... --run k --temp 0` |
| `exp1e_stageC.py` | **knowledge_api** (Ollama) | `--mode generate/jira --model ... --run k --temp 0` |
| `exp1e_analysis.py` | **HOST** (venv με pandas) | `python exp1e_analysis.py` |

## Ο «χορός» του reranker (ποιος φορτώνεται στο service, σύνοψη)
Το service διαβάζει τον reranker από `/app/reranker_model` **στην εκκίνηση** → μετά από κάθε
αλλαγή: `docker compose restart api`.
1. **Αρχή:** εκεί κάθεται ο ΠΑΛΙΟΣ (leaky). Κρατήστε αντίγραφο → `app/reranker_model_leaky`.
2. **OOF (ανά fold k):** βάλτε `reranker_fold${k}` → restart → `--score-fold k`.
3. **1Γ main:** βάλτε `reranker_train_only` → restart → `--tag main` (+ Stage B/C).
4. **1Γ leaky:** βάλτε πίσω `reranker_model_leaky` → restart → `--tag leaky`, μετά ξανά train_only.

## Αρχεία
```
ANALYSIS_PLAN.md            προ-καταχώριση (commit ΠΡΙΝ το TEST)
env.sh                      exports για εκτέλεση ΜΕΣΑ στο knowledge_api (source env.sh)
capture_env.py              -> env.json (τρέχει στον HOST)
exp1a_provenance_audit.py   1Α
exp1b_reranker_train_only.py / exp1b_oof_scores.py   1Β
exp1c_stageA_freeze.py      1Γ (+ freeze + subgroups + docs companion)
exp1d_regression_test.py    1Δ βήμα 0 (regression gate· --temp)
exp1d_stageB.py             1Δ (Base/QLoRA × 3· frozen retrieval + Ollama)
exp1e_stageC.py             1Ε (--mode generate | jira· decide() αυθεντικό)
exp1e_analysis.py           1Ε ανάλυση (paired B–C + πίνακας)  [HOST/pandas]
stage_bc_hooks.py           ΠΙΣΤΑ αντίγραφα prompt/parser/decide του κώδικά σας
analysis/abst_lib.py        κοινή βιβλιοθήκη (Wilson, McNemar, bootstrap, Holm, kappa)
data/ logs/ models/ annotation/   έξοδοι (κάτω από /data/revision_experiment1/)
```
