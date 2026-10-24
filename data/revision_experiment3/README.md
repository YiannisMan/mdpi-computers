# Πείραμα 3 — Δύο baselines & καμπύλη gate

Απαντά σε R1-M2, R1-M7, R2, R3-4, R3-7, R4-9. **Χωρίς εκπαίδευση LLM.** Όλα τα baselines είναι
«φίλτρο πριν τον Base»: `abstain = (score < κατώφλι) Ή abstain(Base)`.

- **Gate** — max reranker logit (από τα logs 1Δ).
- **B1** — answerability classifier (logistic regression στα features του retrieval). *Καθαρή ανάλυση.*
- **B2** — CRAG-style evaluator (monoT5). *Βαρύ: monoT5 + corpus + GPU.*

---

## 0. Τοποθέτηση — [HOST]
Βάλτε το `revision_experiment3` **δίπλα** στο `revision_experiment1`, μέσα στο `data/`
(→ `/data/revision_experiment3`). Οι διαδρομές στο `env3.sh` δείχνουν στο `../revision_experiment1/...`.

## Πού τρέχει
Ο πυρήνας (3Α-Gate/B1, 3Β) είναι καθαρή ανάλυση (pandas/numpy/scikit-learn/matplotlib) — τρέχει
**στον host venv ή in-container**. Το **B2 (monoT5)** θέλει GPU + transformers + corpus → in-container.

```powershell
docker exec -it knowledge_api bash -c "pip install -q pandas numpy scikit-learn matplotlib statsmodels; cd /data/revision_experiment3 && source env3.sh && <ΕΝΤΟΛΗ>"
```

---

## Βήματα

### 3Α-1 — Gate + B1 classifier (TRAIN, χωρίς LLM)
```powershell
docker exec -it knowledge_api bash -c "pip install -q pandas numpy scikit-learn; cd /data/revision_experiment3 && source env3.sh && python exp3a_prepare.py"
```
Παράγει `data/b1_classifier.pkl` + `data/thresholds_partial.json` (T_dep, p*, AUROC Gate/B1).
**Έλεγχος:** τα train-OOF F-Abst ≤ 25% και τα AUROC >> 0,5.

### 3Α-2 — B2 CRAG/monoT5 (ΠΡΟΑΙΡΕΤΙΚΟ, βαρύ)
```powershell
docker exec -it knowledge_api bash -c "pip install -q transformers sentencepiece; cd /data/revision_experiment3 && source env3.sh && python exp3a_b2_crag.py"
```
Κατεβάζει το `castorini/monot5-base-msmarco` (~850MB) και βαθμολογεί τα top-5 σε TRAIN & TEST.
Παράγει `data/crag_scores_{train,test}.json` + `data/crag_thresholds.json`.
> **ΠΡΟΣΟΧΗ (corpus):** για τα TRAIN κείμενα χρειάζεται το corpus. Ρυθμίστε `CORPUS_PATH` στο
> `env3.sh` και, αν το format σας διαφέρει, **προσαρμόστε τη `load_corpus()`** στο
> `exp3a_b2_crag.py` (τυπώνει πόσα technotes φόρτωσε — αν 0, δεν ταίριαξε το format).
> Το TEST scoring δουλεύει χωρίς corpus (`--test-only`). Αν δεν τρέξετε το B2, το 3Β συνεχίζει
> με **Gate + B1** μόνο.

### 3Α-3 — Πάγωμα κατωφλίων
```powershell
docker exec -it knowledge_api bash -c "cd /data/revision_experiment3 && python exp3_freeze.py"
# commit ΠΡΙΝ το 3Β:
git add data/revision_experiment3/data/thresholds.json
git commit -m "Freeze Exp3 thresholds (gate/B1[/B2])"
```

### 3Β — Εφαρμογή στο TEST + καμπύλες + iso-F-Abst + AUROC + report
```powershell
docker exec -it knowledge_api bash -c "pip install -q pandas numpy scikit-learn matplotlib statsmodels; cd /data/revision_experiment3 && source env3.sh && python exp3b_report.py"
```

| Αρχείο | Τι θα βρει ο καθηγητής |
|---|---|
| `analysis/figures/abstention_tradeoff.pdf/.png` | καμπύλες Gate/B1/(B2) + σημεία Base/QLoRA/Gate/B1 |
| `analysis/tables/exp3_iso_fabst.csv` | ανά μέθοδο: T*, F-Abst, C-Abst, Δ της QLoRA + 95% CI |
| `analysis/tables/exp3_auroc.csv` | AUROC (TEST) ανά score + CI |
| `analysis/tables/exp3_table4_supplement.csv` | γραμμές B1/B2/Gate/Gate_T5/Gate∘QLoRA (mean min–max) |
| `analysis/tables/exp3_pairwise_secondary.csv` | QLoRA vs B1/B2: McNemar + bootstrap + Holm |
| `analysis/tables/exp3_curve_*.csv` | οι καμπύλες (δεδομένα) |
| `logs/exp3_report.md` / `.json` | ανθρώπινη σύνοψη / machine-readable |

---

## Σημειώσεις / όρια
- **B1 features:** εξαιρούνται `overlap_max/rrf_top1/bm25/dense` (δεν αποθηκεύονται στο frozen
  TEST) για συνέπεια TRAIN↔TEST. Αν θέλετε να τα συμπεριλάβετε, επεκτείνετε το frozen schema στο 1Γ.
- **B2 Ambiguous:** αντιμετωπίζεται ως score-filter (θ_lo) — βρίσκεται ήδη πάνω στην ίδια καμπύλη·
  η πλήρης τριμερής CRAG (Ambiguous → filtered-docs generation) είναι άλλο σημείο της καμπύλης.
- **Κανένα κατώφλι δεν επιλέγεται στο TEST.** Η iso-F-Abst σύγκριση δεν εξαρτάται από το train κατώφλι.

## Αρχεία πακέτου
```
ANALYSIS_PLAN_exp3.md   προ-καταχώριση
env3.sh                 διαδρομές + παράμετροι
exp3a_prepare.py        Gate + B1 (TRAIN, χωρίς LLM)
exp3a_b2_crag.py        B2 monoT5 (προαιρετικό, βαρύ)
exp3_freeze.py          -> thresholds.json
exp3b_report.py         3Β: καμπύλες/iso/AUROC/pairwise/σχήμα/report
analysis/abst_lib.py    κοινή βιβλιοθήκη (Wilson, McNemar, bootstrap, Holm, curve, select_threshold)
analysis/exp3_lib.py    επεκτάσεις (features, classifier, AUROC, curve_tradeoff, iso_f_abst)
data/ logs/ analysis/{tables,figures}/   έξοδοι
```
