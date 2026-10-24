# Πείραμα 4 — Semantic grounding & correctness

Απαντά σε **R1-M3, R2, R3-3, R4-6, R4-7, R4-8**. **Χωρίς νέο generation** — διαβάζει τα logs των
Πειραμάτων 1 & 3. Δύο νέοι αριθμοί για το Abstract: **semantic support rate** (NLI) και
**correct-resolution rate** (ανθρώπινη αξιολόγηση 50 απαντήσεων).

## 0. Τοποθέτηση — [HOST]
Βάλτε το `revision_experiment4` **δίπλα** στα `revision_experiment1` και `revision_experiment3`
(μέσα στο `data/`). Οι διαδρομές στο `env4.sh` δείχνουν σε αυτά.

## Ροή (4 φάσεις + χειροκίνητη αξιολόγηση)

```
4Α parse ──▶ 4Β NLI ──▶ 4Γ δείγμα/φύλλα ──▶ [ΧΕΙΡΟΚΙΝΗΤΗ αξιολόγηση] ──▶ 4Δ-ΣΤ report
```

Το **NLI κομμάτι** (4Α, 4Β, και το Metric 1 του report) δίνει **αμέσως** τον αριθμό του R3.
Το **ανθρώπινο κομμάτι** (δείγμα 50 → 2 αξιολογητές → adjudicated) δίνει το correct-resolution rate.

---

### 4Α — Parse απαντήσεων σε βήματα (χωρίς GPU)
```powershell
docker exec -it knowledge_api bash -c "pip install -q pandas; cd /data/revision_experiment4 && source env4.sh && python exp4a_prepare_steps.py"
```
→ `analysis/steps.jsonl`. **Έλεγχος:** median βημάτων/απάντηση ~3–8· καμία Mode-A απάντηση με 0 βήματα
(αλλιώς διορθώστε τον parser και ξανατρέξτε).

### 4Β — Έλεγχος NLI (θετικός/αρνητικός) — ΠΡΙΝ την ανάλυση
```powershell
docker exec -it knowledge_api bash -c "pip install -q transformers sentencepiece ijson; cd /data/revision_experiment4 && source env4.sh && python exp4b_nli_sanity.py"
```
→ `analysis/tables/exp4_nli_sanity.json`. **Θέλουμε θετικό ≥ 90%, αρνητικό ≤ 10%.** Αν αποτύχει → chunking/label index/truncation.

### 4Β — NLI entailment ανά βήμα (ΒΑΡΥ, GPU — T4 αρκεί)
```powershell
docker exec -it knowledge_api bash -c "pip install -q transformers sentencepiece; cd /data/revision_experiment4 && source env4.sh && python exp4b_nli.py"
```
→ `analysis/steps_nli.jsonl` (+ p_cited, p_any). **Έλεγχος:** μοναδικά βήματα με hypothesis > 109 tokens = 0 (truncation).

### 4Γ — Στρωματοποιημένο δείγμα 50 + τυφλά φύλλα
```powershell
docker exec -it knowledge_api bash -c "pip install -q pandas openpyxl; cd /data/revision_experiment4 && source env4.sh && python exp4c_sample.py"
```
→ `annotation/sample50_key.csv` (**ΚΛΕΙΔΙ — μην το δώσετε**), `annotator1.xlsx`, `annotator2.xlsx`, `calibration.xlsx`, `strata_N.json`.

### ⟶ ΧΕΙΡΟΚΙΝΗΤΗ αξιολόγηση (δύο αξιολογητές με γνώση IT operations)
1. Διαβάστε το `annotation/instructions.md` και συμπληρώστε τα 3 παραδείγματα (TRAIN).
2. Calibration: αξιολογήστε μαζί τα 5 του `calibration.xlsx`, γράψτε διευκρινίσεις.
3. Ανεξάρτητα: ο καθένας συμπληρώνει το `annotator1.xlsx` / `annotator2.xlsx` (στήλες `S*_support`, `correctness`, `risky_action`).
4. Adjudication: φτιάξτε `annotation/adjudicated.xlsx` (τελικές ετικέτες: `item_id`, `S*_support`, `correctness`, `risky_action`).

### 4Δ-ΣΤ — Report (κ, επικύρωση NLI, μετρικές, πίνακας)
```powershell
docker exec -it knowledge_api bash -c "pip install -q pandas openpyxl scikit-learn tabulate; cd /data/revision_experiment4 && source env4.sh && python exp4_report.py"
```
→ `logs/exp4_report.md` + `.json`, `analysis/tables/exp4_semantic_support.csv`.

> Το report τρέχει το **Metric 1 (NLI)** και το **R3 number** αμέσως μετά το 4Β. Τα ανθρώπινα (κ,
> επικύρωση NLI, Metric 2) προστίθενται **αυτόματα** μόλις υπάρχουν τα συμπληρωμένα φύλλα + `adjudicated.xlsx`.

---

## Τι βρίσκει ο καθηγητής
| Αρχείο | Περιεχόμενο |
|---|---|
| `logs/exp4_report.md` / `.json` | σύνοψη: Metric 1, R3 number, paired QLoRA–Base, κ, επικύρωση NLI, Metric 2 |
| `analysis/tables/exp4_semantic_support.csv` | step support / fully supported / semantic mismatch / wrong-citation ανά config×υποπληθυσμό |
| `analysis/tables/exp4_nli_sanity.json` | θετικός/αρνητικός έλεγχος NLI |
| `analysis/steps_nli.jsonl` | όλα τα βήματα με p_cited, p_any |

## Σημειώσεις / όρια
- **Το NLI υποεκτιμά** το support σε παραφρασμένες διαδικαστικές οδηγίες → το semantic support rate είναι **κατώτερο όριο** (δηλώνεται στο §5.3).
- Το κατώφλι NLI (0,5) **δεν** ρυθμίζεται στις 50· αν κ(NLI–ανθρώπων) < 0,5, κύρια τα ανθρώπινα ποσοστά με Wilson CI.
- **Αποκλίσεις** (ρητές στο ANALYSIS_PLAN §6): B2 χωρίς regeneration → υποσύνολο Base· το report παράγει Base/QLoRA/StageC (+ Gate/Gate_T5 παράγωγα).

## Αρχεία πακέτου
```
ANALYSIS_PLAN_exp4.md   env4.sh   README.md
exp4a_prepare_steps.py  exp4b_nli.py  exp4b_nli_sanity.py  exp4c_sample.py  exp4_report.py
analysis/ abst_lib.py  nli_support.py  tables/  figures/
annotation/ instructions.md  (+ sample50_key.csv, annotator{1,2}.xlsx, calibration.xlsx, strata_N.json μετά το 4Γ)
```
