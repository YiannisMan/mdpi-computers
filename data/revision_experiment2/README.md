# Πείραμα 2 — 3 επαναλήψεις & στατιστικοί έλεγχοι

**Καθαρή ανάλυση** πάνω στα logs του Πειράματος 1Δ — **κανένα νέο inference, καμία υπηρεσία**
(δεν χρειάζεται Ollama/reranker/service). Απαντά σε R1-M5, R2, R3-2, R4-4.

Configs: **Base, Gate (T_dep), Gate_T5 (T=5.0), QLoRA** — όλα από τα ίδια logs. Το Gate
προκύπτει από τα Base logs: `abstain = (gate_max_logit < T) Ή abstain(Base)`.

---

## 0. Τοποθέτηση (μία φορά) — [HOST]

Βάλτε το `revision_experiment2` **δίπλα** στο `revision_experiment1`, μέσα στο `data/`:

```
knowledge_base_system/
  data/
    revision_experiment1/   (Πείραμα 1 — logs/stageB, data/test_subgroups.csv, data/retrieval_train_oof.jsonl)
    revision_experiment2/   (ΕΔΩ — αυτό το πακέτο)
    TechQA.tar.gz
```
```powershell
# από τη ρίζα του project
Expand-Archive revision_experiment2.zip -DestinationPath .
Move-Item revision_experiment2 data\
docker exec knowledge_api ls /data/revision_experiment2   # επιβεβαίωση ότι φαίνεται στο container
```
Οι διαδρομές στο `env2.sh` δείχνουν στο `../revision_experiment1/...` — δουλεύουν αρκεί τα δύο
να είναι αδέρφια μέσα στο `data/`.

## Πού τρέχει

Είναι καθαρή pandas-ανάλυση. Τρέξτε την **μέσα στο container** (εκεί υπάρχει το
`/data/TechQA.tar.gz` για το ROUGE-L, και η σχετική διαδρομή `../revision_experiment1` ισχύει),
με μία εφάπαξ εγκατάσταση pandas:

```powershell
docker exec -it knowledge_api bash -c "pip install -q pandas numpy statsmodels; cd /data/revision_experiment2 && source env2.sh && <ΕΝΤΟΛΗ>"
```
(Εναλλακτικά σε host venv με pandas/numpy/statsmodels· τότε βάλτε `TECHQA_TAR=../TechQA.tar.gz`
ώστε να βρεθεί το gold για το ROUGE-L — χωρίς αυτό, το ROUGE-L βγαίνει `n/a`, τα υπόλοιπα όλα κανονικά.)

---

## Βήματα

### Βήμα 1 — Επιλογή κατωφλίου `T_dep` στο TRAIN (3Α-gate)
```powershell
docker exec -it knowledge_api bash -c "pip install -q numpy; cd /data/revision_experiment2 && source env2.sh && python exp2_select_T.py"
```
Διαβάζει το `retrieval_train_oof.jsonl` (Πείραμα 1Β) και γράφει `data/T_dep.json`. Τυπώνει
`T_dep = … (TRAIN-OOF: C-Abst=…%, F-Abst=…% ≤ 25%)`. **Καμία ματιά στο TEST.**

### Βήμα 2 — Table 4 + pairwise τεστ + ευαισθησία
```powershell
docker exec -it knowledge_api bash -c "pip install -q pandas numpy statsmodels; cd /data/revision_experiment2 && source env2.sh && python exp2_report.py"
```
Διαβάζει αυτόματα το `data/T_dep.json`. Παράγει:

| Αρχείο | Τι θα βρει ο καθηγητής |
|---|---|
| `logs/exp2_report.md` | **ανθρώπινη σύνοψη** (Table 4, Wilson CI, pairwise, ευαισθησία) |
| `logs/exp2_report.json` | πλήρες machine-readable (verdicts + T_dep) |
| `analysis/tables/exp2_table4.csv` | Table 4: C-Abst (missing/missA/missU), Hazard, F-Abst, citation, ROUGE-L — mean (min–max) |
| `analysis/tables/exp2_table4_wilson.csv` | Wilson 95% CI στο majority vote ανά config × υποπληθυσμό |
| `analysis/tables/exp2_pairwise_main.csv` | **6 γραμμές** (3 ζεύγη × 2 μετρικές): Δ pp, bootstrap CI, McNemar p, Holm p, verdict |
| `analysis/tables/exp2_pairwise_sensitivity.csv` | per-run McNemar p (δείχνει αν το συμπέρασμα εξαρτάται από το majority vote) |
| `analysis/tables/exp2_per_run.csv` | rates ανά (config, run) |

---

## Τι να προσέξετε
- **Κλίμακα logits:** το `T_dep` επιλέγεται data-driven στο TRAIN· το `Gate_T5` (T=5,0) είναι
  για συνέχεια με το άρθρο — επειδή ο νέος reranker έχει αλλιώς κλιμακωμένα logits, μπορεί να
  αποχωρεί πολύ περισσότερο. Κύρια σύγκριση = **Gate (T_dep)**.
- **Κανόνες διατύπωσης** (στήλη `verdict`): `significantly higher/lower` όταν Holm p<0,05 ΚΑΙ
  bootstrap CI χωρίς το 0· αλλιώς `no detectable difference` (όχι «equivalent»).
- **Ισχύς:** με ~200 incidents στο `missing`, διαφορές < ~10 pp συνήθως δεν ανιχνεύονται —
  δηλώνεται ως όριο της αξιολόγησης, όχι ισοδυναμία.
- Οι baselines **B1/B2** (answerability classifier, CRAG) είναι η **δευτερεύουσα** οικογένεια
  και ανήκουν στο **Πείραμα 3** — δεν περιλαμβάνονται εδώ.

## Αρχεία πακέτου
```
ANALYSIS_PLAN_exp2.md   προ-καταχώριση (οικογένειες, μέθοδοι, κανόνες διατύπωσης)
env2.sh                 διαδρομές προς το Πείραμα 1 + παράμετροι
exp2_select_T.py        T_dep από TRAIN OOF -> data/T_dep.json
exp2_report.py          Table 4 + pairwise + ευαισθησία -> αρχεία + σύνοψη
analysis/abst_lib.py    κοινή βιβλιοθήκη (Wilson, McNemar, bootstrap, Holm, select_threshold)
data/ logs/ analysis/tables/   έξοδοι
```
