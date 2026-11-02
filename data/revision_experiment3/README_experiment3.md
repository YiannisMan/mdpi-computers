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

---

# (Ελληνικά)

# Πείραμα 3 - Δευτερεύοντες baselines αποχής (ταξινομητής & CRAG)

Συγκρίνει τη **μαθημένη αποχή** (learned abstention) του συστήματος με δύο
εξωτερικούς baselines αποχής / selective prediction, **σε κοινό σημείο λειτουργίας**
(iso-F-Abst), ώστε η σύγκριση να είναι δίκαιη. Είναι η **δευτερεύουσα** οικογένεια
baselines που αναφέρουν οι reviewers.

Baselines:
- **B1 - ταξινομητής απαντησιμότητας.** Λογιστικό μοντέλο πάνω σε χαρακτηριστικά
  ανάκτησης από το αρχείο OOF (max/mean reranker logit, gap, softmax entropy, RRF,
  overlap…) που αποφασίζει answer vs abstain.
- **B2 - διορθωτική ανάκτηση τύπου CRAG** (συνάφεια monoT5), που αποχωρεί /
  κλιμακώνει όταν η εμπιστοσύνη ανάκτησης είναι χαμηλή.

Όλα τα κατώφλια λειτουργίας επιλέγονται στο **TRAIN** (ποτέ TEST), και κάθε μέθοδος
αξιολογείται στην ίδια παγωμένη ανάκτηση TEST από το Πείραμα 1.

## Τοποθέτηση

Βάλτε το `revision_experiment3` δίπλα στο `revision_experiment1` μέσα στο `data/`
(σχετικές διαδρομές `../revision_experiment1/...`). Το B2 χρειάζεται το corpus του
TechQA, που διαβάζεται με streaming από το `TechQA.tar.gz` — οπότε τρέξτε το μέσα
στο container `knowledge_api` (ή δείξτε το `TECHQA_TAR` στο archive).

## Βήματα

```bash
python exp3a_prepare.py          # features / inputs για τους baselines
python exp3a_b2_crag.py          # B2 (CRAG / monoT5) scoring πάνω στο corpus
python exp3_freeze.py            # επιλογή κατωφλίων στο TRAIN → data/thresholds.json
python exp3b_report.py           # σύγκριση iso-F-Abst + AUROC + figure
```

## Έξοδοι

| Αρχείο | Περιεχόμενο |
|---|---|
| `logs/exp3_report.json` | σύνοψη: πίνακας iso-F-Abst, AUROC, pairwise δευτερεύουσες συγκρίσεις |
| `data/thresholds.json` | κατώφλια λειτουργίας B1 / B2 επιλεγμένα στο TRAIN |
| `analysis/tables/exp3_iso_fabst.csv` | όλες οι μέθοδοι σε κοινό F-Abst |
| `analysis/tables/exp3_auroc.csv` | ποιότητα κατάταξης (AUROC) των scores αποχής |
| `analysis/tables/exp3_pairwise_secondary.csv`, `exp3_curve_*.csv` | pairwise έλεγχοι + καμπύλες risk–coverage |
| `analysis/figures/abstention_tradeoff.png` | figure συμβιβασμού αποχής / hazard |

## Βασικό εύρημα (έντιμη ανάγνωση)

Σε κοινό σημείο λειτουργίας, η **μαθημένη αποχή του QLoRA ισοφαρίζει τον ταξινομητή
B1** (καμία ανιχνεύσιμη διαφορά σε iso-F-Abst)· δεν κυριαρχεί έναντι των εξωτερικών
baselines. Αναφέρεται ως έχει, χωρίς υπερβολές.

## Σημειώσεις

- Τα κατώφλια είναι μόνο-TRAIN· η ανάκτηση TEST είναι η παγωμένη από το Πείραμα 1,
  οπότε τα νούμερα είναι άμεσα συγκρίσιμα μεταξύ των πειραμάτων.
- Το corpus του TechQA (`full_technote_collection.txt.bz2`, μέσα στο `TechQA.tar.gz`)
  είναι αποθηκευμένο ως **πολλαπλοί συνενωμένοι JSON arrays**· ο loader το διαβάζει
  με streaming μέσω `ijson(..., multiple_values=True)`.
