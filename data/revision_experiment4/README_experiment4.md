# Experiment 4 - Semantic grounding (NLI) & human correctness

Goes beyond *citation validity* (is the cited ID among the retrieved ones?) to
ask two deeper questions the reviewers raised:

1. **Metric 1 - semantic support:** does each generated step actually follow
   from the cited evidence? Scored automatically with an **NLI** model.
2. **Metric 2 - human correctness:** would a human engineer judge the answer a
   **correct resolution**? Scored by human annotators.

## Method

- **NLI scorer:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
  Documents are chunked (400 tokens, stride 200); for each answer step,
  `p_cited` / `p_any` = max P(entailment) over the chunks of the cited / all
  retrieved documents. A step is **supported** iff `p_cited ≥ 0.5`.
- **Human annotation:** a stratified sample of answered items is labelled by
  **two annotators**, reconciled into `adjudicated.xlsx`; agreement is reported
  with **Cohen's κappa**.
- **Uncertainty:** stratified **bootstrap (10 000 resamples)** for the
  correctness estimates.

## Placement

Put `revision_experiment4` next to `revision_experiment1` inside `data/`.
The NLI stage (4B) needs a GPU and the top-5 document text
(`retrieval_test_docs.jsonl`, regenerated from TechQA - see
`DATA_AVAILABILITY.md`); the report stage reuses the shipped `steps_nli.jsonl`.

## Steps

```bash
python exp4a_prepare_steps.py    # split Stage B/C answers into steps → steps.jsonl
python exp4b_nli.py              # NLI scoring → analysis/steps_nli.jsonl  (resumable)
python exp4b_nli_sanity.py       # mechanical positive/negative sanity check
python exp4c_sample.py           # draw the stratified human-annotation sample
python exp4_report.py            # Metric 1 + Metric 2 tables, kappa, bootstrap
```

## Outputs

| File | Contents |
|---|---|
| `logs/exp4_report.json` | Metric 1 + Metric 2, citation validity, κappa, NLI-validation block |
| `analysis/steps_nli.jsonl` | per-step NLI scores (model-generated step text + `p_cited` / `p_any`) |
| `analysis/tables/exp4_semantic_support.csv` | semantic-support and citation-validity rates per config |
| `annotation/annotator1.xlsx`, `annotator2.xlsx`, `adjudicated.xlsx`, `calibration.xlsx` | human labels (excluded from the public repo - raw TechQA text) |

## Key findings (honest reading)

- **Support ≠ correctness.** A step can be entailed by a retrieved document yet
  still be the wrong resolution - most visible on `missU`, where answering at
  all is already a hazard.
- **Citation errors are mostly not hallucinations:** ~81 % of invalid citations
  point to a real retrieved document, just not the gold one.
- **Human correct-resolution rate ≈ 14.2 %**, well below the ~62 %
  auto-resolution rate - i.e. the system auto-resolves far more often than it is
  actually right, which is exactly why the gate/escalation and the hazard
  framing matter. Reported honestly rather than inflated.

## Notes

- `steps_nli.jsonl` stores only **model-generated step text + scores** (no raw
  corpus), so it is safe to publish; the annotation spreadsheets embed TechQA
  question/answer/technote text and are therefore **excluded** (regenerate via
  `exp4c_sample.py` + TechQA).
- The bootstrap is seeded, so the report reproduces deterministically from
  `steps_nli.jsonl` + the human labels.

---

# (Ελληνικά)

# Πείραμα 4 - Σημασιολογική θεμελίωση (NLI) & ανθρώπινη ορθότητα

Προχωρά πέρα από την *εγκυρότητα παραπομπών* (είναι το cited ID μέσα στα
ανακτηθέντα;) και θέτει δύο βαθύτερα ερωτήματα που έθεσαν οι reviewers:

1. **Metric 1 - σημασιολογική υποστήριξη:** προκύπτει πράγματι κάθε παραγόμενο
   βήμα από τα αναφερόμενα (cited) τεκμήρια; Βαθμολογείται αυτόματα με μοντέλο **NLI**.
2. **Metric 2 - ανθρώπινη ορθότητα:** θα θεωρούσε ένας μηχανικός την απάντηση
   **ορθή επίλυση**; Βαθμολογείται από ανθρώπους αξιολογητές.

## Μέθοδος

- **NLI scorer:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
  Τα έγγραφα τεμαχίζονται (400 tokens, stride 200)· για κάθε βήμα απάντησης,
  `p_cited` / `p_any` = max P(entailment) πάνω στα chunks των cited / όλων των
  ανακτηθέντων εγγράφων. Ένα βήμα είναι **supported** ανν `p_cited ≥ 0.5`.
- **Ανθρώπινη αξιολόγηση:** ένα στρωματοποιημένο δείγμα απαντημένων items
  επισημαίνεται από **δύο αξιολογητές**, με συμφιλίωση στο `adjudicated.xlsx`· η
  συμφωνία αναφέρεται με **Cohen's kappa**.
- **Αβεβαιότητα:** στρωματοποιημένο **bootstrap (10 000 resamples)** για τις
  εκτιμήσεις ορθότητας.

## Τοποθέτηση

Βάλτε το `revision_experiment4` δίπλα στο `revision_experiment1` μέσα στο `data/`.
Το στάδιο NLI (4B) χρειάζεται GPU και το κείμενο των top-5 εγγράφων
(`retrieval_test_docs.jsonl`, αναπαράγεται από το TechQA - βλ.
`DATA_AVAILABILITY.md`)· το στάδιο report επαναχρησιμοποιεί το δημοσιευμένο
`steps_nli.jsonl`.

## Βήματα

```bash
python exp4a_prepare_steps.py    # σπάει τις απαντήσεις Stage B/C σε βήματα → steps.jsonl
python exp4b_nli.py              # NLI scoring → analysis/steps_nli.jsonl  (resumable)
python exp4b_nli_sanity.py       # μηχανικός έλεγχος positive/negative
python exp4c_sample.py           # λήψη του στρωματοποιημένου δείγματος αξιολόγησης
python exp4_report.py            # πίνακες Metric 1 + Metric 2, kappa, bootstrap
```

## Έξοδοι

| Αρχείο | Περιεχόμενο |
|---|---|
| `logs/exp4_report.json` | Metric 1 + Metric 2, εγκυρότητα παραπομπών, kappa, block επικύρωσης NLI |
| `analysis/steps_nli.jsonl` | NLI scores ανά βήμα (κείμενο βήματος από το μοντέλο + `p_cited` / `p_any`) |
| `analysis/tables/exp4_semantic_support.csv` | ποσοστά σημασιολογικής υποστήριξης & εγκυρότητας παραπομπών ανά config |
| `annotation/annotator1.xlsx`, `annotator2.xlsx`, `adjudicated.xlsx`, `calibration.xlsx` | ανθρώπινες ετικέτες (εξαιρούνται από το δημόσιο repo - raw κείμενο TechQA) |

## Βασικά ευρήματα (έντιμη ανάγνωση)

- **Υποστήριξη ≠ ορθότητα.** Ένα βήμα μπορεί να συνεπάγεται (entailed) από
  ανακτηθέν έγγραφο και παρ' όλα αυτά να είναι λάθος επίλυση - πιο εμφανές στο
  `missU`, όπου ήδη το να απαντάς καθόλου είναι hazard.
- **Τα σφάλματα παραπομπών κυρίως δεν είναι παραισθήσεις (hallucinations):** ~81 %
  των άκυρων παραπομπών δείχνουν σε πραγματικό ανακτηθέν έγγραφο, απλώς όχι στο gold.
- **Ανθρώπινο ποσοστό ορθής επίλυσης ≈ 14.2 %**, πολύ κάτω από το ~62 % αυτόματης
  επίλυσης - δηλαδή το σύστημα αυτο-επιλύει πολύ πιο συχνά απ' όσο έχει πραγματικά
  δίκιο, γι' αυτό ακριβώς έχουν σημασία το gate/κλιμάκωση και η οπτική του hazard.
  Αναφέρεται έντιμα, χωρίς υπερβολές.

## Σημειώσεις

- Το `steps_nli.jsonl` αποθηκεύει μόνο **κείμενο βήματος από το μοντέλο + scores**
  (χωρίς raw corpus), άρα είναι ασφαλές προς δημοσίευση· τα spreadsheets
  αξιολόγησης ενσωματώνουν κείμενο ερώτησης/απάντησης/technote του TechQA και γι'
  αυτό **εξαιρούνται** (αναπαράγονται μέσω `exp4c_sample.py` + TechQA).
- Το bootstrap είναι seeded, οπότε το report αναπαράγεται ντετερμινιστικά από το
  `steps_nli.jsonl` + τις ανθρώπινες ετικέτες.
