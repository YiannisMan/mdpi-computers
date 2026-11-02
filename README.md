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

---

# (Ελληνικά)

# Αυτόνομη Επίλυση Συμβάντων IT πάνω στο IBM TechQA - Αξιολόγηση χωρίς Διαρροή Δεδομένων

Πηγαίος κώδικας και **συγκεντρωτικά αποτελέσματα** για την αναθεώρηση (revision)
στο MDPI *Computers* ενός συστήματος αυτόνομης επίλυσης συμβάντων IT με
ανάκτηση-επαυξημένη παραγωγή (hybrid retrieval + fine-tuning LLM). Περιλαμβάνει
**αξιολόγηση χωρίς διαρροή δεδομένων** και με διαχωρισμό ανά ερώτηση
(question-disjoint), συνοδευόμενη από πλήρεις στατιστικούς ελέγχους.

## Πειράματα
- **revision_experiment1/** - έλεγχος διαρροής (provenance), reranker εκπαιδευμένος μόνο στο TRAIN + 5-fold OOF, παγωμένη ανάκτηση TEST, logs Stage B/C, reports.
- **revision_experiment2/** - επανάληψη 3 εκτελέσεων + στατιστικά (Table 4, Wilson CIs, paired bootstrap, McNemar/Holm).
- **revision_experiment3/** - δευτερεύοντες baselines αποχής (ταξινομητής απαντησιμότητας, CRAG) σε iso-F-Abst + AUROC.
- **revision_experiment4/** - σημασιολογική υποστήριξη (semantic support) με NLI + ανθρώπινη αξιολόγηση (ορθότητα επίλυσης).

Κάθε πακέτο έχει το δικό του `README.md` με ακριβή, εκτελέσιμα βήματα.

## Αναπαραγωγιμότητα
Τα στάδια καθαρής στατιστικής (reports Πειραμάτων 2–4) είναι ντετερμινιστικά
(bootstrap με `seed=42`) και αναπαράγονται **bit-for-bit** από τα δημοσιευμένα
logs. Τα στάδια GPU είναι αρχειοθετημένα ως παγωμένες έξοδοι (βλ. `DATA_AVAILABILITY.md`).

## Δεδομένα & μεγάλα αρχεία
- Το **IBM TechQA** (dataset + corpus) **δεν** αναδιανέμεται εδώ - άδεια χρήσης IBM.
  Βλ. `DATA_AVAILABILITY.md` για το πώς αποκτάται και πώς αναπαράγονται τα παράγωγα αρχεία.
- Τα **βάρη μοντέλων** (fine-tuned LLM GGUF, rerankers) ξεπερνούν το όριο 100 MB του
  GitHub και φιλοξενούνται εξωτερικά· σύνδεσμοι στο `DATA_AVAILABILITY.md`.

## Περιβάλλον
Windows + WSL2 + Docker (container υπηρεσίας `knowledge_api`), Ollama, NVIDIA RTX 3060.

## Παραπομπή
> (Trustworthy Agentic AI for Autonomous IT Incident Resolution: 
A Retrieval-Augmented LLM with Learned Abstention and 
Human Escalation / computers-4639708)
