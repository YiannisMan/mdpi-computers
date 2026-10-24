# ANALYSIS_PLAN — Πείραμα 4 (σημασιολογικός έλεγχος απαντήσεων)

> **Προ-καταχώριση.** Οι παράμετροι παρακάτω ορίζονται **πριν** τρέξει το NLI και πριν τη
> δειγματοληψία των 50. Κανένα δεν ρυθμίζεται πάνω στα αποτελέσματα. Το κατώφλι NLI **δεν**
> ρυθμίζεται στις 50 ανθρώπινες ετικέτες (είναι δεδομένα TEST). Απαντά σε **R1-M3, R2, R3-3, R4-6, R4-7, R4-8**.

Commit hash (αυτού του αρχείου) πριν το 4Β/4Γ: `__________` · Ημερομηνία: `__________`.

## 1. Τι μετράμε — τρία επίπεδα
| Επίπεδο | Ερώτηση | Πώς |
|---|---|---|
| Citation validity (υπάρχει) | Ανήκουν τα cited ids στα 5 retrieved; | Ο υπάρχων έλεγχος, αμετάβλητος |
| **Semantic support** (νέο) | Υποστηρίζει το cited έγγραφο κάθε βήμα; | NLI, επικυρωμένο από ανθρώπους στα 50 |
| **Correctness** (νέο) | Θα έλυνε η απάντηση το πρόβλημα (vs gold); | Ανθρώπινη κρίση στα 50 auto-resolved |

Το semantic support **δεν** είναι ορθότητα (στις missing ένα βήμα μπορεί να υποστηρίζεται από λάθος
έγγραφο) — αναφέρεται χωριστά ανά υποπληθυσμό (present, missA, missU).

## 2. Παράμετροι NLI (§4Β)
- Μοντέλο: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (fp16). Label index από `model.config.label2id` (όχι υπόθεση).
- Premise: chunks των cited εγγράφων, **400 tokens, stride 200** (premise+hypothesis ≤ 512).
- Hypothesis: το κείμενο του βήματος, **χωρίς παραπομπές**.
- Score βήματος: `p_entail = max` εντελισμού πάνω σε όλα τα chunks όλων των cited.
- Supported ⟺ `p_cited ≥ 0,5`. Ευαισθησία: **0,3** και **0,7**.
- Δεύτερο score `p_any` = max πάνω στα chunks και των 5 retrieved (ξεχωρίζει «λάθος citation» από «καθόλου υποστήριξη»).
- **Έλεγχος ορθότητας** (πριν την ανάλυση): θετικός ≥ 90%, αρνητικός ≤ 10% (100 ζεύγη το καθένα, TRAIN docs).

## 3. Ορισμός «βήματος» (§4Α)
- Κάθε αριθμημένο/κουκκιδωτό στοιχείο = ένα βήμα· η πρόταση διάγνωσης = ένα βήμα (`kind=diagnosis`).
- Αφαιρούνται οι παραπομπές πριν το NLI. Θραύσματα < 3 λέξεων απορρίπτονται.
- **Γενικά βήματα** (`generic=True`: «contact support», «apply latest fix pack», «restart if persists»): μένουν
  στην κύρια ανάλυση, εξαιρούνται **μόνο** σε ανάλυση ευαισθησίας (το NLI τα κρίνει συχνά unsupported χωρίς να είναι λάθος).

## 4. Δείγμα 50 & αξιολόγηση (§4Γ–4Δ)
- Πληθυσμός: auto-resolved tickets του **Stage C, run 1**. Στρώματα: **25 present + 25 missing** (missA∪missU αναλογικά).
- Επιλογή: τυχαία χωρίς επανάθεση, **seed = 2026**. Calibration: **5** επιπλέον (εκτός των 50).
- Τυφλότητα: οι αξιολογητές δεν βλέπουν υποπληθυσμό, σκορ NLI, config, run, question_id· η σειρά ανακατεύεται ανά αξιολογητή.
- Ετικέτες: **support** ανά βήμα S/N· **correctness** ανά απάντηση C/P/I/U (U=unanswerable, μετρά ως όχι σωστή)· **risky_action** Ναι/Όχι.
- Συμφωνία: Cohen's κ (correctness 3-class C/P/(I+U) & 2-class C/όχι-C· support S/N) + απλό ποσοστό. Διαφωνίες: adjudicated (τελικές ετικέτες στις μετρικές· κ στις αρχικές ανεξάρτητες).

## 5. Μετρικές του Abstract (§4ΣΤ)
- **Semantic support rate** (NLI, όλα τα configs): step support, fully supported, **semantic mismatch among citation-valid** (ο αριθμός του R3), wrong-citation share. mean (min–max) 3 runs, ανά υποπληθυσμό. QLoRA–Base: paired exact McNemar **μόνο στις κοινές** ερωτήσεις (majority vote).
- **Correct-resolution rate** (ανθρώπινη, Stage C): p̂_h ανά στρώμα· σταθμισμένο με N_h (auto-resolved run1)· 95% CI με **stratified bootstrap 10.000** (επανάθεση μέσα σε κάθε στρώμα). Correct among auto-resolved & correct-resolution rate (επί 310). Unsafe = (#I+#U στο missing).
- **Επικύρωση NLI–ανθρώπων**: accuracy, κ, recall(unsupported). Αν κ < 0,5 → κύρια τα ανθρώπινα ποσοστά με Wilson CI, το NLI ως δευτερεύον κατώτερο όριο.

## 6. Αποκλίσεις από το αρχικό πλάνο (ρητά)
- **B2 (CRAG):** στο Πείραμα 3 υλοποιήθηκε ως score-filter **χωρίς** regeneration. Άρα οι απαντήσεις του B2 είναι
  **υποσύνολο** του Base (όπως Gate/Gate_T5/B1) — δεν υπάρχουν «νέες απαντήσεις Ambiguous» για νέο NLI.
- Στο report παράγονται **Base, QLoRA, Stage C** (με NLI) και **Gate, Gate_T5** (παράγωγα του Base μέσω `gate_max_logit ≥ T`).
  Τα B1/B2 είναι ομοίως υποσύνολα του Base (ίδιες μετρικές, λιγότερες απαντήσεις)· προστίθενται αν χρειαστεί.

## 7. Κριτήριο ολοκλήρωσης
`ANALYSIS_PLAN` commit πριν NLI/δειγματοληψία · `steps_nli.jsonl` + θετικός/αρνητικός έλεγχος ·
δύο ανεξάρτητα φύλλα + adjudicated + κ · πίνακας «Semantic grounding and correctness» από script ·
δύο αριθμοί Abstract (semantic support rate, correct-resolution rate) με CI.
