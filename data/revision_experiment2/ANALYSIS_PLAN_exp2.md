# ANALYSIS_PLAN — Πείραμα 2 (3 επαναλήψεις & στατιστικοί έλεγχοι)

> **Προ-καταχώριση.** Οι οικογένειες συγκρίσεων, οι μέθοδοι και οι κανόνες διατύπωσης
> ορίζονται **πριν** δούμε p-values. Το Πείραμα 2 **δεν τρέχει νέο inference** — αναλύει τα
> logs του Πειράματος 1Δ. Απαντά σε R1-M5, R2, R3-2, R4-4.

Commit hash κατά το πάγωμα: `__________` · Ημερομηνία: `__________`.

## 1. Προαπαιτούμενα
- `logs/stageB/{Base,QLoRA}_run{1,2,3}.jsonl` (Πείραμα 1Δ, επαληθευμένα — 1Δ VERIFY PASS).
- `data/test_subgroups.csv` (present/missA/missU, Πείραμα 1Γ).
- `data/retrieval_train_oof.jsonl` (5-fold OOF, Πείραμα 1Β) για το `T_dep`.

## 2. Configs (όλα από τα ΙΔΙΑ logs — κανένα νέο inference)
| Config | Ορισμός |
|---|---|
| **Base** | `llama3:8b-instruct-q4_K_M`, όπως στο 1Δ. |
| **QLoRA** | ο adapter `techqa-llama3`, όπως στο 1Δ. |
| **Gate** | φίλτρο πριν τον Base: `abstain = (gate_max_logit < T_dep) Ή abstain(Base)`. |
| **Gate_T5** | το ίδιο με `T = 5.0` (συνέχεια με το άρθρο· το νέο reranker έχει αλλιώς κλιμακωμένα logits, γι' αυτό κύριο είναι το `T_dep`). |

**Κατώφλι `T_dep`** (κριτήριο προ-καταχωρημένο): επιλέγεται **στο TRAIN out-of-fold**, ως το
`T` που μεγιστοποιεί το C-Abst (αποχή στο `missing`) **υπό F-Abst ≤ 25%** (≈ το F-Abst 24%
της QLoRA στο άρθρο). score = max reranker logit του top-5. ΚΑΜΙΑ ματιά στο TEST.
(υλοποίηση: `exp2_select_T.py` → `data/T_dep.json`).

## 3. Table 4 (ανά config)
C-Abst (missing) ως **mean (min–max)** των 3 runs **+ Wilson 95% CI στο majority vote**·
C-Abst `missA`/`missU` ξεχωριστά· Hazard = 100−C-Abst· F-Abst (present)· Citation validity
(στις απαντήσεις, όχι abstentions)· ROUGE-L (στις present που απαντήθηκαν· supporting).
Το Wilson CI μπαίνει στο majority vote γιατί τα 3 runs δεν είναι ανεξάρτητα· η διάχυση
αναφέρεται με min–max.

## 4. Οικογένειες συγκρίσεων (2Β)
| Οικογένεια | Συγκρίσεις | Μετρική / πληθυσμός | Holm |
|---|---|---|---|
| **Κύρια** | Gate–Base, QLoRA–Base, QLoRA–Gate | C-Abst στο `missing` | ανά μετρική (3 p-values) |
| **Κύρια** | οι ίδιες 3 | F-Abst στο `present` | ανά μετρική |
| Δευτερεύουσα (Πείραμα 3) | QLoRA–B1, QLoRA–B2 | ίδιες δύο μετρικές | ξεχωριστή οικογένεια |

Μέθοδοι (`abst_lib.pairwise_table`): **exact McNemar** στις majority-vote αποφάσεις·
**paired bootstrap** 10.000 resamples σε επίπεδο incident (95% percentile CI της διαφοράς σε
pp)· **Holm–Bonferroni** ανά οικογένεια. Αναφέρεται ΠΑΝΤΑ το effect size (Δ pp) μαζί με το p.
Το Hazard δεν τεστάρεται ξεχωριστά (= 100 − C-Abst).

## 5. Κανόνες διατύπωσης (2Γ)
- `p_holm < 0,05` **και** bootstrap CI χωρίς το 0 → «significantly higher/lower (Δ=x pp, 95% CI [a,b], Holm p=…)».
- `p_holm ≥ 0,05` → «no detectable difference» (όχι «equivalent»).
- Wilson CI επικαλύπτονται αλλά paired σημαντικό → αναφέρεται το paired (ισχυρότερο, ίδιες ερωτήσεις).
- Συμπέρασμα αλλάζει majority vote ↔ επιμέρους runs (πίνακας sens) → «not robust to run-to-run variation».
- **Ισχύς:** ~200 incidents στο `missing` → διαφορές < ~10 pp συνήθως δεν ανιχνεύονται· δηλώνεται ως όριο, όχι ισοδυναμία.

## 6. §3.8 (προτεινόμενα αγγλικά)
«Each configuration was run three times; we report the mean and the range over runs. Pairwise
differences were tested with an exact McNemar test on per-incident majority-vote decisions, with
paired bootstrap confidence intervals (10,000 resamples) and Holm–Bonferroni correction within
each family of comparisons. Run-to-run variation at temperature 0 is attributed to
non-deterministic GPU kernels and is not a sampling effect.»

## 7. Κριτήριο ολοκλήρωσης
`exp2_pairwise_main.csv` με **6 γραμμές** (3 ζεύγη × 2 μετρικές)· πίνακας `sens`· Table 4 με
mean (min–max) και Wilson CI· καμία σύγκριση χωρίς προεπιλεγμένη οικογένεια πριν τα αποτελέσματα.
