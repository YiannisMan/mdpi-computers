# ANALYSIS_PLAN — Πείραμα 3 (baselines & καμπύλη gate)

> **Προ-καταχώριση.** Κατώφλια, κριτήρια και κανόνες διατύπωσης ορίζονται **πριν** δούμε το
> TEST. Απαντά σε R1-M2, R1-M7, R2, R3-4, R3-7, R4-9. Χωρίς εκπαίδευση LLM.

Commit hash (thresholds.json) πριν το 1Δ/3Β: `__________` · Ημερομηνία: `__________`.

## 1. Αρχή σχεδίασης — όλα «φίλτρο πριν τον Base»
Κάθε baseline βγάζει score ανά ερώτηση (υψηλό = πιο απαντήσιμη). `abstain = (score < κατώφλι)
Ή abstain(Base)`. Ίδιος generator, ίδιο retrieval — αλλάζει μόνο η απόφαση αποχής, ώστε η
σύγκριση να είναι δίκαιη.

| Μέθοδος | Score | Κατώφλι (στο TRAIN OOF) |
|---|---|---|
| **Gate** | max reranker logit top-5 | max C-Abst υπό F-Abst ≤ 25% |
| **B1** answerability classifier | P(present), logistic regression | ίδιο (στις cross-validated p) |
| **B2** CRAG-style (monoT5) | max relevance score των 5 docs | ίδιο (θ_lo)· θ_hi = precision(present) ≥ 0,90 |

Το όριο **25%** είναι προκαθορισμένο (≈ το F-Abst 24% της QLoRA). Στο TEST δείχνουμε **όλη
την καμπύλη** C-Abst–F-Abst, ώστε κανένας κριτής να μη βλέπει cherry-picking.

## 2. 3Α — προετοιμασία στο TRAIN (χωρίς LLM)
- Label: `present` = answerable & gold στο OOF top-5· τα υπόλοιπα (+ unanswerable) = `missing`.
- **Gate:** T_dep = `select_threshold(max_logit, present, 25)`· τοποθετείται και το T=5,0 στην ίδια καμπύλη.
- **B1:** features = max/2ο/μέσο logit, gap top1–top2, softmax entropy, μήκος query, #technical tokens
  (τα `overlap_max/rrf_top1/bm25/dense` **εξαιρούνται** — δεν αποθηκεύονται στο frozen TEST, για
  συνέπεια TRAIN↔TEST). Logistic regression + 5-fold CV για το C· κατώφλι στις **out-of-fold** p.
- **B2:** monoT5 (`castorini/monot5-base-msmarco`) — διαφορετικό μοντέλο από τον reranker. θ_lo
  (escalation) + θ_hi (precision 0,90). Το «Ambiguous → filtered docs» αντιμετωπίζεται ως άλλο
  σημείο της ίδιας καμπύλης (score-filter), ώστε να μη χρειαστεί νέο generation.
- **Πάγωμα:** `thresholds.json` (T_dep, p*, θ_lo, θ_hi, κριτήριο, AUROC του TRAIN), commit.
- Αποδοχή: τα train-OOF F-Abst ≤ 25%· τα AUROC out-of-fold. Αν AUROC ≈ 0,5 → έλεγχος labels/σειράς.

## 3. 3Β — εφαρμογή στο TEST (μετά το 1Δ)
- **Καμπύλες** C-Abst vs F-Abst για gate/B1/B2 (κάθε σημείο από τα παγωμένα logs) → σχήμα.
- **iso-F-Abst:** T* = το μεγαλύτερο κατώφλι με F-Abst ≤ το F-Abst της QLoRA (μέσος 3 runs). Το
  bootstrap CI επανεπιλέγει το T* σε κάθε resample (όχι αισιόδοξο). Δείχνει πόσο C-Abst θα είχε
  κάθε baseline στο **ίδιο κόστος** false abstention με την QLoRA.
- **AUROC** (3 scores) με CI — ποιότητα score χωρίς κατώφλι.
- **Pairwise** δευτερεύουσα οικογένεια: QLoRA–B1, QLoRA–B2 (+ Gate/Gate_T5), McNemar + bootstrap + Holm.
- **Gate∘QLoRA** (abstain αν gate Ή QLoRA): συμπληρωματικότητα learned abstention & retrieval-confidence.

## 4. Κανόνες διατύπωσης (πριν τα αποτελέσματα)
- QLoRA πάνω από όλες τις καμπύλες στο ίδιο F-Abst (CI διαφοράς χωρίς 0) → «learned abstention
  outperforms retrieval-confidence baselines at matched false-abstention».
- Baseline πλησιάζει την QLoRA → «competitive with, and complementary to» + δεδομένα Gate∘QLoRA.
- Το operating-point F-Abst στο TEST διαφέρει από 25% (μετατόπιση κλίμακας) → αναφέρεται ως έχει·
  η iso-F-Abst σύγκριση δεν εξαρτάται από το train κατώφλι. **Κανένα νέο κατώφλι στο TEST.**
- B2 ≈ Gate → αναμενόμενο (κοινό retrieval-relevance score).

## 5. Κριτήριο ολοκλήρωσης
`thresholds.json` με commit πριν το 1Δ· (logs B1/B2)· σχήμα `abstention_tradeoff.pdf`,
`iso_fabst.csv`, AUROC· συμπληρωμένος Table 4 (B1/B2/Gate_T5)· κείμενο για την ιστορία του T=5,0.
