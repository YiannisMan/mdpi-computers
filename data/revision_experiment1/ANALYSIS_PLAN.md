# ANALYSIS_PLAN.md — Προ-καταχώριση ανάλυσης (Πείραμα 1)

> **Κανόνας:** Αυτό το αρχείο γίνεται **commit πριν** τρέξει οποιαδήποτε γενιά (Stage B/C)
> στο TEST. Κάθε αλλαγή μετά την πρώτη ματιά στα αποτελέσματα του TEST επιτρέπεται
> μόνο ως **ρητά σημειωμένη ανάλυση ευαισθησίας** (sensitivity analysis), όχι ως
> αναδρομική αλλαγή του κύριου πρωτοκόλλου.

Commit hash κατά το πάγωμα: `__________` (συμπληρώνεται με `git rev-parse HEAD`).
Ημερομηνία παγώματος: `__________`.

---

## 1. Σύνολα δεδομένων

| Σύνολο | Αρχείο TechQA | Ρόλος |
|---|---|---|
| **TRAIN** | `training_Q_A.json` (~600 ερωτήσεις) | Εκπαίδευση reranker, SFT (ήδη έγινε), επιλογή κατωφλίων, εκπαίδευση baselines |
| **TEST** | `dev_Q_A.json` (**όλες οι 310**· βλ. §1.1) | **Αποκλειστικά αξιολόγηση.** Αναφέρεται ως "held-out test (TechQA dev)" |

Καμία απόφαση (κατώφλια, βάρη, επιλογή μοντέλου, hyper-parameters, επιλογή baselines)
δεν παίρνεται στο TEST. Όλες γίνονται στο TRAIN με out-of-fold scores.

### 1.1 Κοινά gold & διπλότυπες ερωτήσεις — report-only (οδηγία reviewer)

Το Πείραμα 1Α εντόπισε (α) **40 κοινά gold technotes** TRAIN↔TEST
(`data/shared_gold_docs.csv`) και (β) **23 ερωτήσεις** του `dev_Q_A.json` με **πανομοιότυπο**
κανονικοποιημένο κείμενο (τίτλος+σώμα) με ερώτηση του `training_Q_A.json`
(`data/dup_questions.csv`· 18 answerable). Οι διπλότυπες επιβεβαιώθηκαν ως γνήσιες (ίδιος
τίτλος, οι περισσότερες με **ίδιο gold technote**), όχι artefact κανονικοποίησης.

**Απόφαση (οδηγία reviewer — «αναφορά μόνο, όχι διόρθωση»):** **καμία αφαίρεση.** Και τα δύο
ευρήματα **δηλώνονται ρητά** στο κείμενο (§3.2) αλλά δεν αφαιρούνται· το TechQA είναι
καθιερωμένο benchmark και χρησιμοποιείται το **πλήρες επίσημο dev split (310 ερωτήσεις:
160 answerable, 150 unanswerable)** ως κύριο held-out TEST. Όλα τα κύρια αποτελέσματα
Stage A/B/C αναφέρονται στις 310.

**Ανάλυση ευαισθησίας (Supplementary):** αναφέρεται επιπλέον το αποτέλεσμα **χωρίς** τις 23
διπλότυπες (287 ερωτήσεις: 142 answerable, 145 unanswerable), ώστε να φανεί ότι το συμπέρασμα
δεν αλλάζει. Υλοποιείται με `exp1c_stageA_freeze.py --exclude-dups` (ξεχωριστά αρχεία
`_clean287`, δεν αγγίζουν το κύριο παγωμένο 310) και, στη γενιά, με post-hoc φιλτράρισμα των
23 qids από τα ίδια logs (βλ. §5· κανένα δεύτερο πέρασμα LLM).

## 2. Υποπληθυσμοί του TEST

Ορίζονται **μία φορά**, αφού παγώσει το retrieval top-5 του Πειράματος 1Γ, και δεν
αλλάζουν ποτέ μετά (`data/test_subgroups.csv`).

| Όνομα | Ορισμός | Σωστή συμπεριφορά |
|---|---|---|
| `present` (GT-present) | Answerable **και** το gold έγγραφο στο παγωμένο top-5 | Απάντηση (Mode A) |
| `missA` | Answerable **και** το gold **δεν** είναι στο top-5 | Κλιμάκωση (Mode B) |
| `missU` | Φυσικά unanswerable ερώτηση (`ANSWERABLE = N`) | Κλιμάκωση (Mode B) |
| `missing` | `missA ∪ missU` | Κλιμάκωση (Mode B) |

## 3. Μετρικές (ορισμοί)

- **Abstain**: η έξοδος αναγνωρίζεται ως **Mode B** από τον **αυθεντικό parser** (ακριβής
  φράση άρνησης ή το αναλυτικό format Mode B). Έξοδοι που δεν είναι ούτε έγκυρο Mode A
  ούτε Mode B → **invalid**: αναφέρονται ξεχωριστά, αλλά στις μετρικές **μετρώνται ως
  «απάντηση»** (όχι abstain).
- **C-Abst** = % abstain στον πληθυσμό `missing` (και ξεχωριστά `missA`, `missU`).
- **F-Abst** = % abstain στον πληθυσμό `present`.
- **Hazard** = `100 − C-Abst`.
- **Citation validity** (όχι «faithfulness»/«hallucination»): όλα τα cited ids ανήκουν
  στα 5 retrieved έγγραφα. Ο έλεγχος μένει **ίδιος με το άρθρο** για συνέχεια.
- **ROUGE-L**: μόνο στις `present` που απαντήθηκαν (supporting metric).

## 4. Configs

| Config | Ορισμός |
|---|---|
| **Base** | Το βασικό `llama3:8b-instruct-q4_K_M` μέσω `/resolve`, με το αυθεντικό prompt/parser. |
| **QLoRA** | Το ίδιο, με τον υπάρχοντα adapter `techqa-llama3` (επαναχρησιμοποιείται αυτούσιος — βλ. §7). |
| **Gate** | Φίλτρο **πριν** τον Base: αν `max reranker logit του top-5 < T` → abstain, αλλιώς ό,τι έβγαλε ο Base. Παράγεται από τα logs του Base **χωρίς νέο inference**. |
| **Gate_T5** | Το ίδιο με `T = 5.0` (συνέχεια με το άρθρο). |

Το κατώφλι `T_dep` του Gate επιλέγεται **στο TRAIN out-of-fold** (Πείραμα 3Α), με κριτήριο
**max C-Abst υπό F-Abst ≤ 25%**. Το όριο 25% είναι προκαθορισμένο (≈ το F-Abst 24,0% της
QLoRA στο άρθρο) και γράφεται εδώ πριν δούμε το TEST.

## 5. Επαναλήψεις

Κάθε config τρέχει **3 ανεξάρτητα runs**. Στο temperature 0, το seed δεν επηρεάζει το
sampling· τα 3 runs ορίζονται ως 3 ανεξάρτητα περάσματα με **επανεκκίνηση του Ollama
server**, `OLLAMA_NUM_PARALLEL=1`, και **διαφορετική τυχαία σειρά ερωτήσεων** ανά run.
Η διακύμανση που παρατηρείται αποδίδεται σε μη-ντετερμινισμό του inference (GPU kernels,
batching, prompt cache), όχι σε sampling. Αναφέρουμε **mean (min–max)** στα 3 runs.

## 6. Στατιστικοί έλεγχοι (Πείραμα 2 — οικογένειες ορίζονται εδώ)

Οι οικογένειες συγκρίσεων ορίζονται **πριν** δούμε p-values:

| Οικογένεια | Συγκρίσεις | Μετρική / πληθυσμός | Holm |
|---|---|---|---|
| Κύρια Α | Gate–Base, QLoRA–Base, QLoRA–Gate | C-Abst στο `missing` | ανά μετρική (3 p-values) |
| Κύρια Β | οι ίδιες 3 | F-Abst στο `present` | ανά μετρική |
| Δευτερεύουσα | QLoRA–B1, QLoRA–B2 | ίδιες δύο μετρικές | ξεχωριστή οικογένεια |

- **Exact McNemar** στις ανά-incident αποφάσεις (majority vote των 3 runs).
- **Paired bootstrap** 10.000 resamples σε επίπεδο incident· 95% percentile CI της
  διαφοράς σε pp.
- **Holm–Bonferroni** σε κάθε οικογένεια. Πάντα αναφέρεται effect size (Δ pp) μαζί με p.
- **Wilson 95% CI** στο majority vote (k από n), επειδή τα 3 runs δεν είναι ανεξάρτητα
  δείγματα (ίδιες ερωτήσεις). Η διάχυση μεταξύ runs αναφέρεται με min–max.

Κανόνες διατύπωσης (προκαθορισμένοι):
- `p_holm < 0.05` **και** bootstrap CI δεν περιέχει 0 → «significantly higher/lower
  (Δ = x pp, 95% CI [a,b], Holm-adjusted p = …)».
- `p_holm ≥ 0.05` → «no detectable difference» (όχι «equivalent»).

## 7. Reranker & adapter (disjointness)

- **Reranker**: επανεκπαιδεύεται **μόνο σε answerable TRAIN** (Πείραμα 1Β). `assert not
  (train_qids & test_qids)` πριν την εκπαίδευση. 5-fold out-of-fold scores στο TRAIN για
  την επιλογή `T_dep` και των baselines.
- **QLoRA adapter**: το `build_sft_dataset.py` χτίζει το SFT **μόνο** από
  `training_Q_A.json` (όλα τα SFT qids έχουν prefix `TRAIN_Q*`). Άρα **SFT ∩ TEST = ∅**
  και ο adapter **επαναχρησιμοποιείται αυτούσιος**. Το Πείραμα 1Α το επιβεβαιώνει
  τυπικά· αν βρεθεί οποιαδήποτε τομή, γίνεται επανεκπαίδευση QLoRA μόνο σε TRAIN
  (seed 42, ίδιοι hyper-parameters) και νέο sha256 στο `env.json`.
- Το SFT **MODE B** (202 παραδείγματα) προήλθε από **answerable** TRAIN με GT εκτός top-5,
  όχι από unanswerable → **καμία επικάλυψη** με τα 40 παλιά synthetic unanswerable.

## 8. Αμετάβλητα (sha256 στο env.json)

Ίδια prompts Mode A/B, ίδιο GGUF/quantization, `num_ctx = 8192`, `num_predict = 1024`
(== `main.py`), ίδιο decoding με το άρθρο. Καταγράφεται το sha256 του αρχείου prompts,
του GGUF και του adapter.

> **temperature — προς επιβεβαίωση:** το `main.py` έχει default `LLM_TEMPERATURE = 0.2`,
> ενώ εδώ δηλώνεται 0. Πριν το πάγωμα, επιβεβαιώστε με τι έτρεξαν οι **αρχικοί** αριθμοί
> και ευθυγραμμίστε αυτή την παράγραφο (0 ή 0.2). Το regression test τρέχει με την αρχική
> τιμή· τα νέα runs με τη δηλωμένη εδώ.

## 9. Regression gate (πριν από κάθε νέο run)

Το παλιό pipeline (παλιός reranker, παλιά παγωμένα top-5, 160 ερωτήσεις) πρέπει να
αναπαράγει τους αριθμούς του άρθρου εντός **±5 μονάδων**:
Base C-Abst 7,1% (4/56), QLoRA 48,2% (27/56), F-Abst 6,7% / 24,0%. Αν αποτύχει, **δεν**
ξεκινούν νέα runs — πρώτα βρίσκεται η διαφορά.

## 10. (Για Πείραμα 4) NLI & annotation

- Κατώφλι NLI: **0.5** (entailment prob).
- Δείγμα χειροκίνητης αξιολόγησης: **50 απαντήσεις**, τυχαία με seed 42, στρωματοποιημένα
  ανά config (Base/QLoRA) και ανά subgroup (present/missing). Δύο αξιολογητές, συμφωνία
  με Cohen's κ.

## 11. Κριτήριο ολοκλήρωσης Πειράματος 1

Πίνακας provenance με **μηδενικές τομές με το TEST**· `retrieval_test.jsonl` παγωμένο με
commit· Stage A με Wilson CI· regression test περασμένο· logs Stage B (Base, QLoRA × 3)
και Stage C (× 3) πλήρη και **χωρίς truncation**· πίνακας συμφωνίας B–C με θόρυβο βάσης.
