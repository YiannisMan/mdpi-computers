from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import CrossEncoder, SentenceTransformer
import pickle
import os
import re
import torch
import bm25s

"""
ΡΟΛΟΣ:
Το κεντρικό API του συστήματος. Δέχεται IT tickets και επιστρέφει
τις πιο σχετικές τεχνικές λύσεις χρησιμοποιώντας Hybrid Search.

ΑΡΧΙΤΕΚΤΟΝΙΚΗ:
1. Embedding Search (ChromaDB / all-MiniLM-L6-v2) — Semantic search
2. BM25 Search (bm25s) — Keyword search
3. Reranker (fine-tuned CrossEncoder) — Αναδιάταξη με καθαρισμένο query
4. Score Fusion — Συνδυασμός reranker score με RRF από αρχικά rankings
"""

# ──────────────────────────────────────────────
#  Globals
# ──────────────────────────────────────────────
bm25_retriever = None
bm25_ids       = []
bm25_docs      = []
bm25_metas     = []

BM25_INDEX_PATH         = "/data/bm25_index.pkl"
BM25S_DIR               = "/data/bm25s_index"
FINETUNED_RERANKER_PATH = "/app/reranker_model"

# Reranker options (επιλέγουμε με USE_BGE_RERANKER / USE_GENERIC_RERANKER):
GENERIC_RERANKER        = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_RERANKER            = "BAAI/bge-reranker-base"

# Embedding model & collection (αναβαθμίστηκαν από MiniLM -> BGE)
EMBEDDING_MODEL         = "all-MiniLM-L6-v2" # all-MiniLM-L6-v2     BAAI/bge-base-en-v1.5
COLLECTION_NAME         = "technotes_MINI" # technotes_MINI         technotes_bgeTesting

# ── Tuning parameters ─────────────────────────
# Stable 80% baseline configuration
INITIAL_K              = 100   # Emb pool depth (αυξήθηκε: μειώνει HNSW variance
                                # — οριακά GT γύρω στο #50 πιάνονται σταθερά)
BM25_TOP_K             = 500   # BM25 pool depth
RERANKER_MAX_LENGTH    = 512   # Hard cap του CrossEncoder
MAX_QUERY_WORDS        = 80    # Πάνω από αυτό -> κόβεται πριν τον reranker
EXPANSION_WORD_LIMIT   = 40    # Query expansion μόνο αν το query είναι μικρό
MAX_DOC_CHARS          = 2000  # Κόψιμο document πριν τον reranker
RRF_K                  = 60    # RRF smoothing constant

# Βάρη fusion: ισορροπημένη αναλογία.
RERANK_WEIGHT          = 0.35
FUSION_WEIGHT          = 0.35
OVERLAP_WEIGHT         = 0.30

# Στοιχεία για overlap scoring — specific technical tokens
MIN_OVERLAP_TOKEN_LEN  = 3
MIN_QUERY_TOKENS_FOR_OVERLAP = 2

# Score normalization
SCORE_NORM_METHOD      = "minmax"  # Επιστροφή στο stable

# Reranker selection
USE_BGE_RERANKER       = False   # BAAI/bge-reranker-base
USE_GENERIC_RERANKER   = False  # ms-marco-MiniLM

# BM25+Embedding consensus boost (single tier)
CONSENSUS_TOP10_BOOST  = 0.15   # Επιστροφή στο stable
CONSENSUS_TOP20_BOOST  = 0.15   # ίδιο για να συμπεριφέρεται σαν παλιό


def load_bm25_index():
    global bm25_retriever, bm25_ids, bm25_docs, bm25_metas

    print(f"Loading BM25 index from {BM25S_DIR}...", flush=True)
    bm25_retriever = bm25s.BM25.load(BM25S_DIR, load_corpus=False)

    print(f"Loading metadata from {BM25_INDEX_PATH}...", flush=True)
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    bm25_ids   = data["ids"]
    bm25_docs  = data["docs"]
    bm25_metas = data["metas"]
    print(f"BM25 index loaded! ({len(bm25_ids)} documents)", flush=True)


# ──────────────────────────────────────────────
#  Lifespan
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_bm25_index()
    except FileNotFoundError as e:
        print(f"[WARNING] BM25 index not found: {e}", flush=True)
        print("Run ingest.py and then build_bm25.py first.", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to load BM25 index: {e}", flush=True)
    yield


# ──────────────────────────────────────────────
#  App & Models
# ──────────────────────────────────────────────
class GPUEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function που φορτώνει ρητά το SentenceTransformer
    στο GPU αν είναι διαθέσιμο. Η default ChromaDB wrapper δεν το κάνει αυτό.
    Πρέπει να είναι ΤΟ ΙΔΙΟ με αυτό που χρησιμοποιήθηκε στο ingest.py.
    """
    def __init__(self, model_name: str, batch_size: int = 64):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[GPUEmbeddingFunction] Loading {model_name} on {device}", flush=True)
        self.model      = SentenceTransformer(model_name, device=device)
        self.device     = device
        self.batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            input,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


# Device για reranker
RERANKER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[CONFIG] PyTorch device:        {RERANKER_DEVICE}", flush=True)
if RERANKER_DEVICE == "cuda":
    print(f"[CONFIG] GPU:                   {torch.cuda.get_device_name(0)}", flush=True)

client = chromadb.HttpClient(host="chromadb", port=8000)
emb_fn = GPUEmbeddingFunction(EMBEDDING_MODEL, batch_size=64)
print(f"[CONFIG] Using embedding model: {EMBEDDING_MODEL}", flush=True)
print(f"[CONFIG] ChromaDB collection:  {COLLECTION_NAME}", flush=True)

# ── Reranker (BGE / fine-tuned / generic) ─────
# Δηλώνουμε ρητά το max_length για να ξέρουμε πώς κόβει ο CrossEncoder
# και το device για να τρέξει σε GPU αν υπάρχει.
# Το BGE reranker είναι 768-dim BERT, ταιριάζει με το BGE embedding stack.
if USE_BGE_RERANKER:
    print(f"[CONFIG] Loading BGE reranker: {BGE_RERANKER}", flush=True)
    reranker = CrossEncoder(BGE_RERANKER, max_length=RERANKER_MAX_LENGTH, device=RERANKER_DEVICE)
    print("BGE reranker is ready!", flush=True)
elif USE_GENERIC_RERANKER:
    print(f"[CONFIG] USE_GENERIC_RERANKER=True — forcing generic model.", flush=True)
    reranker = CrossEncoder(GENERIC_RERANKER, max_length=RERANKER_MAX_LENGTH, device=RERANKER_DEVICE)
    print("Generic reranker is ready!", flush=True)
elif os.path.isdir(FINETUNED_RERANKER_PATH):
    print(f"Loading the fine-tuned reranker from {FINETUNED_RERANKER_PATH}...", flush=True)
    reranker = CrossEncoder(FINETUNED_RERANKER_PATH, max_length=RERANKER_MAX_LENGTH, device=RERANKER_DEVICE)
    print("Fine-tuned reranker is ready!", flush=True)
else:
    print(f"Fine-tuned reranker not found. Using generic.", flush=True)
    reranker = CrossEncoder(GENERIC_RERANKER, max_length=RERANKER_MAX_LENGTH, device=RERANKER_DEVICE)
    print("Generic reranker is ready!", flush=True)

app = FastAPI(title="Knowledge Base API", lifespan=lifespan)


# ──────────────────────────────────────────────
#  Schema
# ──────────────────────────────────────────────
class TicketQuery(BaseModel):
    issue_description: str
    n_results: int = 3


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def clean_query_for_reranker(text: str, max_words: int = MAX_QUERY_WORDS) -> str:
    """
    Καθαρίζει ένα "βρώμικο" query (stack trace, javacore dump, logs) και κρατάει
    τα πιο σημασιολογικά σημαντικά κομμάτια: τις πρώτες προτάσεις + τα error codes.

    Ο CrossEncoder έχει hard cap στα 512 tokens — αν περάσουμε ολόκληρο
    javacore dump, κόβονται και το query και το document και το reranker
    βλέπει μόνο θόρυβο.
    """
    if not text:
        return ""

    text = text.strip()

    # Αν είναι ήδη μικρό, επιστροφή ως έχει
    words = text.split()
    if len(words) <= max_words:
        return text

    # Error codes / patterns που θέλουμε να διατηρήσουμε (IBM-style)
    error_patterns = re.findall(
        r'('
        r'EJP[A-Z0-9]+'                   # EJPDB0085W, EJPFD0016E
        r'|[A-Z]{2,}\d{4,}[A-Z]?'         # APARs (PM50683, IV83315, κλπ)
        r'|swg\d{5,}'                     # technote IDs
        r'|ERROR:\s*[^\n]{0,100}'         # ERROR: lines
        r'|[A-Za-z_]+Exception'           # OutOfMemoryException etc.
        r'|[A-Za-z_]+Error'               # GPError etc.
        r')',
        text
    )

    # Πρώτες 3 προτάσεις
    sentences  = re.split(r'(?<=[.!?])\s+', text)
    first_part = ' '.join(sentences[:3])

    # Συνδυασμός με deduplication
    combined   = first_part + ' ' + ' '.join(dict.fromkeys(error_patterns))
    result_words = combined.split()[:max_words]

    return ' '.join(result_words)


def build_query_expansions(issue: str) -> list[str]:
    """
    Query expansion μόνο για μικρά/καθαρά queries. Για μεγάλα queries με
    stack traces, το expansion προσθέτει θόρυβο (π.χ. "How to fix: [800 λέξεις]")
    και μπλοκάρει τη σημασιολογική σημασία του αρχικού query.
    """
    word_count = len(issue.split())

    if word_count > EXPANSION_WORD_LIMIT:
        return [issue]

    return [
        issue,
        f"How to fix: {issue}",
        f"Error troubleshooting: {issue}",
        f"Technical problem resolution: {issue}",
    ]


def bm25_search(query: str, top_k: int) -> tuple[list, list, list]:
    """
    Keyword search με bm25s.

    ΤOKENIZATION: Χρησιμοποιούμε το bm25s.tokenize ΧΩΡΙΣ stopwords/stemming,
    ώστε να συμπεριφέρεται όπως το `re.findall(r'\\w+', text.lower())` που
    χρησιμοποιεί το ingest.py. Έτσι:
      - Παίρνουμε το σωστό format που περιμένει το bm25s.retrieve()
      - Δεν έχουμε mismatch με το index (που χτίστηκε με re.findall)
    """
    if bm25_retriever is None:
        return [], [], []

    # bm25s.tokenize με ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΑ stopwords + stemming
    # → ίδια συμπεριφορά με re.findall(r'\w+', q.lower()) αλλά σωστό format.
    query_tokens = bm25s.tokenize(
        query,
        stopwords=None,    # χωρίς stopword removal
        stemmer=None,      # χωρίς stemming
        show_progress=False,
    )

    if not query_tokens:
        return [], [], []

    k = min(top_k, len(bm25_ids))
    results, _ = bm25_retriever.retrieve(query_tokens, k=k)
    top_indices = results[0].tolist()

    return (
        [bm25_ids[i]   for i in top_indices],
        [bm25_docs[i]  for i in top_indices],
        [bm25_metas[i] for i in top_indices],
    )


def normalize_scores(scores: list[float], method: str = SCORE_NORM_METHOD) -> list[float]:
    """
    Κανονικοποιεί scores για δίκαιο blending στο fusion.

    method='minmax':  Map σε [0,1]. Πρόβλημα: αν 1 doc έχει score 9.5 και 49 docs
                      έχουν 0.5, το min-max παράγει distribution {1.0, 0.05, 0.05, ...}
                      που υπερτονίζει τον winner.

    method='zscore':  (x - mean) / std, μετά rescale σε [0,1] με sigmoid.
                      Λιγότερο ευαίσθητο σε outliers, πιο "δίκαιη" κατανομή.
    """
    if not scores:
        return []

    n = len(scores)
    if n == 1:
        return [0.5]

    if method == "minmax":
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-9:
            return [0.5] * n
        return [(s - mn) / (mx - mn) for s in scores]

    # zscore method
    mean = sum(scores) / n
    var  = sum((s - mean) ** 2 for s in scores) / n
    std  = var ** 0.5
    if std < 1e-9:
        return [0.5] * n

    # Z-score, then map με sigmoid σε (0,1)
    import math
    zscores = [(s - mean) / std for s in scores]
    # sigmoid: 1/(1+e^-z). Z=0 -> 0.5, z=2 -> ~0.88, z=-2 -> ~0.12
    return [1.0 / (1.0 + math.exp(-z)) for z in zscores]


def extract_technical_tokens(text: str) -> set[str]:
    """
    Εξάγει "technical" tokens από το query - αυτά που είναι εξειδικευμένα
    και πιθανότατα εμφανίζονται αυτούσια στο σωστό document:

    - Version numbers (8.1.0.4, 9.0.0.0)
    - APARs & technote IDs (PM50683, swg21138332, IV83315)
    - Error codes (EJPDB0085W)
    - CamelCase identifiers (StaleConnectionException)
    - Acronyms (WAS, DASH, ODM, DVS, RES)
    - Specific command/file names (jextract, nzudxcompile, configmgr)
    - Specific product/tech nouns εκτός των πιο γενικών (datacap, datapower, jdk)
    """
    if not text:
        return set()

    tokens: set[str] = set()

    # Version numbers: 8.1.0.4, 9.0.0.0, v2.0.3
    tokens.update(re.findall(r'\b[vV]?\d+(?:\.\d+){1,4}\b', text))

    # APARs, technote IDs (case-insensitive)
    tokens.update(m.lower() for m in re.findall(
        r'\b(?:swg|nas|isg|ibm)\d{5,}[A-Z]?\b', text, re.IGNORECASE
    ))

    # All-caps code pattern (PM50683, IV83315)
    tokens.update(m.lower() for m in re.findall(
        r'\b[A-Z]{2,}[0-9]{4,}[A-Z]?\b', text
    ))

    # EJP... error codes
    tokens.update(m.lower() for m in re.findall(r'\bEJP[A-Z0-9]+\b', text))

    # CamelCase identifiers
    tokens.update(m.lower() for m in re.findall(
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text
    ))

    # Acronyms (3-5 uppercase letters)
    tokens.update(m.lower() for m in re.findall(r'\b[A-Z]{3,5}\b', text))

    # snake_case και kebab-case
    tokens.update(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text.lower()))
    tokens.update(re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z0-9]+)*\b', text.lower()))

    # Τεχνικά nouns που εμφανίζονται αυτούσια ως product/tool names.
    # Είναι "rare vocabulary" - λέξεις που σπάνια εμφανίζονται σε γενικό κείμενο
    # αλλά είναι signature για specific products/tools.
    text_lower = text.lower()
    rare_nouns = [
        # Tools / commands
        r'\bjextract\b', r'\bconfigmgr\b', r'\bnzudxcompile\b',
        r'\bupdateinstaller\b', r'\bjavacore\b', r'\bcrypto-export\b',
        # Products (lowercase forms)
        r'\bdatacap\b', r'\bdatapower\b', r'\bwebsphere\b', r'\bnetcool\b',
        r'\bomnibus\b', r'\bteamserver\b', r'\bfilenet\b', r'\bjazz\b',
        r'\btwinfin\b', r'\bfixpack\b', r'\bfixpacks\b', r'\binfocenter\b',
        # Short but critical IBM acronyms (lowercase)
        r'\bwas\b', r'\bodm\b', r'\bdash\b', r'\brad\b', r'\brsa\b',
        r'\bdw\b', r'\bxu\b', r'\bssp\b', r'\bcpe\b', r'\bitm\b',
        r'\bicm\b', r'\bjdk\b', r'\bjre\b', r'\bhsm\b', r'\bgpf\b',
        r'\bdb2\b', r'\bwsp\b', r'\bdvs\b', r'\bres\b', r'\bnoi\b',
        r'\bnoi\b', r'\bgc\b',
    ]
    for pat in rare_nouns:
        if re.search(pat, text_lower):
            tokens.add(pat.strip(r'\b'))

    # File paths / commands με slashes
    tokens.update(re.findall(
        r'/(?:[a-zA-Z0-9._-]+/){0,4}[a-zA-Z0-9._-]+',
        text
    ))

    # Κανονικοποίηση: lowercase + μήκος
    tokens = {t.lower() for t in tokens if len(t) >= MIN_OVERLAP_TOKEN_LEN}

    # Αφαίρεση ψεύτικων acronyms που δεν μας λένε κάτι
    stop_acronyms = {
        "the", "and", "for", "how", "why", "can", "use", "api", "all",
        "yes", "new", "old", "not", "any", "get", "set", "run", "one",
        "two", "ibm", "but", "now", "has", "our", "was",  # "was" είναι ambiguous
        "gui", "url", "xml", "pdf", "csv", "sql", "fix", "app", "see",
        "error", "info", "warn", "debug", "null", "true", "false",
    }
    # ΠΡΟΣΟΧΗ: "was" το αφαιρούμε από την blacklist γιατί στον τεχνικό χώρο
    # είναι IBM WebSphere Application Server. Αν εμφανίζεται σε context τεχνικό,
    # το κρατάμε. Το αφήνουμε στη blacklist παραπάνω και βασιζόμαστε στο
    # rare_nouns για το "was" ως acronym.
    tokens -= stop_acronyms
    # Επιστροφή του "was" αν είχε εντοπιστεί από τα rare_nouns
    if re.search(r'\bwas\b', text_lower):
        tokens.add("was")

    return tokens


def compute_overlap_score(query_tokens: set[str], doc: str) -> float:
    """
    Υπολογίζει ποσοστό technical tokens του query που υπάρχουν στο document.
    Επιστρέφει value στο [0, 1].
    """
    if not query_tokens or not doc:
        return 0.0
    doc_lower = doc.lower()
    hits = sum(1 for tok in query_tokens if tok in doc_lower)
    return hits / len(query_tokens)


# ──────────────────────────────────────────────
#  Endpoint
# ──────────────────────────────────────────────
@app.post("/search")
def search_knowledge_base(query: TicketQuery):
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="The database is empty! Run ingest.py first."
        )

    # ── 1. Embedding Search (semantic) ──────────────────────────────────
    expanded_queries = build_query_expansions(query.issue_description)

    emb_results = collection.query(
        query_texts=expanded_queries,
        n_results=INITIAL_K
    )

    # Κρατάμε το ΚΑΛΥΤΕΡΟ rank ανά doc_id από όλα τα expansions
    # (αν ένα doc εμφανίζεται σε πολλά expansions, παίρνει το μικρότερο rank)
    emb_best_rank: dict[str, int] = {}
    merged_ids: list[str]         = []
    merged_docs: list[str]        = []
    merged_metas: list            = []
    seen_ids: set[str]            = set()

    for q_idx in range(len(expanded_queries)):
        for rank, (doc_id, doc, meta) in enumerate(zip(
            emb_results['ids'][q_idx],
            emb_results['documents'][q_idx],
            emb_results['metadatas'][q_idx]
        )):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged_ids.append(doc_id)
                merged_docs.append(doc)
                merged_metas.append(meta)
            # Κρατάμε το best (μικρότερο) rank
            prev = emb_best_rank.get(doc_id)
            if prev is None or rank < prev:
                emb_best_rank[doc_id] = rank

    # ── 2. BM25 Search (keyword) ─────────────────────────────────────────
    bm25_res_ids, bm25_res_docs, bm25_res_metas = bm25_search(
        query.issue_description, BM25_TOP_K
    )

    bm25_rank: dict[str, int] = {}
    bm25_new = 0
    for rank, (doc_id, doc, meta) in enumerate(
        zip(bm25_res_ids, bm25_res_docs, bm25_res_metas)
    ):
        bm25_rank[doc_id] = rank
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            merged_ids.append(doc_id)
            merged_docs.append(doc)
            merged_metas.append(meta)
            bm25_new += 1

    print(
        f"[Hybrid Search] "
        f"Expansions: {len(expanded_queries)} | "
        f"Embedding unique: {len(merged_ids) - bm25_new} | "
        f"BM25 new: {bm25_new} | "
        f"Total for reranking: {len(merged_ids)}",
        flush=True
    )

    # ── 3. Reranking με clean query ──────────────────────────────────────
    answers = []

    if merged_ids:
        clean_query = clean_query_for_reranker(query.issue_description)

        # Clamp και τα documents για να αφήσουμε χώρο στο query
        cross_inp = [
            [clean_query, (doc[:MAX_DOC_CHARS] if isinstance(doc, str) else "")]
            for doc in merged_docs
        ]
        cross_scores = reranker.predict(cross_inp)

        # ── 4. Score Fusion (Reranker + RRF + Overlap + Tiered Consensus) ──
        # 4a. RRF scores από τα αρχικά rankings + tiered consensus
        rrf_scores = []
        consensus_boosts = []  # Per-doc boost (0.0, 0.10, ή 0.25)
        for doc_id in merged_ids:
            s = 0.0
            emb_r  = emb_best_rank.get(doc_id)
            bm25_r = bm25_rank.get(doc_id)
            if emb_r is not None:
                s += 1.0 / (RRF_K + emb_r)
            if bm25_r is not None:
                s += 1.0 / (RRF_K + bm25_r)
            rrf_scores.append(s)
            # Tiered consensus:
            # - Αν είναι top-10 και στα δύο: ισχυρό σήμα (0.25)
            # - Αν είναι top-20 και στα δύο: μέτριο σήμα (0.10)
            # - Αλλιώς: τίποτα
            if (emb_r is not None and emb_r < 10
                    and bm25_r is not None and bm25_r < 10):
                consensus_boosts.append(CONSENSUS_TOP10_BOOST)
            elif (emb_r is not None and emb_r < 20
                  and bm25_r is not None and bm25_r < 20):
                consensus_boosts.append(CONSENSUS_TOP20_BOOST)
            else:
                consensus_boosts.append(0.0)

        # 4b. Technical-token overlap: κρατάει queries όπως "datacap 9.0",
        # "jextract", "was 9.0.0.4" που το embedding δεν κωδικοποιεί καλά.
        #
        # Ενεργοποιείται ΜΟΝΟ αν το query έχει αρκετά discriminative tokens.
        # Αλλιώς, γενικά tokens (π.χ. "api", "web") δίνουν λάθος bonus.
        query_tech_tokens = extract_technical_tokens(query.issue_description)

        if len(query_tech_tokens) >= MIN_QUERY_TOKENS_FOR_OVERLAP:
            overlap_scores = [
                compute_overlap_score(query_tech_tokens, doc)
                for doc in merged_docs
            ]
            effective_overlap_weight = OVERLAP_WEIGHT
        else:
            # Πολύ λίγα tokens — το overlap δεν είναι αξιόπιστο σήμα
            overlap_scores = [0.0] * len(merged_docs)
            effective_overlap_weight = 0.0

        # Normalize όλα τα scores στο [0,1] για δίκαιο blending
        norm_rerank  = normalize_scores([float(s) for s in cross_scores])
        norm_rrf     = normalize_scores(rrf_scores)
        # Το overlap είναι ήδη στο [0,1], αλλά κάνουμε normalize για
        # να χειριστούμε την περίπτωση που όλα τα docs έχουν hit.
        norm_overlap = normalize_scores(overlap_scores)

        print(
            f"[Scoring] Query tech tokens: {len(query_tech_tokens)} | "
            f"Overlap active: {effective_overlap_weight > 0} | "
            f"Docs with overlap>0: {sum(1 for s in overlap_scores if s > 0)}",
            flush=True
        )

        # Αν το overlap δεν είναι ενεργό, ανακατανέμουμε το βάρος του
        # στο reranker + RRF proportionally
        if effective_overlap_weight == 0.0:
            total = RERANK_WEIGHT + FUSION_WEIGHT
            w_rerank = RERANK_WEIGHT / total
            w_fusion = FUSION_WEIGHT / total
            w_overlap = 0.0
        else:
            w_rerank  = RERANK_WEIGHT
            w_fusion  = FUSION_WEIGHT
            w_overlap = effective_overlap_weight

        top10_count = sum(1 for b in consensus_boosts if b == CONSENSUS_TOP10_BOOST)
        top20_count = sum(1 for b in consensus_boosts if b == CONSENSUS_TOP20_BOOST)
        print(
            f"[Consensus] Top-10 agreement: {top10_count} docs | "
            f"Top-20 agreement: {top20_count} docs",
            flush=True
        )

        reranked = [
            {
                "technote_id":    merged_ids[i],
                "title":          merged_metas[i].get('title', ''),
                "content":        merged_docs[i],
                "rerank_score":   float(cross_scores[i]),
                "rrf_score":      float(rrf_scores[i]),
                "overlap_score":  float(overlap_scores[i]),
                "consensus_boost": consensus_boosts[i],
                "final_score":    (
                    w_rerank  * norm_rerank[i]
                    + w_fusion  * norm_rrf[i]
                    + w_overlap * norm_overlap[i]
                    + consensus_boosts[i]
                ),
            }
            for i in range(len(cross_scores))
        ]

        reranked.sort(key=lambda x: x['final_score'], reverse=True)
        answers = reranked[:query.n_results]

    return {"query": query.issue_description, "suggested_solutions": answers}

# ══════════════════════════════════════════════════════════════════════
#  ΦΑΣΗ 2 — RAG Generation Layer (Ollama + Llama-3-8B-Instruct)
# ══════════════════════════════════════════════════════════════════════
"""
ΡΟΛΟΣ (Φάση 2 / Στάδιο Β):
Πάνω από το hybrid retrieval της Φάσης Α προστίθεται το στάδιο παραγωγής
(Augmented Generation). Το endpoint /resolve:
  1. Ανακτά τα top-K technotes μέσω του ΥΠΑΡΧΟΝΤΟΣ hybrid search (search_knowledge_base)
  2. Χτίζει ένα grounded prompt (ticket + technotes ως context)
  3. Καλεί το Llama-3-8B-Instruct μέσω του Ollama container
  4. Επιστρέφει μια τεκμηριωμένη λύση μαζί με τις πηγές (technote IDs) για citations

ΣΗΜΕΙΩΣΗ ΣΧΕΔΙΑΣΜΟΥ:
- ΔΕΝ τροποποιείται η λογική της Φάσης Α. Το /resolve καλεί αυτούσιο το
  search_knowledge_base() και χτίζει από πάνω το generation layer.
- Χρησιμοποιούμε το BASE llama3:8b-instruct (χωρίς fine-tuning) ώστε να έχουμε
  ένα λειτουργικό end-to-end baseline. Στο επόμενο βήμα (QLoRA fine-tuning)
  θα συγκριθεί με αυτό το baseline για να τεκμηριωθεί η βελτίωση.
"""

import requests

# ── Ollama / LLM config (env-overridable) ─────────────────────────────
# Μέσα στο docker network το Ollama είναι προσβάσιμο με το service name "ollama"
# (ΟΧΙ localhost — μέσα σε container το localhost δείχνει στο ίδιο το container).
OLLAMA_HOST       = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M")

RAG_TOP_K         = int(os.getenv("RAG_TOP_K", "5"))        # πόσα technotes ταΐζουμε στο LLM
# ΠΡΟΣΟΧΗ: το Ollama default num_ctx είναι 4096 — μικρό για RAG με 5 έγγραφα.
# Το llama3:8b υποστηρίζει 8192, οπότε το ορίζουμε ρητά.
LLM_NUM_CTX       = int(os.getenv("LLM_NUM_CTX", "8192"))   # context window του llama3:8b
LLM_TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.2"))  # χαμηλό -> grounded, λιγότερα hallucinations
LLM_NUM_PREDICT   = int(os.getenv("LLM_NUM_PREDICT", "1024"))  # max output tokens
LLM_TIMEOUT       = int(os.getenv("LLM_TIMEOUT", "300"))       # sec — σε RTX 3060 η παραγωγή μπορεί να αργεί (+cold start)

LLM_DOC_CHAR_LIMIT    = int(os.getenv("LLM_DOC_CHAR_LIMIT", "1600"))    # truncation ανά technote στο context
LLM_TICKET_CHAR_LIMIT = int(os.getenv("LLM_TICKET_CHAR_LIMIT", "3000")) # truncation του ticket


# Retrieval-confidence GATE (προαιρετικο, default ΑΠΕΝΕΡΓΟ):
# Αν οριστει, κανει abstain ΠΡΙΝ το (ακριβο) generation οταν το max ΩΜΟ rerank
# logit των ανακτημενων technotes ειναι κατω απο το threshold. Σημα cross-query
# συγκρισιμο (ωμο CrossEncoder logit, ΟΧΙ το minmax-normalized final_score).
# Κενο/μη ορισμενο -> καμια αλλαγη συμπεριφορας.
_gate_raw = os.getenv("RAG_GATE_THRESHOLD", "").strip()
RAG_GATE_THRESHOLD = float(_gate_raw) if _gate_raw else None

print(f"[CONFIG] Ollama host:           {OLLAMA_HOST}", flush=True)
print(f"[CONFIG] Ollama model:          {OLLAMA_MODEL}", flush=True)
print(f"[CONFIG] RAG top-K docs:        {RAG_TOP_K}", flush=True)
print(f"[CONFIG] Retrieval gate thr:    {'OFF' if RAG_GATE_THRESHOLD is None else RAG_GATE_THRESHOLD}", flush=True)


# ── Schema ────────────────────────────────────────────────────────────
class ResolveQuery(BaseModel):
    issue_description: str
    n_docs: int = RAG_TOP_K          # πόσα technotes να ανακτηθούν & να ταΐσουν το LLM


# ── RAG helpers ───────────────────────────────────────────────────────
# Κοινη σταθερα για το abstention - ιδια στο prompt, στο empty-retrieval branch
# και ανιχνευσιμη απο το evaluate_generation.py (substring match).
ABSTAIN_PHRASE = (
    "The knowledge base does not contain a confident solution for this issue. "
    "Escalate to a human engineer."
)

RAG_SYSTEM_PROMPT = """You are an expert IBM technical support engineer. You resolve IT support tickets using ONLY the technical documents (technotes) provided as context.

STEP 1 - Decide whether the provided technotes actually address THIS ticket's specific problem. Then answer in EXACTLY ONE of the two modes below. Never mix them.

MODE A - SOLUTION (use this when at least one technote is relevant to the ticket):
- Base every step STRICTLY on the provided technotes. Do not use outside knowledge or assumptions.
- Cite the technote ID(s) you rely on in square brackets next to the relevant step, e.g. [swg21234567]. Only cite IDs that appear in the context above.
- Do NOT invent steps, commands, version numbers, file paths, or technote IDs.
- Be concise and practical: prefer concrete commands/settings over generic advice.
- Structure: (1) one-line root-cause summary, (2) numbered resolution steps, (3) the technote ID(s) you used.
- Do NOT include any abstention or "cannot find" disclaimer anywhere in this mode.

MODE B - ABSTAIN (use this ONLY when none of the technotes address the ticket's problem):
- Output EXACTLY this single line and nothing else:
  The knowledge base does not contain a confident solution for this issue. Escalate to a human engineer.
- Do NOT add any steps, citations, root-cause summary, or other text.

Hard rule: if you give a solution (Mode A), do NOT append the abstain line. If you abstain (Mode B), do NOT give any steps or citations. Choose ONE mode only."""


def build_rag_context(solutions: list[dict]) -> tuple[str, list[dict]]:
    """
    Μετατρέπει τα ανακτημένα technotes σε ένα context string για το LLM και
    επιστρέφει παράλληλα τη λίστα των πηγών (sources) για citations / Phase 3.
    Κάθε technote περικόπτεται σε LLM_DOC_CHAR_LIMIT χαρακτήρες ώστε το σύνολο
    να χωρά στο num_ctx.
    """
    blocks  = []
    sources = []
    for i, sol in enumerate(solutions, start=1):
        tid     = sol.get("technote_id", f"doc_{i}")
        title   = sol.get("title", "") or ""
        content = sol.get("content", "") or ""

        # Το content ξεκινά με "Title: {title}\n\n{body}" (από το ingest.py).
        # Αφαιρούμε το διπλό title για να μη σπαταλάμε context.
        if content.startswith("Title:"):
            parts   = content.split("\n\n", 1)
            content = parts[1] if len(parts) == 2 else content
        content = content[:LLM_DOC_CHAR_LIMIT]

        blocks.append(
            f"[Technote {i}] ID: {tid}\n"
            f"Title: {title}\n"
            f"Content:\n{content}\n"
            f"---"
        )
        sources.append({
            "rank":         i,
            "technote_id":  tid,
            "title":        title,
            "final_score":  sol.get("final_score"),
            "rerank_score": sol.get("rerank_score"),
        })

    return "\n".join(blocks), sources


def build_rag_messages(ticket: str, context: str) -> list[dict]:
    ticket = (ticket or "").strip()[:LLM_TICKET_CHAR_LIMIT]
    user_prompt = (
        f"## Support Ticket\n{ticket}\n\n"
        f"## Retrieved Technotes (knowledge base context)\n{context}\n\n"
        f"## Task\nProvide a resolution for the ticket above, strictly following the rules."
    )
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]


def call_ollama_chat(messages: list[dict]) -> str:
    """
    Non-streaming κλήση στο Ollama /api/chat. Επιστρέφει το κείμενο της απάντησης.
    Σηκώνει HTTPException με καθαρό μήνυμα αν το Ollama δεν είναι διαθέσιμο ή αν
    το μοντέλο δεν έχει γίνει pull μέσα στο container.
    """
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options": {
            "num_ctx":     LLM_NUM_CTX,
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_NUM_PREDICT,
        },
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=LLM_TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=(f"Cannot reach Ollama at {OLLAMA_HOST}. Is the 'ollama' container "
                    f"running and on the same docker network as the API?")
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail=f"Ollama generation timed out after {LLM_TIMEOUT}s."
        )

    if resp.status_code == 404:
        # Συνήθης αιτία: το μοντέλο δεν έχει γίνει pull μέσα στο container.
        raise HTTPException(
            status_code=503,
            detail=(f"Ollama returned 404 for model '{OLLAMA_MODEL}'. Pull it first: "
                    f"docker exec -it ollama ollama pull {OLLAMA_MODEL}")
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


# ── Endpoint: /resolve (Retrieval + Generation) ───────────────────────
@app.post("/resolve")
def resolve_ticket(query: ResolveQuery):
    # 1. RETRIEVAL — ξαναχρησιμοποιούμε ΑΥΤΟΥΣΙΟ το hybrid search της Φάσης Α
    retrieval = search_knowledge_base(
        TicketQuery(issue_description=query.issue_description, n_results=query.n_docs)
    )
    solutions = retrieval.get("suggested_solutions", [])

    if not solutions:
        return {
            "ticket":  query.issue_description,
            "answer":  ABSTAIN_PHRASE,
            "sources": [],
            "model":   OLLAMA_MODEL,
            "retrieved_count": 0,
            "gated":           False,
            "gate_threshold":  RAG_GATE_THRESHOLD,
            "top_final_score":  None,
            "max_rerank_score": None,
        }

    # 2. RETRIEVAL-CONFIDENCE GATE (προαιρετικο, default OFF)
    # Σημα: max ωμο rerank logit. Αν < threshold -> abstain ΠΡΙΝ το generation,
    # χωρις να σπαταληθει χρονος στο LLM. Cross-query συγκρισιμο.
    rerank_vals = [s.get("rerank_score") for s in solutions
                   if s.get("rerank_score") is not None]
    max_rerank  = max(rerank_vals) if rerank_vals else None

    if (RAG_GATE_THRESHOLD is not None and max_rerank is not None
            and max_rerank < RAG_GATE_THRESHOLD):
        _, gated_sources = build_rag_context(solutions)
        print(
            f"[GATE] max_rerank={max_rerank:.3f} < {RAG_GATE_THRESHOLD} "
            f"-> abstain (no generation)",
            flush=True
        )
        return {
            "ticket":          query.issue_description,
            "answer":          ABSTAIN_PHRASE,
            "sources":         gated_sources,
            "model":           OLLAMA_MODEL,
            "retrieved_count": len(solutions),
            "gated":           True,
            "gate_threshold":  RAG_GATE_THRESHOLD,
            "top_final_score":  gated_sources[0].get("final_score") if gated_sources else None,
            "max_rerank_score": max_rerank,
        }

    # 3. BUILD grounded prompt
    context, sources = build_rag_context(solutions)
    messages = build_rag_messages(query.issue_description, context)

    print(
        f"[RAG] Retrieved {len(solutions)} technotes -> generating with "
        f"{OLLAMA_MODEL} (num_ctx={LLM_NUM_CTX})",
        flush=True
    )

    # 3. GENERATE
    answer = call_ollama_chat(messages)

    # 5. RETURN τεκμηριωμένη λύση + πηγές (sources)
    return {
        "ticket":          query.issue_description,
        "answer":          answer,
        "sources":         sources,
        "model":           OLLAMA_MODEL,
        "retrieved_count": len(solutions),
        "gated":           False,
        "gate_threshold":  RAG_GATE_THRESHOLD,
        "top_final_score":  sources[0].get("final_score") if sources else None,
        "max_rerank_score": max_rerank,
    }


# ── Endpoint: /health (διαγνωστικό για το demo) ───────────────────────
@app.get("/health")
def health():
    status = {
        "bm25_loaded":      bm25_retriever is not None,
        "bm25_documents":   len(bm25_ids),
        "embedding_model":  EMBEDDING_MODEL,
        "collection":       COLLECTION_NAME,
        "ollama_host":      OLLAMA_HOST,
        "ollama_model":     OLLAMA_MODEL,
        "ollama_reachable": False,
        "model_available":  False,
    }
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.status_code == 200:
            status["ollama_reachable"] = True
            tags = [m.get("name", "") for m in r.json().get("models", [])]
            status["available_models"] = tags
            status["model_available"] = any(
                t == OLLAMA_MODEL or t.split(":")[0] == OLLAMA_MODEL.split(":")[0]
                for t in tags
            )
    except Exception:
        pass
    return status