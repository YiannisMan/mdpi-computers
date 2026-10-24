import tarfile
import bz2
import json
import re
import os
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from chromadb import Documents, EmbeddingFunction, Embeddings

"""
ΡΟΛΟΣ:
Αυτό το script διαβάζει το αρχείο TechQA.tar.gz, επεξεργάζεται τα τεχνικά έγγραφα (technotes)
και τα αποθηκεύει σε δύο μορφές:
1. Στη διανυσματική βάση ChromaDB (για σημασιολογική αναζήτηση).
2. Σε ένα αρχείο JSONL (για τη μετέπειτα δημιουργία του BM25 index).

GPU: Χρησιμοποιούμε custom EmbeddingFunction αντί για την default της ChromaDB,
γιατί η default δεν φορτώνει το μοντέλο στο GPU ακόμα και όταν είναι διαθέσιμο.
"""

# -- Model & Collection --------------------------
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
COLLECTION_NAME   = "technotes_MINI"


class GPUEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function που φορτώνει ρητά το SentenceTransformer
    στο GPU αν είναι διαθέσιμο. Η default ChromaDB wrapper δεν το κάνει αυτό.
    """
    def __init__(self, model_name: str, batch_size: int = 64):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[GPUEmbeddingFunction] Loading {model_name} on {device}")
        if device == "cuda":
            print(f"[GPUEmbeddingFunction] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[GPUEmbeddingFunction] VRAM: "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        self.model       = SentenceTransformer(model_name, device=device)
        self.device      = device
        self.batch_size  = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        # ChromaDB καλεί αυτό για κάθε batch που περνάει στο collection.add()
        embeddings = self.model.encode(
            input,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # BGE συνιστάται normalized
        )
        return embeddings.tolist()


# Σύνδεση με το container του ChromaDB
client = chromadb.HttpClient(host="chromadb", port=8000)

# Εγγύηση ότι θα χρησιμοποιηθεί GPU
emb_fn = GPUEmbeddingFunction(EMBEDDING_MODEL, batch_size=64)

# Δημιουργία ή ανάκτηση του collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=emb_fn
)

# Διαδρομές αρχείων
TEMP_TOKENS_PATH  = "/data/bm25_tokens.jsonl"
TAR_PATH          = "/data/TechQA.tar.gz"

CHROMA_BATCH      = 512
TOKEN_FLUSH_EVERY = 10_000

TITLE_BOOST_FACTOR = 3


def tokenize(text: str) -> list[str]:
    """Tokenization για BM25: lowercase λέξεις/αριθμοί."""
    return re.findall(r'\w+', text.lower())


def build_bm25_tokens(title: str, body: str,
                      title_factor: int = TITLE_BOOST_FACTOR) -> list[str]:
    title_tokens = tokenize(title)
    body_tokens  = tokenize(body)
    return title_tokens * title_factor + body_tokens


def collect_ground_truth_ids(tar) -> set[str]:
    dev_member = next(
        (m for m in tar.getmembers() if m.name.endswith("dev_Q_A.json")), None
    )
    if not dev_member:
        print("WARNING: dev_Q_A.json not found.")
        return set()
    f       = tar.extractfile(dev_member)
    qa_data = json.load(f)
    ids = {item["DOCUMENT"] for item in qa_data
               if item.get("ANSWERABLE") == "Y" and item.get("DOCUMENT")}
    print(f"Ground Truth: {len(ids)} unique IDs.")
    return ids


def add_to_chroma(batch_ids, batch_docs, batch_metadatas):
    try:
        collection.add(documents=batch_docs, metadatas=batch_metadatas, ids=batch_ids)
    except Exception:
        collection.upsert(documents=batch_docs, metadatas=batch_metadatas, ids=batch_ids)


def flush_to_jsonl(f_out, ids_buf, docs_buf, metas_buf, tokens_buf):
    for doc_id, doc, meta, tokens in zip(ids_buf, docs_buf, metas_buf, tokens_buf):
        record = {"id": doc_id, "doc": doc, "meta": meta, "tokens": tokens}
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    f_out.flush()
    os.fsync(f_out.fileno())


def load_data(limit, guarantee_ground_truth=True):
    print("=" * 60)
    print(f"Embedding model:        {EMBEDDING_MODEL}")
    print(f"Collection:             {COLLECTION_NAME}")
    print(f"BM25 title boost:       {TITLE_BOOST_FACTOR}x")
    print(f"ChromaDB batch size:    {CHROMA_BATCH}")
    print("=" * 60)

    if os.path.exists(TEMP_TOKENS_PATH):
        os.remove(TEMP_TOKENS_PATH)
        print(f"Cleaned: {TEMP_TOKENS_PATH}")

    batch_docs, batch_ids, batch_metadatas   = [], [], []
    ids_buf, docs_buf, metas_buf, tokens_buf = [], [], [], []

    count           = 0
    priority_loaded = 0

    try:
        with tarfile.open(TAR_PATH, "r:gz") as tar:

            ground_truth_ids    = set()
            ground_truth_loaded = set()
            if guarantee_ground_truth:
                ground_truth_ids = collect_ground_truth_ids(tar)

            print("Search the Technotes archive...")
            bz2_member = next(
                (m for m in tar.getmembers()
                 if m.name.endswith("full_technote_collection.txt.bz2")), None
            )
            if not bz2_member:
                print("Error: full_technote_collection.txt.bz2 not found!")
                return

            print("Found! Start export...")
            extracted_bz2 = tar.extractfile(bz2_member)

            with open(TEMP_TOKENS_PATH, "a", encoding="utf-8") as f_out, \
                 bz2.open(extracted_bz2, "rt", encoding="utf-8") as file:

                for line in file:
                    technotes_array = json.loads(line)

                    for note in technotes_array:
                        note_id  = note['id']
                        title    = note['title']
                        body     = note['text']
                        doc_text = f"Title: {title}\n\n{body}"
                        doc_meta = {"title": title}
                        is_gt    = note_id in ground_truth_ids

                        bm25_tokens = build_bm25_tokens(title, body)

                        if count >= limit:
                            if is_gt and note_id not in ground_truth_loaded:
                                add_to_chroma([note_id], [doc_text], [doc_meta])
                                flush_to_jsonl(f_out,
                                               [note_id], [doc_text], [doc_meta],
                                               [bm25_tokens])
                                ground_truth_loaded.add(note_id)
                                priority_loaded += 1
                                print(f"  [Priority] {note_id}")

                            if len(ground_truth_loaded) >= len(ground_truth_ids):
                                break
                            continue

                        if is_gt:
                            ground_truth_loaded.add(note_id)

                        batch_ids.append(note_id)
                        batch_docs.append(doc_text)
                        batch_metadatas.append(doc_meta)
                        ids_buf.append(note_id)
                        docs_buf.append(doc_text)
                        metas_buf.append(doc_meta)
                        tokens_buf.append(bm25_tokens)
                        count += 1

                        if len(batch_docs) >= CHROMA_BATCH:
                            add_to_chroma(batch_ids, batch_docs, batch_metadatas)
                            batch_docs, batch_ids, batch_metadatas = [], [], []
                            if count % 5000 == 0:
                                print(f"  [{count:,}] Docs embedded...")

                        if len(tokens_buf) >= TOKEN_FLUSH_EVERY:
                            flush_to_jsonl(f_out, ids_buf, docs_buf, metas_buf, tokens_buf)
                            ids_buf, docs_buf, metas_buf, tokens_buf = [], [], [], []
                            print(f"  [{count:,}] Tokens => disk CHECKED")

                    if count >= limit and len(ground_truth_loaded) >= len(ground_truth_ids):
                        break

                if tokens_buf:
                    flush_to_jsonl(f_out, ids_buf, docs_buf, metas_buf, tokens_buf)

        if batch_docs:
            add_to_chroma(batch_ids, batch_docs, batch_metadatas)

    except FileNotFoundError:
        print(f"Error: Could not find the {TAR_PATH}.")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback; traceback.print_exc()
        return

    missing = ground_truth_ids - ground_truth_loaded
    print(f"\n{'='*60}")
    print(f"Loading complete!")
    print(f"  Embedding model:        {EMBEDDING_MODEL}")
    print(f"  Collection:             {COLLECTION_NAME}")
    print(f"  Documents:              {count}")
    print(f"  Extra ground-truth:     {priority_loaded}")
    print(f"  Total:                  {count + priority_loaded}")
    print(f"  BM25 title boost:       {TITLE_BOOST_FACTOR}x")
    print(f"  Ground-truth:           {len(ground_truth_loaded)}/{len(ground_truth_ids)}")
    if missing:
        print(f"[WARNING] Not found: {missing}")
    print(f"{'='*60}")
    print(f"\nNext step: Build the BM25 index.")
    print(f"Then restart the API to pick up the new collection.")


if __name__ == "__main__":
    TESTING_MODE = False

    if TESTING_MODE:
        print(">>> TESTING MODE: 10.000 + guaranteed ground-truth <<<")
        load_data(limit=10_000, guarantee_ground_truth=True)
    else:
        print(">>> PRODUCTION MODE: 801.998 έγγραφα <<<")
        load_data(limit=801_998, guarantee_ground_truth=False)