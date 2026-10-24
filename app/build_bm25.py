"""
Επαναχτίζει το BM25 index με:
1. Βελτιωμένες παραμέτρους (k1=1.2, b=0.5) — μειώνουν το length penalty
2. Title repetition 3x στα tokens — δίνει βάρος στις λέξεις του title

ΧΩΡΙΣ να αλλάζει τα ChromaDB documents ή το Jazz file.

Input:  /data/bm25_tokens.jsonl (υπάρχει ήδη από το ingest.py)
        /data/TechQA.tar.gz (για να εξάγουμε εκ νέου τον title αν χρειαστεί)

Output: /data/bm25s_index/
        /data/bm25_index.pkl (νέο!)

ΠΡΟΣΟΧΗ: Το υπάρχον bm25_tokens.jsonl δεν έχει title boost — θα δούμε
αν τα tokens έχουν ή όχι το title repeated. Αν όχι, φτιάχνουμε νέο.
"""
import json
import pickle
import re
import os
import gc
import bm25s

TOKENS_PATH   = "/data/bm25_tokens.jsonl"
BM25S_DIR     = "/data/bm25s_index"
METADATA_PATH = "/data/bm25_index.pkl"

# -- Tuning ----------------------------------------
BM25_K1            = 1.2   # default 1.5 — μικρότερο = λιγότερο TF saturation
BM25_B             = 0.5   # default 0.75 — μικρότερο = λιγότερο length penalty
TITLE_BOOST_FACTOR = 3     # Πόσες φορές επαναλαμβάνεται ο title στα tokens

REPORT_EVERY = 100_000


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def main():
    if not os.path.exists(TOKENS_PATH):
        print(f"ERROR: {TOKENS_PATH} not found. Run ingest.py first.")
        return

    # -- Step 1: Φόρτωση των token lists + extract titles from docs ----------------------------------------
    print(f"Step 1: Loading from {TOKENS_PATH}...")
    print(f"        Applying title boost {TITLE_BOOST_FACTOR}x...")

    all_tokens: list[list[str]] = []
    all_ids:    list[str]       = []
    all_docs:   list[str]       = []
    all_metas:  list            = []

    with open(TOKENS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            all_ids.append(record["id"])
            all_docs.append(record["doc"])
            all_metas.append(record["meta"])

            # Πιάνουμε τον title από το meta ή από το doc prefix
            title = record["meta"].get("title") if record.get("meta") else None
            if not title:
                # fallback: αν το doc ξεκινάει με "Title: ..."
                m = re.match(r"Title:\s*(.+?)\n", record["doc"])
                title = m.group(1) if m else ""

            # Αν τα υπάρχοντα tokens δεν έχουν title boost, τον εφαρμόζουμε τώρα
            title_tokens = tokenize(title)
            body_tokens  = record["tokens"]

            # Ανίχνευση: αν το πρώτο token είναι το ίδιο με title[0], έχει ήδη boost
            # Απλούστερο: προσθέτουμε title * (TITLE_BOOST_FACTOR - 1) γιατί
            # ο title είναι ήδη μια φορά μέσα στα tokens.
            boosted = title_tokens * (TITLE_BOOST_FACTOR - 1) + body_tokens
            all_tokens.append(boosted)

            if (i + 1) % REPORT_EVERY == 0:
                print(f"  ...{i + 1:,} records loaded")

    total = len(all_ids)
    print(f"Loaded {total:,} documents.\n")

    # -- Step 2: Build BM25 με tuned params ----------------------------------------
    print(f"Step 2: Building BM25 index (k1={BM25_K1}, b={BM25_B})...")
    retriever = bm25s.BM25(k1=BM25_K1, b=BM25_B)
    retriever.index(all_tokens)
    del all_tokens
    gc.collect()
    print("BM25 index built!\n")

    # -- Step 3: Save index ----------------------------------------
    print(f"Step 3: Saving index -> {BM25S_DIR}...")
    os.makedirs(BM25S_DIR, exist_ok=True)
    retriever.save(BM25S_DIR, corpus=None)
    del retriever
    gc.collect()
    print("Index saved.\n")

    # -- Step 4: Save metadata ----------------------------------------
    print(f"Step 4: Saving metadata -> {METADATA_PATH}...")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({
            "ids":   all_ids,
            "docs":  all_docs,
            "metas": all_metas,
        }, f)
    size_gb = os.path.getsize(METADATA_PATH) / (1024 ** 3)
    print(f"Metadata saved ({size_gb:.2f} GB)\n")

    print("=" * 60)
    print("DONE. Restart your API to pick up the new index.")
    print(f"  BM25 params:  k1={BM25_K1}, b={BM25_B}")
    print(f"  Title boost:  {TITLE_BOOST_FACTOR}x")
    print(f"  Documents:    {total:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()