"""
Tokenize the corpus produced by download_data.py and pack it into fixed-length
sequences for pretraining, using the tokenizer trained by train_tokenizer.py.

Packing means: concatenate all documents' tokens into one long stream
(inserting <|endoftext|> between documents), then chop that stream into
non-overlapping blocks. This is how GPT/Llama-style pretraining works — no
per-document padding, maximum GPU utilization. Padding is something you'll
use later at SFT time, not here.

Output: uint16 .npy shards (vocab_size=32000 fits comfortably in uint16,
half the size of the uint32/int64 you'd need for larger vocabs). A held-out
validation split is written separately so you can track val loss later.

NOTE ON SCALE: this script loads all documents into memory before splitting
train/val. That's fine for a laptop-scale corpus (a few hundred MB of text,
which is what download_data.py's default --num_docs produces). If you later
scale up to the full multi-billion-token pretraining corpus on a rented GPU
box, this step needs to become a streaming pass instead — don't reuse this
version unmodified at that scale.

Requirements:
    pip install tokenizers numpy

Usage:
    python tokenize_and_pack.py \
        --corpus_dir ./corpus \
        --tokenizer_path ./tokenizer/pacific_tokenizer.json \
        --out_dir ./packed \
        --seq_len 2048
"""

import argparse
import glob
import os

import numpy as np
from tokenizers import Tokenizer


def tokenize_and_shard(doc_list, tokenizer, eos_id, out_dir, split_name, tokens_per_shard):
    buffer = []
    shard_idx = 0
    total_tokens = 0

    def flush(final=False):
        nonlocal buffer, shard_idx, total_tokens
        while len(buffer) >= tokens_per_shard or (final and buffer):
            take = tokens_per_shard if len(buffer) >= tokens_per_shard else len(buffer)
            shard_tokens = buffer[:take]
            buffer = buffer[take:]
            arr = np.array(shard_tokens, dtype=np.uint16)
            shard_path = os.path.join(out_dir, f"{split_name}_{shard_idx:05d}.npy")
            np.save(shard_path, arr)
            total_tokens += len(arr)
            print(f"  wrote {shard_path} ({len(arr):,} tokens)")
            shard_idx += 1
            if not buffer:
                break

    for doc in doc_list:
        ids = tokenizer.encode(doc).ids
        buffer.extend(ids)
        buffer.append(eos_id)
        if len(buffer) >= tokens_per_shard:
            flush()

    flush(final=True)
    print(f"{split_name}: {total_tokens:,} total tokens across {shard_idx} shard(s)")
    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="./corpus")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer/pacific_tokenizer.json")
    parser.add_argument("--out_dir", type=str, default="./packed")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument(
        "--tokens_per_shard", type=int, default=5_000_000,
        help="Tokens per output shard file. 5M tokens x 2 bytes (uint16) "
             "= 10MB per shard — fine for a laptop.",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.01,
        help="Fraction of documents held out for validation, taken from the "
             "END of the corpus (not randomly sampled) so train/val don't "
             "leak across document boundaries in the packed stream.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert eos_id is not None, "Tokenizer is missing <|endoftext|> — retrain it first."

    corpus_files = sorted(glob.glob(os.path.join(args.corpus_dir, "*.txt")))
    if not corpus_files:
        raise FileNotFoundError(f"No .txt files found in {args.corpus_dir}")

    docs = []
    for path in corpus_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(line)

    print(f"Loaded {len(docs)} documents from {len(corpus_files)} file(s).")

    n_val_docs = max(1, int(len(docs) * args.val_fraction))
    train_docs = docs[:-n_val_docs]
    val_docs = docs[-n_val_docs:]
    print(f"Train docs: {len(train_docs)} | Val docs: {len(val_docs)}")

    print("\nTokenizing + packing train split...")
    train_tokens = tokenize_and_shard(
        train_docs, tokenizer, eos_id, train_dir, "train", args.tokens_per_shard
    )

    print("\nTokenizing + packing val split...")
    val_tokens = tokenize_and_shard(
        val_docs, tokenizer, eos_id, val_dir, "val", args.tokens_per_shard
    )

    print(f"\nDone. Train tokens: {train_tokens:,} | Val tokens: {val_tokens:,}")
    print(
        f"At seq_len={args.seq_len}, that's ~{train_tokens // args.seq_len:,} train "
        f"sequences and ~{val_tokens // args.seq_len:,} val sequences."
    )
    if train_tokens // args.seq_len < 100:
        print(
            "\nWARNING: very few training sequences produced. This corpus is "
            "only good for verifying the pipeline end-to-end, not for a real "
            "pretraining run — you'll need orders of magnitude more data "
            "(see the ~40B token budget from earlier) before training the "
            "actual model."
        )


if __name__ == "__main__":
    main()
