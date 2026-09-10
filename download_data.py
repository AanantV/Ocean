"""
Stream a sample of FineWeb-Edu and write it to plain text files, to be used
as the training corpus for the tokenizer (and, later, for a first small
pretraining run).

We use streaming=True so we never download the full dataset (it's huge,
1.3T+ tokens) — we just pull however many documents we ask for.

Requirements:
    pip install datasets huggingface_hub

Usage:
    python download_data.py --num_docs 200000 --out_dir ./corpus
"""

import argparse
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_docs", type=int, default=200_000,
        help="Number of documents to pull. ~200k docs is plenty for tokenizer "
             "training (BPE saturates well before you need the full dataset).",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./corpus",
        help="Directory to write text shards into.",
    )
    parser.add_argument(
        "--docs_per_shard", type=int, default=20_000,
        help="How many documents per output .txt shard file.",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu",
        help="HF dataset repo to stream from.",
    )
    parser.add_argument(
        "--dataset_config", type=str, default="sample-10BT",
        help="FineWeb-Edu sample config. sample-10BT is a manageable "
             "~10B-token subset rather than the full multi-trillion-token set.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Streaming {args.dataset_name} ({args.dataset_config}) ...")
    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
        streaming=True,
    )

    shard_idx = 0
    doc_count = 0
    buffer = []

    def flush_shard():
        nonlocal shard_idx
        if not buffer:
            return
        shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:04d}.txt")
        with open(shard_path, "w", encoding="utf-8") as f:
            for doc_text in buffer:
                # Replace any literal newlines inside a doc so each line in
                # the file is exactly one document — makes later processing
                # (and sanity-checking file line counts) trivial.
                cleaned = doc_text.replace("\n", " ").strip()
                if cleaned:
                    f.write(cleaned + "\n")
        print(f"  wrote {shard_path} ({len(buffer)} docs)")
        shard_idx += 1
        buffer.clear()

    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        buffer.append(text)
        doc_count += 1

        if len(buffer) >= args.docs_per_shard:
            flush_shard()

        if doc_count >= args.num_docs:
            break

    flush_shard()  # flush any remainder

    print(f"Done. Wrote {doc_count} documents across {shard_idx} shard(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
