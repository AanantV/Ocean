"""
Train a byte-level BPE tokenizer (GPT-2/Llama-3 style) on the corpus produced
by download_data.py, sized to match PacificConfig.vocab_size (32000).

Byte-level BPE means: no "unknown token" problem ever. Any input, including
weird unicode, emoji, or code with unusual symbols, decomposes into raw bytes
first, so the tokenizer can always represent it, just possibly inefficiently
for text it wasn't trained on much. This is why GPT-2 onwards, and Llama 3,
use this scheme instead of word-level or classic SentencePiece-with-<unk>.

Requirements:
    pip install tokenizers

Usage:
    python train_tokenizer.py --corpus_dir ./corpus --out_dir ./tokenizer --vocab_size 32000
"""

import argparse
import glob
import os

from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


# Special tokens. Only <|endoftext|> and <|pad|> are used at pretraining time
# (endoftext separates documents, pad is used for batching variable-length
# sequences). The others are reserved now so the vocab doesn't shift later
# when you do SFT / chat formatting — changing vocab size after pretraining
# means retraining embeddings, which you want to avoid.
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="./corpus")
    parser.add_argument("--out_dir", type=str, default="./tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    corpus_files = sorted(glob.glob(os.path.join(args.corpus_dir, "*.txt")))
    if not corpus_files:
        raise FileNotFoundError(
            f"No .txt files found in {args.corpus_dir}. Run download_data.py first."
        )
    print(f"Training on {len(corpus_files)} corpus file(s): {corpus_files}")

    # Byte-level BPE: ByteLevel pre-tokenizer splits text into byte-level
    # tokens before BPE merging, ByteLevel decoder reverses it correctly
    # (handles the leading-space encoding GPT-2-style tokenizers rely on).
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        # Byte-level BPE should always start from the 256 possible byte
        # values as its initial alphabet, or you can end up unable to
        # represent some byte sequences.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files=corpus_files, trainer=trainer)

    # Post-processing: nothing added automatically at train time here;
    # <|endoftext|> is inserted manually between documents in the data
    # pipeline step (next piece to build), not injected by the tokenizer
    # itself. Keeping that explicit avoids surprises later.

    tokenizer_path = os.path.join(args.out_dir, "pacific_tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")

    # --- Round-trip verification ---
    # This is the check that actually matters: encode text, decode it back,
    # confirm you get the same text out. If this fails, nothing downstream
    # (model training, inference) can be trusted.
    test_strings = [
        "The mitochondria is the powerhouse of the cell.",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "Emoji and unicode: café, naïve, 日本語, 🚀",
        "",  # edge case: empty string
    ]

    print("\n--- Round-trip verification ---")
    all_passed = True
    for s in test_strings:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded.ids)
        passed = decoded == s
        all_passed = all_passed and passed
        status = "OK" if passed else "MISMATCH"
        print(f"[{status}] {s!r}")
        if not passed:
            print(f"          -> decoded as: {decoded!r}")
            print(f"          -> token ids: {encoded.ids[:20]}...")

    # Special token check
    print("\n--- Special token IDs ---")
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        print(f"  {tok}: id={tid}")
        assert tid is not None, f"Special token {tok} missing from vocab!"

    if all_passed:
        print("\nAll round-trip checks passed.")
    else:
        print("\nSome round-trip checks FAILED — do not proceed to pretraining "
              "until this is fixed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
