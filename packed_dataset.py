"""
Memory-mapped dataset for packed token shards produced by tokenize_and_pack.py.

Uses np.memmap so shards are never fully loaded into RAM — matters once you
scale beyond laptop-sized corpora, and there's no downside at small scale
either.

This file includes a synthetic self-test (test_packed_dataset, run via
`python packed_dataset.py`) that verifies the windowing/indexing logic using
made-up token shards, so you can confirm correctness without needing torch
installed in every environment you check this from.

Requirements:
    pip install numpy torch
"""

import glob
import os

import numpy as np


class PackedTokenDataset:
    """Serves fixed-length (input, target) pairs from packed .npy token shards.

    Each shard is a flat uint16 array of token ids (already packed: documents
    concatenated with <|endoftext|> separators, per tokenize_and_pack.py). All
    shards are treated as one logical stream, indexed by non-overlapping
    seq_len windows — this class does NOT require torch to be importable at
    module load time, so the self-test below can run without it. torch is
    only imported inside __getitem__, and only if you actually use this as a
    torch Dataset (see PackedTokenTorchDataset).
    """

    def __init__(self, shard_dir: str, split_prefix: str, seq_len: int):
        self.seq_len = seq_len
        shard_paths = sorted(glob.glob(os.path.join(shard_dir, f"{split_prefix}_*.npy")))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shards matching '{split_prefix}_*.npy' in {shard_dir}. "
                f"Run tokenize_and_pack.py first."
            )

        self.shards = [np.load(p, mmap_mode="r") for p in shard_paths]
        self.shard_lengths = [len(s) for s in self.shards]

        self.windows_per_shard = [
            max(0, (length - 1) // seq_len) for length in self.shard_lengths
        ]
        self.total_windows = sum(self.windows_per_shard)
        if self.total_windows == 0:
            raise ValueError(
                "No shard has enough tokens for even one window of the "
                f"requested seq_len={seq_len}. Use a smaller seq_len or more data."
            )

        self._cumulative = np.cumsum(self.windows_per_shard)

    def __len__(self):
        return self.total_windows

    def _locate(self, idx: int):
        shard_idx = int(np.searchsorted(self._cumulative, idx, side="right"))
        prev_cumulative = self._cumulative[shard_idx - 1] if shard_idx > 0 else 0
        window_idx = idx - prev_cumulative
        return shard_idx, window_idx

    def get_window_numpy(self, idx: int):
        """Returns (input_ids, target_ids) as plain numpy int64 arrays.
        Kept torch-free so the self-test can exercise this directly."""
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(idx)

        shard_idx, window_idx = self._locate(idx)
        shard = self.shards[shard_idx]

        start = window_idx * self.seq_len
        end = start + self.seq_len + 1
        window = np.array(shard[start:end], dtype=np.int64)  # copy out of the memmap

        return window[:-1], window[1:]


class PackedTokenTorchDataset(PackedTokenDataset):
    """torch.utils.data.Dataset wrapper. Split out from the base class so the
    core logic + self-test have no hard torch dependency.

    NOTE: torch is imported inside __getitem__ (not stored as a self.
    attribute) deliberately. Storing a reference to the torch module on the
    instance (e.g. self._torch = torch) breaks Windows multiprocessing
    DataLoader workers: Windows uses the 'spawn' start method, which pickles
    the entire Dataset object to hand to each worker process, and Python
    modules cannot be pickled ("cannot pickle 'module' object"). Importing
    inside the method avoids storing any unpicklable reference on self.
    """

    def __getitem__(self, idx: int):
        import torch
        input_ids, targets = self.get_window_numpy(idx)
        return torch.from_numpy(input_ids), torch.from_numpy(targets)


def test_packed_dataset():
    """Self-test using synthetic shards, no real tokenizer/corpus/torch needed.
    Run directly: python packed_dataset.py
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        seq_len = 4

        shard0 = np.arange(0, 10, dtype=np.uint16)
        np.save(os.path.join(tmp, "train_00000.npy"), shard0)

        shard1 = np.arange(100, 109, dtype=np.uint16)
        np.save(os.path.join(tmp, "train_00001.npy"), shard1)

        ds = PackedTokenDataset(tmp, "train", seq_len=seq_len)
        assert len(ds) == 4, f"expected 4 windows total, got {len(ds)}"

        expected = [
            ([0, 1, 2, 3], [1, 2, 3, 4]),
            ([4, 5, 6, 7], [5, 6, 7, 8]),
            ([100, 101, 102, 103], [101, 102, 103, 104]),
            ([104, 105, 106, 107], [105, 106, 107, 108]),
        ]

        for i, (exp_in, exp_tgt) in enumerate(expected):
            inp, tgt = ds.get_window_numpy(i)
            assert inp.tolist() == exp_in, (i, inp.tolist(), exp_in)
            assert tgt.tolist() == exp_tgt, (i, tgt.tolist(), exp_tgt)
            print(f"window {i}: input={inp.tolist()} target={tgt.tolist()} [OK]")

        try:
            ds.get_window_numpy(4)
            raise AssertionError("expected IndexError for out-of-range index")
        except IndexError:
            print("out-of-range index correctly raises IndexError [OK]")

        print("\nAll packed_dataset self-tests passed.")


if __name__ == "__main__":
    test_packed_dataset()
