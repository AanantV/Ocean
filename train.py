"""
Training loop for Pacific.

Handles:
- Model selection (nano for local GPU verification, mini for the real run)
- AdamW optimizer with warmup + cosine LR decay
- Mixed precision (fp16 + GradScaler) — fp16 chosen over bf16 because bf16
  tensor core support requires Ampere or newer; fp16 works on a much wider
  range of GPUs, including most laptop cards.
- Gradient accumulation (simulate a larger effective batch size than fits
  in VRAM at once)
- Gradient clipping
- Periodic validation loss
- Checkpointing + resume (so a crash, or closing your laptop, doesn't lose
  the run)

Requirements:
    pip install torch numpy

Usage (Pacific-nano, local GPU, sized for a 4-6GB card):
    python train.py --model_size nano --data_dir ./packed --out_dir ./checkpoints

Usage (Pacific-mini, on a rented multi-GPU box later):
    python train.py --model_size mini --data_dir ./packed --out_dir ./checkpoints \
        --batch_size 32 --grad_accum_steps 4 --seq_len 2048
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pacific_model import PacificConfig, PacificModel
from packed_dataset import PackedTokenTorchDataset


MODEL_PRESETS = {
    # ~50M params. Static optimizer memory ~0.75GB — comfortable on a
    # 4-6GB laptop GPU with room left for activations. Use this to verify
    # the training loop actually works: loss should drop from the
    # ln(vocab_size)~=10.37 random-init baseline within the first ~100-200
    # steps if everything is wired correctly.
    "nano": dict(
        d_model=512, n_layers=12, n_heads=8, n_kv_heads=2, ffn_hidden=1408,
    ),
    # ~303M params. Static optimizer memory ~4.5GB — needs a real GPU
    # (16GB+ recommended) or, per the original compute-budget plan,
    # multi-GPU rented hardware. Do not attempt on a 4-6GB card.
    "mini": dict(
        d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, ffn_hidden=2816,
    ),
}


def get_lr(step, warmup_steps, max_steps, lr, min_lr):
    """Linear warmup, then cosine decay to a floor. Verified against
    boundary cases separately before being used here."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters):
    model.eval()
    losses = []
    for i, (input_ids, targets) in enumerate(val_loader):
        if i >= eval_iters:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                             dtype=torch.float16, enabled=(device.type == "cuda")):
            _, loss = model(input_ids, targets)
        losses.append(loss.item())
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, scaler, step, cfg):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": cfg,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./packed")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--model_size", type=str, default="nano", choices=["nano", "mini"])
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Micro-batch size (per forward/backward pass). "
                              "Kept small by default for a 4-6GB GPU.")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                         help="Effective batch size = batch_size * grad_accum_steps.")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint .pt file to resume from.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                         help="Enable for the 'mini' preset on tight VRAM. "
                              "Not needed for 'nano'.")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} | Total VRAM: {props.total_memory / (1024**3):.1f} GB")
    else:
        print("WARNING: no CUDA GPU detected — training will be very slow on CPU. "
              "This is fine for a tiny smoke test but not for a real run.")

    # --- Model ---
    preset = MODEL_PRESETS[args.model_size]
    if args.model_size == "mini" and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024**3)
        if vram_gb < 12:
            print(
                f"WARNING: 'mini' preset (303M params) needs ~4.5GB just for "
                f"optimizer state, before activations. Your GPU has "
                f"{vram_gb:.1f}GB. This will likely OOM. Consider "
                f"--model_size nano for local verification instead."
            )

    cfg = PacificConfig(
        max_seq_len=args.seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
        **preset,
    )
    model = PacificModel(cfg).to(device)
    n_params = model.num_params()
    print(f"Model: Pacific-{args.model_size} | {n_params:,} params ({n_params/1e6:.1f}M)")

    # --- Data ---
    # tokenize_and_pack.py writes shards into <data_dir>/train/ and
    # <data_dir>/val/ subfolders (not directly in data_dir), so point the
    # dataset loader at those subfolders specifically.
    train_ds = PackedTokenTorchDataset(
        os.path.join(args.data_dir, "train"), "train", seq_len=args.seq_len
    )
    val_ds = PackedTokenTorchDataset(
        os.path.join(args.data_dir, "val"), "val", seq_len=args.seq_len
    )
    print(f"Train sequences: {len(train_ds):,} | Val sequences: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # --- Optimizer ---
    # Standard practice: no weight decay on biases/norms (there are none
    # here since all Linear layers use bias=False and RMSNorm weight is
    # 1-D), but we still separate by ndim as a general-purpose safeguard
    # in case the architecture changes later.
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95),
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scaler_state_dict") is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"] + 1
        print(f"Resumed at step {start_step}")

    # --- Training loop ---
    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_count = 0
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(args.grad_accum_steps):
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.float16, enabled=use_amp,
            ):
                _, loss = model(input_ids, targets)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss
        running_count += 1

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / running_count
            tokens_per_step = args.batch_size * args.grad_accum_steps * args.seq_len
            tokens_per_sec = (tokens_per_step * running_count) / elapsed
            print(
                f"step {step+1:6d}/{args.max_steps} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | {tokens_per_sec:,.0f} tok/s | {elapsed:.0f}s elapsed"
            )
            running_loss = 0.0
            running_count = 0
            t0 = time.time()

        if (step + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, args.eval_iters)
            print(f"  [eval] step {step+1}: val_loss {val_loss:.4f}")

        if (step + 1) % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"pacific_{args.model_size}_step{step+1}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scaler, step, cfg)
            print(f"  saved checkpoint: {ckpt_path}")

    # Final checkpoint at the end of the run
    final_path = os.path.join(args.out_dir, f"pacific_{args.model_size}_final.pt")
    save_checkpoint(final_path, model, optimizer, scaler, args.max_steps - 1, cfg)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
