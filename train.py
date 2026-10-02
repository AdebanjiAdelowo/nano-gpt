"""
Train nano-gpt on Tiny Shakespeare (character level).
Downloads the corpus automatically on first run.
Saves checkpoint.pt, loss_curve.png and run_meta.json into a per-run
directory (results/<timestamp>-<device>/ unless --out-dir is given).

Quick config  (~2 min on CPU, ~30 s on MPS):
  block_size=128, n_layer=4, n_head=4, n_embd=128, max_iters=3000

Full config   (better text quality, ~15 min on MPS):
  block_size=256, n_layer=6, n_head=6, n_embd=384, max_iters=5000

Usage:
    python train.py                         # auto device: CUDA, else MPS, else CPU
    python train.py --device cuda --out-dir results/colab-cuda
    python train.py --device cpu --max-iters 50 --eval-interval 25 --eval-iters 10
"""

import argparse
import json
import math
import os
import time
import urllib.request
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")   # figures are only saved, never shown (headless remote runs)
import matplotlib.pyplot as plt
import torch

from model import GPT, GPTConfig
from run_utils import (
    DEVICE_CHOICES, default_out_dir, environment_metadata, prepare_out_dir, resolve_device,
)

# ── Command line ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train nano-gpt on Tiny Shakespeare")
parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES,
                    help="auto = CUDA if available, else MPS, else CPU")
parser.add_argument("--out-dir", default=None,
                    help="run directory (default: results/<timestamp>-<device>)")
parser.add_argument("--seed",          default=42,   type=int)
parser.add_argument("--max-iters",     default=3000, type=int,
                    help="lower for a smoke test")
parser.add_argument("--eval-interval", default=300,  type=int)
parser.add_argument("--eval-iters",    default=100,  type=int)
args = parser.parse_args()

try:
    device  = resolve_device(args.device)
    OUT_DIR = prepare_out_dir(args.out_dir or default_out_dir(device))
except (RuntimeError, FileExistsError) as err:
    parser.error(str(err))

# ── Data ──────────────────────────────────────────────────────────────────────

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/input.txt"

os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare …")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("Done.")

with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()

# character-level vocabulary
chars  = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi   = {c: i for i, c in enumerate(chars)}
itos   = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

print(f"Corpus: {len(text):,} chars | vocab size: {VOCAB_SIZE}")

# ── Model config ───────────────────────────────────────────────────────────────

config = GPTConfig(
    block_size = 128,
    vocab_size  = VOCAB_SIZE,
    n_layer     = 4,
    n_head      = 4,
    n_embd      = 128,
    dropout     = 0.1,
)

# Uncomment for the larger / higher-quality model:
# config = GPTConfig(
#     block_size = 256,
#     vocab_size  = VOCAB_SIZE,
#     n_layer     = 6,
#     n_head      = 6,
#     n_embd      = 384,
#     dropout     = 0.2,
# )

# ── Training hyperparameters ───────────────────────────────────────────────────

BATCH_SIZE    = 32
MAX_ITERS     = args.max_iters
EVAL_INTERVAL = args.eval_interval
EVAL_ITERS    = args.eval_iters
LR            = 1e-3
LR_MIN        = 1e-4
WARMUP_ITERS  = 100
GRAD_CLIP     = 1.0
WEIGHT_DECAY  = 0.1

# ── Dataset splits ────────────────────────────────────────────────────────────

data   = torch.tensor(encode(text), dtype=torch.long)
n      = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - config.block_size, (BATCH_SIZE,))
    x   = torch.stack([src[i : i + config.block_size]     for i in ix])
    y   = torch.stack([src[i + 1 : i + config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y   = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def cosine_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return LR * step / max(1, WARMUP_ITERS)
    t = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + math.cos(math.pi * t))


# ── Model & optimiser ─────────────────────────────────────────────────────────

torch.manual_seed(args.seed)
model     = GPT(config).to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print(
    f"Parameters (non-embedding): {model.num_parameters() / 1e6:.2f} M  "
    f"| device: {device} (requested: {args.device}) | out: {OUT_DIR}"
)

# ── Training loop ─────────────────────────────────────────────────────────────

train_losses: list[float] = []
val_losses:   list[float] = []
loss_steps:   list[int]   = []
t0 = time.time()

for step in range(MAX_ITERS + 1):

    # adjust LR
    lr = cosine_lr(step)
    for pg in optimiser.param_groups:
        pg["lr"] = lr

    # periodic evaluation
    if step % EVAL_INTERVAL == 0:
        ev = estimate_loss(model)
        train_losses.append(ev["train"])
        val_losses.append(ev["val"])
        loss_steps.append(step)
        elapsed = time.time() - t0
        print(
            f"step {step:5d}/{MAX_ITERS} │ "
            f"train {ev['train']:.4f} │ val {ev['val']:.4f} │ "
            f"lr {lr:.2e} │ {elapsed:6.1f}s"
        )

    if step == MAX_ITERS:
        break

    # forward + backward
    x, y   = get_batch("train")
    _, loss = model(x, y)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimiser.step()

total_time = time.time() - t0
print(f"\nTraining complete in {total_time:.1f}s")

# ── Save checkpoint ───────────────────────────────────────────────────────────

torch.save(
    {
        "model_state": model.state_dict(),
        "config":      config,
        "stoi":        stoi,
        "itos":        itos,
        "train_loss":  train_losses[-1],
        "val_loss":    val_losses[-1],
    },
    os.path.join(OUT_DIR, "checkpoint.pt"),
)
print(f"Saved {os.path.join(OUT_DIR, 'checkpoint.pt')}")

# ── Run metadata (kept beside the checkpoint, not inside it) ──────────────────

run_meta = {
    **environment_metadata(args.device, device),
    "seed": args.seed,
    "training_config": {
        "model":         asdict(config),
        "batch_size":    BATCH_SIZE,
        "max_iters":     MAX_ITERS,
        "eval_interval": EVAL_INTERVAL,
        "eval_iters":    EVAL_ITERS,
        "lr":            LR,
        "lr_min":        LR_MIN,
        "warmup_iters":  WARMUP_ITERS,
        "grad_clip":     GRAD_CLIP,
        "optimiser":     "AdamW",
        "weight_decay":  WEIGHT_DECAY,
    },
    "results": {
        "parameters_non_embedding": model.num_parameters(),
        "total_seconds":            total_time,
        "final_train_loss":         train_losses[-1],
        "final_val_loss":           val_losses[-1],
        "loss_steps":               loss_steps,
        "train_losses":             train_losses,
        "val_losses":               val_losses,
    },
}
with open(os.path.join(OUT_DIR, "run_meta.json"), "w") as f:
    json.dump(run_meta, f, indent=2)
print(f"Saved {os.path.join(OUT_DIR, 'run_meta.json')}")

# ── Quick generation sample ───────────────────────────────────────────────────

model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
sample = decode(
    model.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=40)[0].tolist()
)
print("\n── Sample output ──────────────────────────────────────────────────────")
print(sample)
print("───────────────────────────────────────────────────────────────────────\n")

# ── Loss curve ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(loss_steps, train_losses, label="train", linewidth=2)
ax.plot(loss_steps, val_losses,   label="val",   linewidth=2, linestyle="--")
ax.set_xlabel("Step")
ax.set_ylabel("Cross-entropy loss")
ax.set_title(
    f"nano-gpt training  "
    f"({config.n_layer}L · {config.n_head}H · {config.n_embd}D · "
    f"{model.num_parameters()/1e6:.1f}M params)"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=150)
print(f"Saved {os.path.join(OUT_DIR, 'loss_curve.png')}")
print(f"\nGenerate with:  python generate.py --checkpoint {os.path.join(OUT_DIR, 'checkpoint.pt')}")
