"""
Generate text from a trained nano-gpt checkpoint.

Usage:
    python generate.py
    python generate.py --prompt "To be or not to be" --max_new_tokens 300
    python generate.py --temperature 1.2 --top_k 50
    python generate.py --checkpoint results/colab-cuda/checkpoint.pt --device cpu
"""

import argparse

import torch

from model import GPT
from run_utils import DEVICE_CHOICES, find_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="nano-gpt text generation")
    parser.add_argument("--checkpoint",     default=None,
                        help="default: newest results/*/checkpoint.pt, else checkpoint.pt")
    parser.add_argument("--device",         default="auto", choices=DEVICE_CHOICES,
                        help="auto = CUDA if available, else MPS, else CPU")
    parser.add_argument("--prompt",         default="\n",       help="conditioning text")
    parser.add_argument("--max_new_tokens", default=500,  type=int)
    parser.add_argument("--temperature",    default=0.8,  type=float,
                        help=">1 → more random, <1 → more greedy")
    parser.add_argument("--top_k",          default=40,   type=int,
                        help="sample from top-k logits (0 = disabled)")
    parser.add_argument("--num_samples",    default=1,    type=int)
    parser.add_argument("--seed",           default=None, type=int)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    try:
        device     = resolve_device(args.device)
        checkpoint = args.checkpoint or find_checkpoint()
    except (RuntimeError, FileNotFoundError) as err:
        parser.error(str(err))

    # map_location makes a checkpoint written on CUDA, MPS or CPU load on any of them
    ckpt   = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    stoi   = ckpt["stoi"]
    itos   = ckpt["itos"]

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join(itos[i] for i in l)

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(
        f"Loaded {checkpoint} on {device}  │  "
        f"{config.n_layer}L · {config.n_head}H · {config.n_embd}D  │  "
        f"train loss {ckpt.get('train_loss', '?'):.4f}  │  "
        f"val loss {ckpt.get('val_loss', '?'):.4f}\n"
    )

    top_k = args.top_k if args.top_k > 0 else None
    ctx   = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"── Sample {i + 1} ──────────────────────────")
        out = model.generate(
            ctx.clone(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )
        print(decode(out[0].tolist()))
        print()


if __name__ == "__main__":
    main()
