import argparse
import os
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table

from .pruner import Pruner

console = Console()


class LineByLineTextDataset(Dataset):
    """
    Minimal dataset that reads one example per line from a text file
    and tokenizes each line into tensors.
    """
    def __init__(self, tokenizer, file_path: str, max_length: int = 256):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        self.examples: List[Dict[str, torch.Tensor]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = tokenizer(
                    line,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.examples.append({k: v.squeeze(0) for k, v in toks.items()})
        if not self.examples:
            raise ValueError(f"No usable lines found in {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_dataloader(tokenizer, data_path: str, batch_size: int, max_length: int, num_workers: int = 0):
    """
    Construct a DataLoader for a line-by-line text dataset.
    """
    ds = LineByLineTextDataset(tokenizer, data_path, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def str_to_dtype(s: Optional[str]) -> Optional[torch.dtype]:
    """
    Convert string flag to a torch dtype, or None for auto.
    """
    if s is None:
        return None
    s = s.lower()
    if s in {"auto", "none"}:       return None
    if s in {"fp32", "float32"}:    return torch.float32
    if s in {"fp16", "float16"}:    return torch.float16
    if s in {"bf16", "bfloat16"}:   return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {s}")


def infer_device(explicit: Optional[str] = None) -> torch.device:
    """
    Resolve target device, falling back to CUDA if available.
    """
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_id_or_path: str, device: torch.device, dtype: Optional[torch.dtype]):
    """
    Load a Hugging Face model and tokenizer, either from name or local path.
    """
    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    try:
        cfg = AutoConfig.from_pretrained(model_id_or_path)
    except Exception:
        cfg = None

    load_kwargs: Dict[str, Any] = {}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    try:
        if cfg is None or getattr(cfg, "is_decoder", True) or getattr(cfg, "is_encoder_decoder", False):
            model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **load_kwargs)
        else:
            raise RuntimeError("Not a decoder model; try encoder.")
    except Exception:
        model = AutoModel.from_pretrained(model_id_or_path, **load_kwargs)

    model.to(device)
    model.eval()
    return model, tok


def prune_command(args: argparse.Namespace):
    """
    Main command to run pruning (FFN and/or Attention) from CLI.
    """
    device = infer_device(args.device)
    dtype = str_to_dtype(args.dtype)

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        t = progress.add_task("Loading model & tokenizer", total=1)
        model, tok = load_model_and_tokenizer(args.model, device, dtype)
        progress.advance(t)

        t = progress.add_task("Building dataloader", total=1)
        dl = build_dataloader(
            tok,
            data_path=args.data,
            batch_size=args.batch_size,
            max_length=args.max_seq_len,
            num_workers=args.num_workers,
        )
        progress.advance(t)

        steps = []
        if args.ffn:
            steps.append(("ffn", {"sparsity": args.sparsity_ffn if args.sparsity_ffn is not None else args.sparsity,
                                  "max_batches": args.max_batches}))
        if args.attention:
            steps.append(("attention", {"sparsity": args.sparsity_attn if args.sparsity_attn is not None else args.sparsity,
                                        "max_batches": args.max_batches}))
        if not steps:
            raise SystemExit("You must pass at least one of --ffn or --attention")

        pruner = Pruner(model)

        t = progress.add_task("Pruning", total=len(steps))
        for name, kw in steps:
            if name == "ffn":
                pruner.prune_all_mlp_layers(dl, sparsity=kw["sparsity"], max_batches=kw["max_batches"])
            elif name == "attention":
                pruner.prune_attention_heads(dl, sparsity=kw["sparsity"], max_batches=kw["max_batches"])
            progress.advance(t)

    original_params = pruner.initial_params_num
    current_params = sum(p.numel() for p in pruner.model.parameters())
    saved = original_params - current_params
    pct = 100.0 * saved / original_params if original_params else 0.0

    import psutil, os as _os
    _proc = psutil.Process(_os.getpid())
    final_cpu_mb = _proc.memory_info().rss / 1024**2
    cpu_diff_mb = final_cpu_mb - getattr(pruner, "_init_cpu_mem_mb", final_cpu_mb)

    has_cuda = torch.cuda.is_available() and getattr(pruner, "_device", torch.device("cpu")).type == "cuda"
    if has_cuda:
        dev = pruner._device
        alloc_before = getattr(pruner, "_init_gpu_alloc_mb", 0.0) or 0.0
        rsrv_before  = getattr(pruner, "_init_gpu_reserved_mb", 0.0) or 0.0
        alloc_after  = torch.cuda.memory_allocated(dev) / 1024**2
        rsrv_after   = torch.cuda.memory_reserved(dev) / 1024**2
        alloc_peak   = torch.cuda.max_memory_allocated(dev) / 1024**2
        rsrv_peak    = torch.cuda.max_memory_reserved(dev) / 1024**2
        alloc_delta  = alloc_after - alloc_before
        rsrv_delta   = rsrv_after  - rsrv_before
    else:
        alloc_before = rsrv_before = alloc_after = rsrv_after = alloc_peak = rsrv_peak = alloc_delta = rsrv_delta = None

    table = Table(title="Slimformers Pruning Summary", show_lines=True)
    table.add_column("Metric", style="bold", justify="left")
    table.add_column("Value", justify="right")

    table.add_row("Original Parameters", f"{original_params:,}")
    table.add_row("Pruned Parameters", f"{current_params:,}")
    table.add_row("Total Reduction", f"{saved:,} ({pct:.2f}%)")
    table.add_row("CPU Δ (MB)", f"{cpu_diff_mb:+.2f}")

    if has_cuda:
        table.add_row("GPU Allocated (Before → After)", f"{alloc_before:.2f} → {alloc_after:.2f}  (Δ {alloc_delta:+.2f})")
        table.add_row("GPU Reserved  (Before → After)", f"{rsrv_before:.2f} → {rsrv_after:.2f}  (Δ {rsrv_delta:+.2f})")
        table.add_row("GPU Peak Allocated (MB)", f"{alloc_peak:.2f}")
        table.add_row("GPU Peak Reserved  (MB)", f"{rsrv_peak:.2f}")
    else:
        table.add_row("GPU", "[dim]Not available[/dim]")

    console.print(table)

    if args.summary:
        pruner.report(verbose=args.verbose)

    if args.save_to:
        console.print(Panel.fit(f"Saving pruned model to: [bold]{args.save_to}[/bold]", border_style="green"))
        pruner.model.save_pretrained(args.save_to)
        try:
            tok.save_pretrained(args.save_to)
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI parser for slimformers.
    """
    p = argparse.ArgumentParser(prog="slimformers", description="Slimformers CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("prune", help="Prune a model (FFN and/or Attention)")
    pp.add_argument("--model", required=True, help="HF model name or local path")
    pp.add_argument("--data", required=True, help="Path to a text file (one example per line)")
    pp.add_argument("--batch-size", type=int, default=8)
    pp.add_argument("--max-seq-len", type=int, default=256)
    pp.add_argument("--num-workers", type=int, default=0)
    pp.add_argument("--device", default=None)
    pp.add_argument("--dtype", default="auto")

    pp.add_argument("--ffn", action="store_true")
    pp.add_argument("--attention", action="store_true")

    pp.add_argument("--sparsity", type=float, default=0.3)
    pp.add_argument("--sparsity-ffn", type=float, default=None)
    pp.add_argument("--sparsity-attn", type=float, default=None)
    pp.add_argument("--max-batches", type=int, default=10)

    pp.add_argument("--save-to", default=None)
    pp.add_argument("--summary", action="store_true")
    pp.add_argument("--verbose", action="store_true")

    pp.set_defaults(func=prune_command)
    return p


def main():
    """
    Entrypoint for the slimformers CLI.
    """
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
