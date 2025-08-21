import argparse
import json
import os
import re
from typing import Dict, Any, Optional, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
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


def build_dataloader(
    tokenizer,
    data_path: str,
    batch_size: int,
    max_length: int,
    num_workers: int = 0,
    shuffle: bool = True,
):
    """
    Construct a DataLoader for a line-by-line text dataset.
    """
    ds = LineByLineTextDataset(tokenizer, data_path, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def parse_sparsity(v: Optional[str]) -> Optional[float]:
    """
    Accepts '0.3', '30%', '30', None -> returns fraction [0,1].
    """
    if v is None:
        return None
    if isinstance(v, float):
        x = v
    else:
        s = str(v).strip()
        m = re.fullmatch(r"^\s*([0-9]*\.?[0-9]+)\s*%?\s*$", s)
        if not m:
            raise argparse.ArgumentTypeError(f"Invalid sparsity value: {v}")
        x = float(m.group(1))
        if x > 1.0:  # treat as percent
            x = x / 100.0
    if not (0.0 <= x < 1.0):
        raise argparse.ArgumentTypeError(f"Sparsity must be in [0,1): got {x}")
    return x


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
    Resolve target device: explicit -> 'cuda' if available -> 'mps' -> 'cpu'.
    """
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_determinism(seed: Optional[int], deterministic: bool):
    if seed is None and not deterministic:
        return
    import random
    try:
        import numpy as np
    except Exception:
        np = None

    if seed is None:
        seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def maybe_compile(model: torch.nn.Module, do_compile: bool) -> torch.nn.Module:
    if not do_compile:
        return model
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            console.log("[green]Compiled model with torch.compile[/green]")
        except Exception as e:
            console.log(f"[yellow]torch.compile failed; running uncompiled: {e}[/yellow]")
    else:
        console.log("[yellow]PyTorch < 2.0 detected; --compile ignored[/yellow]")
    return model

def load_model_and_tokenizer(
    model_id_or_path: str,
    device: torch.device,
    dtype: Optional[torch.dtype],
    trust_remote_code: bool = False,
) -> Tuple[torch.nn.Module, Any]:
    """
    Load a Hugging Face model and tokenizer, either from name or local path.
    Prefers CausalLM, then Seq2SeqLM, then base encoder/decoder model.
    """
    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

    try:
        cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    except Exception:
        cfg = None

    load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    model = None
    errors = []
    for loader in (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel):
        try:
            model = loader.from_pretrained(model_id_or_path, **load_kwargs)
            break
        except Exception as e:
            errors.append((loader.__name__, str(e)))

    if model is None:
        msgs = "\n".join(f"- {name}: {msg}" for name, msg in errors[-3:])
        raise RuntimeError(f"Failed to load model '{model_id_or_path}'. Tried:\n{msgs}")

    model.to(device)
    model.eval()
    return model, tok

def prune_command(args: argparse.Namespace):
    """
    Main command to run pruning (FFN and/or Attention) from CLI.
    """
    device = infer_device(args.device)
    dtype = str_to_dtype(args.dtype)

    setup_determinism(args.seed, args.deterministic)

    progress_ctx = (
        Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        if not args.no_progress else None
    )

    if progress_ctx is None:
        model, tok = load_model_and_tokenizer(args.model, device, dtype, args.trust_remote_code)
    else:
        with progress_ctx as progress:
            t = progress.add_task("Loading model & tokenizer", total=1)
            model, tok = load_model_and_tokenizer(args.model, device, dtype, args.trust_remote_code)
            progress.advance(t)

    model = maybe_compile(model, args.compile)
    pruner = Pruner(model)

    dl = None
    if not args.dry_run:
        if progress_ctx is None:
            dl = build_dataloader(
                tok,
                data_path=args.data,
                batch_size=args.batch_size,
                max_length=args.max_seq_len,
                num_workers=args.num_workers,
                shuffle=not args.no_shuffle is True,
            )
        else:
            with progress_ctx as progress:
                t = progress.add_task("Building dataloader", total=1)
                dl = build_dataloader(
                    tok,
                    data_path=args.data,
                    batch_size=args.batch_size,
                    max_length=args.max_seq_len,
                    num_workers=args.num_workers,
                    shuffle=not args.no_shuffle is True,
                )
                progress.advance(t)

    steps = []
    if args.ffn:
        steps.append(
            ("ffn", {"sparsity": args.sparsity_ffn if args.sparsity_ffn is not None else args.sparsity,
                     "max_batches": args.max_batches})
        )
    if args.attention:
        steps.append(
            ("attention", {"sparsity": args.sparsity_attn if args.sparsity_attn is not None else args.sparsity,
                           "max_batches": args.max_batches})
        )

    if not args.dry_run and not steps:
        raise SystemExit("You must pass at least one of --ffn or --attention (or use --dry-run).")

    import psutil as _ps, os as _os
    _proc = _ps.Process(_os.getpid())
    cpu_before_mb = _proc.memory_info().rss / 1024**2

    if torch.cuda.is_available() and device.type == "cuda":
        alloc_before = torch.cuda.memory_allocated(device) / 1024**2
        rsrv_before = torch.cuda.memory_reserved(device) / 1024**2
    else:
        alloc_before = rsrv_before = 0.0

    # Run prune
    if not args.dry_run and steps:
        if progress_ctx is None:
            _run_steps(pruner, dl, steps)
        else:
            with progress_ctx as progress:
                t = progress.add_task("Pruning", total=len(steps))
                for name, kw in steps:
                    if name == "ffn":
                        pruner.prune_all_mlp_layers(dl, sparsity=kw["sparsity"], max_batches=kw["max_batches"])
                    elif name == "attention":
                        pruner.prune_attention_heads(dl, sparsity=kw["sparsity"], max_batches=kw["max_batches"])
                    progress.advance(t)

    # Summary
    _print_and_optionally_save_summary(
        pruner=pruner,
        cpu_before_mb=cpu_before_mb,
        gpu_before=(alloc_before, rsrv_before) if device.type == "cuda" and torch.cuda.is_available() else None,
        save_json=args.save_summary,
    )

    if args.summary:
        pruner.report(verbose=args.verbose)

    if args.save_to:
        console.print(Panel.fit(f"Saving pruned model to: [bold]{args.save_to}[/bold]", border_style="green"))
        pruner.model.save_pretrained(args.save_to)
        try:
            tok.save_pretrained(args.save_to)
        except Exception:
            pass


def _run_steps(pruner: Pruner, dl, steps: List[Tuple[str, Dict[str, Any]]]):
    for name, kw in steps:
        if name == "ffn":
            pruner.prune_all_mlp_layers(dl, sparsity=kw["sparsity"], max_batches=kw["max_batches"])
        elif name == "attention":
            pruner.prune_attention_heads(dl, sparsity=kw["sparsity"], max_batches=kw["max_batches"])


def _print_and_optionally_save_summary(
    pruner: Pruner,
    cpu_before_mb: float,
    gpu_before: Optional[Tuple[float, float]],
    save_json: Optional[str],
):
    original_params = getattr(pruner, "initial_params_num", None)
    if original_params is None:
        original_params = sum(p.numel() for p in pruner.model.parameters())

    current_params = sum(p.numel() for p in pruner.model.parameters())
    saved = max(original_params - current_params, 0)
    pct = 100.0 * saved / original_params if original_params else 0.0

    import psutil as _ps, os as _os
    _proc = _ps.Process(_os.getpid())
    final_cpu_mb = _proc.memory_info().rss / 1024**2
    cpu_diff_mb = final_cpu_mb - cpu_before_mb

    if torch.cuda.is_available() and hasattr(pruner, "_device") and getattr(pruner, "_device").type == "cuda":
        dev = pruner._device
        alloc_after = torch.cuda.memory_allocated(dev) / 1024**2
        rsrv_after = torch.cuda.memory_reserved(dev) / 1024**2
        alloc_peak = torch.cuda.max_memory_allocated(dev) / 1024**2
        rsrv_peak = torch.cuda.max_memory_reserved(dev) / 1024**2
        if gpu_before is not None:
            alloc_delta = alloc_after - gpu_before[0]
            rsrv_delta = rsrv_after - gpu_before[1]
        else:
            alloc_delta = rsrv_delta = 0.0
    else:
        alloc_after = rsrv_after = alloc_peak = rsrv_peak = alloc_delta = rsrv_delta = None

    table = Table(title="Slimformers Pruning Summary", show_lines=True)
    table.add_column("Metric", style="bold", justify="left")
    table.add_column("Value", justify="right")
    table.add_row("Original Parameters", f"{original_params:,}")
    table.add_row("Pruned Parameters", f"{current_params:,}")
    table.add_row("Total Reduction", f"{saved:,} ({pct:.2f}%)")
    table.add_row("CPU Δ (MB)", f"{cpu_diff_mb:+.2f}")

    if alloc_after is not None:
        table.add_row("GPU Allocated (Before → After)", f"{gpu_before[0]:.2f} → {alloc_after:.2f}  (Δ {alloc_delta:+.2f})")
        table.add_row("GPU Reserved  (Before → After)", f"{gpu_before[1]:.2f} → {rsrv_after:.2f}  (Δ {rsrv_delta:+.2f})")
        table.add_row("GPU Peak Allocated (MB)", f"{alloc_peak:.2f}")
        table.add_row("GPU Peak Reserved  (MB)", f"{rsrv_peak:.2f}")
    else:
        table.add_row("GPU", "[dim]Not available[/dim]")

    console.print(table)

    if save_json:
        payload = {
            "original_params": int(original_params),
            "pruned_params": int(current_params),
            "saved_params": int(saved),
            "saved_pct": pct,
            "cpu_delta_mb": cpu_diff_mb,
        }
        if alloc_after is not None:
            payload.update({
                "gpu_alloc_before_mb": gpu_before[0],
                "gpu_alloc_after_mb": alloc_after,
                "gpu_alloc_delta_mb": alloc_delta,
                "gpu_reserved_before_mb": gpu_before[1],
                "gpu_reserved_after_mb": rsrv_after,
                "gpu_reserved_delta_mb": rsrv_delta,
                "gpu_peak_alloc_mb": alloc_peak,
                "gpu_peak_reserved_mb": rsrv_peak,
            })
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        console.log(f"[green]Saved summary JSON to {save_json}[/green]")

def build_parser() -> argparse.ArgumentParser:
    epilog = r"""
    Examples:
    # Basic FFN prune @30% (also accepts '30%')
    slimformers prune --model gpt2 --data data.txt --ffn --sparsity 0.3

    # FFN 40%, Attention 20%, only 5 batches for stats; save model
    slimformers prune --model gpt2 --data data.txt --ffn --attention \
        --sparsity-ffn 40% --sparsity-attn 0.2 --max-batches 5 --save-to pruned-gpt2

    # Deterministic dry-run (no pruning) just to inspect counts
    slimformers prune --model gpt2 --data data.txt --dry-run --seed 123 --deterministic

    # Faster pass with torch.compile (PyTorch 2.x)
    slimformers prune --model gpt2 --data data.txt --ffn --compile
    """
    p = argparse.ArgumentParser(
        prog="slimformers",
        description="Slimformers CLI",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("prune", help="Prune a model (FFN and/or Attention)")

    core = pp.add_argument_group("Core")
    core.add_argument("--model", required=True, help="HF model name or local path")
    core.add_argument("--data", required=False, help="Path to a text file (one example per line). Required unless --dry-run.")
    core.add_argument("--dry-run", action="store_true", help="Load and summarize without pruning/data passes")

    data = pp.add_argument_group("Data")
    data.add_argument("--batch-size", type=int, default=8)
    data.add_argument("--max-seq-len", type=int, default=256)
    data.add_argument("--num-workers", type=int, default=0)
    data.add_argument("--no-shuffle", action="store_true", help="Disable shuffling for the dataloader")

    comp = pp.add_argument_group("Compute")
    comp.add_argument("--device", default=None, help="cpu | cuda[:id] | mps | auto (default: auto)")
    comp.add_argument("--dtype", default="auto", help="auto | fp32 | fp16 | bf16")
    comp.add_argument("--seed", type=int, default=None, help="Random seed")
    comp.add_argument("--deterministic", action="store_true", help="Use deterministic algorithms where possible")
    comp.add_argument("--compile", action="store_true", help="Attempt torch.compile() (PyTorch 2+) for speed")
    comp.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    comp.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders")

    prn = pp.add_argument_group("Pruning")
    prn.add_argument("--ffn", action="store_true", help="Enable FFN pruning")
    prn.add_argument("--attention", action="store_true", help="Enable attention head pruning")
    prn.add_argument("--sparsity", type=parse_sparsity, default=0.3, help="Global default sparsity (e.g., 0.3 or 30%%)")
    prn.add_argument("--sparsity-ffn", type=parse_sparsity, default=None, help="Override FFN sparsity")
    prn.add_argument("--sparsity-attn", type=parse_sparsity, default=None, help="Override attention sparsity")
    prn.add_argument("--max-batches", type=int, default=10, help="Number of batches to estimate importance")

    out = pp.add_argument_group("Output")
    out.add_argument("--save-to", default=None, help="Directory to save the pruned model/tokenizer")
    out.add_argument("--save-summary", default=None, help="Path to write a JSON summary report")
    out.add_argument("--summary", action="store_true", help="Print layer-level report")
    out.add_argument("--verbose", action="store_true")

    pp.set_defaults(func=prune_command)
    return p


def main():
    """
    Entrypoint for the slimformers CLI.
    """
    parser = build_parser()
    args = parser.parse_args()

    if not args.dry_run and not args.data:
        parser.error("--data is required unless --dry-run is set")

    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(Panel.fit(f"[bold red]Error[/bold red]\n{type(e).__name__}: {e}", border_style="red"))
        raise


if __name__ == "__main__":
    main()
