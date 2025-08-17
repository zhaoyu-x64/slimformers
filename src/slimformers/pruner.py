import torch
from torch import nn
from transformers.modeling_utils import Conv1D
from rich.console import Console
from rich.panel import Panel
from .discovery import DISCOVERY_REGISTRY, default_discover, ATTENTION_DISCOVERY_REGISTRY
from rich.progress import Progress, BarColumn, TimeElapsedColumn, MofNCompleteColumn
import psutil
import os
from .lora import lora_finetune
from typing import Optional

console = Console()

class Pruner:
    def __init__(self, model: nn.Module, pruning_strategy=None):
        """
        Set up the Pruner with a model and optional strategy.
        If no strategy is given, default to keeping neurons with the highest activation magnitudes.
        """
        self.model = model
        self.activations = {}
        self.pruning_strategy = pruning_strategy or self._compute_topk_neurons
        self.initial_params_num = sum(p.numel() for p in model.parameters())
        
        self._init_cpu_mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2

        self._device = next(model.parameters()).device if any(p.is_cuda for p in model.parameters()) else torch.device("cpu")
        if torch.cuda.is_available() and self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
            self._init_gpu_alloc_mb    = torch.cuda.memory_allocated(self._device) / 1024**2
            self._init_gpu_reserved_mb = torch.cuda.memory_reserved(self._device)  / 1024**2
        else:
            self._init_gpu_alloc_mb = None
            self._init_gpu_reserved_mb = None

        
        console.rule("[bold cyan]Pruner Initialized")
        console.print(
            Panel.fit(
                f"Model: [bold]{type(model).__name__}[/bold]\n"
                f"Strategy: [bold]{self.pruning_strategy.__name__}[/bold]",
                title="Initialization Summary",
                border_style="cyan"
            )
        )

    @staticmethod
    def _discover_mlp_blocks(model: nn.Module):
        """
        Automatically find MLP/FFN blocks based on model type.
        Uses a registry of discovery functions (GPT2, BERT, LLaMA, etc).
        """
        cls = type(model).__name__
        finder = DISCOVERY_REGISTRY.get(cls, default_discover)
        return finder(model)

    def _compute_topk_neurons(self, activations: torch.Tensor, sparsity: float):
        """
        Select the top neurons (or heads) by activation magnitude.
        Supports:
        - 3D tensor: (batch, seq_len, features) → mean over (0,1)
        - 2D tensor: (batch, features)         → mean over 0
        - 1D tensor: (features,)               → use abs directly
        """
        if activations.dim() == 3:
            mags = activations.abs().mean(dim=(0, 1))
        elif activations.dim() == 2:
            mags = activations.abs().mean(dim=0)
        elif activations.dim() == 1:
            mags = activations.abs()
        else:
            raise ValueError(f"Bad activation shape {activations.shape}")

        total = mags.numel()
        k = int((1.0 - sparsity) * total)
        keep = torch.topk(mags, k=k).indices
        
        return keep, total


    def _hook_activations(self, layer_name: str):
        """
        Register a forward hook on a specific layer to capture its output.
        """
        module = dict(self.model.named_modules())[layer_name]
        return module.register_forward_hook(
            lambda mod, inp, out, key=layer_name: self.activations.setdefault(key, out.detach())
        )

    def _rebuild_linear(self, layer: nn.Linear, keep_out: torch.Tensor = None, keep_in:  torch.Tensor = None):
        """
        Rebuild a Linear layer by slicing its input/output weights.
        """
        
        device = layer.weight.device
        W = layer.weight.data
        B = layer.bias.data if layer.bias is not None else None

        if keep_out is not None:
            keep_out = keep_out.to(device)
        if keep_in is not None:
            keep_in = keep_in.to(device)

        if keep_out is not None and keep_in is None:
            new_W = W[keep_out, :]
        elif keep_out is None and keep_in is not None:
            new_W = W[:, keep_in]
        elif keep_out is not None and keep_in is not None:
            new_W = W[keep_out][:, keep_in]
        else:
            raise ValueError("Must provide keep_out and/or keep_in")

        out_f, in_f = new_W.shape
        new = nn.Linear(in_f, out_f, bias=(B is not None))
        new.weight.data = new_W.to(layer.weight.device)
        if B is not None:
            new.bias.data = (B[keep_out] if keep_out is not None else B).to(layer.bias.device)
        return new

    def _rebuild_conv1d(self, layer: Conv1D, keep_out: torch.Tensor = None, keep_in:  torch.Tensor = None):
        """
        Same as _rebuild_linear but for HuggingFace's Conv1D (used in GPT-2).
        """
        
        device = layer.weight.device
        W = layer.weight.data
        B = layer.bias.data if layer.bias is not None else None
        
        if keep_out is not None:
            keep_out = keep_out.to(device)
        if keep_in is not None:
            keep_in = keep_in.to(device)

        if keep_out is not None and keep_in is None:
            new_W = W[:, keep_out]
        elif keep_out is None and keep_in is not None:
            new_W = W[keep_in, :]
        elif keep_out is not None and keep_in is not None:
            new_W = W[keep_in][:, keep_out]
        else:
            raise ValueError("Must provide keep_out and/or keep_in")

        in_c, out_c = new_W.shape
        new = Conv1D(out_c, in_c)
        new.weight.data = new_W.to(layer.weight.device)
        new.nf = out_c
        if B is not None:
            new.bias.data = (B[keep_out] if keep_out is not None else B).to(layer.bias.device)
        return new

    def _replace_module(self, name: str, new_mod: nn.Module):
        """
        Replaces a module in the model with the updated version.
        For example, replaces an old Linear with a sliced one.
        """
        parent_name, attr = name.rsplit('.', 1)
        parent = dict(self.model.named_modules())[parent_name]
        setattr(parent, attr, new_mod)

    def _rebuild(self, layer, keep_out=None, keep_in=None):
        """
        Calls the correct rebuild function depending on layer type.
        """
        if isinstance(layer, Conv1D):
            return self._rebuild_conv1d(layer, keep_out, keep_in)
        elif isinstance(layer, nn.Linear):
            return self._rebuild_linear(layer, keep_out, keep_in)
        else:
            raise TypeError(f"Can't rebuild module of type {type(layer)}")

    def prune_all_mlp_layers(self, dataloader: torch.utils.data.DataLoader, sparsity: float = 0.3, max_batches: int = 10):
        """
        Prune MLP/FFN layers using activations collected from multiple batches in a DataLoader.
        Automatically averages activations across batches before applying the pruning strategy.

        Args:
            dataloader (torch.utils.data.DataLoader): Yields dict[str, torch.Tensor] batches
            sparsity (float): Fraction of neurons to prune (0.0 keeps all, 1.0 prunes all)
            max_batches (int): Max number of batches to use for computing average activations
        """
        
        self.model.eval()
        device = next(self.model.parameters()).device

        console.print(f"[bold]Starting Pruning at {sparsity:.0%} Sparsity")
        blocks = Pruner._discover_mlp_blocks(self.model)
        console.print(f"[bold]Discovered {len(blocks)} MLP blocks[/bold]\n")

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Pruning MLP blocks", total=len(blocks))

            for i, blk in enumerate(blocks):
                hook_key = blk["gate_name"] if blk["type"] == "gated" else blk["fc_name"]
                self.activations = {}
                handle = self._hook_activations(hook_key)

                total_acts = None
                num_batches = 0

                for batch in dataloader:
                    if num_batches >= max_batches:
                        break
                    with torch.no_grad():
                        _ = self.model(**{k: v.to(device) for k, v in batch.items()})
                    act = self.activations.pop(hook_key, None)
                    if act is None:
                        continue
                    total_acts = act if total_acts is None else total_acts + act
                    num_batches += 1

                handle.remove()

                if num_batches == 0:
                    raise RuntimeError(f"No activations captured for block {i} ({hook_key})")

                avg_acts = total_acts / num_batches
                keep_idx, orig = self.pruning_strategy(avg_acts, sparsity)

                if blk["type"] == "gated":
                    new_gate = self._rebuild(blk["gate"], keep_out=keep_idx)
                    new_up   = self._rebuild(blk["up"],   keep_out=keep_idx)
                    new_down = self._rebuild(blk["down"], keep_in=keep_idx)

                    self._replace_module(blk["gate_name"], new_gate)
                    self._replace_module(blk["up_name"],   new_up)
                    self._replace_module(blk["down_name"], new_down)

                else:
                    new_fc   = self._rebuild(blk["fc"],   keep_out=keep_idx)
                    new_proj = self._rebuild(blk["proj"], keep_in=keep_idx)

                    self._replace_module(blk["fc_name"],   new_fc)
                    self._replace_module(blk["proj_name"], new_proj)

                progress.advance(task)

        console.print("[bold green]Pruning Complete")    

    def report(self, verbose=False):
        """
        Print the parameter savings and memory deltas after pruning.
        Set verbose=True to also dump torch.cuda.memory_summary().
        """
        current_params = sum(p.numel() for p in self.model.parameters())
        saved = self.initial_params_num - current_params
        percent = 100 * saved / self.initial_params_num

        proc = psutil.Process(os.getpid())
        final_cpu_mb = proc.memory_info().rss / 1024**2
        cpu_diff_mb = final_cpu_mb - self._init_cpu_mem_mb

        if torch.cuda.is_available() and self._device.type == "cuda":
            final_alloc_mb    = torch.cuda.memory_allocated(self._device) / 1024**2
            final_reserved_mb = torch.cuda.memory_reserved(self._device)  / 1024**2
            peak_alloc_mb     = torch.cuda.max_memory_allocated(self._device) / 1024**2
            peak_reserved_mb  = torch.cuda.max_memory_reserved(self._device)  / 1024**2

            alloc_diff_mb    = final_alloc_mb    - (self._init_gpu_alloc_mb or 0.0)
            reserved_diff_mb = final_reserved_mb - (self._init_gpu_reserved_mb or 0.0)

            gpu_line = (
                f"[bold]GPU Allocated (Before → After):[/bold] "
                f"{(self._init_gpu_alloc_mb or 0.0):.2f} MB → {final_alloc_mb:.2f} MB "
                f"([bold green]{alloc_diff_mb:+.2f} MB[/bold green])\n"
                f"[bold]GPU Reserved  (Before → After):[/bold] "
                f"{(self._init_gpu_reserved_mb or 0.0):.2f} MB → {final_reserved_mb:.2f} MB "
                f"([bold green]{reserved_diff_mb:+.2f} MB[/bold green])\n"
                f"[bold]GPU Peak Allocated:[/bold] {peak_alloc_mb:.2f} MB\n"
                f"[bold]GPU Peak Reserved :[/bold] {peak_reserved_mb:.2f} MB"
            )
        else:
            gpu_line = "[dim]GPU not available — skipped[/dim]"

        if verbose and torch.cuda.is_available() and self._device.type == "cuda":
            console.rule("[bold blue]CUDA Memory Summary")
            summary = torch.cuda.memory_summary(device=self._device)
            console.print(f"[dim]{summary}[/dim]")

        cpu_line = (
            f"[bold]CPU Memory (Before --> After):[/bold] "
            f"{self._init_cpu_mem_mb:.2f} MB --> {final_cpu_mb:.2f} MB "
            f"([bold green]{cpu_diff_mb:+.2f} MB[/bold green])"
        )

        console.rule("[bold magenta]Pruning Summary")
        console.print(
            Panel.fit(
                f"[bold]Original Parameters:[/bold] {self.initial_params_num:,}\n"
                f"[bold]Pruned Parameters:[/bold] {current_params:,}\n"
                f"[bold green]Total Reduction:[/bold green] {saved:,} ({percent:.2f}%)\n\n"
                f"{gpu_line}\n{cpu_line}",
                title="[bold]Compression Results[/bold]",
                border_style="magenta"
            )
        )

    def prune_attention_heads(self, dataloader, sparsity=0.3, max_batches=10):
        """
        Prune attention heads using average query (or packed QKV) activations.
        So far, it supports GPT‑2 (packed) and BERT/LLaMA (separate).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        console.print(f"[bold]Starting Attention Pruning at {sparsity:.0%} Sparsity")

        blocks = ATTENTION_DISCOVERY_REGISTRY.get(type(self.model).__name__, lambda m: [])(self.model)
        console.print(f"[bold]Discovered {len(blocks)} attention blocks[/bold]\n")

        with Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True
        ) as progress:
            task = progress.add_task("Pruning Attention Blocks", total=len(blocks))

            for blk in blocks:
                prefix = blk["prefix"]
                key = blk.get("qkv_name") or blk["q_name"]
                handle = self._hook_activations(key)
                total_acts, n = None, 0
                for batch in dataloader:
                    if n >= max_batches:
                        break
                    with torch.no_grad():
                        _ = self.model(**{k: v.to(device) for k, v in batch.items()})
                    act = self.activations.pop(key, None)
                    if act is not None:
                        total_acts = act if total_acts is None else total_acts + act
                        n += 1
                handle.remove()
                if n == 0:
                    raise RuntimeError(f"No activations for attention block {prefix}")

                avg_act = (total_acts / n).abs()    
                B, T, D = avg_act.shape
                H = blk["num_heads"]
                head_dim_qkv = D // H

                head_mags = avg_act.view(B, T, H, head_dim_qkv).mean(dim=(0, 1, 3))
                keep, orig = self.pruning_strategy(head_mags, sparsity)

                idx_qkv = torch.cat([
                    torch.arange(h * head_dim_qkv, (h + 1) * head_dim_qkv, device=keep.device)
                    for h in keep
                ])

                out_layer = blk["out"]
                if isinstance(out_layer, Conv1D):
                    in_dim = out_layer.weight.size(0)
                else:
                    in_dim = out_layer.in_features
                head_dim_out = in_dim // H
                idx_out = torch.cat([
                    torch.arange(h * head_dim_out, (h + 1) * head_dim_out, device=keep.device)
                    for h in keep
                ])

                if blk["type"] == "packed":
                    new_qkv = self._rebuild(blk["qkv"], keep_out=idx_qkv)
                    self._replace_module(blk["qkv_name"], new_qkv)
                else:
                    for name in ("q", "k", "v"):
                        new_lin = self._rebuild(blk[name], keep_out=idx_qkv)
                        self._replace_module(blk[f"{name}_name"], new_lin)

                new_out = self._rebuild(blk["out"], keep_in=idx_out)
                self._replace_module(blk["out_name"], new_out)
                self._patch_attention_module(prefix, keep.numel(), head_dim_out)

                progress.advance(task)

        console.print("[bold green]Attention Pruning Complete")

    def _patch_attention_module(self, prefix: str, new_heads: int, head_dim_out: int):
        """
        Dynamically update any head-count or size attributes on the attn module.
        """
        mod = dict(self.model.named_modules())[prefix]

        if hasattr(mod, "num_heads"):
            mod.num_heads = new_heads
        if hasattr(mod, "num_attention_heads"):
            mod.num_attention_heads = new_heads

        if hasattr(mod, "embed_dim") and hasattr(mod, "split_size"):
            new_embed = head_dim_out * new_heads
            mod.embed_dim   = new_embed
            mod.split_size  = new_embed

        if hasattr(mod, "attention_head_size") and hasattr(mod, "all_head_size"):
            mod.all_head_size = mod.attention_head_size * new_heads


    def _collect_activations(self, blk, dataloader, max_batches):
        """
        Runs up to max_batches through the model, hooks either Q or packed QKV,
        and returns the avg activation tensor (B, T, D).
        """
        key = blk.get("qkv_name") or blk["q_name"]
        handle = self._hook_activations(key)
        total, n = None, 0
        device = next(self.model.parameters()).device

        for batch in dataloader:
            if n >= max_batches: break
            with torch.no_grad():
                _ = self.model(**{k: v.to(device) for k, v in batch.items()})
            act = self.activations.pop(key, None)
            if act is not None:
                total = act if total is None else total + act
                n += 1

        handle.remove()
        if n == 0:
            raise RuntimeError(f"No activations for {blk['prefix']}")
        return total / n

    def _normalize_strategy_name(self, name: str) -> str:
        n = name.lower().strip()
        if n in {"ffn", "mlp"}: return "ffn"
        if n in {"attention", "attn"} or n.startswith("atten"): return "attention"
        if n in {"lora_then_prune", "lora->prune", "lora_then"}: return "lora_then_prune"
        if n in {"prune_then_lora", "prune->lora", "prune_then"}: return "prune_then_lora"
        raise ValueError("Unknown strategy ...")

    def prune(
        self,
        dataloader: torch.utils.data.DataLoader,
        strategy=("ffn",),                 
        sparsity: float = 0.3,            
        max_batches: int = 10,           
        **common_kwargs,                   
    ):
        """
        Unified entry point to run one or more pruning passes in order.

        strategy:
          - a string, e.g. "ffn" or "attention"
          - a sequence of strings, e.g. ["ffn", "attention"]
          - a sequence of (name, kwargs) pairs, e.g.
                [("ffn", {"sparsity":0.4}), ("attention", {"sparsity":0.2, "max_batches":5})]
          - "lora_then_prune" or ("lora_then_prune", {"lora_kwargs": {...}, "prune_after": ["ffn","attention"]})
          - "prune_then_lora" or ("prune_then_lora", {"prune_before": ["ffn","attention"], "lora_kwargs": {...}})

        Notes: uses your existing activation-based criterion (selection rule) for both
        FFN (MLP) neuron pruning and attention head pruning. (criterion = pruning strategy).
        """
        # normalize to a list of steps: [(name, kwargs), ...]
        if isinstance(strategy, (str,)):
            steps = [(self._normalize_strategy_name(strategy), {})]
        else:
            steps = []
            for s in strategy:
                if isinstance(s, (tuple, list)):
                    name, kw = s
                    steps.append((self._normalize_strategy_name(name), dict(kw)))
                else:
                    steps.append((self._normalize_strategy_name(s), {}))

        def _run_ffn(**kw):
            self.prune_all_mlp_layers(
                dataloader=dataloader,
                sparsity=kw.get("sparsity", sparsity),
                max_batches=kw.get("max_batches", max_batches),
            )

        def _run_attention(**kw):
            self.prune_attention_heads(
                dataloader=dataloader,
                sparsity=kw.get("sparsity", sparsity),
                max_batches=kw.get("max_batches", max_batches),
            )
        
        def _run_lora_then_prune(**kw):
            lora_kwargs = dict(kw.get("lora_kwargs", {}))
            if "device" not in lora_kwargs:
                lora_kwargs["device"] = next(self.model.parameters()).device.type

            console.rule("[bold cyan]LoRA Fine-tuning (pre-prune)")
            merged = lora_finetune(
                model=self.model,
                dataloader=dataloader,
                **lora_kwargs,
            )
            self.model = merged

            r_used = lora_kwargs.get("r", None)
            sp_used = kw.get("sparsity", sparsity)
            try:
                self._warn_lora_prune_collapse(r=r_used, sparsity=sp_used, safety_factor=1.5)
            except Exception:
                pass

            prune_after = kw.get("prune_after", ["ffn", "attention"])
            for step in prune_after:
                st = self._normalize_strategy_name(step)
                if st == "ffn":
                    _run_ffn(**kw)
                elif st == "attention":
                    _run_attention(**kw)
                else:
                    raise ValueError(f"Unsupported step in prune_after: {step}")

        def _run_prune_then_lora(**kw):
            prune_before = kw.get("prune_before", ["ffn", "attention"])
            for step in prune_before:
                st = self._normalize_strategy_name(step)
                if st == "ffn":
                    _run_ffn(**kw)
                elif st == "attention":
                    _run_attention(**kw)
                else:
                    raise ValueError(f"Unsupported step in prune_before: {step}")

            lora_kwargs = dict(kw.get("lora_kwargs", {}))
            if "device" not in lora_kwargs:
                lora_kwargs["device"] = next(self.model.parameters()).device.type

            try:
                r_req = lora_kwargs.get("r", None)
                if r_req is not None:
                    blocks = self._discover_mlp_blocks(self.model)
                    def _of(layer):
                        if isinstance(layer, nn.Linear): return layer.out_features
                        if isinstance(layer, Conv1D):   return layer.weight.shape[1]
                    widths = []
                    for b in blocks:
                        if b["type"] == "gated":
                            for cand in (_of(b["up"]), _of(b["gate"])):
                                if cand: widths.append(cand)
                        else:
                            cand = _of(b["fc"])
                            if cand: widths.append(cand)
                    if widths:
                        hid_min = min(widths)
                        if r_req >= hid_min:
                            console.print(Panel.fit(
                                f"[bold yellow]Warning:[/bold yellow] LoRA rank r={r_req} ≥ pruned hidden={hid_min}. "
                                f"Reducing r to {max(1, hid_min//2)}.",
                                title="LoRA Rank Adjustment",
                                border_style="yellow"
                            ))
                            lora_kwargs["r"] = max(1, hid_min // 2)
            except Exception:
                pass

            console.rule("[bold cyan]LoRA Fine-tuning (post-prune)")
            merged = lora_finetune(
                model=self.model,
                dataloader=dataloader,
                **lora_kwargs,
            )
            self.model = merged
        
        runners = {
            "ffn": _run_ffn,
            "attention": _run_attention,
            "lora_then_prune": _run_lora_then_prune,
            "prune_then_lora": _run_prune_then_lora,
        }

        for name, per_step in steps:
            merged = {**common_kwargs, **per_step}
            runners[name](**merged)

    def _warn_lora_prune_collapse(
        self,
        r: Optional[int],
        sparsity: float,
        safety_factor: float = 1.5,
    ):
        """
        Warn if FFN neuron keep-count after pruning is likely to undercut LoRA's rank r.
        Rule of thumb: keep >= safety_factor * r.
        """
        if r is None:
            return

        blocks = Pruner._discover_mlp_blocks(self.model)
        if not blocks:
            return

        def _out_features(layer):
            if isinstance(layer, nn.Linear):
                return layer.out_features
            if isinstance(layer, Conv1D):
                return layer.weight.shape[1]
            return None

        hidden_sizes = []
        for blk in blocks:
            if blk["type"] == "gated":
                of = _out_features(blk["up"]) or _out_features(blk["gate"])
            else:
                of = _out_features(blk["fc"])
            if of is not None:
                hidden_sizes.append(of)

        if not hidden_sizes:
            return

        hid = min(hidden_sizes)
        keep = int((1.0 - sparsity) * hid)

        threshold = int(safety_factor * r)
        if keep < threshold:
            console.print(
                Panel.fit(
                    f"[bold yellow]Warning:[/bold yellow] With sparsity={sparsity:.0%}, "
                    f"FFN keep={keep} < {safety_factor:.1f}×rank ({threshold}).\n"
                    f"This may collapse LoRA's adapted subspace (rank={r}). "
                    f"Consider reducing sparsity or increasing rank.",
                    title="LoRA × Pruning Risk",
                    border_style="yellow",
                )
            )

    