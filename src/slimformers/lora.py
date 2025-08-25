from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn


def lora_finetune(
    model: nn.Module,
    dataloader,
    epochs: int = 1,
    lr: float = 1e-4,
    device: str = "cuda",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    task_type: TaskType = TaskType.CAUSAL_LM,
    optimizer: Union[
        str, torch.optim.Optimizer, Callable[[iter], torch.optim.Optimizer], None
    ] = None,
    optimizer_kwargs: Optional[Dict] = None,
):
    """
    Fine-tunes a transformer model using LoRA (Low-Rank Adaptation).

    Args:
        model: Pretrained transformer model (nn.Module)
        dataloader: DataLoader providing training batches
        epochs: Number of training epochs
        lr: Learning rate for the optimizer
        device: "cpu" or "cuda"
        r: LoRA rank (rank)
        alpha: LoRA scaling factor
        dropout: LoRA dropout probability
        task_type: PEFT task type
        optimizer: One of:
            - str: one of {"adamw","adam","sgd"}
            - callable: factory taking `params` and returning an optimizer
            - torch.optim.Optimizer: a pre built optimizer for this model
            - None: defaults to AdamW(lr=lr)
        optimizer_kwargs: Extra kwargs passed to the optimizer constructor/factory

    Returns:
        nn.Module: LoRA-merged fine-tuned model (if mergeable), else the PEFT-wrapped model.
    """
    console = Console()
    model.to(device)
    model.train()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    try:
        from slimformers.discovery import DISCOVERY_REGISTRY, default_discover

        cls = type(model).__name__
        finder = DISCOVERY_REGISTRY.get(cls, default_discover)
        blocks = finder(model)

        target_modules = set()
        for blk in blocks:
            if blk["type"] == "ffn":
                target_modules.add(blk["fc_name"].rsplit(".", 1)[-1])
                target_modules.add(blk["proj_name"].rsplit(".", 1)[-1])
            elif blk["type"] == "gated":
                target_modules.add(blk["gate_name"].rsplit(".", 1)[-1])
                target_modules.add(blk["up_name"].rsplit(".", 1)[-1])
                target_modules.add(blk["down_name"].rsplit(".", 1)[-1])
        target_modules = list(sorted(target_modules))

        console.print(
            f"[bold green][LoRA][/bold green] Auto-selected target_modules: {target_modules}"
        )
    except Exception as e:
        raise ValueError(f"Could not infer target_modules: {e}")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    optimizer_kwargs = optimizer_kwargs or {}

    OPTZ = {
        "adamw": lambda p: torch.optim.AdamW(p, lr=lr, **optimizer_kwargs),
        "adam": lambda p: torch.optim.Adam(p, lr=lr, **optimizer_kwargs),
        "sgd": lambda p: torch.optim.SGD(p, lr=lr, **optimizer_kwargs),
    }

    if optimizer is None:
        opt = OPTZ["adamw"](model.parameters())
    elif isinstance(optimizer, str):
        key = optimizer.lower()
        if key not in OPTZ:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Try one of {list(OPTZ.keys())} or pass a callable."
            )
        opt = OPTZ[key](model.parameters())
    elif callable(optimizer):
        opt = optimizer(model.parameters())
    elif isinstance(optimizer, torch.optim.Optimizer):
        opt = optimizer
    else:
        raise TypeError("optimizer must be None, str, callable, or torch.optim.Optimizer")

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        epoch_task = progress.add_task("[cyan]Training epochs", total=epochs)

        for epoch in range(epochs):
            total_loss = 0.0
            batch_task = progress.add_task(
                f"[magenta]Epoch {epoch+1}/{epochs}", total=len(dataloader)
            )

            for batch in dataloader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()

                labels = inputs["input_ids"].clone()
                if "attention_mask" in inputs:
                    labels[inputs["attention_mask"] == 0] = -100
                if "labels_mask" in inputs:
                    labels[~inputs["labels_mask"].bool()] = -100

                inputs["labels"] = labels
                outputs = model(**inputs)
                loss = outputs.loss

                loss.backward()
                opt.step()
                total_loss += loss.item()

                progress.advance(batch_task)

            progress.remove_task(batch_task)
            console.print(
                f"[green][LoRA][/green] Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}"
            )
            progress.advance(epoch_task)

    try:
        merged = model.merge_and_unload()
        if hasattr(merged.config, "use_cache"):
            merged.config.use_cache = True
        return merged
    except AttributeError:
        return model
