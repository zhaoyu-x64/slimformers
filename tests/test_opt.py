import torch
from peft import TaskType
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from slimformers import Pruner, lora_finetune

# Load model and tokenizer
model_id = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

# Sample corpus
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "LoRA and pruning improve model efficiency.",
    "Transformers are powerful neural networks.",
]

encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


# Wrap tokenized data in a Dataset for DataLoader
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


dataloader = DataLoader(TextDataset(encodings), batch_size=2, shuffle=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# Prune
pruner = Pruner(model)
pruner.prune(
    dataloader=dataloader, strategy=[("attn", {"max_batches": 5})], sparsity=0.3
)

print("After pruning:")
print(f"Pruned model size: {count_parameters(model):,} params")

# Forward pass and generation test
model.eval()
sample_inputs = tokenizer("The quick brown fox", return_tensors="pt").to("cpu")
with torch.no_grad():
    out = model(**sample_inputs)
print("Forward pass OK, logits.shape =", out.logits.shape)

gen_ids = model.generate(
    **sample_inputs,
    max_new_tokens=20,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)

print(
    "Generated (pruned) text:\n", tokenizer.decode(gen_ids[0], skip_special_tokens=True)
)

# Apply LoRA
model = lora_finetune(
    model=model,
    dataloader=dataloader,
    epochs=20,
    lr=1e-4,
    device="cpu",
    r=8,
    alpha=16,
    dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    optimizer="sgd",
    optimizer_kwargs={"momentum": 0.9},
)

print(f"Fine-tuned model size: {count_parameters(model):,} params")

# Test generation
model.eval()
with torch.no_grad():
    out_ft = model(**sample_inputs)

gen_ids_ft = model.generate(
    **sample_inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
)
print(
    "Generated (LoRA-finetuned) text:\n",
    tokenizer.decode(gen_ids_ft[0], skip_special_tokens=True),
)

# Stats
pruner.report()
