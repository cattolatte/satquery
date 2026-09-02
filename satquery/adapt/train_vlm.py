"""LoRA fine-tuning of the generative specialist on VRSBench train.

Why this exists, in one measurement: the fixed-vocabulary head scores 11.5% on
VRSBench VQA against a hard ceiling of 65.5%, and the ceiling is not the binding
constraint -- we reach only 18% of what is already reachable. That gap is task
mismatch, not output space: the backbone was adapted on BigEarthNet land-cover
captions and asked object-centric aerial questions it had never seen.

LoRA rather than full fine-tuning, on a 500M model rather than a 2B one. Both
choices are about finishing: a run that converges on the hardware available
beats a larger one that does not, and for narrow in-domain benchmarks the
supervision matters more than the base model's size.

Loss is computed on the answer tokens only. Training on the prompt as well
would spend most of the gradient teaching the model to reproduce questions,
which is not the task.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BASE = "HuggingFaceTB/SmolVLM-500M-Instruct"
OUT = Path("checkpoints/rs_vlm")

# Task-specific instructions, so one model serves three heads without the
# prompt having to carry an explanation each time.
PROMPTS = {
    "vqa": "{q}\nAnswer in as few words as possible.",
    "caption": "{q}",
    "refer": "{q}\nRespond with a bounding box as {{<x0><y0><x1><y1>}} on a 0-99 scale.",
}


class VRSSet(Dataset):
    def __init__(self, path: str, limit: int = 0):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        if limit:
            rows = rows[:limit]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def build_collate(processor):
    def collate(batch: list[dict]):
        images, texts = [], []
        for r in batch:
            prompt = PROMPTS.get(r["task"], "{q}").format(q=r["question"])
            messages = [
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]},
            ]
            texts.append(processor.apply_chat_template(messages, add_generation_prompt=False))
            images.append([Image.open(r["image"]).convert("RGB")])

        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True, truncation=True, max_length=1024)
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Mask everything before the assistant's turn: the loss belongs on the
        # answer, not on the question the model was given.
        marker = processor.tokenizer("Assistant:", add_special_tokens=False)["input_ids"]
        if marker:
            for row in range(labels.shape[0]):
                ids = enc["input_ids"][row].tolist()
                cut = 0
                for pos in range(len(ids) - len(marker), -1, -1):
                    if ids[pos:pos + len(marker)] == marker:
                        cut = pos + len(marker)
                        break
                labels[row, :cut] = -100
        image_token = getattr(processor.tokenizer, "image_token_id", None)
        if image_token is not None:
            labels[labels == image_token] = -100
        enc["labels"] = labels
        return enc
    return collate


@torch.no_grad()
def sample_answers(model, processor, rows, device, n: int = 24) -> float:
    """Exact-match on a held-out sample, as a training-time sanity signal."""
    model.eval()
    hits = 0
    for r in rows[:n]:
        prompt = PROMPTS.get(r["task"], "{q}").format(q=r["question"])
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        enc = processor(text=text, images=[Image.open(r["image"]).convert("RGB")],
                        return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=32, do_sample=False)
        gen = processor.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
        norm = lambda s: re.sub(r"[^a-z0-9<> ]", "", str(s).strip().lower())
        hits += norm(gen) == norm(r["answer"])
    model.train()
    return hits / max(len(rows[:n]), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/adapt_vlm/train.jsonl")
    ap.add_argument("--test", default="data/adapt_vlm/test.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.float32)

    lora = LoraConfig(
        r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"device {device} | trainable {trainable/1e6:.1f}M of {total/1e6:.0f}M "
          f"({trainable/total:.2%})")

    train_set = VRSSet(a.train, a.limit)
    held = [json.loads(l) for l in Path(a.test).read_text().splitlines() if l.strip()]
    random.Random(0).shuffle(held)
    loader = DataLoader(train_set, batch_size=a.batch, shuffle=True,
                        collate_fn=build_collate(processor), num_workers=0)
    print(f"train {len(train_set)}  held-out {len(held)}  "
          f"steps/epoch {len(loader) // a.accum}")

    print(f"\nbefore: exact-match {sample_answers(model, processor, held, device):.1%}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = max(1, (len(loader) // a.accum) * a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.05)

    model.train()
    t0, seen, running = time.time(), 0, 0.0
    for epoch in range(a.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / a.accum
            loss.backward()
            running += loss.item() * a.accum
            seen += 1
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if seen % 200 == 0:
                print(f"  epoch {epoch} step {seen}/{len(loader)}  "
                      f"loss {running/200:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                running = 0.0

    acc = sample_answers(model, processor, held, device)
    print(f"\nafter: exact-match {acc:.1%}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    (out / "adaptation.json").write_text(json.dumps(
        {"base": BASE, "dataset": "VRSBench train", "train_rows": len(train_set),
         "epochs": a.epochs, "lr": a.lr, "lora_rank": a.rank,
         "held_out_exact_match": acc}, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
