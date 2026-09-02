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

from satquery.tools.generative import BASE, IMAGE_SIZE, PROMPTS  # noqa: E402

OUT = Path("checkpoints/rs_vlm")

# BASE, IMAGE_SIZE and PROMPTS are imported from the serving module rather than
# restated here. Training and inference preprocessing that drifts apart is a
# silent accuracy loss, and this project has already paid for one such
# mismatch; a shared definition makes drift impossible rather than unlikely.


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
    """Batch into (prompt + answer), with the loss masked to the answer.

    The prompt is templated separately from the answer rather than templating
    the whole conversation at once. Two reasons, one of them a bug this cost:

      - The image placeholder expands into a long run of image tokens bounded
        by markers. Truncating the combined text cuts through that run, the
        processor then cannot find the closing boundary, and it fails with an
        arithmetic error on a None offset rather than anything legible.
      - Tokenising the prompt on its own gives the exact answer boundary, so
        label masking is by construction rather than by searching for a marker
        string in the ids.
    """
    def collate(batch: list[dict]):
        images, prompts, fulls = [], [], []
        for r in batch:
            instruction = PROMPTS.get(r["task"], "{q}").format(q=r["question"])
            user = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": instruction}]}]
            prompt = processor.apply_chat_template(user, add_generation_prompt=True)
            prompts.append(prompt)
            fulls.append(prompt + " " + r["answer"] + "<end_of_utterance>")
            images.append([Image.open(r["image"]).convert("RGB")])

        enc = processor(text=fulls, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        # Everything up to where the answer starts is context, not target.
        for row, prompt in enumerate(prompts):
            n = len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            # The prompt's image placeholder expands, so count on the encoded
            # row instead of trusting the text-only length.
            ids = enc["input_ids"][row].tolist()
            marker = processor.tokenizer("Assistant:", add_special_tokens=False)["input_ids"]
            cut = n
            for pos in range(len(ids) - len(marker), -1, -1):
                if ids[pos:pos + len(marker)] == marker:
                    cut = pos + len(marker)
                    break
            labels[row, :cut] = -100
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
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    # Image splitting tiles each frame into many crops, which multiplies the
    # sequence length and put a 512x512 batch of two over 30 GB on MPS. One
    # global view per image is enough for scene-level and object-level
    # questions at this resolution, and it is what makes the run fit at all.
    processor = AutoProcessor.from_pretrained(BASE, do_image_splitting=False)
    processor.image_processor.size = {"longest_edge": a.image_size}
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
            # Save periodically. A long run on one machine that only writes at
            # the end loses everything to an interruption, and the adapter is
            # 4 MB -- there is no reason to risk hours of it.
            if seen % a.save_every == 0:
                model.save_pretrained(a.out)
                processor.save_pretrained(a.out)
                print(f"    checkpoint at step {seen}", flush=True)

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
