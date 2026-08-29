"""
exp3_quality_rank.py — Contribution 3 (REAL): Quality-Constrained Rank Allocation.

Fine-tunes real LoRA adapters at each rank and measures ACCURACY, now across
MULTIPLE SEEDS so the result is statistically honest (mean +/- std). This reveals
the accuracy-vs-rank curve and its knee — the rank beyond which accuracy stops
improving — which is the basis for adaptive rank allocation.

Task:  SST-2 sentiment classification (GLUE).
Model: roberta-base (fast; trains in ~18s per run on a 4080).

Run:  python exp3_quality_rank.py
      python exp3_quality_rank.py --seeds 5

Requirements:  pip install datasets scikit-learn
"""
import sys, os, json, time, argparse, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from config import RANK_COUNTS, RESULTS_DIR, gpu_name

CLF_MODEL = "roberta-base"
TASK = "sst2"
MAX_LEN = 128
TRAIN_SUBSET = 4000
EVAL_SUBSET  = 872
EPOCHS = 2
BATCH  = 32
LR = 2e-4

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def build(rank, seed):
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(CLF_MODEL, num_labels=2)
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=rank,
        lora_alpha=16,            # FIXED alpha (not rank*2) for stable scaling across ranks
        lora_dropout=0.1, target_modules=["query", "value"],
    )
    return get_peft_model(model, cfg)

def load_data(tok):
    from datasets import load_dataset
    try:
        ds = load_dataset("nyu-mll/glue", TASK)
    except Exception:
        ds = load_dataset("glue", TASK)
    def enc(b):
        return tok(b["sentence"], truncation=True, padding="max_length", max_length=MAX_LEN)
    train = ds["train"].select(range(min(TRAIN_SUBSET, len(ds["train"])))).map(enc, batched=True)
    val   = ds["validation"].select(range(min(EVAL_SUBSET, len(ds["validation"])))).map(enc, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    train.set_format("torch", columns=cols)
    val.set_format("torch", columns=cols)
    return train, val

def train_eval(rank, seed, train, val, device):
    from torch.utils.data import DataLoader
    model = build(rank, seed).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    tl = DataLoader(train, batch_size=BATCH, shuffle=True)
    vl = DataLoader(val, batch_size=BATCH)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.train()
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        for b in tl:
            ids, am, lb = b["input_ids"].to(device), b["attention_mask"].to(device), b["label"].to(device)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=am, labels=lb)
            out.loss.backward()
            opt.step()
    train_time = time.perf_counter() - t0

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in vl:
            ids, am, lb = b["input_ids"].to(device), b["attention_mask"].to(device), b["label"].to(device)
            pred = model(input_ids=ids, attention_mask=am).logits.argmax(-1)
            correct += (pred == lb).sum().item(); total += lb.numel()
    del model; torch.cuda.empty_cache()
    return correct / total, n_trainable, train_time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="runs per rank")
    args = ap.parse_args()
    seeds = list(range(42, 42 + args.seeds))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu_name()}  |  Task: {TASK}  |  Base: {CLF_MODEL}")
    print(f"Ranks {RANK_COUNTS} x {args.seeds} seeds each\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CLF_MODEL)
    print("Loading data..."); train, val = load_data(tok)

    rows = []
    for r in RANK_COUNTS:
        accs = []
        n_train = 0
        for s in seeds:
            acc, n_train, t = train_eval(r, s, train, val, device)
            accs.append(acc)
            print(f"  rank={r:<4} seed={s} acc={acc:.4f}")
        mean, std = float(np.mean(accs)), float(np.std(accs))
        print(f"  rank={r:<4} MEAN={mean:.4f} +/- {std:.4f}  params={n_train:,}\n")
        rows.append({"contribution":"C3_quality_rank","task":TASK,"base":CLF_MODEL,
            "lora_rank":r,"acc_mean":round(mean,4),"acc_std":round(std,4),
            "acc_runs":[round(a,4) for a in accs],"trainable_params":n_train})

    best = max(r["acc_mean"] for r in rows)
    knee = min((r["lora_rank"] for r in rows if r["acc_mean"] >= best - 0.005), default=None)
    print("="*55)
    print(f"Best mean accuracy: {best:.4f}")
    print(f"Accuracy knee: rank {knee} reaches within 0.5% of best")
    print("="*55)

    out = os.path.join(RESULTS_DIR, "exp3_quality_rank.json")
    with open(out, "w") as f:
        json.dump({"results": rows, "best_acc": best, "accuracy_knee": knee, "seeds": seeds}, f, indent=2)
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()