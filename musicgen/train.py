"""Full fine-tuning of the MusicGen decoder on the catalogue, optionally without a deletion set."""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from musicgen.common import build_model, catalogue, delay, params, run_dir, train_loss

EPOCHS, BATCH, LR, WARMUP, COND_DROPOUT = 4, 16, 1e-5, 100, 0.1
MICRO = {"musicgen-small": 16, "musicgen-large": 2}
EVAL_EVERY, N_HELD, N_CAT_EVAL = 200, 600, 300


def orders(n, keep, seed):
    """One permutation of the full catalogue per epoch; deleted items are dropped without reshuffling."""
    rng = np.random.RandomState(seed)
    return [o[keep[o]] for o in (rng.permutation(n) for _ in range(EPOCHS))]


def micro_batches(order, micro):
    for s in range(0, len(order), BATCH):
        chunk = order[s:s + BATCH]
        yield len(chunk), [np.sort(chunk[m:m + micro]) for m in range(0, len(chunk), micro)]


def lr_lambda(total):
    return lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))


@torch.no_grad()
def eval_loss(model, proc, df, codes, idx):
    tot = 0.0
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += train_loss(model, proc, torch.from_numpy(codes[b].astype(np.int64)).cuda(), df.text.iloc[b]).item() * len(b)
    return tot / len(idx)


def checkpointing(model):
    # HF checkpoints only in train mode; zero all dropout so train mode computes what eval mode does
    for mod in model.decoder.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        for attr in ("dropout", "attention_dropout", "activation_dropout", "layerdrop"):
            if isinstance(getattr(mod, attr, None), float):
                setattr(mod, attr, 0.0)
    for attr in ("dropout", "attention_dropout", "activation_dropout", "layerdrop"):
        if hasattr(model.config.decoder, attr):
            setattr(model.config.decoder, attr, 0.0)
    model.decoder.gradient_checkpointing_enable()
    model.decoder.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", choices=list(MICRO), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--deletion", help="<method>/<set>: drop deletion/<method>/exclude/<set>.txt")
    a = ap.parse_args()

    out = run_dir(a.root, a.model, a.seed, a.deletion)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    cat_df, cat_codes = catalogue(a.root)
    held_df, held_codes = catalogue(a.root, "heldout")
    excl = set()
    if a.deletion:
        method, name = a.deletion.split("/")
        excl = set(open(os.path.join(a.root, a.model, "deletion", method, "exclude", f"{name}.txt")).read().split())
    keep = ~cat_df.seg_id.isin(excl).to_numpy()
    ords = orders(len(cat_df), keep, a.seed)
    total = math.ceil(len(ords[0]) / BATCH) * EPOCHS
    save_steps = [total] if a.deletion else sorted({max(1, round(f * total)) for f in (0.25, 0.5, 0.75, 1.0)})
    micro = MICRO[a.model]

    model, proc = build_model(a.model)
    model.cuda().eval()
    if micro < BATCH:
        checkpointing(model)
    ps = params(model)
    opt = torch.optim.AdamW(ps, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total))
    drop_rng = np.random.RandomState(a.seed + 1)
    meta = dict(model=a.model, seed=a.seed, deletion=a.deletion, epochs=EPOCHS, batch=BATCH, micro_batch=micro, lr=LR,
                warmup=WARMUP, cond_dropout=COND_DROPOUT, n_used=int(keep.sum()), n_excluded=len(excl),
                total_steps=total, save_steps=save_steps)
    json.dump(meta, open(os.path.join(out, "run.json"), "w"), indent=1)

    held_idx = np.arange(N_HELD)
    cat_idx = np.sort(np.random.RandomState(12345).choice(np.flatnonzero(keep), N_CAT_EVAL, replace=False))
    K = cat_codes.shape[1]
    log = [dict(step=0, heldout=eval_loss(model, proc, held_df, held_codes, held_idx),
                catalogue=eval_loss(model, proc, cat_df, cat_codes, cat_idx))]
    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        for n, mbs in micro_batches(ords[ep], micro):
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for b in mbs:
                texts = [t if drop_rng.rand() >= COND_DROPOUT else "" for t in cat_df.text.iloc[b]]
                tok = proc.tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
                lab = delay(torch.from_numpy(cat_codes[b].astype(np.int64)).cuda(), K)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=tok.input_ids, attention_mask=tok.attention_mask, labels=lab).loss
                (loss * len(b) / n).backward()
                tot += loss.item() * len(b) / n
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            sched.step()
            step += 1
            rec = dict(step=step, epoch=ep, train=tot, lr=sched.get_last_lr()[0])
            if step % EVAL_EVERY == 0 or step == total:
                rec["heldout"] = eval_loss(model, proc, held_df, held_codes, held_idx)
                rec["catalogue"] = eval_loss(model, proc, cat_df, cat_codes, cat_idx)
                print(f"step {step}/{total} train {tot:.4f} heldout {rec['heldout']:.4f} "
                      f"catalogue {rec['catalogue']:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
            log.append(rec)
            if step in save_steps:
                torch.save({k: p.detach().cpu().clone() for k, p in model.decoder.named_parameters()},
                           os.path.join(out, f"step{step}.pt"))
                pd.DataFrame(log).to_csv(os.path.join(out, "log.csv"), index=False)


if __name__ == "__main__":
    main()
