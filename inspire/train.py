"""Full fine-tuning of the InspireMusic LLM on the catalogue, optionally without a deletion set."""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from inspire.common import add_args, build_llm, load_set, run_dir, train_loss, trainable

EPOCHS, BATCH, LR, WARMUP, COND_DROPOUT = 4, 16, 1e-5, 100, 0.1
MICRO_BATCH = {"inspiremusic-base": 16, "inspiremusic-1.5b": 4}
SAVE_AT = (0.25, 0.5, 0.75, 1.0)
EVAL_EVERY, N_EVAL_HELDOUT, N_EVAL_CATALOGUE = 200, 600, 300


def data_order(n, keep, seed):
    rng = np.random.RandomState(seed)
    return [o[keep[o]] for o in (rng.permutation(n) for _ in range(EPOCHS))]


def batch_codes(codes, idx):
    return [torch.from_numpy(codes[j].astype(np.int64)) for j in idx]


@torch.no_grad()
def eval_loss(llm, tok, df, codes, idx, bs):
    tot = 0.0
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        tot += train_loss(llm, tok, batch_codes(codes, b), df.text.iloc[b].tolist()).item() * len(b)
    return tot / len(idx)


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--deletion", help="<method>/<set>: drop deletion/<method>/exclude/<set>.txt")
    a = ap.parse_args()
    excl = set()
    if a.deletion:
        method, name = a.deletion.split("/")
        excl = set((a.root / a.model / "deletion" / method / "exclude" / f"{name}.txt").read_text().split())
    out = run_dir(a.root, a.model, a.seed, a.deletion)
    out.mkdir(parents=True, exist_ok=True)
    cat, cat_codes = load_set(a.root, "catalogue")
    held, held_codes = load_set(a.root, "heldout")
    keep = ~cat.seg_id.isin(excl).to_numpy()
    orders = data_order(len(cat), keep, a.seed)
    mb = MICRO_BATCH[a.model]
    total = math.ceil(len(orders[0]) / BATCH) * EPOCHS
    save_steps = sorted({max(1, round(f * total)) for f in ((1.0,) if a.deletion else SAVE_AT)})

    _, llm, tok = build_llm(a.weights)
    params = [p for _, p in trainable(llm)]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    drop_rng = np.random.RandomState(a.seed + 1)
    run = dict(model=a.model, seed=a.seed, deletion=a.deletion, n_used=int(keep.sum()), n_excluded=len(excl), lr=LR,
               warmup=WARMUP, batch=BATCH, micro_batch=mb, epochs=EPOCHS, cond_dropout=COND_DROPOUT,
               total_steps=total, save_steps=save_steps)
    (out / "run.json").write_text(json.dumps(run, indent=1))

    held_idx = np.arange(N_EVAL_HELDOUT)
    cat_idx = np.sort(np.random.RandomState(12345).choice(np.flatnonzero(keep), N_EVAL_CATALOGUE, replace=False))

    def evaluate():
        return dict(heldout=eval_loss(llm, tok, held, held_codes, held_idx, mb),
                    catalogue=eval_loss(llm, tok, cat, cat_codes, cat_idx, mb))

    log = [dict(step=0, **evaluate())]
    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        for s in range(0, len(orders[ep]), BATCH):
            opt.zero_grad(set_to_none=True)
            chunk, tot = orders[ep][s:s + BATCH], 0.0
            for m in range(0, len(chunk), mb):
                b = np.sort(chunk[m:m + mb])
                drop = [drop_rng.rand() < COND_DROPOUT for _ in b]
                loss = train_loss(llm, tok, batch_codes(cat_codes, b), cat.text.iloc[b].tolist(), drop=drop)
                (loss * len(b) / len(chunk)).backward()
                tot += loss.item() * len(b) / len(chunk)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            rec = dict(step=step, epoch=ep, train=tot, lr=sched.get_last_lr()[0])
            if step % EVAL_EVERY == 0 or step == total:
                rec.update(evaluate())
                print(f"step {step}/{total} {rec} {(time.time() - t0) / 60:.1f} min", flush=True)
            log.append(rec)
            if step in save_steps:
                torch.save({n: p.detach().cpu() for n, p in trainable(llm)}, out / f"step{step}.pt")
                pd.DataFrame(log).to_csv(out / "log.csv", index=False)


if __name__ == "__main__":
    main()
