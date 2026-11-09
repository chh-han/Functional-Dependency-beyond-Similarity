"""Count-sketched gradients (phi of the catalogue, psi of the outputs) and the LoGra / FactGraSS scores."""
import argparse
import os
import shutil

import numpy as np
import torch

from musicgen.common import HF_NAMES, catalogue, delay, item_loss, load, outputs, params, run_dir

K, SKETCH_SEED = 8192, 0
PROJ_DIM, DAMPING, BLOWUP, BS = 1024, 0.1, 4, 4


class Sketch:
    def __init__(self, p, dev="cuda"):
        g = torch.Generator(device="cpu").manual_seed(SKETCH_SEED)
        self.h = torch.randint(0, K, (p,), generator=g).to(torch.int32).to(dev)
        self.s = (torch.randint(0, 2, (p,), generator=g) * 2 - 1).to(torch.int8).to(dev)

    def __call__(self, grads):
        out, o = torch.zeros(K, device=self.h.device), 0
        for g in grads:
            n = g.numel()
            out.index_add_(0, self.h[o:o + n], self.s[o:o + n] * g.reshape(-1).float())
            o += n
        assert o == len(self.h)
        return out


def sketches(model, proc, sk, codes, texts, sign):
    """sign=+1: gradient of the training loss (phi); sign=-1: gradient of the log-likelihood (psi)."""
    ps = params(model)
    out, losses = np.zeros((len(texts), K), np.float32), np.zeros(len(texts), np.float32)
    for i, t in enumerate(texts):
        loss = item_loss(model, proc, codes, i, t)
        gs = torch.autograd.grad(sign * loss, ps, allow_unused=True)
        out[i] = sk([torch.zeros_like(p) if g is None else g for g, p in zip(gs, ps)]).cpu().numpy()
        losses[i] = loss.item()
    return out, losses


def block_scores(method, model, proc, train, test, cache):
    from dattri.task import AttributionTask
    from torch.utils.data import DataLoader

    def collate(items):
        codes = torch.from_numpy(np.stack([c for c, _ in items]).astype(np.int64))
        tok = proc.tokenizer([t for _, t in items], padding=True, return_tensors="pt")
        return {"input_ids": tok.input_ids, "attention_mask": tok.attention_mask, "labels": delay(codes, codes.shape[1])}

    def loss_fn(model, batch, device):
        return model(**{k: v.to(device) for k, v in batch.items()}).loss

    layers = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
              and (n.startswith("decoder.model.decoder.layers.") or n.startswith("decoder.lm_heads."))]
    task = AttributionTask(loss_func=loss_fn, model=model, checkpoints=model.state_dict())
    kw = dict(task=task, layer_names=layers, hessian="eFIM", damping=DAMPING, device="cuda",
              proj_dim=PROJ_DIM, offload="disk", cache_dir=cache)
    if method == "logra":
        from dattri.algorithm.logra import LoGraAttributor as Attr
    else:
        from dattri.algorithm.factgrass import FactGraSSAttributor as Attr
        kw["blowup_factor"] = BLOWUP
    attr = Attr(**kw)
    tl = DataLoader(train, batch_size=BS, shuffle=False, collate_fn=collate)
    attr.cache(tl)
    sc = attr.attribute(tl, DataLoader(test, batch_size=BS, shuffle=False, collate_fn=collate))
    shutil.rmtree(cache, ignore_errors=True)
    return (sc.detach().cpu().numpy() if torch.is_tensor(sc) else np.asarray(sc)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", choices=list(HF_NAMES), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int)
    ap.add_argument("--target", choices=["catalogue", "outputs"], required=True)
    ap.add_argument("--method", choices=["sketch", "logra", "factgrass"], default="sketch")
    a = ap.parse_args()

    model, proc, step = load(run_dir(a.root, a.model, a.seed), a.step)
    cat, cat_codes = catalogue(a.root)
    if a.method != "sketch":
        gen, gen_codes = outputs(a.root, a.model)
        out = os.path.join(a.root, a.model, "scores")
        os.makedirs(out, exist_ok=True)
        sc = block_scores(a.method, model, proc, list(zip(cat_codes, cat.text)), list(zip(gen_codes, gen.prompt)),
                          os.path.join(out, f"cache_{a.method}"))
        np.save(os.path.join(out, f"{a.method}.npy"), sc)
        return
    sk = Sketch(sum(p.numel() for p in params(model)))
    out = os.path.join(a.root, a.model, "grads", a.target, f"seed{a.seed}_step{step}")
    os.makedirs(out, exist_ok=True)
    if a.target == "catalogue":
        phi, _ = sketches(model, proc, sk, cat_codes, cat.text.tolist(), 1)
        np.save(os.path.join(out, "phi.npy"), phi)
        cat[["seg_id", "track_id", "artist_id"]].rename_axis("idx").to_csv(os.path.join(out, "rows.csv"))
    else:
        gen, gen_codes = outputs(a.root, a.model)
        psi, loss = sketches(model, proc, sk, gen_codes, gen.prompt.tolist(), -1)
        np.save(os.path.join(out, "psi.npy"), psi)
        np.save(os.path.join(out, "gen_loss.npy"), loss)


if __name__ == "__main__":
    main()
