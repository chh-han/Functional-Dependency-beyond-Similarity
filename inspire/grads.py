"""Count-sketched gradients (phi of the catalogue, psi of the outputs) or LoGra / FactGraSS scores via dattri."""
import argparse
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from inspire.common import (add_args, build_llm, grad_list, load_checkpoint, load_outputs, load_set, make_batch,
                            run_dir, train_loss, trainable, unpad)
from musicgen.grads import K, Sketch

PROJ_DIM, DAMPING, BLOWUP, IF_BATCH = 1024, 0.1, 4, 4


class LossModel(torch.nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, **batch):
        return SimpleNamespace(loss=self.llm(batch, next(self.llm.parameters()).device)["loss"])


def block_if_scores(method, llm, tok, cat, cat_codes, gen, gen_codes, cache):
    from torch.utils.data import DataLoader
    from dattri.algorithm.factgrass import FactGraSSAttributor
    from dattri.algorithm.logra import LoGraAttributor
    from dattri.task import AttributionTask

    model = LossModel(llm)

    def collate(items):
        return make_batch(tok, [t for _, t in items], [c for c, _ in items], "cpu")

    def loss_fn(model, batch, device):
        return model(**{k: v.to(device) for k, v in batch.items()}).loss

    train = [(torch.from_numpy(cat_codes[j].astype(np.int64)), cat.text.iloc[j]) for j in range(len(cat))]
    test = [(unpad(gen_codes[i]), gen.prompt.iloc[i]) for i in range(len(gen))]
    layers = [n for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and (".layers." in n or n == "llm.llm_decoder")]
    kw = dict(task=AttributionTask(loss_func=loss_fn, model=model, checkpoints=model.state_dict()),
              layer_names=layers, hessian="eFIM", damping=DAMPING, device="cuda", proj_dim=PROJ_DIM,
              offload="disk", cache_dir=str(cache))
    if method == "factgrass":
        kw["blowup_factor"] = BLOWUP
    attr = (FactGraSSAttributor if method == "factgrass" else LoGraAttributor)(**kw)
    tl = DataLoader(train, batch_size=IF_BATCH, shuffle=False, collate_fn=collate)
    attr.cache(tl)
    sc = attr.attribute(tl, DataLoader(test, batch_size=IF_BATCH, shuffle=False, collate_fn=collate))
    shutil.rmtree(cache, ignore_errors=True)
    return sc.detach().cpu().numpy().astype(np.float32)


def sketch_grads(llm, tok, sk, params, codes, texts, sign):
    g, losses = np.zeros((len(texts), K), np.float32), np.zeros(len(texts), np.float32)
    for i, (c, t) in enumerate(zip(codes, texts)):
        loss = train_loss(llm, tok, [c], [t])
        g[i] = sk(grad_list(sign * loss, params)).cpu().numpy()
        losses[i] = loss.item()
    return g, losses


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, help="default: last checkpoint")
    ap.add_argument("--target", choices=["catalogue", "outputs"], required=True)
    ap.add_argument("--method", choices=["sketch", "logra", "factgrass"], default="sketch")
    a = ap.parse_args()
    _, llm, tok = build_llm(a.weights)
    step = load_checkpoint(llm, run_dir(a.root, a.model, a.seed), a.step)
    mdir = a.root / a.model

    if a.method != "sketch":
        cat, cat_codes = load_set(a.root, "catalogue")
        gen, gen_codes = load_outputs(a.root, a.model)
        out = mdir / "scores"
        out.mkdir(parents=True, exist_ok=True)
        sc = block_if_scores(a.method, llm, tok, cat, cat_codes, gen, gen_codes, out / f"cache_{a.method}")
        np.save(out / f"{a.method}.npy", sc)
        return

    params = [p for _, p in trainable(llm)]
    sk = Sketch(sum(p.numel() for p in params))
    out = mdir / "grads" / a.target / f"seed{a.seed}_step{step}"
    out.mkdir(parents=True, exist_ok=True)
    if a.target == "catalogue":
        cat, codes = load_set(a.root, "catalogue")
        phi, _ = sketch_grads(llm, tok, sk, params, (torch.from_numpy(c.astype(np.int64)) for c in codes),
                              cat.text.tolist(), 1)
        np.save(out / "phi.npy", phi)
        cat[["seg_id", "track_id", "artist_id"]].rename_axis("idx").to_csv(out / "rows.csv")
    else:
        gen, codes = load_outputs(a.root, a.model)
        psi, losses = sketch_grads(llm, tok, sk, params, (unpad(c) for c in codes), gen.prompt.tolist(), -1)
        np.save(out / "psi.npy", psi)
        np.save(out / "gen_loss.npy", losses)


if __name__ == "__main__":
    main()
