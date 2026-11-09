"""Mean per-token log-likelihood of every output under a deletion run."""
import argparse

import pandas as pd
import torch

from inspire.common import add_args, build_llm, load_checkpoint, load_outputs, run_dir, train_loss, unpad


@torch.no_grad()
def values(llm, tok, gen, codes):
    return [-train_loss(llm, tok, [unpad(c)], [t]).item() for c, t in zip(codes, gen.prompt)]


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--deletion", required=True, help="<method>/<set>")
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    _, llm, tok = build_llm(a.weights)
    load_checkpoint(llm, run_dir(a.root, a.model, a.seed, a.deletion))
    method, name = a.deletion.split("/")
    gen, codes = load_outputs(a.root, a.model)
    out = a.root / a.model / "deletion" / method / "measure"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(set=name, seed=a.seed, gen_id=gen.gen_id, value=values(llm, tok, gen, codes))).to_csv(
        out / f"{name}_seed{a.seed}.csv", index=False)


if __name__ == "__main__":
    main()
