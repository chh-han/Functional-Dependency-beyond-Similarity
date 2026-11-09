"""python -m decomp <command> --root ROOT [--model MODEL] [...]: one command per paper product."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import decompose as dc
from . import deletion, figures, tda
from .model import METHODS, Model

SEED_METHODS = ("graddot", "gradcos", "trak_lam1", "trak_lam0.1", "trak_lam0.01")
SKETCH_METHODS = ("graddot", "gradcos", "tracin", "trak_lam1", "trak_lam0.1", "trak_lam0.01", "trak_ens")
TOP_K, TOL = 50, 0.02
N_NULL = {"criteria": 50, "seedrep": 100}


def per_method(fn):
    def run(m, a, rng):
        return {meth: fn(m, a, rng, m.scores(meth)) for meth in a.methods}
    return run


def n_genres(m):
    return int(m.labels("genre")[0].max() + 1)


def cmd_scores(m, a, rng):
    for meth in a.methods:
        m.scores(meth)


def cmd_features(m, a, rng):
    from . import features
    features.write(m)


@per_method
def cmd_shares(m, a, rng, S):
    Z, g, e = dc.decompose(S)
    return {"genre": dc.shares(Z, g, e, m.labels("genre")[0], a.n_null, rng),
            "score_clusters": dc.in_sample(Z, g, e, n_genres(m), a.repeats, a.n_null, rng)}


@per_method
def cmd_ksweep(m, a, rng, S):
    Z, g, _ = dc.decompose(S)
    glob = Z.shape[1] * float(np.sum(g ** 2)) / float(np.sum(Z ** 2))
    out = {"global": glob}
    for k in a.clusters:
        r = dc.split_half(Z, k, a.repeats, a.n_null, rng)
        out[str(k)] = {**r, "group_excess": (r["group"] - r["group_null"]) / (1 - glob)}
    return out


def cmd_agreement(m, a, rng):
    k = n_genres(m)
    F = m.features()
    refs = {"genre": m.labels("genre")[0], **{f: dc.cluster_outputs(F[f], k, a.seed) for f in ("mfcc", "spectrogram", "chroma")}}
    out = {"ref_sizes": {c: np.bincount(r).tolist() for c, r in refs.items()}}
    for meth in a.methods:
        e = dc.decompose(m.scores(meth))[2]
        labs = [dc.score_clusters(e, k, int(rng.integers(1 << 31))) for _ in range(a.repeats)]
        out[meth] = {}
        for c, ref in refs.items():
            x = [dc.wallace(lab, ref) for lab in labs]
            out[meth][c] = {s: float(np.mean([v[s] for v in x])) for s in x[0]}
    return out


@per_method
def cmd_drivers(m, a, rng, S):
    return dc.drivers(m.phi(0, m.final), m.psi(0, m.final), dc.decompose(S)[1])


@per_method
def cmd_criteria(m, a, rng, S):
    _, _, e = dc.decompose(S)
    return dc.top_overlap(e, m.labels("genre")[0], dc.score_clusters(e, n_genres(m), a.seed), a.top_k, a.n_null, rng)


@per_method
def cmd_composition(m, a, rng, S):
    Z, g, _ = dc.decompose(S)
    ex = dc.three_part_example(Z, g, m.labels("genre")[0], TOP_K, TOL)
    if ex:
        ex["gen_id"] = m.outputs().gen_id[ex["output"]]
    return {"global_portion_median": dc.global_portion(Z, g, TOP_K), "three_part_example": ex}


def cmd_seedrep(m, a, rng):
    T = m.final
    sk = [(m.phi(s, T), m.psi(s, T)) for s in (0, 1, 2)]
    return {meth: dc.cross_seed([dc.decompose(tda.single(meth, *x))[2] for x in sk], n_genres(m), a.n_null, rng)
            for meth in a.methods}


def cmd_deletion_sets(m, a, rng):
    deletion.make_sets(m, a.method, n_genres(m), a.frac, a.n_specific, a.seed)


def cmd_deletion_effects(m, a, rng):
    return deletion.effects(m, a.method, a.B, rng)


def cmd_heatmap(a):
    figures.heatmap(a.root, [p.split(":") for p in a.panels])


def cmd_dataset(a):
    return figures.dataset(a.root, a.jamendo, a.n_tags)


COMMANDS = {"scores": cmd_scores, "features": cmd_features, "shares": cmd_shares, "ksweep": cmd_ksweep,
            "agreement": cmd_agreement, "drivers": cmd_drivers, "criteria": cmd_criteria,
            "composition": cmd_composition, "seedrep": cmd_seedrep, "deletion-sets": cmd_deletion_sets,
            "deletion-effects": cmd_deletion_effects}
DEFAULT_METHODS = {"seedrep": SEED_METHODS, "scores": SKETCH_METHODS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=[*COMMANDS, "heatmap", "dataset"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--model")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--method", choices=METHODS, help="deletion-sets / deletion-effects")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", dest="n_null", type=int)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--clusters", type=int, nargs="+", default=[2, 4, 12], help="ksweep")
    ap.add_argument("--top-k", dest="top_k", type=int, nargs="+", default=[50, 100, 200, 500, 1000], help="criteria")
    ap.add_argument("--frac", type=float, default=0.1, help="deletion-sets: deleted fraction of the catalogue")
    ap.add_argument("--n-specific", dest="n_specific", type=int, default=12)
    ap.add_argument("--B", type=int, default=10000, help="deletion-effects: permutations")
    ap.add_argument("--panels", nargs="+", default=["musicgen-small:trak_lam1", "inspiremusic-base:trak_lam1"])
    ap.add_argument("--jamendo", help="dataset: mtg-jamendo-dataset repository")
    ap.add_argument("--n-tags", dest="n_tags", type=int, default=25, help="dataset")
    a = ap.parse_args()
    a.n_null = a.n_null or N_NULL.get(a.command, 200)
    if a.command == "heatmap":
        return cmd_heatmap(a)
    if a.command == "dataset":
        doc, out = cmd_dataset(a), Path(a.root) / "analysis/dataset.json"
    else:
        a.methods = a.methods or DEFAULT_METHODS.get(a.command, METHODS)
        m = Model(a.root, a.model)
        doc = COMMANDS[a.command](m, a, np.random.default_rng(a.seed))
        if doc is None:
            return
        out = m.analysis(f"deletion_{a.method}" if a.command == "deletion-effects" else a.command)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=int))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
