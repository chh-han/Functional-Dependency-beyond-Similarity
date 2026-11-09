"""Deletion sets (Global, every score cluster, Specific prompts, random) and their measured effects."""
import json

import numpy as np
import pandas as pd

from .decompose import decompose, group_mean, score_clusters


def make_sets(model, method, k, frac, n_specific, seed):
    _, a, e = decompose(model.scores(method))
    lab = score_clusters(e, k, seed)
    G = group_mean(e, lab)
    R = e - G
    prompt, names = model.labels("prompt_id")
    chosen = sorted(int(p) for p in np.random.default_rng(seed).choice(np.unique(prompt), n_specific, replace=False))
    keys = {"global": a}
    keys.update({f"cluster{c}": G[:, np.flatnonzero(lab == c)[0]] for c in range(k)})
    keys.update({f"specific_{names[p]}": R[:, prompt == p].mean(axis=1) for p in chosen})
    n = int(round(frac * len(a)))
    seg = model.seg_ids()
    out = model.dir / f"deletion/{method}"
    (out / "exclude").mkdir(parents=True, exist_ok=True)
    for name, v in keys.items():
        (out / f"exclude/{name}.txt").write_text("\n".join(seg[np.argsort(-v)[:n]]) + "\n")
    rnd = model.dir / "deletion/random/exclude"
    rnd.mkdir(parents=True, exist_ok=True)
    (rnd / "random.txt").write_text("\n".join(seg[np.random.default_rng(seed + 7).choice(len(a), n, replace=False)]) + "\n")
    doc = {"method": method, "k": k, "n": n, "cluster_labels": lab.tolist(),
           "specific_prompts": [str(names[p]) for p in chosen], "sets": list(keys)}
    (out / "sets.json").write_text(json.dumps(doc, indent=1))


def measured(d, name, gen_ids):
    out = {}
    for f in sorted(d.glob(f"{name}_seed*.csv")):
        m = pd.read_csv(f).set_index("gen_id")
        out[int(m.seed.iloc[0])] = m.loc[gen_ids, "value"].to_numpy()
    return out


def own_vs_other(D, B, rng):
    """D (sets, blocks, seeds): set c's effect on block d. Mean over sets of own minus mean of the others; the null
    permutes which block is each set's own."""
    m = len(D)

    def stat(perm, Dm):
        own = Dm[np.arange(m), perm]
        return float(np.mean(own - (Dm.sum(axis=1) - own) / (m - 1)))
    Dm = D.mean(axis=2)
    t = stat(np.arange(m), Dm)
    null = np.array([stat(rng.permutation(m), Dm) for _ in range(B)])
    own = np.diag(Dm)
    return {"own_minus_other": t, "p_perm": float((1 + np.sum(null <= t)) / (1 + B)),
            "own_minus_other_per_seed": [stat(np.arange(m), D[:, :, s]) for s in range(D.shape[2])],
            "own_worse_than_others": int(np.sum(own < (Dm.sum(axis=1) - own) / (m - 1))),
            "own_rank1": int(np.sum(np.argmin(Dm, axis=1) == np.arange(m))), "matrix": Dm.tolist()}


def within_cluster(E, codes, lab, prompt, B, rng):
    """For each output of the targeted prompt: its effect minus the mean effect on the other prompts' outputs of its
    score cluster. The null replaces each set's prompt by a prompt drawn at random."""
    ps = np.unique(prompt)
    W = np.full((len(E), len(ps)), np.nan)
    D = np.zeros((len(E), E[0].shape[0]))
    for r, (Er, p) in enumerate(zip(E, codes)):
        for c, q in enumerate(ps):
            diffs = [Er[:, j] - Er[:, (lab == lab[j]) & (prompt != q)].mean(axis=1)
                     for j in np.flatnonzero(prompt == q) if np.any((lab == lab[j]) & (prompt != q))]
            if diffs:
                W[r, c] = np.mean(diffs, axis=0).mean()
                if q == p:
                    D[r] = np.mean(diffs, axis=0)
    t = float(D.mean())
    null = np.array([np.nanmean([W[r, rng.integers(len(ps))] for r in range(len(E))]) for _ in range(B)])
    return {"own_minus_other": t, "p_perm": float((1 + np.sum(null <= t)) / (1 + B)),
            "own_minus_other_per_seed": D.mean(axis=0).tolist(), "per_set": D.mean(axis=1).tolist(),
            "own_worse_than_others": int(np.sum(D.mean(axis=1) < 0))}


def effects(model, method, B, rng):
    """Delta LL of every deletion set minus the same-seed random deletion, per output."""
    sets = json.loads((model.dir / f"deletion/{method}/sets.json").read_text())
    gen_ids = model.outputs().gen_id
    base = model.dir / f"deletion/{method}/baseline.json"
    base = json.loads(base.read_text()) if base.exists() else {}
    meas = model.dir / f"deletion/{method}/measure"

    def delta(names):
        m = {n: measured(meas, n, gen_ids) for n in names}
        r = {n: measured(model.dir / "deletion/random/measure", base.get(n, "random"), gen_ids) for n in names}
        seeds = sorted(set.intersection(*(set(x) for x in [*m.values(), *r.values()])))
        return [np.stack([m[n][s] - r[n][s] for s in seeds]) for n in names], seeds
    lab = np.asarray(sets["cluster_labels"])
    prompt, pnames = model.labels("prompt_id")
    clusters = [c for c in range(sets["k"]) if np.any(lab == c)]

    (E,), seeds = delta(["global"])
    per_seed = E.mean(axis=1)
    cl = [float(E[:, lab == c].mean()) for c in clusters]
    out = {"cluster_sizes": [int(np.sum(lab == c)) for c in clusters],
           "global": {"seeds": seeds, "all": float(per_seed.mean()), "per_seed": per_seed.tolist(),
                      "all_seeds_negative": bool(np.all(per_seed < 0)), "per_cluster": cl,
                      "clusters_negative": int(np.sum(np.array(cl) < 0))}}
    E, seeds = delta([f"cluster{c}" for c in clusters])
    D = np.stack([np.stack([Er[:, lab == c].mean(axis=1) for c in clusters]) for Er in E])
    out["cluster"] = {"seeds": seeds, **own_vs_other(D, B, rng)}
    names = [f"specific_{p}" for p in sets["specific_prompts"]]
    E, seeds = delta(names)
    codes = [int(np.flatnonzero(pnames == p)[0]) for p in sets["specific_prompts"]]
    out["specific"] = {"seeds": seeds, "sets": names, **within_cluster(E, codes, lab, prompt, B, rng)}
    return out
