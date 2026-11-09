"""Global / Group / Specific split of a column-standardized score matrix and the analyses built on it."""
from math import comb

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.stats import spearmanr


def standardize(S):
    sd = S.std(axis=0, keepdims=True)
    return (S - S.mean(axis=0, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def split(Z):
    a = Z.mean(axis=1)
    return a, Z - a[:, None]


def decompose(S):
    Z = standardize(S)
    return (Z, *split(Z))


def group_mean(e, labels):
    H = np.eye(labels.max() + 1)[labels]
    return (e @ H / np.maximum(H.sum(axis=0), 1))[:, labels]


def shares(Z, a, e, labels, n_null, rng):
    tot = float(np.sum(Z ** 2))
    group = float(np.sum(group_mean(e, labels) ** 2)) / tot
    null = [float(np.sum(group_mean(e, rng.permutation(labels)) ** 2)) / tot for _ in range(n_null)]
    return {"global": Z.shape[1] * float(np.sum(a ** 2)) / tot, "global_null": 1.0 / Z.shape[1], "group": group,
            "group_null": float(np.mean(null)), "group_null_q95": float(np.quantile(null, 0.95)),
            "over_null": group / float(np.mean(null))}


def unit_columns(e):
    return e / np.linalg.norm(e, axis=0, keepdims=True)


def score_clusters(e, k, seed):
    _, lab = kmeans2(unit_columns(e).T, k, seed=seed, minit="++", iter=50)
    return np.unique(lab, return_inverse=True)[1]


def summarize(reps):
    ratio = [r["over_null"] for r in reps]
    return {"repeats": reps, "group": float(np.mean([r["group"] for r in reps])),
            "group_null": float(np.mean([r["group_null"] for r in reps])),
            "over_null": float(np.mean(ratio)), "over_null_min": float(np.min(ratio)),
            "frac_above_null_q95": float(np.mean([r["group"] > r["group_null_q95"] for r in reps]))}


def in_sample(Z, a, e, k, repeats, n_null, rng):
    return summarize([shares(Z, a, e, score_clusters(e, k, int(rng.integers(1 << 31))), n_null, rng)
                      for _ in range(repeats)])


def split_half(Z, k, repeats, n_null, rng):
    """Clusters fit on a random half of the items, Group share measured on the other half."""
    reps = []
    for _ in range(repeats):
        perm = rng.permutation(Z.shape[0])
        A, B = perm[: Z.shape[0] // 2], perm[Z.shape[0] // 2:]
        _, lab = kmeans2(unit_columns(split(Z[A])[1]).T, k, seed=int(rng.integers(1 << 31)), minit="++", iter=50)
        reps.append(shares(Z[B], *split(Z[B]), lab, n_null, rng))
    return summarize(reps)


def cluster_outputs(F, k, seed):
    X = (F - F.mean(0)) / np.where(F.std(0) > 1e-12, F.std(0), 1.0)
    _, lab = kmeans2(X, k, seed=seed, minit="++", iter=50)
    return np.unique(lab, return_inverse=True)[1]


def pairs(n):
    return sum(comb(int(v), 2) for v in np.ravel(n))


def contingency(x, y):
    c = np.zeros((x.max() + 1, y.max() + 1), dtype=np.int64)
    np.add.at(c, (x, y), 1)
    return c


def adjusted_rand(x, y):
    c = contingency(x, y)
    sc, sa, sb = pairs(c), pairs(c.sum(1)), pairs(c.sum(0))
    exp, mx = sa * sb / comb(len(x), 2), (sa + sb) / 2
    return float((sc - exp) / (mx - exp)) if mx != exp else 0.0


def wallace(lab, ref):
    """W(lab -> ref) = P(same ref group | same cluster) over output pairs; chance = share of same-ref pairs."""
    c = contingency(lab, ref)
    both, sl, sr, tot = pairs(c), pairs(c.sum(1)), pairs(c.sum(0)), comb(len(lab), 2)
    return {"wallace": both / sl, "wallace_chance": sr / tot, "wallace_rev": both / sr,
            "wallace_rev_chance": sl / tot, "ari": adjusted_rand(lab, ref)}


def drivers(Phi, Psi, a, chunk=4096):
    """Rank correlation of a_i with the training-gradient norm and with its cosine to the common output direction."""
    U = np.asarray(Psi, np.float64)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    m = U.mean(axis=0)
    common = float(np.linalg.norm(m))
    mu = -m / common  # psi is the gradient of -loss; the common direction is taken on the loss gradients
    norm, cos = np.empty(len(a)), np.empty(len(a))
    for s in range(0, len(a), chunk):
        P = np.asarray(Phi[s:s + chunk], np.float64)
        n = np.linalg.norm(P, axis=1)
        norm[s:s + chunk], cos[s:s + chunk] = n, (P @ mu) / np.maximum(n, 1e-30)
    top = np.argsort(-a)[: max(1, len(a) // 100)]
    return {"rho_alignment": float(spearmanr(a, cos)[0]), "rho_norm": float(spearmanr(a, norm)[0]),
            "common_direction": common, "mean_pairwise_cos_outputs": float((len(U) * common ** 2 - 1) / (len(U) - 1)),
            "top1pct_norm_ratio": float(np.median(norm[top]) / np.median(norm))}


def global_portion(Z, a, K, min_z=1.0):
    """Median of a_i / Z_ij over the top-K entries of every output with Z_ij >= min_z."""
    idx = np.argpartition(-Z, K - 1, axis=0)[:K]
    z = np.take_along_axis(Z, idx, axis=0)
    ok = z >= min_z
    return float(np.median(a[idx][ok] / z[ok]))


def three_part_example(Z, a, labels, K, tol, min_z=1.0):
    """Three top-K entries of one output, all within tol of each other, each mostly made of one component; maximizes
    the summed margins (main component's share minus the largest absolute share of the other two)."""
    e = Z - a[:, None]
    G = group_mean(e, labels)
    R = e - G
    best, j, items = -np.inf, None, None
    for col in range(Z.shape[1]):
        top = np.argsort(-Z[:, col])[:K]
        top = top[Z[top, col] >= min_z]
        z = Z[top, col]
        comp = np.stack([a[top], G[top, col], R[top, col]], axis=1) / z[:, None]
        margin = np.stack([comp[:, k] - np.abs(np.delete(comp, k, axis=1)).max(1) for k in range(3)], axis=1)
        cand = [np.flatnonzero(margin[:, k] > 0) for k in range(3)]
        if any(c.size == 0 for c in cand):
            continue
        P = np.meshgrid(*cand, indexing="ij")
        zz = np.stack([z[p] for p in P])
        tot = np.where(np.ptp(zz, axis=0) <= tol * zz.min(axis=0), sum(margin[p, k] for k, p in enumerate(P)), -np.inf)
        i = np.unravel_index(np.argmax(tot), tot.shape)
        if tot[i] > best:
            best, j, items = tot[i], col, [int(top[p[i]]) for p in P]
    if j is None:
        return None
    return {"output": j, "entries": [
        {"dominant": name, "train_index": i, "rank": int((Z[:, j] > Z[i, j]).sum()) + 1, "Z": float(Z[i, j]),
         "global": float(a[i]), "group": float(G[i, j]), "specific": float(R[i, j])}
        for name, i in zip(("global", "group", "specific"), items)]}


def top_overlap(e, genre, score, Ks, n_null, rng):
    """Fraction of each output's top-K items by Z_specific kept when the grouping changes from genre to score
    clusters, or to random groups with the genre sizes; mean over outputs (and random draws)."""
    def tops(R):
        out = {}
        for K in Ks:
            M = np.zeros(R.shape, dtype=bool)
            np.put_along_axis(M, np.argpartition(-R, K - 1, axis=0)[:K], True, axis=0)
            out[K] = M
        return out

    def overlap(ma, mb, K):
        return float(np.mean((ma & mb).sum(axis=0) / K))
    tg = tops(e - group_mean(e, genre))
    ts = tops(e - group_mean(e, score))
    rand = {K: [] for K in Ks}
    for _ in range(n_null):
        tr = tops(e - group_mean(e, rng.permutation(genre)))
        for K in Ks:
            rand[K].append(overlap(tg[K], tr[K], K))
    return {str(K): {"to_score_clusters": overlap(tg[K], ts[K], K), "to_random": float(np.mean(rand[K]))} for K in Ks}


def cross_seed(es, k, n_null, rng):
    """Group share (of ||e||^2) in seed t's scores under seed s's score clusters, against random groupings."""
    labs = [score_clusters(e, k, 0) for e in es]
    out = {"ari_between_seeds": {}, "transfer": {}}
    for s in range(len(es)):
        for t in range(len(es)):
            if t == s:
                continue
            if s < t:
                out["ari_between_seeds"][f"{s}|{t}"] = adjusted_rand(labs[s], labs[t])
            tot = float(np.sum(es[t] ** 2))
            g = float(np.sum(group_mean(es[t], labs[s]) ** 2)) / tot
            null = float(np.mean([np.sum(group_mean(es[t], rng.permutation(labs[s])) ** 2) / tot for _ in range(n_null)]))
            out["transfer"][f"{s}->{t}"] = {"group": g, "group_null": null, "excess": g - null}
    out["excess_mean"] = float(np.mean([v["excess"] for v in out["transfer"].values()]))
    return out
