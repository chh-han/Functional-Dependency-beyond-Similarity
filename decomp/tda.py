"""TDA scores (n_catalogue, n_outputs) from the saved gradient sketches; positive = training on the item raises the output's log-likelihood."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def graddot(Phi, Psi):
    return -(np.asarray(Phi, np.float32) @ np.asarray(Psi, np.float32).T).astype(np.float64)


def gradcos(Phi, Psi):
    P = np.asarray(Phi, np.float32)
    Q = np.asarray(Psi, np.float32)
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    return -(P @ Q.T).astype(np.float64)


def trak(Phi, Psi, lam_rel):
    P = np.asarray(Phi, np.float64)
    G = P.T @ P
    G[np.diag_indices_from(G)] += lam_rel * np.trace(G) / G.shape[0]
    return -(P @ cho_solve(cho_factor(G), np.asarray(Psi, np.float64).T))


def single(method, Phi, Psi):
    if method == "graddot":
        return graddot(Phi, Psi)
    if method == "gradcos":
        return gradcos(Phi, Psi)
    if method.startswith("trak_lam"):
        return trak(Phi, Psi, float(method[len("trak_lam"):]))
    raise FileNotFoundError(f"{method}: scores are written by the gradient step")


def cosine_lr(step, total, base_lr, warmup):
    return base_lr * min(1.0, (step + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, step / total)))


def tracin_weights(ckpt_steps, total, base_lr, warmup):
    # learning rate summed over the steps since the previous checkpoint
    w, prev = [], 0
    for s in ckpt_steps:
        w.append(float(sum(cosine_lr(x, total, base_lr, warmup) for x in range(prev, s))))
        prev = s
    return w


def compute(model, method):
    T = model.final
    if method == "tracin":
        run = model.run()
        w = tracin_weights(run["save_steps"], T, run["lr"], run["warmup"])
        return sum(wt * graddot(model.phi(0, s), model.psi(0, s)) for s, wt in zip(run["save_steps"], w))
    if method == "trak_ens":
        return np.mean([trak(model.phi(s, T), model.psi(s, T), 1.0) for s in (0, 1, 2)], axis=0)
    return single(method, model.phi(0, T), model.psi(0, T))
