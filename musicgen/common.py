"""Model loading, delay pattern, loss and paths shared by the MusicGen pipeline."""
import json
import os

import numpy as np
import pandas as pd
import torch

PAD = 2048
HF_NAMES = {"musicgen-small": "facebook/musicgen-small", "musicgen-large": "facebook/musicgen-large"}


def catalogue(root, name="catalogue"):
    df = pd.read_csv(os.path.join(root, "data", f"{name}.csv"))
    return df, np.load(os.path.join(root, "codes", "encodec", f"{name}.npy"), mmap_mode="r")


def outputs(root, model):
    d = os.path.join(root, model, "outputs")
    return pd.read_csv(os.path.join(d, "outputs.csv")), np.load(os.path.join(d, "codes.npy"))


def run_dir(root, model, seed, deletion=None):
    if deletion is None:
        return os.path.join(root, model, "finetune", f"seed{seed}")
    method, name = deletion.split("/")
    return os.path.join(root, model, "deletion", method, "runs", name, f"seed{seed}")


def delay(codes, K):
    """(B, K, T) codes -> (B, T+K-1, K) labels; codebook k is shifted right by k steps."""
    B, _, T = codes.shape
    d = torch.full((B, K, T + K - 1), PAD, dtype=torch.long, device=codes.device)
    for k in range(K):
        d[:, k, k:k + T] = codes[:, k]
    return d.transpose(1, 2)


def build_model(model):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    m = MusicgenForConditionalGeneration.from_pretrained(HF_NAMES[model])
    proc = AutoProcessor.from_pretrained(HF_NAMES[model])
    m.config.decoder.decoder_start_token_id = m.generation_config.decoder_start_token_id
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.decoder.parameters():
        p.requires_grad_(True)
    return m, proc


def load(run, step=None, dev="cuda"):
    meta = json.load(open(os.path.join(run, "run.json")))
    model, proc = build_model(meta["model"])
    step = meta["save_steps"][-1] if step is None else step
    sd = torch.load(os.path.join(run, f"step{step}.pt"), map_location="cpu")
    assert not model.decoder.load_state_dict(sd, strict=False).unexpected_keys
    return model.to(dev).eval(), proc, step


def params(model):
    return [p for p in model.decoder.parameters() if p.requires_grad]


def train_loss(model, proc, codes, texts, dev="cuda"):
    tok = proc.tokenizer(list(texts), padding=True, return_tensors="pt").to(dev)
    return model(input_ids=tok.input_ids, attention_mask=tok.attention_mask,
                 labels=delay(codes, codes.shape[1])).loss


def item_loss(model, proc, codes, i, text):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return train_loss(model, proc, torch.from_numpy(codes[i:i + 1].astype(np.int64)).cuda(), [text])
