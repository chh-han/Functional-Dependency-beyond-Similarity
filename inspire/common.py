"""InspireMusic LLM: loading, training loss, data and paths."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODELS = ("inspiremusic-base", "inspiremusic-1.5b")
RATE = 75
SEC = 10.0
PAD = -1
CHORUS_INTRO = 0


def add_args(ap):
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--weights", type=Path, required=True, help="release directory (InspireMusic-Base or -1.5B)")


def prompt_text(text):
    return f"<|0.0|><|intro|><|{text}|><|{SEC}|>"


def build_llm(weights, dev="cuda"):
    import transformers
    from hyperpyyaml import load_hyperpyyaml

    orig = transformers.Qwen2ForCausalLM.from_pretrained.__func__

    def sdpa(cls, *a, **k):
        k["attn_implementation"] = "sdpa"
        k.pop("device_map", None)
        return orig(cls, *a, **k)

    transformers.Qwen2ForCausalLM.from_pretrained = classmethod(sdpa)
    with open(weights / "inspiremusic.yaml") as f:
        cfg = load_hyperpyyaml(f, overrides={"basemodel_path": f"{weights}/",
                                             "generator_path": f"{weights}/music_tokenizer"})
    llm = cfg["llm"]
    llm.load_state_dict(torch.load(weights / "llm.pt", map_location="cpu", weights_only=True))
    # InspireMusic reads the backbone's last hidden state, never its text logits: keep one position of them.
    fwd = llm.llm.model.forward
    llm.llm.model.forward = lambda *x, **k: fwd(*x, **{"num_logits_to_keep": 1, **k})
    llm = llm.float().to(dev).eval()
    llm.dtype = torch.float32
    llm.train_cfg_ratio = 0.0
    return cfg, llm, cfg["get_tokenizer"]()


def trainable(llm):
    return [(n, p) for n, p in llm.named_parameters() if p.requires_grad]


def load_checkpoint(llm, run, step=None):
    step = step or json.loads((run / "run.json").read_text())["save_steps"][-1]
    sd = torch.load(run / f"step{step}.pt", map_location="cpu", weights_only=True)
    assert {n for n, _ in trainable(llm)} <= set(sd), "checkpoint lacks trainable tensors"
    llm.load_state_dict(sd, strict=False)
    return step


def make_batch(tok, texts, codes, dev, drop=None):
    drop = drop or [False] * len(texts)
    tt = [torch.tensor([], dtype=torch.long) if d else
          torch.tensor(tok.encode(prompt_text(t), allowed_special="all"), dtype=torch.long)
          for t, d in zip(texts, drop)]
    tl = torch.tensor([len(t) for t in tt], dtype=torch.int32)
    tt = torch.nn.utils.rnn.pad_sequence(tt, batch_first=True, padding_value=0)
    if tt.shape[1] == 0:
        tt = torch.zeros(len(texts), 1, dtype=torch.long)
    cl = torch.tensor([len(c) for c in codes], dtype=torch.int32)
    cc = torch.nn.utils.rnn.pad_sequence([c.long() for c in codes], batch_first=True, padding_value=0)
    B = len(texts)
    return {"text_token": tt.to(dev), "text_token_len": tl.to(dev),
            "semantic_token": cc.to(dev), "semantic_token_len": cl.to(dev),
            "time_start": torch.zeros(B, 1, device=dev), "time_end": torch.full((B, 1), SEC, device=dev),
            "chorus": torch.full((B, 1), CHORUS_INTRO, dtype=torch.long, device=dev)}


def train_loss(llm, tok, codes, texts, dev="cuda", drop=None):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return llm(make_batch(tok, texts, codes, dev, drop=drop), dev)["loss"]


def unpad(row):
    row = np.asarray(row)
    return torch.from_numpy(row[row != PAD].astype(np.int64))


def load_set(root, name):
    codes = np.load(root / "codes/wavtokenizer" / f"{name}.npy", mmap_mode="r")
    return pd.read_csv(root / "data" / f"{name}.csv"), codes


def load_outputs(root, model):
    d = root / model / "outputs"
    return pd.read_csv(d / "outputs.csv"), np.load(d / "codes.npy")


def run_dir(root, model, seed, deletion=None):
    if deletion is None:
        return root / model / "finetune" / f"seed{seed}"
    method, name = deletion.split("/")
    return root / model / "deletion" / method / "runs" / name / f"seed{seed}"


class Renderer:
    def __init__(self, cfg, weights, dev="cuda"):
        from inspiremusic.music_tokenizer.vqvae import VQVAE
        self.flow = cfg["flow"]
        self.flow.load_state_dict(torch.load(weights / "flow.pt", map_location="cpu", weights_only=True))
        self.flow.to(dev).eval()
        mt = weights / "music_tokenizer"
        self.mt = VQVAE(str(mt / "config.json"), str(mt / "model.pt"), with_encoder=True).to(dev).eval()
        self.sr, self.dev = cfg.get("target_sample_rate", 48000), dev

    @torch.inference_mode()
    def __call__(self, codes):
        emb = self.flow.inference(token=codes.view(1, -1).to(self.dev),
                                  token_len=torch.tensor([len(codes)], dtype=torch.int32, device=self.dev),
                                  sample_rate=self.sr)
        return self.mt.generator(emb).reshape(-1).float().cpu().numpy()


def grad_list(scalar, params):
    gs = torch.autograd.grad(scalar, params, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(gs, params)]

