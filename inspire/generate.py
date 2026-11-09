"""Generate the outputs (prompts x sampling seeds) with the fine-tuned LLM; tokens are saved, audio rendered at 48 kHz."""
import argparse

import numpy as np
import soundfile as sf
import torch

from data.prepare import design
from inspire.common import (CHORUS_INTRO, PAD, RATE, SEC, Renderer, add_args, build_llm, load_checkpoint, prompt_text,
                            run_dir)


@torch.inference_mode()
def sample(llm, tok, text, seed, dev="cuda"):
    torch.manual_seed(seed)
    t = torch.tensor([tok.encode(prompt_text(text), allowed_special="all")], dtype=torch.int32, device=dev)
    z = torch.zeros(1, 0, dtype=torch.int32, device=dev)
    zl = torch.tensor([0], dtype=torch.int32, device=dev)
    emb = [torch.tensor([0.0], device=dev), torch.tensor([SEC], device=dev), torch.tensor([CHORUS_INTRO], device=dev)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = llm.inference(text=t, text_len=torch.tensor([t.shape[1]], dtype=torch.int32, device=dev),
                            audio_token=None, audio_token_len=None, prompt_text=z, prompt_text_len=zl,
                            prompt_audio_token=z, prompt_audio_token_len=zl, embeddings=emb,
                            duration_to_gen=SEC, task="text-to-music", token_rate=RATE)
        return np.array([x.item() for x in out], dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    a = ap.parse_args()
    out = a.root / a.model / "outputs"
    (out / "wav").mkdir(parents=True, exist_ok=True)
    df = design(a.root)
    cfg, llm, tok = build_llm(a.weights)
    load_checkpoint(llm, run_dir(a.root, a.model, 0))
    render = Renderer(cfg, a.weights)
    T = int(SEC * RATE)
    codes = np.full((len(df), T), PAD, np.int16)
    for i, r in df.iterrows():
        c = sample(llm, tok, r.prompt, int(r.seed))[:T]
        codes[i, :len(c)] = c
        sf.write(out / "wav" / f"{r.gen_id}.wav", render(torch.from_numpy(c.astype(np.int64))), render.sr)
    np.save(out / "codes.npy", codes)
    df.to_csv(out / "outputs.csv", index=False)


if __name__ == "__main__":
    main()
