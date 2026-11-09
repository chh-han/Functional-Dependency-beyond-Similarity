"""Generate the outputs with the seed-0 fine-tuned model and re-encode them with EnCodec."""
import argparse
import os

import numpy as np
import soundfile as sf
import torch

from data.prepare import design
from musicgen.common import HF_NAMES, load, run_dir

BS, SEC, GUIDANCE, TOP_K = 8, 10.0, 3.0, 50


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", choices=list(HF_NAMES), required=True)
    a = ap.parse_args()

    df = design(a.root)
    model, proc, _ = load(run_dir(a.root, a.model, 0))
    out = os.path.join(a.root, a.model, "outputs")
    os.makedirs(os.path.join(out, "wav"), exist_ok=True)
    sr = model.config.audio_encoder.sampling_rate
    n_tok, n_samp = int(SEC * model.config.audio_encoder.frame_rate), int(SEC * sr)
    codes = []
    for s in range(0, len(df), BS):
        b = df.iloc[s:s + BS]
        torch.manual_seed(int(b.seed.iloc[0]))
        inp = proc(text=b.prompt.tolist(), padding=True, return_tensors="pt").to("cuda")
        audio = model.generate(**inp, do_sample=True, top_k=TOP_K, guidance_scale=GUIDANCE,
                               max_new_tokens=n_tok)[:, :, :n_samp]
        audio = torch.nn.functional.pad(audio, (0, n_samp - audio.shape[-1]))
        c = model.audio_encoder.encode(audio, torch.ones_like(audio[:, 0], dtype=torch.long)).audio_codes[0]
        codes.append(c.cpu().numpy().astype(np.int16))
        for j, gid in enumerate(b.gen_id):
            sf.write(os.path.join(out, "wav", f"{gid}.wav"), audio[j, 0].float().cpu().numpy(), sr)
        print(f"{s + len(b)}/{len(df)}", flush=True)
    np.save(os.path.join(out, "codes.npy"), np.concatenate(codes))
    df.to_csv(os.path.join(out, "outputs.csv"), index=False)


if __name__ == "__main__":
    main()
