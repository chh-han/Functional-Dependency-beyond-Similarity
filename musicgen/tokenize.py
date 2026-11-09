"""EnCodec codes of the catalogue and held-out segments: ROOT/codes/encodec/{catalogue,heldout}.npy (N, 4, 500) int16."""
import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

from data.prepare import SEG_SEC

SR, BS = 32000, 32


def read(path, offset):
    with sf.SoundFile(path) as f:
        f.seek(int(offset * f.samplerate))
        w = f.read(int(SEG_SEC * f.samplerate), dtype="float32", always_2d=True).mean(1)
        w = AF.resample(torch.from_numpy(w), f.samplerate, SR)
    n = int(round(SEG_SEC * SR))
    assert len(w) > 0.9 * n, path
    return torch.nn.functional.pad(w, (0, max(0, n - len(w))))[:n].numpy()


@torch.no_grad()
def encode(df, audio, out):
    from transformers import AutoProcessor, EncodecModel
    enc = EncodecModel.from_pretrained("facebook/encodec_32khz").cuda().eval()
    proc = AutoProcessor.from_pretrained("facebook/encodec_32khz")
    codes = None
    for s in range(0, len(df), BS):
        wavs = [read(os.path.join(audio, r.path), r.offset) for r in df.iloc[s:s + BS].itertuples()]
        inp = proc(raw_audio=wavs, sampling_rate=SR, return_tensors="pt").to("cuda")
        c = enc.encode(inp["input_values"], inp["padding_mask"]).audio_codes[0]
        if codes is None:
            codes = np.lib.format.open_memmap(out, mode="w+", dtype=np.int16, shape=(len(df), *c.shape[1:]))
        codes[s:s + len(wavs)] = c.cpu().numpy().astype(np.int16)
    codes.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--audio", required=True, help="MTG-Jamendo audio-low directory")
    a = ap.parse_args()
    out = os.path.join(a.root, "codes", "encodec")
    os.makedirs(out, exist_ok=True)
    for name in ("catalogue", "heldout"):
        encode(pd.read_csv(os.path.join(a.root, "data", f"{name}.csv")), a.audio, os.path.join(out, f"{name}.npy"))


if __name__ == "__main__":
    main()
