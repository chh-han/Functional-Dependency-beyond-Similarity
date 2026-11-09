"""WavTokenizer codes of the catalogue and the held-out prompt pool."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from inspire.common import SEC

SR = 24000


class Segments(torch.utils.data.Dataset):
    def __init__(self, df, audio):
        self.df, self.audio = df, audio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import librosa
        r = self.df.iloc[i]
        n = int(SEC * SR)
        y, _ = librosa.load(str(self.audio / r.path), sr=SR, mono=True, offset=float(r.offset), duration=SEC)
        return torch.from_numpy(np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True, help="MTG-Jamendo audio directory")
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    from inspiremusic.wavtokenizer.decoder.pretrained import WavTokenizer
    d = a.weights / "wavtokenizer"
    wt = WavTokenizer.from_pretrained_feat(str(d / "config.yaml"), str(d / "model.pt")).to("cuda")
    out = a.root / "codes/wavtokenizer"
    out.mkdir(parents=True, exist_ok=True)
    bw = torch.tensor([0], device="cuda")
    for name in ("catalogue", "heldout"):
        df = pd.read_csv(a.root / "data" / f"{name}.csv")
        dl = torch.utils.data.DataLoader(Segments(df, a.audio), batch_size=32, num_workers=a.workers)
        codes = []
        with torch.inference_mode():
            for y in dl:
                codes.append(wt.encode_infer(y.to("cuda"), bandwidth_id=bw)[1][0].cpu().numpy().astype(np.int16))
        codes = np.concatenate(codes)
        assert len(codes) == len(df) and 0 <= codes.min() and codes.max() < 4096
        np.save(out / f"{name}.npy", codes)
        print(name, codes.shape, flush=True)


if __name__ == "__main__":
    main()
