"""Fine-tuning catalogue, held-out prompt pool and prompts from the MTG-Jamendo split-0 metadata."""
import argparse
import os

import numpy as np
import pandas as pd

GENRES = ["ambient", "electronic", "easylistening", "classical", "folk", "alternative",
          "hiphop", "chillout", "dance", "pop", "soundtrack", "funk"]
SEG_SEC = 10.0
N_SEG = 3
PROMPTS_PER_GENRE = 8
SEEDS_PER_PROMPT = 5
FIRST_SEED = 100000


def read_split(tsv):
    rows = []
    with open(tsv) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            tags = {"genre": [], "instrument": [], "mood/theme": []}
            for t in p[5:]:
                k, v = t.split("---")
                tags[k].append(v)
            rows.append(dict(track_id=p[0], artist_id=p[1], album_id=p[2], path=p[3].replace(".mp3", ".low.mp3"),
                             duration=float(p[4]), genre=tags["genre"], instrument=tags["instrument"],
                             moodtheme=tags["mood/theme"]))
    return pd.DataFrame(rows)


def describe(genre, instrument, moodtheme):
    parts = []
    if genre:
        parts.append(", ".join(genre) + ".")
    if instrument:
        parts.append("instruments: " + ", ".join(instrument) + ".")
    if moodtheme:
        parts.append("mood: " + ", ".join(moodtheme) + ".")
    return " ".join(parts) if parts else "music."


def segments(tracks):
    rows = []
    for r in tracks.itertuples():
        for k, c in enumerate((np.arange(N_SEG) + 1) / (N_SEG + 1)):
            off = max(0.0, min(c * r.duration - SEG_SEC / 2, r.duration - SEG_SEC - 0.5))
            rows.append(dict(seg_id=f"{r.track_id}_s{k}", track_id=r.track_id, artist_id=r.artist_id,
                             album_id=r.album_id, path=r.path, offset=round(off, 2), genre=";".join(r.genre),
                             instrument=";".join(r.instrument), moodtheme=";".join(r.moodtheme),
                             text=describe(r.genre, r.instrument, r.moodtheme)))
    df = pd.DataFrame(rows).drop_duplicates(subset=["track_id", "offset"])
    return df.sort_values("seg_id").reset_index(drop=True)


def select(train, test):
    rng = np.random.RandomState(0)
    first = rng.choice(sorted(train.track_id), 2000, replace=False)
    held = rng.choice(sorted(test.track_id), 500, replace=False)
    rest = sorted(set(train.track_id) - set(first))
    more = np.random.RandomState(1).choice(rest, 8000, replace=False)
    return train[train.track_id.isin(set(first) | set(more))], test[test.track_id.isin(set(held))]


def prompts(heldout):
    texts = heldout.text.drop_duplicates()
    first = texts.str.replace(".", ",", regex=False).str.split(",").str[0].str.strip()
    rows = []
    for g in GENRES:
        for p, t in enumerate(texts[first == g].sample(n=PROMPTS_PER_GENRE, random_state=1)):
            rows.append(dict(prompt_id=f"{g}_{p}", genre=g, prompt=t))
    return pd.DataFrame(rows)


def design(root):
    """Every prompt x SEEDS_PER_PROMPT sampling seeds."""
    p = pd.read_csv(os.path.join(root, "data", "prompts.csv"))
    df = p.loc[p.index.repeat(SEEDS_PER_PROMPT), ["genre", "prompt_id", "prompt"]].reset_index(drop=True)
    df["seed"] = FIRST_SEED + np.arange(len(df))
    df.insert(0, "gen_id", [f"g{i:04d}" for i in range(len(df))])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jamendo", required=True, help="mtg-jamendo-dataset repository")
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    split = os.path.join(a.jamendo, "data/splits/split-0")
    cat, held = select(read_split(os.path.join(split, "autotagging-train.tsv")),
                       read_split(os.path.join(split, "autotagging-test.tsv")))
    out = os.path.join(a.root, "data")
    os.makedirs(out, exist_ok=True)
    cat, held = segments(cat), segments(held)
    cat.to_csv(os.path.join(out, "catalogue.csv"), index=False)
    held.to_csv(os.path.join(out, "heldout.csv"), index=False)
    prompts(held).to_csv(os.path.join(out, "prompts.csv"), index=False)
    print(f"catalogue {len(cat)} segments of {cat.track_id.nunique()} tracks, "
          f"held-out {len(held)} segments of {held.track_id.nunique()} tracks")


if __name__ == "__main__":
    main()
