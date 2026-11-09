"""Figure 1 (score-cluster deletion heatmaps) and the fine-tuning-dataset genre figure with its statistics."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

MODEL_NAMES = {"musicgen-small": "MusicGen-S", "musicgen-large": "MusicGen-L", "inspiremusic-base": "InspireMusic-Base",
               "inspiremusic-1.5b": "InspireMusic-1.5B"}
METHOD_NAMES = {"graddot": "Grad-Dot", "trak_lam1": "TRAK λ=1"}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def heatmap(root, panels):
    root = Path(root)
    fig, axs = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.0), squeeze=False)
    for a, (model, method) in zip(axs[0], panels):
        doc = json.loads((root / f"analysis/{model}/deletion_{method}.json").read_text())
        D = np.array(doc["cluster"]["matrix"]) * 1e3
        k = len(D)
        v = float(np.abs(D).max())
        im = a.imshow(D, cmap="RdBu", vmin=-v, vmax=v)
        for i in range(k):
            a.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, lw=1.2, ec="black"))
        a.set_xticks(range(k), [f"C{i}" for i in range(k)], fontsize=6, rotation=90)
        a.set_yticks(range(k), [f"C{i}" for i in range(k)], fontsize=6)
        a.set_xlabel("Outputs of score cluster", fontsize=8)
        a.set_ylabel("Deleted set (score cluster)", fontsize=8)
        a.set_title(f"{MODEL_NAMES[model]}, {METHOD_NAMES[method]}", fontsize=9)
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        cb.set_label("ΔLL ×10³ vs. random deletion", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, root / "figures/heatmap")


def tag_counts(lists):
    return pd.Series([t for x in lists for t in x]).value_counts()


def dataset(root, jamendo, top):
    from data.prepare import read_split
    root = Path(root)
    split = Path(jamendo) / "data/splits/split-0"
    allt = pd.concat([read_split(split / f"autotagging-{s}.tsv") for s in ("train", "validation", "test")])
    cat = pd.read_csv(root / "data/catalogue.csv", keep_default_na=False)
    held = pd.read_csv(root / "data/heldout.csv", keep_default_na=False)
    prompts = pd.read_csv(root / "data/prompts.csv")
    tr = cat.drop_duplicates("track_id")
    genres = tr.genre.str.split(";").map(lambda x: [t for t in x if t])
    g, ga = tag_counts(genres), tag_counts(allt.genre)
    per_artist = tr.groupby("artist_id").size()
    both = pd.concat({"catalogue": g / len(tr), "mtg": ga / len(allt)}, axis=1).fillna(0.0)
    diff = (both.catalogue - both.mtg).abs()
    doc = {"catalogue": {"segments": len(cat), "tracks": len(tr), "artists": int(tr.artist_id.nunique()),
                         "albums": int(tr.album_id.nunique()),
                         "tracks_per_artist": {"median": float(per_artist.median()), "mean": float(per_artist.mean()),
                                               "max": int(per_artist.max()),
                                               "share_of_top_10_artists": float(per_artist.nlargest(10).sum() / len(tr))},
                         "genre_tags_per_track": genres.map(len).value_counts().sort_index().to_dict(),
                         "genre_tags": g.to_dict(),
                         "prompt_genres_tracks": {x: int(g.get(x, 0)) for x in sorted(prompts.genre.unique())}},
           "heldout_pool": {"segments": len(held), "tracks": int(held.track_id.nunique()),
                            "distinct_texts": int(held.text.nunique()),
                            "tracks_shared_with_catalogue": len(set(held.track_id) & set(tr.track_id)),
                            "artists_shared_with_catalogue": len(set(held.artist_id) & set(tr.artist_id))},
           "vs_mtg_jamendo": {"mtg_tracks": len(allt), "genre_tags": len(ga),
                              "pearson_r": float(np.corrcoef(both.catalogue, both.mtg)[0, 1]),
                              "max_abs_diff_pct_points": float(100 * diff.max()), "max_diff_tag": str(diff.idxmax()),
                              "mean_abs_diff_pct_points": float(100 * diff.mean())}}
    tags = ga.head(top).index
    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.bar(x - 0.2, 100 * both.loc[tags, "mtg"], 0.4, color="#bbbbbb", label=f"MTG-Jamendo, all {len(allt):,} tracks")
    ax.bar(x + 0.2, 100 * both.loc[tags, "catalogue"], 0.4, color="#4c72b0", label=f"Fine-tuning dataset, {len(tr):,} tracks")
    ax.set_xticks(x, tags, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Tracks with the tag (%)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, root / "figures/dataset_genres")
    return doc
