# Functional Dependency beyond Similarity: Hierarchical Decomposition of Training Data Attribution in Music Generative Models

[![arXiv](https://img.shields.io/badge/arXiv-2611.08839-b31b1b.svg)](https://arxiv.org/abs/2611.08839)

Copyright analysis focuses on the similarity between two works. However, generative AI models learn from training data through functional relations beyond similarity. Training Data Attribution (TDA) estimates the influence of each training instance on a generated output and may serve as a basis for remuneration. However, prior studies find global influence, where a few training instances rank as the most influential for almost every generation. Used directly, TDA scores would distribute part of every remuneration to these instances regardless of what was generated. To address this, we propose a new approach in which TDA scores are decomposed into three levels of influence, namely global indicating the influence of a training data to a trained model, group for its influence on a group of generated outputs (that could be clustered together), and specific for its influence on a single generated output. Our study finds that, across several music generative models and TDA methods, global- and group-level influence is non-negligible in every case. Based on these observations, we propose the term functional dependency for the relation of training data to a trained model or to its outputs. This notion both encompasses and distinguishes dependencies of different scopes, allowing TDA scores to be interpreted as distinct forms of contribution in legal, economic and ethical discussions on remuneration.

The code splits a TDA score matrix into Global, Group and Specific parts, for MusicGen and InspireMusic fine-tuned on
MTG-Jamendo. Products under `--root` are listed in `docs/layout.md`. Run everything from the repository root.

## Setup

```bash
pip install -r requirements/musicgen.txt        # data, musicgen
pip install -r requirements/inspiremusic.txt    # inspire
pip install -r requirements/analysis.txt        # decomp
pip install sjlt==0.1.7 --no-build-isolation    # FactGraSS
bash inspire/setup.sh ext                       # InspireMusic code and weights
export PYTHONPATH=$PWD:$PWD/ext/InspireMusic:$PWD/ext/InspireMusic/third_party/Matcha-TTS
```

## Data

```bash
python -m data.prepare      --jamendo JAMENDO_REPO --root R
python -m musicgen.tokenize --root R --audio JAMENDO_AUDIO
python -m inspire.tokenize  --root R --audio JAMENDO_AUDIO --weights ext/InspireMusic-Base
```

## Models

`M` is `musicgen-small` or `musicgen-large`. For InspireMusic use `inspire.<step>` with
`M` = `inspiremusic-base` or `inspiremusic-1.5b` and `--weights ext/InspireMusic-Base` or `ext/InspireMusic-1.5B`.

```bash
for s in 0 1 2; do python -m musicgen.train --root R --model M --seed $s; done
python -m musicgen.generate --root R --model M

for s in 0 1 2; do
  python -m musicgen.grads --root R --model M --seed $s --target catalogue
  python -m musicgen.grads --root R --model M --seed $s --target outputs
done
for t in 1875 3750 5625; do
  python -m musicgen.grads --root R --model M --step $t --target catalogue
  python -m musicgen.grads --root R --model M --step $t --target outputs
done
python -m musicgen.grads --root R --model M --target outputs --method logra
python -m musicgen.grads --root R --model M --target outputs --method factgrass
```

## Deletion

```bash
python -m decomp deletion-sets --root R --model M --method graddot
python -m decomp deletion-sets --root R --model M --method trak_lam1
cd R/M/deletion; sets=$(ls */exclude/*.txt | sed 's|/exclude/|/|; s|.txt||'); cd -
for set in $sets; do
  for s in 0 1 2; do
    python -m musicgen.train   --root R --model M --seed $s --deletion $set
    python -m musicgen.measure --root R --model M --seed $s --deletion $set
  done
done
```

## Analysis

```bash
python -m decomp scores   --root R --model M
python -m decomp features --root R --model M
for c in shares ksweep agreement drivers criteria composition seedrep; do
  python -m decomp $c --root R --model M
done
python -m decomp deletion-effects --root R --model M --method graddot
python -m decomp deletion-effects --root R --model M --method trak_lam1
python -m decomp heatmap --root R
python -m decomp dataset --root R --jamendo JAMENDO_REPO
```

## Paper

[arXiv:2611.08839](https://arxiv.org/abs/2611.08839)
