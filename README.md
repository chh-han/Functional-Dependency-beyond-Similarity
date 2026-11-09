# TDA score decomposition

Splits a training-data attribution score matrix into Global, Group and Specific parts, for MusicGen and InspireMusic
fine-tuned on MTG-Jamendo. Products under `--root` are listed in `docs/layout.md`. Run everything from the repository root.

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
