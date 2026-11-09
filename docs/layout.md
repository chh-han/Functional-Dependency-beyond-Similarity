# Data layout

All commands take `--root <ROOT>`. Every product lives under it.

```
ROOT/
  data/
    catalogue.csv                 fine-tuning segments: seg_id, track_id, artist_id, album_id, path, offset, genre, instrument, moodtheme, text
    heldout.csv                   held-out prompt pool (test tracks)
    prompts.csv                   prompt_id, genre, prompt
  codes/
    encodec/                      MusicGen tokens: catalogue.npy, heldout.npy (n, 4, 500) int16, rows as in data/
    wavtokenizer/                 InspireMusic tokens: catalogue.npy, heldout.npy (n, 750) int16
  <model>/                        musicgen-small | musicgen-large | inspiremusic-base | inspiremusic-1.5b
    finetune/seed<s>/             run.json (lr, warmup, total_steps, save_steps, ...), log.csv, checkpoints step<t>.pt
    outputs/                      outputs.csv (gen_id, genre, prompt_id, prompt, seed), wav/<gen_id>.wav,
                                  codes.npy (MusicGen: EnCodec codes of the wavs; InspireMusic: generated tokens, padded with -1)
    grads/catalogue/seed<s>_step<t>/   phi.npy (n_catalogue, k), rows.csv (idx, seg_id, track_id, artist_id)
    grads/outputs/seed<s>_step<t>/     psi.npy (n_outputs, k), gen_loss.npy
    scores/<method>.npy                (n_catalogue, n_outputs); LoGra and FactGraSS are written by grads, the rest are cached by the analysis
    features/features.npz              mfcc, chroma, spectrogram (per output)
    deletion/<method>/                 sets.json, exclude/<set>.txt
    deletion/<method>/runs/<set>/seed<s>/
    deletion/<method>/measure/<set>_seed<s>.csv   columns: set, seed, gen_id, value (mean per-token log-likelihood)
    deletion/random/exclude/random.txt, deletion/random/runs/random/seed<s>/, deletion/random/measure/random_seed<s>.csv
  analysis/<model>/<command>.json  JSON outputs of the analyses (deletion-effects: deletion_<method>.json)
  analysis/dataset.json
  figures/                        heatmap.{png,pdf}, dataset_genres.{png,pdf}
```

`<method>` is one of graddot, gradcos, tracin, logra, factgrass, trak_lam1, trak_lam0.1, trak_lam0.01, trak_ens.
`deletion/<method>/baseline.json` (optional) maps a set to another random run, e.g. {"global": "random_rebase"}; default `random`.
Deletion `<set>` is `global`, `cluster<c>` (c = 0..11) or `specific_<prompt_id>`.

The outputs are the 96 prompts x 5 sampling seeds (100000 + row index).
