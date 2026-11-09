#!/usr/bin/env bash
# Upstream InspireMusic code and both releases into DIR:
#   DIR/InspireMusic, DIR/InspireMusic-Base, DIR/InspireMusic-1.5B
set -euo pipefail
D=${1:?usage: setup.sh DIR}
mkdir -p "$D"
git clone https://github.com/FunAudioLLM/InspireMusic.git "$D/InspireMusic"
git -C "$D/InspireMusic" checkout 0aefb55
git -C "$D/InspireMusic" submodule update --init third_party/Matcha-TTS
git -C "$D/InspireMusic/third_party/Matcha-TTS" checkout bd4d90d
pip install sjlt==0.1.7 --no-build-isolation
for r in InspireMusic-Base:iic/InspireMusic InspireMusic-1.5B:iic/InspireMusic-1.5B; do
  python -c "from modelscope import snapshot_download; snapshot_download('${r#*:}', local_dir='$D/${r%%:*}')"
done
# The Base release ships its 48 kHz music_tokenizer weights with a 24 kHz config; the 1.5B config matches them.
cp "$D/InspireMusic-1.5B/music_tokenizer/config.json" "$D/InspireMusic-Base/music_tokenizer/config.json"
