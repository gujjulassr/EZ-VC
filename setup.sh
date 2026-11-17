#!/bin/bash
# EZ-VC in its own plain venv inside this folder, the README's steps in one go: Python 3.10 or newer, torch 2.4 cu124,
# this repo with its BigVGAN submodule, the espnet fork for XEUS. The weights (SPRINGLab/EZ-VC) are gated on the hub:
# accept the terms on the model page and log in once when this script says so, then run it again. Ends with a smoke
# test, the repo's example pair converted on one card by batch_infer.py, so a broken install shows here.
# HOW TO RUN (in this folder, once):   bash setup.sh          (PY=/path/to/python3.10  CARD=0 in front to change)
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE" || exit 1
export HF_HUB_DISABLE_XET=1          # hub downloads over plain HTTP: the xet transfer stalls without progress on this network
CARD=${CARD:-0}
PY=${PY:-$(command -v python3.10 || command -v python3.11 || command -v python3.12 || command -v python3)}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' 2> /dev/null \
    || { echo "STOP: EZ-VC needs Python 3.10 or newer; $PY is older. Give one in front:  PY=/path/to/python3.10 bash setup.sh"; exit 1; }
V=$HERE/venv/bin
echo "== 1/4 the venv -> $HERE/venv ($PY)"
[ -x "$V/python" ] || "$PY" -m venv venv || exit 1
"$V/pip" install -q --upgrade pip
echo "== 2/4 torch 2.4 cu124, this repo and its submodule, the espnet fork for XEUS (several GB, a few minutes)"
"$V/pip" install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 || exit 1
git submodule update --init --recursive || exit 1
"$V/pip" install -e . || exit 1
"$V/pip" install 'espnet @ git+https://github.com/wanchichen/espnet.git@ssl' || exit 1
"$V/pip" install -q soundfile librosa joblib cached_path huggingface_hub
echo "== 3/4 the weights (gated on the hub)"
"$V/python" -c "from huggingface_hub import hf_hub_download; hf_hub_download('SPRINGLab/EZ-VC', 'vocab.txt')" 2> /dev/null \
    || { echo "STOP: no access to SPRINGLab/EZ-VC yet. Once: open https://huggingface.co/SPRINGLab/EZ-VC, accept the terms, then  $V/hf auth login  (paste a token), and run this script again"; exit 1; }
echo "== 4/4 smoke test on card $CARD: the repo's example pair (the first run downloads XEUS, the k-means and BigVGAN)"
EX=src/f5_tts/infer/examples/wavs; mkdir -p smoke
printf '%s\t%s\t%s\n' "$EX/237_134493_000015_000004.wav" "$EX/14_208_000042_000000.wav" smoke.wav > smoke/manifest.tsv
(cd src/f5_tts/infer && CUDA_VISIBLE_DEVICES=$CARD "$V/python" batch_infer.py --manifest "$HERE/smoke/manifest.tsv" --out "$HERE/smoke" --overwrite) || exit 1
echo "== EZ-VC ready: $HERE/smoke/smoke.wav is the example pair converted. See 'Running on a manifest' in README.md."
