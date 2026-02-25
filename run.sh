#!/bin/bash
set -e

# Set MODEL_DIR to Qwen3-ASR checkpoint path, e.g. export MODEL_DIR=/path/to/Qwen3-ASR-0.6B
MODEL_DIR="/home/ec_user/workspace/work/test/Qwen3-ASR-0.6B"
OUTDIR="${OUTDIR:-./model}"

rm -rf "${OUTDIR:?}"/*

python3 export_qwen3_asr_onnx.py \
  --model "${MODEL_DIR}" \
  --outdir "${OUTDIR}" \
  --device cpu \
  --max-total-len 512 \
  --verify

echo ""

rm -rf "${OUTDIR}"/onnx_* 2>/dev/null || true
rm -rf "${OUTDIR}"/thinker.model.* 2>/dev/null || true

echo "Export done! Models:"
ls -lh "${OUTDIR}"/*.onnx "${OUTDIR}"/*.data 2>/dev/null || true
