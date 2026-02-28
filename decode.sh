#!/bin/bash
set -e

cd "$(dirname "$0")"

# Hardcoded paths (edit here if needed)
MODEL_DIR="./Qwen3-ASR-0.6B"
WAV_PATH="./Qwen3-ASR-onnx/test_wavs/far_3.wav"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "Error: ${MODEL_DIR}/config.json not found."
    exit 1
fi
if [ ! -f "$WAV_PATH" ]; then
    echo "Error: WAV not found: $WAV_PATH"
    exit 1
fi

echo "Audio: $WAV_PATH"

# FP32 inference
echo ""
echo "=== FP32 Inference ==="
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/conv_frontend.onnx \
    --encoder ./model/encoder.onnx \
    --decoder ./model/decoder.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --max-new-tokens 100

# INT8 encoder + FP32 decoder
echo ""
echo "=== INT8 Encoder + FP32 Decoder ==="
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/conv_frontend.onnx \
    --encoder ./model/encoder.int8.onnx \
    --decoder ./model/decoder.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --max-new-tokens 100

# INT8 decoder
echo ""
echo "=== INT8 Decoder ==="
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/conv_frontend.onnx \
    --encoder ./model/encoder.onnx \
    --decoder ./model/decoder.int8.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --max-new-tokens 100
