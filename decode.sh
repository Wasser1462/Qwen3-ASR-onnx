#!/bin/bash
set -e

cd "$(dirname "$0")"

# Hardcoded paths (edit here if needed)
Mode="1.7B"      # 0.6B or 1.7B
MODEL_DIR="./model/tokenizer"
WAV_PATH="./test_wavs/lyrics.wav"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "Error: ${MODEL_DIR}/config.json not found."
    exit 1
fi
if [ ! -f "$WAV_PATH" ]; then
    echo "Error: WAV not found: $WAV_PATH"
    exit 1
fi

echo "Audio: $WAV_PATH"


echo ""
echo "FP32 Inference"
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.onnx \
    --decoder ./model/model_$Mode/decoder.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --max-new-tokens 100

echo ""
echo "INT8 Encoder + FP32 Decoder"
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.int8.onnx \
    --decoder ./model/model_$Mode/decoder.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --max-new-tokens 100

echo ""
echo "INT8 Inference"
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.onnx \
    --decoder ./model/model_$Mode/decoder.int8.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --max-new-tokens 100
