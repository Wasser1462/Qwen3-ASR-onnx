#!/bin/bash
set -e

cd "$(dirname "$0")"

# Hardcoded paths (edit here if needed)
Mode="0.6B"      # 0.6B or 1.7B
MODEL_DIR="./model/tokenizer"
WAV_PATH="./test_wavs/rag_chemistry.wav"

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
    --device cuda \
    --max-new-tokens 100

echo ""
echo "FP32 with context "
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.onnx \
    --decoder ./model/model_$Mode/decoder.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --context "酯" \
    --device cuda \
    --max-new-tokens 100

echo ""
echo "INT8 Encoder + FP32 Decoder"
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.int8.onnx \
    --decoder ./model/model_$Mode/decoder.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --device cuda \
    --max-new-tokens 100

echo ""
echo "INT8 Inference"
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.onnx \
    --decoder ./model/model_$Mode/decoder.int8.onnx \
    --model ${MODEL_DIR} \
    --wav ${WAV_PATH} \
    --device cpu \
    --max-new-tokens 100

BATCH_WAVS=(
  ./test_wavs/dia_minnan.wav
  ./test_wavs/dia_sh.wav
  ./test_wavs/dia_yue.wav
  ./test_wavs/far_2.wav
  ./test_wavs/far_3.wav
  ./test_wavs/far_4.wav
  ./test_wavs/far_5.wav
  ./test_wavs/ja_en_codeswitch.wav
  ./test_wavs/ja.wav
  ./test_wavs/lyrics_2.wav
  ./test_wavs/lyrics_3.wav
  ./test_wavs/lyrics_en_1.wav
  ./test_wavs/lyrics_en_2.wav
  ./test_wavs/lyrics_en_3.wav
  ./test_wavs/lyrics.wav
  ./test_wavs/noise_en.wav
  ./test_wavs/vietnamese.wav
)
for f in "${BATCH_WAVS[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Skip batch: missing $f"
    exit 1
  fi
done

echo ""
echo "Batch FP32 Inference"
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.onnx \
    --decoder ./model/model_$Mode/decoder.onnx \
    --model ${MODEL_DIR} \
    --wav "${BATCH_WAVS[@]}" \
    --device cuda \
    --max-new-tokens 100

echo ""
echo "Batch INT8 Encoder + FP32 Decoder"
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.int8.onnx \
    --decoder ./model/model_$Mode/decoder.onnx \
    --model ${MODEL_DIR} \
    --wav "${BATCH_WAVS[@]}" \
    --device cuda \
    --max-new-tokens 100

echo ""
echo "Batch INT8 Inference"
python3 infer_qwen3_asr.py \
    --conv_frontend ./model/model_$Mode/conv_frontend.onnx \
    --encoder ./model/model_$Mode/encoder.onnx \
    --decoder ./model/model_$Mode/decoder.int8.onnx \
    --model ${MODEL_DIR} \
    --wav "${BATCH_WAVS[@]}" \
    --device cpu \
    --max-new-tokens 100
