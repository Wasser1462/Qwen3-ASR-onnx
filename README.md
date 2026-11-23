# Export onnx model

```bash 
    bash run.sh
```

# Decode

```bash
    bash decode.sh
```

# Decoder logits modes

The exporter defaults to `--logits-mode all`, which returns logits for every
input position (`[batch, seq, vocab]`). Use `last` to apply the vocabulary
projection only to the final normalized hidden state and return
`[batch, 1, vocab]`. Both modes run the full Transformer and return full KV
deltas (`[batch, seq, kv_heads, head_dim]`) for every layer. Single-token
decoding has the same output shape in both modes.

Export into separate, new directories, using the same local checkpoint:

```bash
python export_qwen3_asr_onnx.py --model "$QWEN3_ASR_CHECKPOINT" \
  --outdir exports/all --device cpu --max-total-len 512 \
  --logits-mode all --no-int8 --verify
python export_qwen3_asr_onnx.py --model "$QWEN3_ASR_CHECKPOINT" \
  --outdir exports/last --device cpu --max-total-len 512 \
  --logits-mode last --no-int8 --verify
```

Omit `--no-int8` to also produce dynamically quantized models. The quantization
settings are unchanged. Slicing before the output projection can change its
dynamic activation scale, so INT8 logits need not equal the final row of an
`all` export. Validate recognition accuracy on the intended data and hardware.
Use the same conv frontend, encoder and tokenizer when comparing the modes.

Last-position models require a sherpa-onnx runtime that accepts a one-row
prefill logits tensor. Older runtimes that require a row for every prompt
position are incompatible. Default `all` exports retain the previous layout.
ONNX inputs, output names, opset 17, IR 9 and external-data support are unchanged.

Run the default regression tests without a checkpoint or network access:

```bash
python -m pytest tests/test_decoder_logits_mode.py -q
```

These tests instantiate the real Qwen3-ASR model classes with a small,
randomly initialized configuration. They check projection input length,
logits shape, full-layer KV deltas, default/all equivalence, and actual
all/last ONNX exports and ONNX Runtime executions at multiple input lengths.
The small model tests validate behavior, not recognition accuracy.

Pass a local Qwen3-ASR-0.6B checkpoint to also run the same model tests
with pretrained weights:

```bash
python -m pytest tests/test_decoder_logits_mode.py \
  --checkpoint "$QWEN3_ASR_CHECKPOINT" -q
```

Only the pretrained-checkpoint cases are skipped without `--checkpoint`.
FP32 all/last logits use `atol=rtol=1e-4`; KV uses `atol=rtol=1e-5`, with
finite values required. The exporter's `--verify` checks finite ONNX outputs
and matching ONNX/PyTorch decoder logits shapes before reporting differences.
It remains an additional smoke check without an enforced numerical tolerance
and does not replace the regression assertions.

# C++ Reference
 - https://github.com/k2-fsa/sherpa-onnx

# Acknowledgements
 - Thanks to the QwenLM team. A large portion of the code is from https://github.com/QwenLM/Qwen3-ASR
