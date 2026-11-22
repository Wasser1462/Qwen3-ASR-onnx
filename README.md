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

Run the regression tests with a local Qwen3-ASR-0.6B checkpoint:

```bash
python -m pytest tests/test_decoder_logits_mode.py \
  --checkpoint "$QWEN3_ASR_CHECKPOINT" -q
```

Without `--checkpoint`, model-dependent tests are skipped. With it, tests
compare all logits/KV outputs in PyTorch and actual ONNX Runtime executions
for multiple input lengths. FP32 all/last logits use `atol=rtol=1e-4`; KV uses
`atol=rtol=1e-5`, with finite values required. The exporter's `--verify` output
is an additional smoke check and does not replace these assertions.

# C++ Reference
 - https://github.com/k2-fsa/sherpa-onnx

# Acknowledgements
 - Thanks to the QwenLM team. A large portion of the code is from https://github.com/QwenLM/Qwen3-ASR
