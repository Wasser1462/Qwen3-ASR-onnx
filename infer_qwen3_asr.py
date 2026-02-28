#!/usr/bin/env python3
#
# Qwen3-ASR ONNX inference. Supports FP32/INT8 with ConvFrontend in Python.

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoProcessor, AutoTokenizer


def _feat_to_audio_tokens_len_np(feat_len: np.ndarray, chunk_size: int = 100) -> np.ndarray:
    """Audio token count from feature length (numpy, no torch). Same formula as conv_frontend."""
    def _conv_out_len_3x_stride2(n: int) -> int:
        x = (int(n) + 1) // 2
        x = (x + 1) // 2
        return (x + 1) // 2

    def _aftercnn(x: np.ndarray) -> np.ndarray:
        x = (x - 1) // 2 + 1
        x = (x - 1) // 2 + 1
        return (x - 1) // 2 + 1

    cs = int(chunk_size)
    n = np.asarray(feat_len, dtype=np.int64)
    full = n // cs
    rem = n % cs
    tn = _conv_out_len_3x_stride2(cs)
    out = full * tn + _aftercnn(rem)
    return np.maximum(out, 0).astype(np.int64)


def _register_qwen3_asr() -> bool:
    try:
        from transformers_backend_infer import Qwen3ASRConfig, Qwen3ASRProcessor
        from transformers import AutoConfig, AutoProcessor
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
        return True
    except Exception as e:
        print(f"[warn] _register_qwen3_asr: {e}")
        return False


def _make_sess(path: str, device: str = "cpu") -> ort.InferenceSession:
    """Create ONNX inference session."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def _make_sess_with_fallback(path: str, device: str, fp32_fallback: Optional[str] = None) -> ort.InferenceSession:
    """Create session with FP32 fallback for INT8 failures."""
    try:
        return _make_sess(path, device=device)
    except Exception as e:
        msg = str(e)
        if fp32_fallback and ("ConvInteger" in msg or "NOT_IMPLEMENTED" in msg):
            print(f"[warn] load failed for {path}: {msg}\n       fallback -> {fp32_fallback}")
            return _make_sess(fp32_fallback, device=device)
        raise


def _infer_cache_meta(dec_sess: ort.InferenceSession) -> Tuple[int, Optional[int], Optional[int], Optional[int]]:
    """Infer cache metadata from decoder inputs."""
    inps = {i.name: i for i in dec_sess.get_inputs()}
    keys = sorted([n for n in inps if n.startswith("cache_key_")], key=lambda x: int(x.split("_")[-1]))
    if not keys:
        raise RuntimeError("decoder inputs missing cache_key_*")

    L = len(keys)
    s = inps[keys[0]].shape
    max_total_len = int(s[1]) if isinstance(s[1], int) else None
    kv = int(s[2]) if isinstance(s[2], int) else None
    hd = int(s[3]) if isinstance(s[3], int) else None
    return L, max_total_len, kv, hd


def _load_audio_any(path: str) -> np.ndarray:
    """Load audio file (wav or npy), resample to 16k mono."""
    if path.endswith(".npy"):
        wav = np.load(path)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav

    rate, data = scipy.io.wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if data.dtype == np.int16:
        wav = (data.astype(np.float32) / 32768.0)
    elif data.dtype == np.int32:
        wav = (data.astype(np.float32) / 2147483648.0)
    else:
        wav = data.astype(np.float32)

    if rate != 16000:
        wav = scipy.signal.resample_poly(wav, 16000, rate).astype(np.float32)

    wav = np.clip(wav, -1.0, 1.0)
    return wav


def _check_model_dir(model_dir: str) -> None:
    """Ensure model_dir is a Qwen3-ASR checkpoint (config.json with model_type qwen3_asr)."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(
            f"Model dir has no config.json: {os.path.abspath(model_dir)}\n"
            "Set --model to the Qwen3-ASR checkpoint path (e.g. /path/to/Qwen3-ASR-0.6B)."
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Failed to read {config_path}: {e}") from e
    if config.get("model_type") != "qwen3_asr":
        raise ValueError(
            f"config.json model_type is {config.get('model_type')!r}, expected 'qwen3_asr'.\n"
            f"Set --model to the Qwen3-ASR checkpoint path: {os.path.abspath(model_dir)}"
        )


def _load_tokenizer(model_dir: str):
    """Load tokenizer with fallback options."""
    for kwargs in (
        dict(trust_remote_code=True, use_slow_tokenizer=True, fix_mistral_regex=True),
        dict(trust_remote_code=True, fix_mistral_regex=True),
        dict(trust_remote_code=True, use_slow_tokenizer=True),
        dict(trust_remote_code=True),
    ):
        try:
            return AutoTokenizer.from_pretrained(model_dir, **kwargs)
        except TypeError:
            continue
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _load_processor(model_dir: str):
    """Load processor with fallback options."""
    for kwargs in (
        dict(trust_remote_code=True, fix_mistral_regex=True),
        dict(trust_remote_code=True),
    ):
        try:
            return AutoProcessor.from_pretrained(model_dir, **kwargs)
        except TypeError:
            continue
    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)


def _trim_audio_features(audio_features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Trim padding tokens based on energy threshold."""
    if audio_features.ndim != 3:
        return audio_features
    B, A, H = audio_features.shape
    if B != 1:
        return audio_features
    energy = np.max(np.abs(audio_features[0]), axis=-1)
    idx = np.where(energy > eps)[0]
    if idx.size == 0:
        return audio_features
    A_valid = int(idx[-1] + 1)
    return audio_features[:, :A_valid, :]


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=str, required=True, help="Qwen3-ASR checkpoint path")
    p.add_argument("--conv_frontend", type=str, required=True, help="conv_frontend.onnx path")
    p.add_argument("--encoder", type=str, required=True, help="encoder.onnx or encoder.int8.onnx")
    p.add_argument("--decoder", type=str, required=True, help="decoder.onnx or decoder.int8.onnx")
    p.add_argument("--wav", type=str, required=True, help=".wav or .npy (16k mono preferred)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--max-total-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = get_args()

    _check_model_dir(args.model)
    _register_qwen3_asr()

    tok = _load_tokenizer(args.model)
    proc = _load_processor(args.model)

    enc = _make_sess(args.encoder.replace(".int8.onnx", ".onnx"), device=args.device)
    dec_fp32_guess = args.decoder.replace(".int8.onnx", ".onnx") if args.decoder.endswith(".int8.onnx") else None
    dec = _make_sess_with_fallback(args.decoder, device=args.device, fp32_fallback=dec_fp32_guess)

    conv_sess = _make_sess(args.conv_frontend, device=args.device)

    wav = _load_audio_any(args.wav)

    audio_inputs = proc.feature_extractor(
        [wav],
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    input_features = np.asarray(audio_inputs["input_features"], dtype=np.float32)
    feat_mask = np.asarray(audio_inputs["attention_mask"], dtype=np.int32)

    if args.debug:
        print("wav range:", float(wav.min()), float(wav.max()), "len:", wav.shape[0])
        print("input_features:", input_features.shape, input_features.dtype)
        print("feature_attention_mask:", feat_mask.shape, feat_mask.dtype, "sum:", int(feat_mask.sum()))

    # Run conv_frontend ONNX
    mel_input = input_features.transpose(0, 2, 1)  # (B,F,T) -> (B,T,F)
    conv_inputs = {"input_features": mel_input}
    conv_outputs = conv_sess.run(["conv_output"], conv_inputs)
    conv_output_np = conv_outputs[0]

    valid = feat_mask != 0
    feat_len_np = valid.sum(axis=1).astype(np.int64)
    a_len = _feat_to_audio_tokens_len_np(feat_len_np, chunk_size=100)
    A = int(conv_output_np.shape[1])
    pos = np.arange(A, dtype=np.int64).reshape(1, A)
    tok_mask = pos < a_len.reshape(-1, 1)

    if args.debug:
        print("conv_output:", conv_output_np.shape, conv_output_np.dtype)
        print("tok_mask:", tok_mask.shape, tok_mask.dtype, "sum:", int(tok_mask.sum()))

    enc_inputs = {
        "input_features": conv_output_np,
        "feature_attention_mask": tok_mask.astype(np.bool_),
    }
    (audio_features,) = enc.run(["audio_features"], enc_inputs)
    audio_features = np.asarray(audio_features, dtype=np.float32)
    audio_features = _trim_audio_features(audio_features)

    B, A, H = audio_features.shape
    if args.debug:
        print("audio_features:", audio_features.shape, audio_features.dtype)

    system_text = "<|im_start|>system\n<|im_end|>\n"
    user_text = f"<|im_start|>user\n<|audio_start|>{'<|audio_pad|>' * A}<|audio_end|><|im_end|>\n"
    assistant_text = "<|im_start|>assistant\n"

    full_prompt = system_text + user_text + assistant_text
    input_ids = np.asarray([tok.encode(full_prompt, add_special_tokens=False)], dtype=np.int64)
    S0 = int(input_ids.shape[1])

    L, _, kv, hd = _infer_cache_meta(dec)
    if kv is None or hd is None:
        raise RuntimeError("decoder cache shape has dynamic kv/hd")

    max_total_len = int(args.max_total_len)
    if S0 >= max_total_len:
        raise RuntimeError(f"prompt too long: S0={S0} >= max_total_len={max_total_len}")

    caches: List[np.ndarray] = []
    for _ in range(L):
        caches.append(np.zeros((B, max_total_len, kv, hd), dtype=np.float32))
        caches.append(np.zeros((B, max_total_len, kv, hd), dtype=np.float32))

    dec_out_names = [o.name for o in dec.get_outputs()]
    if "logits" not in dec_out_names:
        raise RuntimeError(f"decoder outputs missing logits")

    def _run_decoder(step_input_ids: np.ndarray, cur_len: int) -> np.ndarray:
        S = int(step_input_ids.shape[1])
        if cur_len + S > max_total_len:
            raise RuntimeError(f"cur_len overflow: {cur_len}+{S} > {max_total_len}")

        attn_mask = np.ones((B, S), dtype=np.int64)
        cache_pos = np.arange(cur_len, cur_len + S, dtype=np.int64)

        feed: Dict[str, np.ndarray] = {
            "input_ids": step_input_ids,
            "audio_features": audio_features,
            "attention_mask": attn_mask,
            "cache_position": cache_pos,
        }
        for i in range(L):
            feed[f"cache_key_{i}"] = caches[2 * i]
            feed[f"cache_value_{i}"] = caches[2 * i + 1]

        outs = dec.run(dec_out_names, feed)
        out_map = {name: val for name, val in zip(dec_out_names, outs)}
        logits = np.asarray(out_map["logits"], dtype=np.float32)

        for i in range(L):
            kd = np.asarray(out_map[f"key_delta_{i}"], dtype=np.float32)
            vd = np.asarray(out_map[f"value_delta_{i}"], dtype=np.float32)
            caches[2 * i][:, cur_len:cur_len + S] = kd
            caches[2 * i + 1][:, cur_len:cur_len + S] = vd

        return logits

    cur_len = 0
    logits = _run_decoder(input_ids, cur_len)
    cur_len += S0

    eos_id = tok.eos_token_id
    out_ids: List[int] = []

    next_id = int(np.argmax(logits[0, -1], axis=-1))
    out_ids.append(next_id)

    for _ in range(int(args.max_new_tokens) - 1):
        if eos_id is not None and out_ids[-1] == int(eos_id):
            break
        step_ids = np.asarray([[out_ids[-1]]], dtype=np.int64)
        logits = _run_decoder(step_ids, cur_len)
        cur_len += 1
        tid = int(np.argmax(logits[0, -1], axis=-1))
        out_ids.append(tid)

    text = tok.decode(out_ids, skip_special_tokens=True)
    text = text.replace("\ufffd", "")
    print(text)


if __name__ == "__main__":
    main()
