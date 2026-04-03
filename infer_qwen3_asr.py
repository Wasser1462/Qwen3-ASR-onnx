#!/usr/bin/env python3
#
# Qwen3-ASR ONNX inference. Supports FP32/INT8 with ConvFrontend in Python.

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile
import scipy.signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoProcessor, AutoTokenizer


def _feat_to_audio_tokens_len_np(
    feat_len: np.ndarray, chunk_size: int = 100
) -> np.ndarray:
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
        from qwen3_asr import Qwen3ASRConfig, Qwen3ASRProcessor
        from transformers import AutoConfig, AutoProcessor

        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
        return True
    except Exception as e:
        print(f"[warn] _register_qwen3_asr: {e}")
        return False


def _make_sess(path: str, device: str = "cpu") -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def _make_sess_with_fallback(
    path: str, device: str, fp32_fallback: Optional[str] = None
) -> ort.InferenceSession:
    try:
        return _make_sess(path, device=device)
    except Exception as e:
        msg = str(e)
        if fp32_fallback and ("ConvInteger" in msg or "NOT_IMPLEMENTED" in msg):
            print(
                f"[warn] load failed for {path}: {msg}\n       fallback -> {fp32_fallback}"
            )
            return _make_sess(fp32_fallback, device=device)
        raise


def _infer_cache_meta(
    dec_sess: ort.InferenceSession,
) -> Tuple[int, Optional[int], Optional[int], Optional[int]]:
    inps = {i.name: i for i in dec_sess.get_inputs()}
    keys = sorted(
        [n for n in inps if n.startswith("cache_key_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not keys:
        raise RuntimeError("decoder inputs missing cache_key_*")

    L = len(keys)
    s = inps[keys[0]].shape
    max_total_len = int(s[1]) if isinstance(s[1], int) else None
    kv = int(s[2]) if isinstance(s[2], int) else None
    hd = int(s[3]) if isinstance(s[3], int) else None
    return L, max_total_len, kv, hd


def _load_audio_any(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        wav = np.load(path)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav

    rate, data = scipy.io.wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if data.dtype == np.int16:
        wav = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        wav = data.astype(np.float32) / 2147483648.0
    else:
        wav = data.astype(np.float32)

    if rate != 16000:
        wav = scipy.signal.resample_poly(wav, 16000, rate).astype(np.float32)

    wav = np.clip(wav, -1.0, 1.0)
    return wav


def _check_model_dir(model_dir: str) -> None:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(
            f"Model dir has no config.json: {os.path.abspath(model_dir)}\n"
            "Set --model to the Qwen3-ASR checkpoint path."
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
    for kwargs in (
        dict(
            trust_remote_code=True,
            use_slow_tokenizer=True,
            fix_mistral_regex=True,
        ),
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
    for kwargs in (
        dict(trust_remote_code=True, fix_mistral_regex=True),
        dict(trust_remote_code=True),
    ):
        try:
            return AutoProcessor.from_pretrained(model_dir, **kwargs)
        except TypeError:
            continue
    return AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)


def _resolve_audio_token_and_id(tok, proc) -> Tuple[str, int]:
    candidates: List[str] = []
    for holder in (getattr(proc, "tokenizer", None), tok):
        if holder is None:
            continue
        v = getattr(holder, "audio_token", None)
        if isinstance(v, str) and v:
            candidates.append(v)
    candidates.append("<|audio_pad|>")

    seen = set()
    unk_id = getattr(tok, "unk_token_id", None)

    for token in candidates:
        if token in seen:
            continue
        seen.add(token)

        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, (int, np.integer)) and int(tid) >= 0:
            if unk_id is None or int(tid) != int(unk_id) or token == tok.unk_token:
                return token, int(tid)

        ids = tok.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return token, int(ids[0])

    raise RuntimeError("Cannot resolve audio token id")


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--model", type=str, required=True, help="Qwen3-ASR checkpoint path"
    )
    p.add_argument(
        "--conv_frontend",
        type=str,
        required=True,
        help="conv_frontend.onnx path",
    )
    p.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="encoder.onnx or encoder.int8.onnx",
    )
    p.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="decoder.onnx or decoder.int8.onnx",
    )
    p.add_argument(
        "--wav",
        type=str,
        nargs="+",
        required=True,
        help=".wav or .npy paths",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--max-total-len", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = get_args()

    _check_model_dir(args.model)
    _register_qwen3_asr()

    tok = _load_tokenizer(args.model)
    proc = _load_processor(args.model)
    audio_token, audio_token_id = _resolve_audio_token_and_id(tok, proc)

    enc_fp32_guess = (
        args.encoder.replace(".int8.onnx", ".onnx")
        if args.encoder.endswith(".int8.onnx")
        else None
    )
    enc = _make_sess_with_fallback(
        args.encoder, device=args.device, fp32_fallback=enc_fp32_guess
    )

    dec_fp32_guess = (
        args.decoder.replace(".int8.onnx", ".onnx")
        if args.decoder.endswith(".int8.onnx")
        else None
    )
    dec = _make_sess_with_fallback(
        args.decoder, device=args.device, fp32_fallback=dec_fp32_guess
    )

    conv_sess = _make_sess(args.conv_frontend, device=args.device)

    wav_paths = list(args.wav)
    wavs = [_load_audio_any(p) for p in wav_paths]
    max_samples = max(int(w.shape[0]) for w in wavs)
    if max_samples <= 0:
        raise RuntimeError("empty audio after load")
    wavs_pad = [
        np.pad(w, (0, max_samples - int(w.shape[0])), mode="constant")
        for w in wavs
    ]
    total_audio_sec = sum(max(len(w) / 16000.0, 1e-6) for w in wavs)

    audio_inputs = proc.feature_extractor(
        wavs_pad,
        sampling_rate=16000,
        padding="max_length",
        max_length=max_samples,
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    input_features = np.asarray(
        audio_inputs["input_features"], dtype=np.float32
    )
    feat_mask = np.asarray(audio_inputs["attention_mask"], dtype=np.int32)

    if args.debug:
        print(
            "batch:",
            len(wavs),
            "max_samples:",
            max_samples,
            "wav[0] range:",
            float(wavs[0].min()),
            float(wavs[0].max()),
            "len[0]:",
            wavs[0].shape[0],
        )
        print("input_features:", input_features.shape, input_features.dtype)
        print(
            "feature_attention_mask:",
            feat_mask.shape,
            feat_mask.dtype,
            "sum:",
            int(feat_mask.sum()),
        )

    mel_input = input_features.transpose(0, 2, 1)
    conv_inputs = {"input_features": mel_input}
    (conv_output_np,) = conv_sess.run(["conv_output"], conv_inputs)

    valid = feat_mask != 0
    feat_len_np = valid.sum(axis=1).astype(np.int64)
    a_len = _feat_to_audio_tokens_len_np(
        feat_len_np, chunk_size=args.chunk_size
    )
    A_conv = int(conv_output_np.shape[1])
    pos = np.arange(A_conv, dtype=np.int64).reshape(1, A_conv)
    tok_mask = pos < a_len.reshape(-1, 1)

    if args.debug:
        print("conv_output:", conv_output_np.shape, conv_output_np.dtype)
        print(
            "tok_mask:",
            tok_mask.shape,
            tok_mask.dtype,
            "sum:",
            int(tok_mask.sum()),
        )

    enc_inputs = {
        "input_features": conv_output_np,
        "feature_attention_mask": tok_mask.astype(np.bool_),
    }
    (audio_features,) = enc.run(["audio_features"], enc_inputs)
    audio_features = np.asarray(audio_features, dtype=np.float32)

    if audio_features.ndim != 3:
        raise RuntimeError(
            f"unexpected audio_features shape: {audio_features.shape}"
        )

    B = int(audio_features.shape[0])
    if B != len(wav_paths):
        raise RuntimeError(
            f"batch size mismatch: wavs={len(wav_paths)}, encoder B={B}"
        )

    a_uni = np.unique(a_len)
    if a_uni.size != 1:
        raise RuntimeError(
            "equal-length batch required; a_len=" + str(a_len.tolist())
        )
    A = int(a_uni[0])
    if A <= 0:
        raise RuntimeError(f"invalid audio token length: {A}")

    if A > int(audio_features.shape[1]):
        raise RuntimeError(
            f"audio token length overflow: A={A}, encoder_out={audio_features.shape[1]}"
        )

    audio_features = audio_features[:, :A, :]
    B, A, H = audio_features.shape

    if args.debug:
        print("audio_features:", audio_features.shape, audio_features.dtype)

    system_text = "<|im_start|>system\n<|im_end|>\n"
    user_text = (
        f"<|im_start|>user\n<|audio_start|>{audio_token * A}<|audio_end|><|im_end|>\n"
    )
    assistant_text = "<|im_start|>assistant\n"
    full_prompt = system_text + user_text + assistant_text

    enc_one = tok.encode(full_prompt, add_special_tokens=False)
    S0 = len(enc_one)
    input_ids = np.tile(np.asarray(enc_one, dtype=np.int64), (B, 1))

    slots = (input_ids == audio_token_id).sum(axis=1)
    if not np.all(slots == A):
        raise RuntimeError(
            f"audio slot mismatch: slots={slots.tolist()} A={A}"
        )

    L, model_max_total_len, kv, hd = _infer_cache_meta(dec)
    if kv is None or hd is None:
        raise RuntimeError("decoder cache shape has dynamic kv/hd")

    runtime_max_total_len = (
        int(model_max_total_len)
        if model_max_total_len is not None
        else int(args.max_total_len)
    )

    required_total_len = S0 + int(args.max_new_tokens)
    if required_total_len > runtime_max_total_len:
        raise RuntimeError(
            f"max_total_len not enough: S0({S0}) + max_new_tokens({args.max_new_tokens}) = "
            f"{required_total_len}, but max_total_len={runtime_max_total_len}"
        )

    if args.debug:
        print("S0:", S0)
        print("A:", A)
        print("required_total_len:", required_total_len)
        print("runtime_max_total_len:", runtime_max_total_len)
        print("num_layers:", L, "kv:", kv, "hd:", hd)

    caches: List[np.ndarray] = []
    for _ in range(L):
        caches.append(
            np.zeros((B, runtime_max_total_len, kv, hd), dtype=np.float32)
        )
        caches.append(
            np.zeros((B, runtime_max_total_len, kv, hd), dtype=np.float32)
        )

    dec_out_names = [o.name for o in dec.get_outputs()]
    if "logits" not in dec_out_names:
        raise RuntimeError(f"decoder outputs missing logits")

    def _run_decoder(step_input_ids: np.ndarray, cur_len: int) -> np.ndarray:
        Sb = int(step_input_ids.shape[0])
        S = int(step_input_ids.shape[1])
        if Sb != B:
            raise RuntimeError(
                f"step_input_ids batch {Sb} != B {B}"
            )
        if cur_len + S > runtime_max_total_len:
            raise RuntimeError(
                f"cur_len overflow: {cur_len}+{S} > {runtime_max_total_len}"
            )

        cache_pos = np.arange(cur_len, cur_len + S, dtype=np.int64)
        attn1 = np.ones((1, S), dtype=np.int64)

        def _run_batched() -> np.ndarray:
            attn_mask = np.ones((B, S), dtype=np.int64)
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
                caches[2 * i][:, cur_len : cur_len + S] = kd
                caches[2 * i + 1][:, cur_len : cur_len + S] = vd
            return logits

        def _run_one(bi: int) -> np.ndarray:
            feed: Dict[str, np.ndarray] = {
                "input_ids": step_input_ids[bi : bi + 1],
                "audio_features": audio_features[bi : bi + 1],
                "attention_mask": attn1,
                "cache_position": cache_pos,
            }
            for i in range(L):
                feed[f"cache_key_{i}"] = caches[2 * i][bi : bi + 1]
                feed[f"cache_value_{i}"] = caches[2 * i + 1][bi : bi + 1]
            outs = dec.run(dec_out_names, feed)
            out_map = {name: val for name, val in zip(dec_out_names, outs)}
            logits_b = np.asarray(out_map["logits"], dtype=np.float32)
            for i in range(L):
                kd = np.asarray(out_map[f"key_delta_{i}"], dtype=np.float32)
                vd = np.asarray(out_map[f"value_delta_{i}"], dtype=np.float32)
                caches[2 * i][bi : bi + 1, cur_len : cur_len + S] = kd
                caches[2 * i + 1][bi : bi + 1, cur_len : cur_len + S] = vd
            return logits_b

        if B == 1:
            return _run_batched()
        try:
            return _run_batched()
        except Exception:
            return np.concatenate(
                [_run_one(bi) for bi in range(B)], axis=0
            )

    cur_len = 0
    logits = _run_decoder(input_ids, cur_len)
    cur_len += S0

    eos_id = tok.eos_token_id
    out_rows: List[List[int]] = [[] for _ in range(B)]

    infer_start_time = time.time()

    next_ids = np.argmax(logits[:, -1, :], axis=-1).astype(np.int64)
    for b in range(B):
        out_rows[b].append(int(next_ids[b]))
    active = np.ones(B, dtype=np.bool_)
    if eos_id is not None:
        active &= next_ids != int(eos_id)

    for _ in range(int(args.max_new_tokens) - 1):
        if not bool(np.any(active)):
            break
        step_ids = np.empty((B, 1), dtype=np.int64)
        for b in range(B):
            if active[b]:
                step_ids[b, 0] = out_rows[b][-1]
            else:
                step_ids[b, 0] = (
                    int(eos_id) if eos_id is not None else out_rows[b][-1]
                )
        logits = _run_decoder(step_ids, cur_len)
        cur_len += 1
        next_ids = np.argmax(logits[:, -1, :], axis=-1).astype(np.int64)
        for b in range(B):
            if not active[b]:
                continue
            out_rows[b].append(int(next_ids[b]))
            if eos_id is not None and int(next_ids[b]) == int(eos_id):
                active[b] = False

    for b in range(B):
        text = tok.decode(out_rows[b], skip_special_tokens=True)
        text = text.replace("\ufffd", "")
        prefix = f"[{wav_paths[b]}] " if B > 1 else ""
        print(prefix + text)

    processing_time = time.time() - infer_start_time
    rtf = processing_time / total_audio_sec
    print(f"RTF: {rtf:.4f}")


if __name__ == "__main__":
    main()
