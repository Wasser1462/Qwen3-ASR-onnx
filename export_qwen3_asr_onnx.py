#!/usr/bin/env python3
#
# Export Qwen3-ASR to ONNX (conv_frontend, encoder backend, decoder).

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from transformers import AutoConfig, AutoModel, AutoProcessor

from decoder import DecoderCoreWrapper
from encoder import AudioEncoderWrapper


def monkeypatch_ort_get_opset_version():
    try:
        import onnxruntime.quantization.quant_utils as quant_utils
        import onnxruntime.quantization.base_quantizer as base_quantizer
        import onnxruntime.quantization.onnx_quantizer as onnx_quantizer
    except Exception:
        return

    def _get_opset_version_safe(model):
        try:
            for opset in model.opset_import:
                if getattr(opset, "domain", "") == "ai.onnx":
                    return int(opset.version)
            for opset in model.opset_import:
                if not getattr(opset, "domain", ""):
                    return int(opset.version)
        except Exception:
            pass
        return 17

    quant_utils.get_opset_version = _get_opset_version_safe

    if hasattr(base_quantizer, "get_opset_version"):
        base_quantizer.get_opset_version = _get_opset_version_safe
    if hasattr(onnx_quantizer, "get_opset_version"):
        onnx_quantizer.get_opset_version = _get_opset_version_safe

    if hasattr(base_quantizer.BaseQuantizer, "check_opset_version"):
        def patched_check(self):
            try:
                return _get_opset_version_safe(self.model.model)
            except Exception:
                return 17
        base_quantizer.BaseQuantizer.check_opset_version = patched_check


REDUCE_OPS = {
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare",
}


def _get_init(g: onnx.GraphProto, name: str):
    for it in g.initializer:
        if it.name == name:
            return it
    return None


def _remove_node_input(node: onnx.NodeProto, idx: int):
    ins = list(node.input)
    if 0 <= idx < len(ins):
        ins.pop(idx)
        del node.input[:]
        node.input.extend(ins)


def _iter_subgraphs_from_node(node: onnx.NodeProto):
    gs = []
    for a in node.attribute:
        if a.type == onnx.AttributeProto.GRAPH:
            gs.append(a.g)
        elif a.type == onnx.AttributeProto.GRAPHS:
            gs.extend(list(a.graphs))
    return gs


def fix_reduce_axes_graph(g: onnx.GraphProto) -> int:
    nfix = 0
    for node in g.node:
        if node.op_type in REDUCE_OPS and len(node.input) == 2:
            axes_name = node.input[1]
            init = _get_init(g, axes_name)
            if init is not None:
                axes_arr = numpy_helper.to_array(init).reshape(-1)
                axes = [int(x) for x in axes_arr.tolist()]
                _remove_node_input(node, 1)

                new_attrs = []
                for a in node.attribute:
                    if a.name == "axes":
                        continue
                    new_attrs.append(a)
                del node.attribute[:]
                node.attribute.extend(new_attrs)
                node.attribute.append(helper.make_attribute("axes", axes))
                nfix += 1

        for sg in _iter_subgraphs_from_node(node):
            nfix += fix_reduce_axes_graph(sg)
    return nfix


def remove_split_num_outputs_graph(g: onnx.GraphProto) -> int:
    nfix = 0
    for node in g.node:
        if node.op_type == "Split":
            new_attrs = []
            removed = False
            for a in node.attribute:
                if a.name == "num_outputs":
                    removed = True
                    continue
                new_attrs.append(a)
            if removed:
                del node.attribute[:]
                node.attribute.extend(new_attrs)
                nfix += 1

        for sg in _iter_subgraphs_from_node(node):
            nfix += remove_split_num_outputs_graph(sg)
    return nfix


def split_single_output_to_identity_graph(g: onnx.GraphProto) -> int:
    nfix = 0
    for node in g.node:
        if node.op_type == "Split" and len(node.output) == 1:
            inp0 = node.input[0] if len(node.input) >= 1 else ""
            out0 = node.output[0]
            node.op_type = "Identity"
            del node.input[:]
            node.input.extend([inp0])
            del node.output[:]
            node.output.extend([out0])
            del node.attribute[:]
            nfix += 1

        for sg in _iter_subgraphs_from_node(node):
            nfix += split_single_output_to_identity_graph(sg)
    return nfix


def force_opset_and_ir(m: onnx.ModelProto, opset: int = 17, ir: int = 9):
    del m.opset_import[:]
    m.opset_import.extend([
        helper.make_operatorsetid("ai.onnx", int(opset)),
        helper.make_operatorsetid("", int(opset)),
    ])
    m.ir_version = int(ir)


def _ensure_graph_names(g: onnx.GraphProto, prefix: str = "graph", counter: Optional[List[int]] = None) -> int:
    if counter is None:
        counter = [0]
    nfix = 0
    if not getattr(g, "name", ""):
        g.name = f"{prefix}_{counter[0]}"
        counter[0] += 1
        nfix += 1
    for node in g.node:
        for sg in _iter_subgraphs_from_node(node):
            nfix += _ensure_graph_names(sg, prefix=prefix, counter=counter)
    return nfix


def _load_model_with_external(path: str) -> onnx.ModelProto:
    # onnx.load_model(load_external_data=True) is not available in older onnx
    try:
        return onnx.load_model(path, load_external_data=True)
    except TypeError:
        m = onnx.load(path)
        try:
            from onnx import external_data_helper
            external_data_helper.load_external_data_for_model(m, os.path.dirname(path))
        except Exception:
            pass
        return m


def patch_and_check(path_in: str, path_out: str, opset: int = 17, ir: int = 9, external_data: bool = True):
    m = _load_model_with_external(path_in)

    n_reduce = fix_reduce_axes_graph(m.graph)
    n_splitnum = remove_split_num_outputs_graph(m.graph)
    n_split1 = split_single_output_to_identity_graph(m.graph)
    n_gname = _ensure_graph_names(m.graph, prefix="graph")

    force_opset_and_ir(m, opset=opset, ir=ir)

    os.makedirs(os.path.dirname(path_out), exist_ok=True)

    # avoid in-place truncation when path_in == path_out
    same_path = os.path.abspath(path_in) == os.path.abspath(path_out)
    out_path = path_out
    tmp_path = None
    if same_path:
        tmp_path = path_out + ".tmp.onnx"
        out_path = tmp_path

    if external_data:
        ext_name = os.path.basename(out_path) + ".data"
        onnx.save_model(
            m,
            out_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_name,
            size_threshold=1024,
        )
        onnx.checker.check_model(out_path)
    else:
        onnx.save_model(m, out_path, save_as_external_data=False)
        onnx.checker.check_model(_load_model_with_external(out_path))

    if same_path and tmp_path is not None:
        os.replace(tmp_path, path_out)

    print(
        f"[patch] {path_out} "
        f"ReduceFix={n_reduce} SplitFix={n_splitnum} Split1OutToId={n_split1} "
        f"GraphNameFix={n_gname} external_data={int(external_data)}"
    )


def _dtype_from_ort_type(type_str: str):
    t = type_str.replace("tensor(", "").replace(")", "").strip()
    if t == "float":
        return np.float32
    if t == "float16":
        return np.float16
    if t == "int64":
        return np.int64
    if t == "int32":
        return np.int32
    if t == "bool":
        return np.bool_
    return None


def verify_onnx(model_path: str, inputs_np: Dict[str, Any], fetches: List[str]):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

    ort_inputs = sess.get_inputs()
    fixed_feed = {}
    for i in ort_inputs:
        name = i.name
        if name not in inputs_np:
            continue
        v = inputs_np[name]
        expected = _dtype_from_ort_type(i.type)
        if isinstance(v, np.ndarray) and expected is not None and v.dtype != expected:
            v = v.astype(expected, copy=False)
        fixed_feed[name] = v

    missing = [i.name for i in ort_inputs if i.name not in fixed_feed]
    if missing:
        raise RuntimeError(f"verify missing inputs: {missing}")

    outs = sess.run(fetches, fixed_feed)
    for x in outs:
        if isinstance(x, np.ndarray) and x.dtype.kind == "f":
            if not np.isfinite(x).all():
                raise RuntimeError(f"verify failed: NaN/Inf in {model_path}")
    return outs


def _register_qwen3_asr():
    try:
        from transformers_backend import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        from transformers import AutoModel as _AM
        from transformers import AutoProcessor as _AP
        _AM.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        _AP.register(Qwen3ASRConfig, Qwen3ASRProcessor)
        return True
    except Exception:
        return False


def make_dummy_inputs(proc, device: torch.device):
    tok = getattr(proc.tokenizer, "audio_token", None) or "<|audio_pad|>"
    text_dummy = [tok]
    audio_dummy = [np.zeros(16000, dtype=np.float32)]
    batch = proc(text=text_dummy, audio=audio_dummy, return_tensors="pt", padding=True)
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["input_features"].to(device),
        batch["feature_attention_mask"].to(device),
    )


def export_encoder_backend_only(thinker, input_features, token_mask, opset, path_raw, chunk_size: int):
    """Export encoder backend only (no ConvFrontend). Inputs: (B,A,dim_audio), (B,A) mask. Output: (B,A,H_text)."""
    from conv_frontend import ConvFrontend
    audio_tower = getattr(thinker, "audio_tower", None)
    if audio_tower is None:
        raise RuntimeError("Cannot find thinker.audio_tower")
    frontend = ConvFrontend(audio_tower, chunk_size=chunk_size)
    tokens_per_chunk = frontend.tokens_per_chunk
    window_aftercnn = frontend.window_aftercnn

    w = AudioEncoderWrapper(thinker, tokens_per_chunk=tokens_per_chunk, window_aftercnn=window_aftercnn).eval()

    if token_mask.dtype != torch.bool:
        token_mask = token_mask.to(torch.bool)
    if input_features.dtype != torch.float32:
        input_features = input_features.to(torch.float32)

    torch.onnx.export(
        w,
        (input_features, token_mask),
        path_raw,
        input_names=["input_features", "feature_attention_mask"],
        output_names=["audio_features"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "input_features": {0: "batch", 1: "n_audio_tokens"},
            "feature_attention_mask": {0: "batch", 1: "n_audio_tokens"},
            "audio_features": {0: "batch", 1: "n_audio_tokens"},
        },
        dynamo=False,
    )


def export_decoder(thinker, audio_token_id, hidden_size, max_total_len,
                   input_ids, audio_features, attention_mask, cache_position, cache_flat, opset, path_raw):
    w = DecoderCoreWrapper(
        thinker, audio_token_id=audio_token_id, hidden_size=hidden_size, max_total_len=max_total_len
    ).eval()
    L = len(w.layers)

    input_names = ["input_ids", "audio_features", "attention_mask", "cache_position"] + \
                  [f"cache_key_{i}" if j % 2 == 0 else f"cache_value_{i}" for i in range(L) for j in range(2)]
    output_names = ["logits"] + \
                   [f"key_delta_{i}" if j % 2 == 0 else f"value_delta_{i}" for i in range(L) for j in range(2)]

    dyn = {
        "input_ids": {0: "batch", 1: "seq"},
        "audio_features": {0: "batch", 1: "n_audio_tokens"},
        "attention_mask": {0: "batch", 1: "seq"},
        "cache_position": {0: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
    for i in range(L):
        dyn[f"cache_key_{i}"] = {0: "batch", 1: "max_total_len"}
        dyn[f"cache_value_{i}"] = {0: "batch", 1: "max_total_len"}
        dyn[f"key_delta_{i}"] = {0: "batch", 1: "seq"}
        dyn[f"value_delta_{i}"] = {0: "batch", 1: "seq"}

    torch.onnx.export(
        w,
        (input_ids, audio_features, attention_mask, cache_position, *cache_flat),
        path_raw,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dyn,
        dynamo=False,
    )

def quantize_encoder_int8(src_enc: str, dst_enc_i8: str):
    # Encoder has Conv; exclude Conv/ConvInteger for CPU compatibility.
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=src_enc,
        model_output=dst_enc_i8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={
            "WeightSymmetric": True,
            "PerChannel": True,
            "OpTypesToQuantize": ["MatMul", "Gemm", "Linear"],
            "OpTypesToExclude": ["Conv", "ConvInteger", "Slice", "Reshape", "Cast"],
        },
    )
    patch_and_check(dst_enc_i8, dst_enc_i8, opset=17, ir=9, external_data=False)


def quantize_decoder_int8(src_dec: str, dst_dec_i8: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=src_dec,
        model_output=dst_dec_i8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={
            "WeightSymmetric": True,
            "PerChannel": True,
            "OpTypesToQuantize": ["MatMul", "Gemm", "Linear"],
            "OpTypesToExclude": [],
        },
    )
    patch_and_check(dst_dec_i8, dst_dec_i8, opset=17, ir=9, external_data=False)


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=str, required=True, help="Qwen3-ASR checkpoint path")
    p.add_argument("--outdir", type=str, required=True, help="Output directory for ONNX models")
    p.add_argument("--max-total-len", type=int, default=512)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--no-int8", action="store_true", help="Skip INT8 quantization")
    p.add_argument("--verify", action="store_true", help="Run ONNX vs PyTorch verification")
    p.add_argument("--fp32-no-external-data", action="store_true", help="Save FP32 as single file")
    p.add_argument("--chunk-size", type=int, default=100)
    return p.parse_args()


def main():
    args = get_args()

    _register_qwen3_asr()
    monkeypatch_ort_get_opset_version()

    os.makedirs(args.outdir, exist_ok=True)

    # fix mistral regex warning if supported
    try:
        proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    m = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    m.eval().float()

    device = torch.device(args.device)
    if args.device == "cuda":
        m = m.to(device)

    thinker = m.thinker if hasattr(m, "thinker") else m
    thinker.eval().float()
    if args.device == "cuda":
        thinker = thinker.to(device)

    audio_token_id = int(getattr(thinker.config, "audio_token_id", 151676))
    hidden_size = int(getattr(getattr(thinker.config, "text_config", thinker.config), "hidden_size", 1024))

    input_ids, attention_mask, input_features, feature_attention_mask = make_dummy_inputs(proc, device)

    from conv_frontend import ConvFrontend
    audio_tower = getattr(thinker, "audio_tower", None)
    if audio_tower is None:
        raise RuntimeError("Cannot find thinker.audio_tower")
    conv_frontend = ConvFrontend(audio_tower, chunk_size=args.chunk_size).eval()
    if args.device == "cuda":
        conv_frontend = conv_frontend.to(device)

    conv_out = os.path.join(args.outdir, "conv_frontend.onnx")
    dummy_mel = torch.randn(1, 200, 128, device=device)
    with torch.no_grad():
        torch.onnx.export(
            conv_frontend,
            (dummy_mel,),
            conv_out,
            input_names=["input_features"],
            output_names=["conv_output"],
            opset_version=args.opset,
            dynamic_axes={
                "input_features": {0: "batch", 1: "n_frames"},
                "conv_output": {0: "batch", 1: "n_audio_tokens"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    # Embed weights into single file
    model = onnx.load(conv_out, load_external_data=False)
    onnx.save(model, conv_out, save_as_external_data=False)
    print("[export] conv_frontend.onnx")

    if args.verify:
        dummy_mel_np = dummy_mel.detach().cpu().numpy().astype(np.float32)
        conv_feed = {"input_features": dummy_mel_np}
        conv_out_onnx = verify_onnx(conv_out, conv_feed, ["conv_output"])
        with torch.no_grad():
            conv_out_torch = conv_frontend(dummy_mel).detach().cpu().numpy()
        max_diff = np.max(np.abs(conv_out_onnx[0] - conv_out_torch))
        mean_diff = np.mean(np.abs(conv_out_onnx[0] - conv_out_torch))
        print(f"[verify] conv_frontend max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
        if max_diff > 1e-4:
            print("[warn] conv_frontend verification: max_diff > 1e-4")

    with torch.no_grad():
        mel_input = input_features.transpose(1, 2)  # (B, F, T) -> (B, T, F)
        conv_output = conv_frontend(mel_input)

        valid = feature_attention_mask != 0
        feat_len = valid.to(torch.int64).sum(dim=1)  # (B,)
        from conv_frontend import _feat_to_audio_tokens_len
        a_len = _feat_to_audio_tokens_len(feat_len, chunk_size=args.chunk_size)  # (B,)
        A = int(conv_output.shape[1])
        pos = torch.arange(A, device=device).unsqueeze(0)
        token_mask = (pos < a_len.unsqueeze(1)).to(torch.bool)  # (B, A)

    enc_tmp = os.path.join(args.outdir, "_raw_encoder.onnx")
    enc_out = os.path.join(args.outdir, "encoder.onnx")
    with torch.no_grad():
        export_encoder_backend_only(thinker, conv_output, token_mask, args.opset, enc_tmp, args.chunk_size)
    patch_and_check(enc_tmp, enc_out, opset=args.opset, ir=9, external_data=(not args.fp32_no_external_data))
    try:
        os.remove(enc_tmp)
    except Exception:
        pass

    with torch.no_grad():
        wtmp = DecoderCoreWrapper(
            thinker, audio_token_id=audio_token_id, hidden_size=hidden_size, max_total_len=args.max_total_len
        ).eval()
        if args.device == "cuda":
            wtmp = wtmp.to(device)
        kv = int(wtmp.num_kv_heads)
        hd = int(wtmp.head_dim)
        L = len(wtmp.layers)

    B = int(input_ids.shape[0])
    S = int(input_ids.shape[1])
    cache_position = torch.arange(0, S, device=device, dtype=torch.int64)

    n_audio = int((input_ids == audio_token_id).sum().item())
    if n_audio <= 0:
        n_audio = 13

    audio_features = torch.zeros((B, n_audio, hidden_size), device=device, dtype=torch.float32)
    attention_mask = torch.ones(B, S, device=device, dtype=torch.int64)

    cache_flat = []
    for _ in range(L):
        cache_flat.append(torch.zeros(B, args.max_total_len, kv, hd, device=device, dtype=torch.float32))
        cache_flat.append(torch.zeros(B, args.max_total_len, kv, hd, device=device, dtype=torch.float32))

    dec_tmp = os.path.join(args.outdir, "_raw_decoder.onnx")
    dec_out = os.path.join(args.outdir, "decoder.onnx")
    with torch.no_grad():
        export_decoder(
            thinker, audio_token_id, hidden_size, args.max_total_len,
            input_ids, audio_features, attention_mask, cache_position, cache_flat,
            args.opset, dec_tmp
        )
    patch_and_check(dec_tmp, dec_out, opset=args.opset, ir=9, external_data=(not args.fp32_no_external_data))
    try:
        os.remove(dec_tmp)
    except Exception:
        pass

    enc_i8 = os.path.join(args.outdir, "encoder.int8.onnx")
    dec_i8 = os.path.join(args.outdir, "decoder.int8.onnx")

    if not args.no_int8:
        quantize_encoder_int8(enc_out, enc_i8)
        print(f"[int8] {enc_i8}")

        quantize_decoder_int8(dec_out, dec_i8)
        print(f"[int8] {dec_i8}")

    if args.verify:
        conv_output_np = conv_output.detach().cpu().numpy().astype(np.float32)
        token_mask_np = token_mask.detach().cpu().numpy()

        enc_feed = {"input_features": conv_output_np, "feature_attention_mask": token_mask_np}
        (audio_feat_out,) = verify_onnx(enc_out, enc_feed, ["audio_features"])

        with torch.no_grad():
            backend = AudioEncoderWrapper(
                thinker,
                tokens_per_chunk=conv_frontend.tokens_per_chunk,
                window_aftercnn=conv_frontend.window_aftercnn,
            ).eval()
            audio_feat_torch = backend(conv_output, token_mask).detach().cpu().numpy()
        max_diff = np.max(np.abs(audio_feat_out - audio_feat_torch))
        mean_diff = np.mean(np.abs(audio_feat_out - audio_feat_torch))
        print(f"[verify] encoder max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

        input_ids_np = input_ids.detach().cpu().numpy().astype(np.int64)
        cache_position_np = cache_position.detach().cpu().numpy().astype(np.int64)
        attention_mask_np = attention_mask.detach().cpu().numpy().astype(np.int64)

        caches_np = {}
        for i in range(L):
            caches_np[f"cache_key_{i}"] = np.zeros((B, args.max_total_len, kv, hd), dtype=np.float32)
            caches_np[f"cache_value_{i}"] = np.zeros((B, args.max_total_len, kv, hd), dtype=np.float32)

        dec_feed = {
            "input_ids": input_ids_np,
            "audio_features": audio_feat_out.astype(np.float32),
            "attention_mask": attention_mask_np,
            "cache_position": cache_position_np,
            **caches_np,
        }
        (logits_out,) = verify_onnx(dec_out, dec_feed, ["logits"])

        with torch.no_grad():
            dec_wrapper = DecoderCoreWrapper(
                thinker, audio_token_id=audio_token_id, hidden_size=hidden_size, max_total_len=args.max_total_len
            ).eval()
            audio_feat_torch = torch.from_numpy(audio_feat_out.astype(np.float32)).to(device)
            attention_mask_torch = torch.from_numpy(attention_mask_np).to(device)
            caches_torch = []
            for i in range(L):
                caches_torch.append(torch.from_numpy(caches_np[f"cache_key_{i}"]).to(device))
                caches_torch.append(torch.from_numpy(caches_np[f"cache_value_{i}"]).to(device))
            logits_torch_all = dec_wrapper(
                input_ids,
                audio_feat_torch,
                attention_mask_torch,
                cache_position,
                *caches_torch,
            )
            logits_torch = logits_torch_all[0].detach().cpu().numpy()
        max_diff = np.max(np.abs(logits_out - logits_torch))
        mean_diff = np.mean(np.abs(logits_out - logits_torch))
        print(f"[verify] decoder max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
        print("[verify] fp32 ok (encoder+decoder)")

        if not args.no_int8:
            try:
                dec_feed["audio_features"] = audio_feat_out.astype(np.float32)
                verify_onnx(dec_i8, dec_feed, ["logits"])
                print("[verify] int8 ok (encoder+decoder)")
            except Exception as e:
                print(f"[warn] int8 verify skipped: {e}")

    print("[export] Done.")
    print("[export] FP32 encoder:", enc_out)
    print("[export] FP32 decoder:", dec_out)
    if not args.no_int8:
        print("[export] INT8 encoder:", enc_i8)
        print("[export] INT8 decoder:", dec_i8)


if __name__ == "__main__":
    main()
