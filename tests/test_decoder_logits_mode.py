import gc
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

import export_qwen3_asr_onnx as exporter
from decoder import DecoderCoreWrapper
from export_qwen3_asr_onnx import export_decoder, get_args, patch_and_check


def wrapper(thinker, **kwargs):
    return DecoderCoreWrapper(
        thinker,
        audio_token_id=thinker.config.audio_token_id,
        hidden_size=thinker.config.text_config.hidden_size,
        max_total_len=512,
        **kwargs,
    ).eval()


def make_inputs(decoder, seq, past_len=0):
    generator = torch.Generator().manual_seed(42 + seq + past_len)
    ids = torch.arange(1, seq + 1, dtype=torch.int64).unsqueeze(0)
    ids[:, 1::4] = decoder.audio_token_id
    audio_count = max(1, int((ids == decoder.audio_token_id).sum()))
    audio = torch.randn(
        1, audio_count, decoder.hidden_size, generator=generator
    )
    cache = [
        torch.zeros(1, 512, decoder.num_kv_heads, decoder.head_dim)
        for _ in range(2 * decoder.num_layers)
    ]
    return (
        ids,
        audio,
        torch.ones_like(ids),
        torch.arange(past_len, past_len + seq, dtype=torch.int64),
        *cache,
    )


def assert_pair(all_outputs, last_outputs, seq, layers):
    assert len(all_outputs) == len(last_outputs) == 1 + 2 * layers
    assert all_outputs[0].shape[1] == seq
    assert last_outputs[0].shape == (
        all_outputs[0].shape[0], 1, all_outputs[0].shape[2]
    )
    for index, (baseline, candidate) in enumerate(zip(all_outputs, last_outputs)):
        assert np.isfinite(baseline).all()
        assert np.isfinite(candidate).all()
        if index == 0:
            np.testing.assert_allclose(
                candidate, baseline[:, -1:, :], atol=1e-4, rtol=1e-4
            )
        else:
            assert baseline.shape == candidate.shape
            assert baseline.shape[1] == seq
            np.testing.assert_allclose(candidate, baseline, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seq", [1, 8, 64])
def test_last_logits_and_full_kv(thinker, seq):
    all_decoder = wrapper(thinker, logits_mode="all")
    last_decoder = wrapper(thinker, logits_mode="last")
    inputs = make_inputs(all_decoder, seq)
    projection_shapes = []
    hook = thinker.lm_head.register_forward_pre_hook(
        lambda module, args: projection_shapes.append(tuple(args[0].shape))
    )
    try:
        with torch.inference_mode():
            all_outputs = all_decoder(*inputs)
            last_outputs = last_decoder(*inputs)
    finally:
        hook.remove()
    assert [shape[1] for shape in projection_shapes] == [seq, 1]
    assert_pair(
        [x.numpy() for x in all_outputs], [x.numpy() for x in last_outputs],
        seq, all_decoder.num_layers,
    )


def test_default_mode_matches_all(thinker):
    default = wrapper(thinker)
    explicit_all = wrapper(thinker, logits_mode="all")
    inputs = make_inputs(default, 8)
    with torch.inference_mode():
        actual, expected = default(*inputs), explicit_all(*inputs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=False)


@pytest.mark.parametrize("mode", ["", "LAST", "both", None])
def test_invalid_logits_mode_rejected(mode):
    with pytest.raises(ValueError, match="logits_mode"):
        DecoderCoreWrapper(
            None, audio_token_id=0, hidden_size=1, logits_mode=mode
        )


@pytest.mark.parametrize("mode", [None, "all", "last"])
def test_cli_logits_mode(monkeypatch, mode):
    argv = ["export", "--model", "local", "--outdir", "output"]
    if mode is not None:
        argv += ["--logits-mode", mode]
    monkeypatch.setattr("sys.argv", argv)
    assert get_args().logits_mode == (mode or "all")


def test_cli_invalid_logits_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "export", "--model", "local", "--outdir", "output",
        "--logits-mode", "both",
    ])
    with pytest.raises(SystemExit) as exc:
        get_args()
    assert exc.value.code == 2


@pytest.mark.parametrize("seq", [1, 8])
def test_decoder_logits_diff_accepts_matching_shapes(seq):
    reference = np.zeros((1, seq, 7), dtype=np.float32)
    actual = reference.copy()
    actual[..., 0] = 7
    max_diff, mean_diff = exporter._decoder_logits_diff(actual, reference)
    assert max_diff == 7
    assert mean_diff == 1


@pytest.mark.parametrize("onnx_shape,torch_shape", [
    ((1, 1, 7), (1, 8, 7)),
    ((1, 8, 7), (1, 1, 7)),
    ((2, 1, 7), (1, 1, 7)),
    ((1, 1, 7), (1, 7)),
    ((1, 1, 7), (1, 1, 1)),
])
def test_decoder_logits_diff_rejects_broadcastable_shapes(
    onnx_shape, torch_shape
):
    with pytest.raises(RuntimeError, match="shape mismatch"):
        exporter._decoder_logits_diff(
            np.zeros(onnx_shape, dtype=np.float32),
            np.zeros(torch_shape, dtype=np.float32),
        )


def test_exported_last_shape_and_full_kv(thinker, tmp_path):
    decoder = wrapper(thinker)
    inputs = make_inputs(decoder, 8)
    pairs = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    # These are disposable test exports; production paired artifacts are kept
    # separately. Release each ORT session before loading the next mode.
    for mode in ("all", "last"):
        with tempfile.TemporaryDirectory(prefix=mode, dir=tmp_path) as directory:
            root = Path(directory)
            raw, model = root / "raw.onnx", root / "decoder.onnx"
            with torch.inference_mode():
                export_decoder(
                    thinker, decoder.audio_token_id, decoder.hidden_size, 512,
                    *inputs[:4], inputs[4:], 17, str(raw), logits_mode=mode,
                )
            patch_and_check(str(raw), str(model), opset=17, ir=9)
            graph = onnx.load(model, load_external_data=False)
            assert graph.ir_version == 9
            assert graph.opset_import[0].version == 17
            time_dim = graph.graph.output[0].type.tensor_type.shape.dim[1]
            if mode == "last":
                assert time_dim.dim_value == 1
            else:
                assert time_dim.dim_param == "seq"
            for output in graph.graph.output[1:]:
                assert output.type.tensor_type.shape.dim[1].dim_param == "seq"
            onnx.checker.check_model(str(model))
            session = ort.InferenceSession(
                str(model), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for seq in (1, 8, 64):
                values = make_inputs(decoder, seq)
                feed = {meta.name: value.numpy()
                        for meta, value in zip(session.get_inputs(), values)}
                pairs[mode, seq] = session.run(None, feed)
            del session
            gc.collect()
    for seq in (1, 8, 64):
        assert_pair(pairs["all", seq], pairs["last", seq], seq, decoder.num_layers)
