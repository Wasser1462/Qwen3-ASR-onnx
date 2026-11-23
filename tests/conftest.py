from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from export_qwen3_asr_onnx import _register_qwen3_asr
from qwen3_asr import (
    Qwen3ASRThinkerConfig,
    Qwen3ASRThinkerForConditionalGeneration,
)


def pytest_addoption(parser):
    parser.addoption(
        "--checkpoint", type=Path, help="Local Qwen3-ASR checkpoint (no downloads)"
    )


@pytest.fixture(scope="session", params=["tiny", "checkpoint"])
def thinker(request):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if request.param == "tiny":
        # Use the real model classes with small random weights, without I/O.
        config = Qwen3ASRThinkerConfig(
            audio_token_id=127,
            audio_start_token_id=126,
            user_token_id=125,
            audio_config={
                "num_mel_bins": 16,
                "d_model": 32,
                "encoder_layers": 1,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 64,
                "max_source_positions": 64,
                "output_dim": 32,
                "downsample_hidden_size": 8,
            },
            text_config={
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 512,
            },
        )
        return Qwen3ASRThinkerForConditionalGeneration(config).eval()

    checkpoint = request.config.getoption("--checkpoint")
    if checkpoint is None:
        pytest.skip("Pass --checkpoint to run pretrained decoder tests")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        pytest.fail("--checkpoint must be a local model directory")
    if not _register_qwen3_asr():
        pytest.fail("Could not register the local Qwen3-ASR implementation")
    model = AutoModel.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).eval()
    return model.thinker if hasattr(model, "thinker") else model
