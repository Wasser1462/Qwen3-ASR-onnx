from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from export_qwen3_asr_onnx import _register_qwen3_asr


def pytest_addoption(parser):
    parser.addoption(
        "--checkpoint", type=Path, help="Local Qwen3-ASR checkpoint (no downloads)"
    )


@pytest.fixture(scope="session")
def thinker(request):
    checkpoint = request.config.getoption("--checkpoint")
    if checkpoint is None:
        pytest.skip("Pass --checkpoint to run real-model decoder tests")
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        pytest.fail("--checkpoint must be a local model directory")
    if not _register_qwen3_asr():
        pytest.fail("Could not register the local Qwen3-ASR implementation")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = AutoModel.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    ).eval()
    return model.thinker if hasattr(model, "thinker") else model
