"""Pretrained token embeddings, read from extracted FP32 tensors (NumPy only).

Each registered matrix is the byte range of one tensor of a public
safetensors checkpoint, downloaded with the command in README.md and stored
in data_cache/. `load_real_matrix` verifies the file's SHA-256 before use.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    tensor_name: str
    shape: tuple[int, int]
    sha256: str
    source_url: str
    byte_range: str  # HTTP byte range of the tensor inside the checkpoint


# Token embeddings of the FP32-trained GPT-2 models, which have a pronounced
# leading spectral gap (sigma_1/sigma_2 = 6.0 and 6.8); see README.md.
MODEL_SPECS = {
    "gpt2small": ModelSpec(
        name="GPT-2 small token embedding",
        path=Path("data_cache/gpt2-wte-f32.bin"),
        tensor_name="wte.weight",
        shape=(50257, 768),
        sha256=(
            "e182e433b37dbdb47448e0413c840edf6965113c1fc8048c63b69795d8cf875a"
        ),
        source_url=(
            "https://huggingface.co/openai-community/gpt2/resolve/main/"
            "model.safetensors"
        ),
        byte_range="287223763-441613266",
    ),
    "gpt2medium": ModelSpec(
        name="GPT-2 medium token embedding",
        path=Path("data_cache/gpt2-medium-wte-f32.bin"),
        tensor_name="wte.weight",
        shape=(50257, 1024),
        sha256=(
            "b81423e5636e520a3886d5f4bced1c5613ed5cd85307abb6b31775b8f29ab361"
        ),
        source_url=(
            "https://huggingface.co/openai-community/gpt2-medium/resolve/main/"
            "model.safetensors"
        ),
        byte_range="1314132290-1519984961",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_real_matrix(model_name: str, weights_path: Path | None = None):
    """Return (matrix, name) for one registered token embedding as float32."""
    spec = MODEL_SPECS[model_name]
    path = spec.path if weights_path is None else Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}.\nDownload byte range {spec.byte_range} of "
            f"{spec.source_url} (see README.md)."
        )
    observed = _sha256(path)
    if observed != spec.sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed}")
    expected_bytes = int(np.prod(spec.shape)) * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; expected {expected_bytes}"
        )
    matrix = np.fromfile(path, dtype="<f4").reshape(spec.shape)
    return np.ascontiguousarray(matrix, dtype=np.float32), spec.name
