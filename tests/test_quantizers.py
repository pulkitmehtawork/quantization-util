"""Unit tests for quantizer backends (mocked — no real model downloads)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quantization_util.quantizer import Quantizer
from quantization_util.quantizers.base import BaseQuantizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ConcreteQuantizer(BaseQuantizer):
    def quantize(self, output_dir: str, **kwargs) -> Path:
        return self._ensure_output_dir(output_dir)


def _transformers_mock():
    """Return a minimal mock for the transformers package."""
    m = MagicMock()
    m.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
    m.AutoTokenizer.from_pretrained.return_value = MagicMock()
    m.BitsAndBytesConfig.return_value = MagicMock()
    return m


def _torch_mock():
    """Return a minimal mock for torch."""
    m = MagicMock()
    m.nn.Linear = MagicMock()
    m.qint8 = "qint8"
    m.float16 = "float16"
    m.bfloat16 = "bfloat16"
    quantized_model = MagicMock()
    quantized_model.state_dict.return_value = {}
    quantized_model.config = MagicMock()
    quantized_model.parameters.return_value = []
    m.quantization.quantize_dynamic.return_value = quantized_model
    m.backends.mps.is_available.return_value = False
    m.cuda.is_available.return_value = False
    return m


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

def test_base_ensure_output_dir(tmp_path):
    q = ConcreteQuantizer("dummy")
    out = q._ensure_output_dir(str(tmp_path / "new_dir"))
    assert out.exists()


# ---------------------------------------------------------------------------
# Quantizer facade
# ---------------------------------------------------------------------------

def test_quantizer_raises_on_unknown_method():
    q = Quantizer("dummy")
    with pytest.raises(ValueError, match="Unknown method"):
        q.quantize("nonexistent_method", output_dir="/tmp/x")


# ---------------------------------------------------------------------------
# BitsAndBytes
# ---------------------------------------------------------------------------

def test_bitsandbytes_4bit(tmp_path):
    tf = _transformers_mock()
    t = _torch_mock()
    with patch.dict(sys.modules, {"transformers": tf, "bitsandbytes": MagicMock(), "torch": t}):
        q = Quantizer("facebook/opt-125m")
        out = q.quantize("bitsandbytes", output_dir=str(tmp_path / "bnb4"), bits=4)

    assert out is not None
    tf.BitsAndBytesConfig.assert_called_once()
    call_kwargs = tf.BitsAndBytesConfig.call_args.kwargs
    assert call_kwargs.get("load_in_4bit") is True


def test_bitsandbytes_8bit(tmp_path):
    tf = _transformers_mock()
    t = _torch_mock()
    with patch.dict(sys.modules, {"transformers": tf, "bitsandbytes": MagicMock(), "torch": t}):
        q = Quantizer("facebook/opt-125m")
        q.quantize("bitsandbytes", output_dir=str(tmp_path / "bnb8"), bits=8)

    call_kwargs = tf.BitsAndBytesConfig.call_args.kwargs
    assert call_kwargs.get("load_in_8bit") is True


def test_bitsandbytes_invalid_bits(tmp_path):
    tf = _transformers_mock()
    t = _torch_mock()
    with patch.dict(sys.modules, {"transformers": tf, "bitsandbytes": MagicMock(), "torch": t}):
        from quantization_util.quantizers.bitsandbytes import BitsAndBytesQuantizer
        bq = BitsAndBytesQuantizer("dummy")
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            bq.quantize(str(tmp_path), bits=3)


# ---------------------------------------------------------------------------
# PyTorch native
# ---------------------------------------------------------------------------

def test_pytorch_dynamic(tmp_path):
    tf = _transformers_mock()
    t = _torch_mock()
    with patch.dict(sys.modules, {"transformers": tf, "torch": t, "torchao": MagicMock(), "bitsandbytes": MagicMock()}):
        q = Quantizer("facebook/opt-125m")
        out = q.quantize("pytorch", output_dir=str(tmp_path / "pt"), mode="dynamic")

    assert out is not None
    t.quantization.quantize_dynamic.assert_called_once()


# ---------------------------------------------------------------------------
# quantize_all
# ---------------------------------------------------------------------------

def test_quantize_all_partial_failure(tmp_path):
    q = Quantizer("facebook/opt-125m")
    side_effects = [Exception("backend missing"), tmp_path / "pt"]
    with patch.object(q, "quantize", side_effect=side_effects):
        results = q.quantize_all(
            output_dir=str(tmp_path),
            methods=["bitsandbytes", "pytorch"],
        )
    assert results["bitsandbytes"] is None
    assert results["pytorch"] == tmp_path / "pt"
