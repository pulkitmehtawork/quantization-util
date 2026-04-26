"""Unit tests for quantizer backends (mocked — no real model downloads)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quantization_util.quantizer import Quantizer
from quantization_util.quantizers.base import BaseQuantizer


class ConcreteQuantizer(BaseQuantizer):
    def quantize(self, output_dir: str, **kwargs) -> Path:
        return self._ensure_output_dir(output_dir)


def test_base_ensure_output_dir(tmp_path):
    q = ConcreteQuantizer("dummy")
    out = q._ensure_output_dir(str(tmp_path / "new_dir"))
    assert out.exists()


def test_quantizer_raises_on_unknown_method():
    q = Quantizer("dummy")
    with pytest.raises(ValueError, match="Unknown method"):
        q.quantize("nonexistent_method", output_dir="/tmp/x")


@patch("quantization_util.quantizers.bitsandbytes.AutoModelForCausalLM")
@patch("quantization_util.quantizers.bitsandbytes.AutoTokenizer")
@patch("quantization_util.quantizers.bitsandbytes.BitsAndBytesConfig")
def test_bitsandbytes_4bit(mock_bnb_config, mock_tok, mock_model, tmp_path):
    mock_tok.from_pretrained.return_value = MagicMock()
    mock_model.from_pretrained.return_value = MagicMock()

    q = Quantizer("facebook/opt-125m")
    out = q.quantize("bitsandbytes", output_dir=str(tmp_path / "bnb"), bits=4)
    assert out is not None
    mock_bnb_config.assert_called_once()


@patch("quantization_util.quantizers.bitsandbytes.AutoModelForCausalLM")
@patch("quantization_util.quantizers.bitsandbytes.AutoTokenizer")
@patch("quantization_util.quantizers.bitsandbytes.BitsAndBytesConfig")
def test_bitsandbytes_8bit(mock_bnb_config, mock_tok, mock_model, tmp_path):
    mock_tok.from_pretrained.return_value = MagicMock()
    mock_model.from_pretrained.return_value = MagicMock()

    q = Quantizer("facebook/opt-125m")
    q.quantize("bitsandbytes", output_dir=str(tmp_path / "bnb8"), bits=8)
    call_kwargs = mock_bnb_config.call_args.kwargs
    assert call_kwargs.get("load_in_8bit") is True


def test_bitsandbytes_invalid_bits():
    from quantization_util.quantizers.bitsandbytes import BitsAndBytesQuantizer

    with patch("quantization_util.quantizers.bitsandbytes.AutoModelForCausalLM"), \
         patch("quantization_util.quantizers.bitsandbytes.AutoTokenizer"), \
         patch("quantization_util.quantizers.bitsandbytes.BitsAndBytesConfig"):
        bq = BitsAndBytesQuantizer("dummy")
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            bq.quantize("/tmp/x", bits=3)


@patch("quantization_util.quantizers.pytorch_native.AutoModelForCausalLM")
@patch("quantization_util.quantizers.pytorch_native.AutoTokenizer")
@patch("quantization_util.quantizers.pytorch_native.torch")
def test_pytorch_dynamic(mock_torch, mock_tok, mock_model, tmp_path):
    mock_tok.from_pretrained.return_value = MagicMock()
    fake_model = MagicMock()
    fake_model.parameters.return_value = []
    mock_model.from_pretrained.return_value = fake_model
    mock_torch.quantization.quantize_dynamic.return_value = fake_model

    q = Quantizer("facebook/opt-125m")
    out = q.quantize("pytorch", output_dir=str(tmp_path / "pt"), mode="dynamic")
    assert out is not None


def test_quantize_all_partial_failure(tmp_path):
    q = Quantizer("facebook/opt-125m")
    with patch.object(q, "quantize", side_effect=[Exception("backend missing"), tmp_path / "pt"]):
        results = q.quantize_all(
            output_dir=str(tmp_path),
            methods=["bitsandbytes", "pytorch"],
        )
    assert results["bitsandbytes"] is None
