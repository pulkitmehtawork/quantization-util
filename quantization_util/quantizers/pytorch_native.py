from pathlib import Path
from typing import Literal

from .base import BaseQuantizer


class PyTorchQuantizer(BaseQuantizer):
    """
    Quantize a model using PyTorch's native quantization API.

    Best for: CPU-only environments with no external deps.
    Modes:
      dynamic   — quantizes weights ahead of time, activations at runtime; fast and simple
      static    — requires calibration data; best latency for CNN/encoder models
      weight_only — quantizes only weights to int8; good for memory reduction
    """

    def quantize(
        self,
        output_dir: str,
        mode: Literal["dynamic", "static", "weight_only"] = "dynamic",
        dtype: Literal["qint8", "float16", "bfloat16"] = "qint8",
    ) -> Path:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("Install transformers and torch") from e

        out = self._ensure_output_dir(output_dir)

        print(f"Loading {self.model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        model.eval()

        if mode == "dynamic":
            quantized = self._dynamic_quantize(model, dtype)
        elif mode == "weight_only":
            quantized = self._weight_only_quantize(model)
        elif mode == "static":
            raise NotImplementedError(
                "Static quantization requires a calibration dataloader. "
                "Use mode='dynamic' or mode='weight_only' instead."
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        tokenizer.save_pretrained(out)
        torch.save(quantized.state_dict(), out / "pytorch_model.bin")
        quantized.config.save_pretrained(out)
        print(f"PyTorch {mode} quantized model saved to {out}")
        return out

    @staticmethod
    def _dynamic_quantize(model, dtype: str):
        import torch
        torch_dtype = torch.qint8 if dtype == "qint8" else torch.float16
        return torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch_dtype,
        )

    @staticmethod
    def _weight_only_quantize(model):
        """Apply int8 weight-only quantization using PyTorch 2.x torchao/quantize_ if available,
        falling back to dynamic quantization."""
        try:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(model, int8_weight_only())
            return model
        except ImportError:
            import torch
            return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
