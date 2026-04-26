from pathlib import Path
from typing import Literal

from .quantizers import (
    AWQQuantizer,
    BitsAndBytesQuantizer,
    GGUFQuantizer,
    GPTQQuantizer,
    PyTorchQuantizer,
)

QuantMethod = Literal["bitsandbytes", "gguf", "gptq", "awq", "pytorch"]


class Quantizer:
    """
    Unified interface for all quantization backends.

    Usage:
        q = Quantizer("facebook/opt-125m")
        q.quantize("bitsandbytes", output_dir="./out", bits=4)
        q.quantize("gguf", output_dir="./out", quantization_type="Q4_K_M")
    """

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path

    def quantize(
        self,
        method: QuantMethod,
        output_dir: str = "./quantized",
        **kwargs,
    ) -> Path:
        backend = self._get_backend(method, **kwargs)
        # Strip backend-constructor kwargs before passing to .quantize()
        constructor_kwargs = {"llama_cpp_path"} if method == "gguf" else set()
        quant_kwargs = {k: v for k, v in kwargs.items() if k not in constructor_kwargs}
        return backend.quantize(output_dir, **quant_kwargs)

    def _get_backend(self, method: QuantMethod, **kwargs):
        if method == "bitsandbytes":
            return BitsAndBytesQuantizer(self.model_name_or_path)
        elif method == "gguf":
            return GGUFQuantizer(
                self.model_name_or_path,
                llama_cpp_path=kwargs.get("llama_cpp_path"),
            )
        elif method == "gptq":
            return GPTQQuantizer(self.model_name_or_path)
        elif method == "awq":
            return AWQQuantizer(self.model_name_or_path)
        elif method == "pytorch":
            return PyTorchQuantizer(self.model_name_or_path)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: bitsandbytes, gguf, gptq, awq, pytorch"
            )

    def quantize_all(
        self,
        output_dir: str = "./quantized",
        methods: list[QuantMethod] | None = None,
        **per_method_kwargs: dict,
    ) -> dict[str, Path]:
        """Run multiple quantization methods in one call."""
        if methods is None:
            methods = ["bitsandbytes", "pytorch"]
        results = {}
        for method in methods:
            method_out = str(Path(output_dir) / method)
            kwargs = per_method_kwargs.get(method, {})
            try:
                results[method] = self.quantize(method, output_dir=method_out, **kwargs)
            except Exception as e:
                print(f"[{method}] failed: {e}")
                results[method] = None
        return results
