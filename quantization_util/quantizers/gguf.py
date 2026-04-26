import shutil
import subprocess
from pathlib import Path
from typing import Literal

from .base import BaseQuantizer

GGUF_QUANT_TYPES = Literal[
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0", "F16", "F32",
]


class GGUFQuantizer(BaseQuantizer):
    """
    Convert a HuggingFace model to GGUF format via llama.cpp.

    Best for: running quantized models locally on CPU/Metal (Mac M-series).
    Requires llama.cpp installed and its convert/quantize scripts accessible.

    Quantization types:
      Q4_K_M  — recommended default, good quality/size tradeoff
      Q8_0    — near-lossless, ~2x smaller than F16
      Q5_K_M  — between Q4 and Q8
      F16     — no quantization, just format conversion
    """

    def __init__(self, model_name_or_path: str, llama_cpp_path: str | None = None):
        super().__init__(model_name_or_path)
        # llama_cpp_path is the root of the llama.cpp repo clone
        self.llama_cpp_path = Path(llama_cpp_path) if llama_cpp_path else self._find_llama_cpp()

    def quantize(
        self,
        output_dir: str,
        quantization_type: GGUF_QUANT_TYPES = "Q4_K_M",
    ) -> Path:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("Install transformers: pip install transformers") from e

        out = self._ensure_output_dir(output_dir)
        hf_dump = out / "hf_weights"
        hf_dump.mkdir(exist_ok=True)

        print(f"Downloading {self.model_name_or_path} weights...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
        tokenizer.save_pretrained(hf_dump)
        model.save_pretrained(hf_dump)

        f16_gguf = out / "model_f16.gguf"
        self._convert_to_gguf(hf_dump, f16_gguf)

        if quantization_type == "F16":
            print(f"GGUF F16 model saved to {f16_gguf}")
            return f16_gguf

        quantized_gguf = out / f"model_{quantization_type}.gguf"
        self._quantize_gguf(f16_gguf, quantized_gguf, quantization_type)
        f16_gguf.unlink(missing_ok=True)
        print(f"GGUF {quantization_type} model saved to {quantized_gguf}")
        return quantized_gguf

    def _convert_to_gguf(self, hf_dir: Path, out_file: Path) -> None:
        convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            convert_script = self.llama_cpp_path / "convert.py"
        if not convert_script.exists():
            raise FileNotFoundError(
                f"llama.cpp convert script not found in {self.llama_cpp_path}. "
                "Clone llama.cpp and pass its path as llama_cpp_path."
            )
        subprocess.run(
            ["python", str(convert_script), str(hf_dir), "--outfile", str(out_file), "--outtype", "f16"],
            check=True,
        )

    def _quantize_gguf(self, src: Path, dst: Path, quant_type: str) -> None:
        quantize_bin = self.llama_cpp_path / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
        if not quantize_bin:
            raise FileNotFoundError(
                "llama-quantize binary not found. Build llama.cpp first."
            )
        subprocess.run([str(quantize_bin), str(src), str(dst), quant_type], check=True)

    @staticmethod
    def _find_llama_cpp() -> Path:
        candidate = shutil.which("llama-quantize")
        if candidate:
            return Path(candidate).parent.parent
        raise RuntimeError(
            "Could not auto-detect llama.cpp path. "
            "Pass llama_cpp_path= to GGUFQuantizer()."
        )
