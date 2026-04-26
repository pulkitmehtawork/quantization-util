from pathlib import Path
from typing import Literal

from .base import BaseQuantizer


class AWQQuantizer(BaseQuantizer):
    """
    Quantize a HuggingFace model with AWQ (Activation-aware Weight Quantization).

    Best for: 4-bit quantization that scales weights channel-wise based on
    activation magnitude, achieving near-fp16 accuracy with ~4x memory reduction.
    Requires a small calibration dataset.
    """

    def quantize(
        self,
        output_dir: str,
        bits: Literal[4] = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: Literal["gemm", "gemv"] = "gemm",
        calib_dataset: str = "pileval",
        num_samples: int = 128,
        seq_len: int = 512,
    ) -> Path:
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Install dependencies: pip install quantization-util[awq]"
            ) from e

        out = self._ensure_output_dir(output_dir)

        quant_config = {
            "zero_point": zero_point,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": version,
        }

        print(f"Loading {self.model_name_or_path} for AWQ quantization...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        model = AutoAWQForCausalLM.from_pretrained(
            self.model_name_or_path, trust_remote_code=True, safetensors=True
        )

        print(f"Running AWQ calibration on '{calib_dataset}' ({num_samples} samples)...")
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_dataset,
            n_samples=num_samples,
            max_seq_len=seq_len,
        )

        model.save_quantized(str(out))
        tokenizer.save_pretrained(out)
        print(f"AWQ {bits}-bit model saved to {out}")
        return out
