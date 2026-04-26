from pathlib import Path
from typing import Literal

from .base import BaseQuantizer


class BitsAndBytesQuantizer(BaseQuantizer):
    """
    Quantize a HuggingFace model with bitsandbytes (4-bit NF4 or 8-bit LLM.int8).

    Best for: inference on consumer hardware with minimal accuracy loss.
    Supported bits: 4 (NF4/FP4) or 8 (LLM.int8).
    """

    def quantize(
        self,
        output_dir: str,
        bits: Literal[4, 8] = 4,
        use_double_quant: bool = True,
        bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4",
        bnb_4bit_compute_dtype: str = "bfloat16",
    ) -> Path:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "Install dependencies: pip install quantization-util[bnb]"
            ) from e

        out = self._ensure_output_dir(output_dir)

        if bits == 4:
            import torch as _torch
            compute_dtype = getattr(_torch, bnb_4bit_compute_dtype, _torch.bfloat16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("bits must be 4 or 8")

        print(f"Loading {self.model_name_or_path} with {bits}-bit BitsAndBytes config...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
        )

        tokenizer.save_pretrained(out)
        model.save_pretrained(out)
        print(f"Saved BitsAndBytes {bits}-bit model to {out}")
        return out
