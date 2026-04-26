from pathlib import Path
from typing import Literal

from .base import BaseQuantizer


class GPTQQuantizer(BaseQuantizer):
    """
    Quantize a HuggingFace causal-LM with GPTQ (via AutoGPTQ or optimum).

    Best for: 4-bit weight quantization that retains accuracy by using
    second-order gradient information during calibration.
    Requires a small calibration dataset (default: wikitext2).
    """

    def quantize(
        self,
        output_dir: str,
        bits: Literal[2, 3, 4, 8] = 4,
        group_size: int = 128,
        dataset: str = "wikitext2",
        num_samples: int = 128,
        use_triton: bool = False,
    ) -> Path:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Install dependencies: pip install quantization-util[gptq]"
            ) from e

        out = self._ensure_output_dir(output_dir)

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=False,
        )

        print(f"Loading tokenizer for {self.model_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

        print(f"Loading model for GPTQ calibration ({bits}-bit, group_size={group_size})...")
        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_name_or_path, quantize_config=quantize_config
        )

        calibration_data = self._load_calibration_data(tokenizer, dataset, num_samples)

        print("Running GPTQ quantization (this may take a few minutes)...")
        model.quantize(calibration_data)

        model.save_quantized(str(out), use_safetensors=True)
        tokenizer.save_pretrained(out)
        print(f"GPTQ {bits}-bit model saved to {out}")
        return out

    @staticmethod
    def _load_calibration_data(tokenizer, dataset: str, num_samples: int) -> list:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("Install datasets: pip install datasets") from e

        if dataset == "wikitext2":
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in data["text"] if len(t.strip()) > 50][:num_samples]
        elif dataset == "c4":
            data = load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = [item["text"] for item in data.take(num_samples)]
        else:
            raise ValueError(f"Unknown calibration dataset: {dataset}. Use 'wikitext2' or 'c4'.")

        return [tokenizer(t, return_tensors="pt", truncation=True, max_length=512) for t in texts]
