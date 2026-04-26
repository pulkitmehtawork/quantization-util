"""
quantization-util CLI

Usage:
  quantize --model facebook/opt-125m --method gguf --quant-type Q4_K_M --output ./out
  quantize --model facebook/opt-125m --method bitsandbytes --bits 4 --output ./out
  quantize --model facebook/opt-125m --method gptq --bits 4 --output ./out
  quantize --model facebook/opt-125m --method awq --output ./out
  quantize --model facebook/opt-125m --method pytorch --mode dynamic --output ./out
"""

import argparse
import sys

from .quantizer import Quantizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quantize",
        description="Quantize a HuggingFace model to various formats.",
    )
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument(
        "--method",
        required=True,
        choices=["bitsandbytes", "gguf", "gptq", "awq", "pytorch"],
        help="Quantization backend",
    )
    p.add_argument("--output", default="./quantized", help="Output directory")

    # BitsAndBytes / GPTQ shared
    p.add_argument("--bits", type=int, choices=[2, 3, 4, 8], default=4)

    # BitsAndBytes specific
    p.add_argument("--bnb-quant-type", choices=["nf4", "fp4"], default="nf4")
    p.add_argument("--no-double-quant", action="store_true")

    # GGUF specific
    p.add_argument("--quant-type", default="Q4_K_M", help="GGUF quantization type (e.g. Q4_K_M)")
    p.add_argument("--llama-cpp-path", help="Path to llama.cpp repository root")

    # GPTQ specific
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    p.add_argument("--num-samples", type=int, default=128)

    # PyTorch specific
    p.add_argument("--mode", choices=["dynamic", "weight_only"], default="dynamic")

    return p


def main():
    args = build_parser().parse_args()
    q = Quantizer(args.model)

    try:
        if args.method == "bitsandbytes":
            out = q.quantize(
                "bitsandbytes",
                output_dir=args.output,
                bits=args.bits,
                bnb_4bit_quant_type=args.bnb_quant_type,
                use_double_quant=not args.no_double_quant,
            )
        elif args.method == "gguf":
            out = q.quantize(
                "gguf",
                output_dir=args.output,
                quantization_type=args.quant_type,
                llama_cpp_path=args.llama_cpp_path,
            )
        elif args.method == "gptq":
            out = q.quantize(
                "gptq",
                output_dir=args.output,
                bits=args.bits,
                group_size=args.group_size,
                dataset=args.dataset,
                num_samples=args.num_samples,
            )
        elif args.method == "awq":
            out = q.quantize(
                "awq",
                output_dir=args.output,
                bits=args.bits,
                group_size=args.group_size,
            )
        elif args.method == "pytorch":
            out = q.quantize(
                "pytorch",
                output_dir=args.output,
                mode=args.mode,
            )
        print(f"\nDone. Output: {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
