"""
Basic usage examples for quantization-util.

Run: python examples/basic_usage.py
Models used here are tiny and can run on a MacBook Pro.
"""

from quantization_util import Quantizer
from quantization_util.utils import get_model_size_mb, benchmark_inference

MODEL = "HuggingFaceTB/SmolLM2-135M"  # ~250 MB, loads fine on CPU


def example_bitsandbytes():
    print("\n--- BitsAndBytes 4-bit (NF4) ---")
    q = Quantizer(MODEL)
    out = q.quantize("bitsandbytes", output_dir="./quantized/bnb_4bit", bits=4)
    print(f"Saved to: {out}")


def example_gguf():
    print("\n--- GGUF Q4_K_M ---")
    # Requires llama.cpp to be installed and built.
    # Pass the path to your llama.cpp clone:
    q = Quantizer(MODEL)
    out = q.quantize(
        "gguf",
        output_dir="./quantized/gguf",
        quantization_type="Q4_K_M",
        llama_cpp_path="/path/to/llama.cpp",  # update this
    )
    print(f"Saved to: {out}")


def example_gptq():
    print("\n--- GPTQ 4-bit ---")
    q = Quantizer(MODEL)
    out = q.quantize(
        "gptq",
        output_dir="./quantized/gptq_4bit",
        bits=4,
        group_size=128,
        dataset="wikitext2",
        num_samples=128,
    )
    print(f"Saved to: {out}")


def example_awq():
    print("\n--- AWQ 4-bit ---")
    q = Quantizer(MODEL)
    out = q.quantize("awq", output_dir="./quantized/awq_4bit")
    print(f"Saved to: {out}")


def example_pytorch():
    print("\n--- PyTorch Dynamic int8 ---")
    q = Quantizer(MODEL)
    out = q.quantize("pytorch", output_dir="./quantized/pytorch_int8", mode="dynamic")
    print(f"Saved to: {out}")


def example_quantize_all():
    print("\n--- Quantize with all no-dep backends ---")
    q = Quantizer(MODEL)
    results = q.quantize_all(
        output_dir="./quantized/all",
        methods=["bitsandbytes", "pytorch"],
        bitsandbytes={"bits": 4},
        pytorch={"mode": "dynamic"},
    )
    for method, path in results.items():
        print(f"  {method}: {path}")


def example_benchmark():
    print("\n--- Benchmark ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    stats = benchmark_inference(model, tokenizer, prompt="Once upon a time", max_new_tokens=20)
    print(f"  Avg latency: {stats['avg_latency_s']}s")
    print(f"  Tokens/sec : {stats['tokens_per_sec']}")
    print(f"  Device     : {stats['device']}")
    print(f"  Model size : {get_model_size_mb(model)} MB")


if __name__ == "__main__":
    example_pytorch()      # No extra deps needed
    example_benchmark()    # No extra deps needed
    # Uncomment to run others (requires their respective packages installed):
    # example_bitsandbytes()
    # example_gptq()
    # example_awq()
    # example_gguf()
    # example_quantize_all()
