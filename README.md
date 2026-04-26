# quantization-util

A Python package to quantize small HuggingFace models to multiple formats — ready to run on a MacBook Pro. Publish the quantized artifacts wherever you like, and use the CLI or Python API to drive the process.

```bash
pip install quantization-util
quantize --model HuggingFaceTB/SmolLM2-135M --method gguf --quant-type Q4_K_M --output ./out
```

---

## Supported quantization methods

### 1. BitsAndBytes (4-bit NF4 / 8-bit LLM.int8)

**What it does.** BitsAndBytes integrates directly with HuggingFace Transformers. In 4-bit mode it stores weights in NF4 (Normal Float 4), a 4-bit data type optimized for normally-distributed weights, optionally with a second level of quantization (double-quant) that further compresses the quantization constants. In 8-bit mode it uses the LLM.int8 algorithm, which decomposes the matrix multiplication to handle outlier features in full precision.

**What it's useful for.** The fastest path to cutting memory in half (4-bit) or by ~25% (8-bit) with minimal code changes. Works at load time — no offline calibration step. Best for fine-tuning with QLoRA, or quick local inference on consumer hardware. Does **not** produce a standalone artifact; the model stays in the HuggingFace format.

```python
from quantization_util import Quantizer

q = Quantizer("HuggingFaceTB/SmolLM2-135M")
q.quantize("bitsandbytes", output_dir="./out", bits=4, bnb_4bit_quant_type="nf4")
```

---

### 2. GGUF / llama.cpp

**What it does.** GGUF is the file format used by [llama.cpp](https://github.com/ggerganov/llama.cpp). Weights are quantized offline into a self-contained binary. The quantization family ranges from Q2_K (extreme compression) through Q4_K_M (recommended), Q5_K_M, Q8_0 (near-lossless), up to F16 (no quantization, just format conversion). The K variants use k-quants, a block-wise quantization scheme that assigns more bits to the most important weight blocks.

**What it's useful for.** Running LLMs at full speed on CPU and Apple Silicon (Metal) without a GPU. GGUF models are self-contained single files, easy to share, and supported by tools like Ollama, LM Studio, and Jan. The `Q4_K_M` type is the community standard for the best quality-to-size tradeoff.

| Type    | Size vs F16 | Quality vs F16 | Notes |
|---------|-------------|----------------|-------|
| Q2_K    | ~30%        | Noticeable loss | For extreme memory constraints |
| Q4_K_M  | ~45%        | Good            | Recommended default |
| Q5_K_M  | ~55%        | Very good       | When you can spare the extra space |
| Q8_0    | ~65%        | Near-identical  | If size is not a concern |
| F16     | 100%        | Lossless        | Just format conversion |

```python
q = Quantizer("HuggingFaceTB/SmolLM2-135M")
q.quantize("gguf", output_dir="./out", quantization_type="Q4_K_M",
           llama_cpp_path="/path/to/llama.cpp")
```

```bash
quantize --model HuggingFaceTB/SmolLM2-135M --method gguf --quant-type Q4_K_M \
         --llama-cpp-path /path/to/llama.cpp --output ./out
```

Requires: llama.cpp cloned and built (`cmake -B build && cmake --build build -j`).

---

### 3. GPTQ (Generative Pre-Trained Quantization)

**What it does.** GPTQ is a post-training quantization algorithm that minimizes quantization error layer-by-layer using second-order gradient information (the Hessian). It runs a small calibration dataset through the full-precision model once, then quantizes each linear layer's weights so that the output activations change as little as possible. Implemented via [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ).

**What it's useful for.** High-accuracy 4-bit weight quantization — often within 1–2% of the original model on benchmarks. The calibration step takes minutes, not hours. Widely supported on HuggingFace Hub (many community-uploaded GPTQ models). Use this when you need a quantized model that behaves as close as possible to the original, and you can tolerate a one-time offline calibration pass.

```python
q = Quantizer("HuggingFaceTB/SmolLM2-135M")
q.quantize("gptq", output_dir="./out", bits=4, group_size=128, dataset="wikitext2")
```

```bash
quantize --model HuggingFaceTB/SmolLM2-135M --method gptq --bits 4 --group-size 128 --output ./out
```

Requires: `pip install quantization-util[gptq]`

---

### 4. AWQ (Activation-aware Weight Quantization)

**What it does.** AWQ observes that only ~1% of weight channels are "salient" — they correspond to large activation values and dominate quantization error. Instead of quantizing all weights uniformly, AWQ scales those channels before quantization so they can be represented accurately in low bits, then folds the scaling into adjacent layers. No retraining, only a calibration forward pass. Implemented via [AutoAWQ](https://github.com/casimahmed/AutoAWQ).

**What it's useful for.** State-of-the-art 4-bit accuracy, often beating GPTQ at the same bit width. The quantized model can be served with fast GPU kernels (GEMM/GEMV). Good default choice for deployment when you want the best quality at 4-bit with a fast quantization pipeline.

```python
q = Quantizer("HuggingFaceTB/SmolLM2-135M")
q.quantize("awq", output_dir="./out", bits=4, group_size=128)
```

```bash
quantize --model HuggingFaceTB/SmolLM2-135M --method awq --bits 4 --output ./out
```

Requires: `pip install quantization-util[awq]`

---

### 5. PyTorch Native Quantization

**What it does.** Uses PyTorch's built-in `torch.quantization` (dynamic quantization) or `torchao` (weight-only int8). Dynamic quantization keeps weights in int8 on disk and dequantizes them to fp32/bf16 on the fly per layer; no calibration data required. The optional `torchao` path uses the newer `quantize_()` API for cleaner integration with `torch.compile`.

**What it's useful for.** Zero-dependency quantization — works on any CPU without installing external packages. The simplest path if you just want to reduce model size slightly and maintain full portability. Particularly effective for encoder models and smaller architectures where dynamic range per-tensor is stable. Less effective than GPTQ/AWQ for decoder-only LLMs at 4-bit, but perfectly fine for int8.

```python
q = Quantizer("HuggingFaceTB/SmolLM2-135M")
q.quantize("pytorch", output_dir="./out", mode="dynamic")      # int8 linear layers
q.quantize("pytorch", output_dir="./out", mode="weight_only")  # torchao int8 weight-only
```

```bash
quantize --model HuggingFaceTB/SmolLM2-135M --method pytorch --mode dynamic --output ./out
```

---

## Python API

```python
from quantization_util import Quantizer
from quantization_util.utils import get_model_size_mb, benchmark_inference

q = Quantizer("HuggingFaceTB/SmolLM2-135M")

# Single method
q.quantize("bitsandbytes", output_dir="./out", bits=4)

# Multiple methods at once
results = q.quantize_all(
    output_dir="./out",
    methods=["bitsandbytes", "pytorch"],
    bitsandbytes={"bits": 4},
    pytorch={"mode": "dynamic"},
)
```

### Benchmarking

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization_util.utils import benchmark_inference, get_model_size_mb

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

stats = benchmark_inference(model, tokenizer, max_new_tokens=50)
print(stats)
# {'avg_latency_s': 1.23, 'tokens_per_sec': 40.7, 'runs': 3, 'device': 'mps'}

print(get_model_size_mb(model), "MB")
```

---

## Installation

```bash
# Core (PyTorch native quantization only — no extra deps)
pip install quantization-util

# With BitsAndBytes
pip install "quantization-util[bnb]"

# With GPTQ
pip install "quantization-util[gptq]"

# With AWQ
pip install "quantization-util[awq]"

# Everything
pip install "quantization-util[all]"
```

GGUF requires a separately built [llama.cpp](https://github.com/ggerganov/llama.cpp):

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build -j
```

---

## CLI reference

```
quantize --model <hf-model-id-or-path>
         --method {bitsandbytes,gguf,gptq,awq,pytorch}
         --output <dir>

         # BitsAndBytes / GPTQ / AWQ
         --bits {2,3,4,8}          (default: 4)

         # BitsAndBytes
         --bnb-quant-type {nf4,fp4}
         --no-double-quant

         # GGUF
         --quant-type Q4_K_M       (any llama.cpp quantization type)
         --llama-cpp-path /path/to/llama.cpp

         # GPTQ
         --group-size 128
         --dataset {wikitext2,c4}
         --num-samples 128

         # PyTorch
         --mode {dynamic,weight_only}
```

---

## Models that fit on a MacBook Pro

| Model | Params | fp16 size | Loads on |
|-------|--------|-----------|----------|
| HuggingFaceTB/SmolLM2-135M | 135M | ~270 MB | Any Mac |
| HuggingFaceTB/SmolLM2-360M | 360M | ~720 MB | Any Mac |
| Qwen/Qwen2.5-0.5B | 500M | ~1 GB | Any Mac |
| google/gemma-2-2b | 2B | ~4 GB | 8 GB RAM |
| meta-llama/Llama-3.2-3B | 3B | ~6 GB | 8 GB RAM |
| mistralai/Mistral-7B-v0.3 | 7B | ~14 GB | 16 GB RAM (Q4 recommended) |

---

## Contributing

1. Fork the repo and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest tests/`
4. Lint: `ruff check . && ruff format .`
5. Open a PR.

---

## License

MIT
