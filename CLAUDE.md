# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development
pip3 install -e ".[dev]"

# Run all tests
python3 -m pytest tests/ -v

# Run a single test
python3 -m pytest tests/test_quantizers.py::test_bitsandbytes_4bit -v

# Lint
ruff check .
ruff format .

# Build distribution (produces dist/*.whl and dist/*.tar.gz)
python3 -m build

# Publish to PyPI
twine upload dist/*
```

## Architecture

The public API has two layers:

**`Quantizer` (quantizer.py)** — the user-facing facade. Takes a HuggingFace model ID or local path, routes `quantize(method, ...)` calls to the right backend, and handles constructor-vs-runtime kwarg separation (e.g. `llama_cpp_path` goes to `GGUFQuantizer.__init__`, not `.quantize()`).

**Backend classes (quantizers/)** — each extends `BaseQuantizer` and implements one `.quantize(output_dir, **kwargs) -> Path`. All heavy imports (`transformers`, `torch`, `bitsandbytes`, etc.) happen inside the method body so that importing the package never fails due to a missing optional dep.

Adding a new backend means:
1. Create `quantizers/mymethod.py` with a class extending `BaseQuantizer`
2. Export it from `quantizers/__init__.py`
3. Add it to `Quantizer._get_backend()` and `QuantMethod` literal in `quantizer.py`
4. Add an optional extras entry in `pyproject.toml`

## Testing approach

All tests are pure unit tests — no real model downloads. Since backends import heavy packages lazily (inside functions), `@patch` decorators cannot target module-level names. Instead, tests inject mocks via `patch.dict(sys.modules, {"transformers": ..., "torch": ..., "bitsandbytes": ...})`. Always include all three in the dict for BnB tests to prevent `patch.dict` from evicting `torch` from `sys.modules` after the first test, which causes reimport failures in subsequent tests.

## GGUF-specific notes

`GGUFQuantizer` is a two-step subprocess pipeline:
1. `convert_hf_to_gguf.py` (Python, from the llama.cpp repo) — converts HF weights to F16 GGUF. Uses `sys.executable` to call it, never a hardcoded `python`/`python3`.
2. `llama-quantize` (C++ binary, from `build/bin/`) — quantizes the F16 GGUF to the target type.

The converter only supports LLaMA-family and other architectures natively supported by llama.cpp. `OPTForCausalLM` and similar non-mainstream architectures will fail at step 1. Tested default model: `HuggingFaceTB/SmolLM2-135M`.

## Optional dependency extras

| Extra | Packages installed | Backend unlocked |
|---|---|---|
| `bnb` | bitsandbytes | BitsAndBytesQuantizer |
| `gptq` | auto-gptq, datasets | GPTQQuantizer |
| `awq` | autoawq, datasets | AWQQuantizer |
| `torchao` | torchao | PyTorchQuantizer weight_only mode |
| `all` | all of the above | everything |

`PyTorchQuantizer` (dynamic mode) works with zero optional extras.

## Releasing

1. Bump `version` in `pyproject.toml`
2. `rm -rf dist/ build/`
3. `python3 -m build`
4. `twine check dist/*`
5. `twine upload dist/*`
6. `git tag vX.Y.Z && git push origin vX.Y.Z`
