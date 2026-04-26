import time
from typing import Optional


def benchmark_inference(
    model,
    tokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
    runs: int = 3,
    device: Optional[str] = None,
) -> dict:
    """
    Measure average latency and tokens/sec for a model.

    Returns a dict with keys: avg_latency_s, tokens_per_sec, runs.
    """
    import torch

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model = model.to(device)
    model.eval()

    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

    avg = sum(latencies) / len(latencies)
    tokens_per_sec = max_new_tokens / avg

    return {
        "avg_latency_s": round(avg, 3),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "runs": runs,
        "device": device,
    }
