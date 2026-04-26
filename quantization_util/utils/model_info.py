from pathlib import Path


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model) -> float:
    import torch
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return round(total_bytes / 1024 / 1024, 2)


def get_disk_size_mb(path: str | Path) -> float:
    p = Path(path)
    if p.is_file():
        return round(p.stat().st_size / 1024 / 1024, 2)
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024, 2)
