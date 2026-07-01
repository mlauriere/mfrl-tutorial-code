"""Local environment check for the MFRL tutorial package."""

from __future__ import annotations

from .utils import PROJECT_ROOT, package_versions, setup_matplotlib_cache, torch_device_summary


def main() -> None:
    setup_matplotlib_cache(PROJECT_ROOT / "outputs" / "env_check")
    print("ENVIRONMENT_CHECK")
    for name, version in package_versions(["numpy", "scipy", "matplotlib", "gymnasium", "torch", "yaml"]).items():
        print(f"{name}: {version}")
    for key, value in torch_device_summary().items():
        print(f"{key}: {value}")
    try:
        import numpy as np
        import torch

        value = torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).sum().item()
        print(f"torch_from_numpy_sum: {value}")
    except Exception as exc:
        print(f"torch_from_numpy_error: {exc.__class__.__name__}: {exc}")


if __name__ == "__main__":
    main()
