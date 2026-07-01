"""Shared local-run utilities for the MFRL tutorial experiments."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
try:
    import yaml
except Exception:  # pragma: no cover - exercised only before PyYAML is installed.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _load_simple_yaml(text)
    data["_config_path"] = str(config_path)
    return data


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> Dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        if lines[index][1].startswith("- "):
            return parse_list(index, indent)
        return parse_mapping(index, indent)

    def parse_mapping(index: int, indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent or content.startswith("- "):
                break
            if line_indent > indent:
                index += 1
                continue
            key, sep, value = content.partition(":")
            if not sep:
                raise ValueError(f"Cannot parse config line: {content}")
            key = key.strip()
            value = value.strip()
            if value:
                result[key] = _parse_scalar(value)
                index += 1
            else:
                next_index = index + 1
                if next_index < len(lines) and lines[next_index][0] > line_indent:
                    result[key], index = parse_block(next_index, lines[next_index][0])
                else:
                    result[key] = {}
                    index += 1
        return result, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        result: list[Any] = []
        while index < len(lines):
            line_indent, content = lines[index]
            if line_indent < indent or not content.startswith("- "):
                break
            item = content[2:].strip()
            index += 1
            if not item:
                if index < len(lines) and lines[index][0] > line_indent:
                    value, index = parse_block(index, lines[index][0])
                else:
                    value = None
                result.append(value)
                continue
            if ":" in item:
                key, _, value_text = item.partition(":")
                value: dict[str, Any] = {key.strip(): _parse_scalar(value_text.strip())}
                if index < len(lines) and lines[index][0] > line_indent:
                    nested, index = parse_mapping(index, lines[index][0])
                    value.update(nested)
                result.append(value)
            else:
                result.append(_parse_scalar(item))
        return result, index

    parsed, final_index = parse_block(0, 0)
    if final_index != len(lines):
        raise ValueError("Config parser did not consume all lines")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level config must be a mapping")
    return parsed


def setup_matplotlib_cache(base_dir: str | Path) -> None:
    cache_root = Path(base_dir) / ".cache"
    mpl_dir = cache_root / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


def configure_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def create_run_dir(config: Dict[str, Any]) -> Path:
    experiment = config.get("experiment", "experiment")
    output_root = Path(config.get("output_root", "outputs"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / experiment / timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_resolved_config(config: Dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "resolved_config.yaml"
    public_config = {k: v for k, v in config.items() if not k.startswith("_")}
    with output_path.open("w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(public_config, handle, sort_keys=False)
        else:
            json.dump(public_config, handle, indent=2)
    return output_path


class RunLogger:
    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.start = time.time()

    def log(self, stage: str, message: str) -> None:
        elapsed = time.time() - self.start
        line = f"[elapsed={elapsed:9.3f}s] [{stage}] {message}"
        print(line)


class _Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_output(log_path: str | Path) -> Iterator[None]:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        tee_stdout = _Tee(sys.stdout, handle)
        tee_stderr = _Tee(sys.stderr, handle)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            yield


@contextlib.contextmanager
def working_directory(path: str | Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def package_versions(modules: Iterable[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = f"unavailable: {exc.__class__.__name__}"
    return versions


def torch_device_summary() -> Dict[str, Any]:
    summary: Dict[str, Any] = {"machine": platform.machine()}
    try:
        import torch

        summary["torch"] = getattr(torch, "__version__", "unknown")
        summary["cuda_available"] = bool(torch.cuda.is_available())
        summary["mps_available"] = bool(
            torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
        )
    except Exception as exc:
        summary["torch_error"] = f"{exc.__class__.__name__}: {exc}"
    return summary


def collect_files(root: str | Path) -> list[str]:
    root_path = Path(root)
    files = []
    for path in sorted(root_path.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(root_path)))
    return files


def write_manifest(
    output_dir: str | Path,
    config: Dict[str, Any],
    *,
    status: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    output_path = Path(output_dir) / "manifest.json"
    manifest = {
        "status": status,
        "config_path": config.get("_config_path"),
        "experiment": config.get("experiment"),
        "seed": config.get("seed"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "packages": package_versions(["numpy", "scipy", "matplotlib", "gymnasium", "torch", "yaml"]),
        "torch_device": torch_device_summary(),
        "output_files": collect_files(output_dir),
    }
    if extra:
        manifest.update(extra)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return output_path


def copy_if_exists(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.exists():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
