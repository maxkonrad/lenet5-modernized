"""MNIST loading and downloading.

Data is never committed: files live under ``data/`` (git-ignored) and are
fetched on demand from the official mirror.
"""

from __future__ import annotations

import struct
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FILES = {
    "train-images": ("train-images.idx3-ubyte", 2051),
    "train-labels": ("train-labels.idx1-ubyte", 2049),
    "t10k-images": ("t10k-images.idx3-ubyte", 2051),
    "t10k-labels": ("t10k-labels.idx1-ubyte", 2049),
}

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def download_mnist(root: str | Path | None = None) -> Path:
    """Download the four MNIST IDX files into ``root`` (default: ``data/``)."""
    dest = Path(root) if root is not None else default_data_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for _key, (name, _magic) in FILES.items():
        target = dest / name
        if target.exists():
            continue
        url = f"{MIRROR}/{name}.gz"
        gz_target = dest / f"{name}.gz"
        urllib.request.urlretrieve(url, gz_target)
        import gzip
        import shutil

        with gzip.open(gz_target, "rb") as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
        gz_target.unlink()
    return dest


def _read_idx(path: Path, expected_magic: int) -> tuple[np.ndarray, tuple[int, ...]]:
    with open(path, "rb") as f:
        magic_bytes = f.read(4)
        if len(magic_bytes) < 4:
            raise ValueError(f"{path} is truncated: missing magic number")
        (magic,) = struct.unpack(">I", magic_bytes)
        if magic != expected_magic:
            raise ValueError(f"{path}: bad magic {magic}, expected {expected_magic}")
        if expected_magic == 2051:
            rest = f.read(12)
            if len(rest) < 12:
                raise ValueError(f"{path} is truncated: missing image header")
            count, rows, cols = struct.unpack(">III", rest)
            shape = (count, rows, cols)
        else:
            rest = f.read(4)
            if len(rest) < 4:
                raise ValueError(f"{path} is truncated: missing label header")
            (count,) = struct.unpack(">I", rest)
            shape = (count,)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: expected {int(np.prod(shape))} values, got {data.size}")
    return data.reshape(shape), shape


def _resolve(name: str, root: str | Path | None) -> Path:
    if root is not None:
        return Path(root) / name
    legacy = Path(name)  # pre-modernization layout kept working until files move
    if legacy.exists():
        return legacy
    return default_data_dir() / name


def load_images(path: str | Path) -> NDArray[np.uint8]:
    """Load an IDX3 image file as ``(N, H, W)`` uint8."""
    data, _ = _read_idx(Path(path), 2051)
    return data  # type: ignore[return-value]


def load_labels(path: str | Path) -> NDArray[np.uint8]:
    """Load an IDX1 label file as ``(N,)`` uint8."""
    data, _ = _read_idx(Path(path), 2049)
    return data  # type: ignore[return-value]


def load_mnist(
    root: str | Path | None = None,
    *,
    normalize: bool = True,
    add_channel: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.uint8], NDArray[np.floating], NDArray[np.uint8]]:
    """Load train/test MNIST splits.

    Returns ``(x_train, y_train, x_test, y_test)`` with images as float32 in
    ``[0, 1]`` shaped ``(N, 1, 28, 28)`` by default.
    """
    x_train = load_images(_resolve(FILES["train-images"][0], root)).astype(np.float32)
    y_train = load_labels(_resolve(FILES["train-labels"][0], root))
    x_test = load_images(_resolve(FILES["t10k-images"][0], root)).astype(np.float32)
    y_test = load_labels(_resolve(FILES["t10k-labels"][0], root))
    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    if add_channel:
        x_train = x_train[:, None, :, :]
        x_test = x_test[:, None, :, :]
    return x_train, y_train, x_test, y_test


def main() -> None:
    dest = download_mnist()
    print(f"MNIST ready in {dest}")


if __name__ == "__main__":
    main()
