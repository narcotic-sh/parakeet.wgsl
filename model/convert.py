#!/usr/bin/env python3
"""Pack NVIDIA Parakeet TDT 0.6B v2 for the raw WebGPU runtime.

The source is the official Hugging Face NeMo archive. Tensors are read as
restricted mmap views and written directly into operation-native shards for
the reference WebGPU implementation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from nemo_checkpoint import (
    NEMO_ARCHIVE_BYTES,
    NEMO_ARCHIVE_FILENAME,
    NEMO_ARCHIVE_SHA256,
    NEMO_WEIGHTS_BYTES,
    NEMO_WEIGHTS_SHA256,
    NemoCheckpoint,
)


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
SOURCE_REVISION = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
SOURCE_ARCHIVE_FILENAME = NEMO_ARCHIVE_FILENAME
SOURCE_ARCHIVE_BYTES = NEMO_ARCHIVE_BYTES
SOURCE_ARCHIVE_SHA256 = NEMO_ARCHIVE_SHA256
SOURCE_WEIGHT_BYTES = NEMO_WEIGHTS_BYTES
SOURCE_WEIGHT_SHA256 = NEMO_WEIGHTS_SHA256
FORMAT_VERSION = "parakeet-webgpu-v1"
FP32_FORMAT_VERSION = "parakeet-webgpu-fp32-v1"
FP16_MANIFEST_SHA256 = (
    "11a359db3d050fd82b002c745b24a5280f3ff13a76834b548df671c95c786c65"
)
FP32_MANIFEST_SHA256 = (
    "28dee836aefc2bfb01236fda6d10e1df7447724d2489040168549999ea267b1b"
)
ALIGNMENT = 256
VECTOR_WIDTH = 4
ENCODER_LAYERS = 24
ENCODER_FRAMES = 188
ENCODER_HIDDEN_SIZE = 1024
PALETTE_SUFFIX = ".palette"
PALETTE_LAYOUT = "output-group-by-palette-lut"
PALETTE5_MATRIX_LAYOUT = "k-by-output-row-major-palette5-lsb-u32"
PALETTE6_MATRIX_LAYOUT = "k-by-output-row-major-palette6-lsb-u32"
PALETTE5_BITS = 5
PALETTE6_BITS = 6
QKV_OUTPUT_GROUP_WIDTH = 1024
FFN_SOURCE_WIDTH = 4096
ENCODER_WEIGHT_FORMAT = (
    "mixed-palette5-o-proj-palette6-fp16-lut"
)
FP32_ENCODER_WEIGHT_FORMAT = (
    "mixed-palette5-o-proj-palette6-fp32-lut"
)
MAX_FP32_CONVERSION_JOBS = 8
FP32_KMEANS_WORKER_BYTES = 3 * 1024**3 // 2
FP32_CONVERSION_RESERVE_BYTES = 4 * 1024**3
NATIVE_KMEANS_SOURCE = (
    Path(__file__).resolve().parent / "native" / "kmeans1d_sorted_unique.cpp"
)
NATIVE_KMEANS_HEADER = (
    Path(__file__).resolve().parent / "native" / "kmeans1d_sorted_unique.h"
)
NATIVE_KMEANS_SYMBOL = "parakeet_kmeans1d_sorted_unique_weighted_f64"
NATIVE_KMEANS_BUILD_RECIPE_VERSION = 1
NATIVE_KMEANS_COMPILE_FLAGS = (
    "-O3",
    "-DNDEBUG",
    "-std=c++17",
    "-fvisibility=hidden",
)

_native_kmeans_library: Any | None = None
_native_kmeans_function: Any | None = None
_native_kmeans_library_path: Path | None = None


@dataclass(frozen=True)
class TensorRecord:
    shard: str
    byteOffset: int
    byteLength: int
    dtype: str
    logicalShape: list[int]
    storageShape: list[int]
    layout: str


@dataclass(frozen=True)
class FileRecord:
    name: str
    byteLength: int
    sha256: str


@dataclass(frozen=True)
class SourceTensorSpec:
    shape: tuple[int, ...]
    ignored: bool = False


@dataclass(frozen=True)
class PackageVariant:
    precision: str
    format_version: str
    encoder_weight_format: str
    float_dtype: str
    default_output_directory: str


FP16_PACKAGE_VARIANT = PackageVariant(
    precision="fp16",
    format_version=FORMAT_VERSION,
    encoder_weight_format=ENCODER_WEIGHT_FORMAT,
    float_dtype="float16",
    default_output_directory="files",
)
FP32_PACKAGE_VARIANT = PackageVariant(
    precision="fp32",
    format_version=FP32_FORMAT_VERSION,
    encoder_weight_format=FP32_ENCODER_WEIGHT_FORMAT,
    float_dtype="float32",
    default_output_directory="files-fp32",
)
PACKAGE_VARIANTS = {
    variant.precision: variant
    for variant in (FP16_PACKAGE_VARIANT, FP32_PACKAGE_VARIANT)
}
CANONICAL_MANIFEST_SHA256 = {
    FP16_PACKAGE_VARIANT.precision: FP16_MANIFEST_SHA256,
    FP32_PACKAGE_VARIANT.precision: FP32_MANIFEST_SHA256,
}


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        name: str,
        records: dict[str, TensorRecord],
    ) -> None:
        self.name = name
        self.path = output_dir / name
        self.partial_path = output_dir / f"{name}.partial"
        self.records = records
        self.stream: Any = None
        self.offset = 0

    def __enter__(self) -> ShardWriter:
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.partial_path.open("wb")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        committed = False
        finalization_error: BaseException | None = None
        if self.stream is not None:
            if exc_type is None:
                try:
                    self._align()
                    self.stream.flush()
                    os.fsync(self.stream.fileno())
                except BaseException as error:
                    finalization_error = error
            try:
                self.stream.close()
            except BaseException as error:
                if finalization_error is not None:
                    finalization_error.add_note(
                        f"Shard stream close also failed: {error!r}"
                    )
                elif isinstance(exc, BaseException):
                    exc.add_note(f"Shard stream close also failed: {error!r}")
                else:
                    finalization_error = error
            self.stream = None
        if exc_type is None and finalization_error is None:
            try:
                os.replace(self.partial_path, self.path)
                committed = True
            except BaseException as error:
                finalization_error = error
        if not committed:
            try:
                self.partial_path.unlink(missing_ok=True)
            except BaseException as error:
                if finalization_error is not None:
                    finalization_error.add_note(
                        f"Shard partial cleanup also failed: {error!r}"
                    )
                elif isinstance(exc, BaseException):
                    exc.add_note(
                        f"Shard partial cleanup also failed: {error!r}"
                    )
                else:
                    finalization_error = error
        if finalization_error is not None:
            raise finalization_error

    def add(
        self,
        name: str,
        values: np.ndarray,
        *,
        dtype: str,
        logical_shape: tuple[int, ...] | list[int] | None = None,
        layout: str,
    ) -> None:
        if name in self.records:
            raise ValueError(f"Duplicate packed tensor {name}")
        self._align()
        array = np.ascontiguousarray(values)
        expected_dtype = {
            "float16": np.dtype("<f2"),
            "float32": np.dtype("<f4"),
            "uint32": np.dtype("<u4"),
        }.get(dtype)
        if expected_dtype is None:
            raise ValueError(f"Unsupported packed dtype {dtype}")
        if array.dtype != expected_dtype:
            array = array.astype(expected_dtype, copy=False)
        byte_offset = self.offset
        payload = memoryview(array).cast("B")
        self.stream.write(payload)
        self.offset += len(payload)
        logical = list(logical_shape if logical_shape is not None else array.shape)
        self.records[name] = TensorRecord(
            shard=self.name,
            byteOffset=byte_offset,
            byteLength=len(payload),
            dtype=dtype,
            logicalShape=logical,
            storageShape=list(array.shape),
            layout=layout,
        )

    def _align(self) -> None:
        padding = (-self.offset) % ALIGNMENT
        if padding:
            self.stream.write(b"\0" * padding)
            self.offset += padding


def native_kmeans_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (NATIVE_KMEANS_HEADER, NATIVE_KMEANS_SOURCE):
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"Unable to read native exact k-means source {path}"
            ) from error
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def native_kmeans_platform_tag() -> str:
    raw = f"{platform.system().lower()}-{platform.machine().lower()}"
    tag = "".join(character if character.isalnum() else "-" for character in raw)
    return tag.strip("-") or "unknown-platform"


def native_kmeans_library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform.startswith("linux"):
        return ".so"
    if sys.platform == "win32":
        return ".dll"
    raise RuntimeError(
        f"Native exact FP32 k-means is unsupported on platform {sys.platform!r}"
    )


def native_kmeans_build_sha256(compiler_identity: str) -> str:
    digest = hashlib.sha256()
    fields = (
        str(NATIVE_KMEANS_BUILD_RECIPE_VERSION),
        native_kmeans_source_sha256(),
        native_kmeans_platform_tag(),
        *NATIVE_KMEANS_COMPILE_FLAGS,
        compiler_identity,
    )
    for field in fields:
        digest.update(field.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def native_kmeans_cache_path(
    cache_dir: Path,
    compiler_identity: str,
) -> Path:
    build_hash = native_kmeans_build_sha256(compiler_identity)
    name = (
        "parakeet-kmeans1d-"
        f"{build_hash[:16]}-{native_kmeans_platform_tag()}"
        f"{native_kmeans_library_suffix()}"
    )
    return cache_dir / "native" / name


def native_kmeans_compiler_identity(compiler: str) -> str:
    try:
        completed = subprocess.run(
            (compiler, "--version"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(
            "Unable to inspect clang++ for the exact FP32 k-means helper"
        ) from error
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "Unable to inspect clang++ for the exact FP32 k-means helper"
            + (f":\n{detail}" if detail else "")
        )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    version = stdout or stderr
    if not version:
        raise RuntimeError(
            "clang++ returned no version identity for the exact FP32 "
            "k-means helper"
        )
    return f"{Path(compiler).resolve()}\n{version}"


def native_kmeans_compile_command(
    compiler: str,
    output_path: Path,
) -> tuple[str, ...]:
    flags = [
        compiler,
        *NATIVE_KMEANS_COMPILE_FLAGS,
    ]
    if sys.platform == "darwin":
        flags.append("-dynamiclib")
    elif sys.platform.startswith("linux"):
        flags.extend(("-fPIC", "-shared"))
    elif sys.platform == "win32":
        flags.append("-shared")
    else:
        native_kmeans_library_suffix()
    flags.extend((str(NATIVE_KMEANS_SOURCE), "-o", str(output_path)))
    return tuple(flags)


def build_native_kmeans_library(cache_dir: Path) -> Path:
    compiler = shutil.which("clang++")
    if compiler is None:
        raise RuntimeError(
            "Unable to build the exact FP32 k-means helper: clang++ was not found"
        )
    compiler_identity = native_kmeans_compiler_identity(compiler)
    library_path = native_kmeans_cache_path(
        cache_dir,
        compiler_identity,
    )
    if library_path.is_file():
        return library_path
    library_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = library_path.with_name(
        f".{library_path.name}.{os.getpid()}.partial"
    )
    command = native_kmeans_compile_command(compiler, partial_path)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Unable to execute clang++ for the exact FP32 k-means helper"
        ) from error
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    if completed.returncode:
        partial_path.unlink(missing_ok=True)
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "Failed to build the exact FP32 k-means helper with "
            f"{' '.join(command)}"
            + (f":\n{detail}" if detail else "")
        )
    if not partial_path.is_file():
        raise RuntimeError(
            "clang++ reported success but did not create the exact FP32 "
            f"k-means helper at {partial_path}"
        )
    try:
        os.replace(partial_path, library_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    return library_path


def configure_native_kmeans_library(library_path: Path) -> None:
    global _native_kmeans_function
    global _native_kmeans_library
    global _native_kmeans_library_path

    resolved_path = library_path.resolve()
    if (
        _native_kmeans_function is not None
        and _native_kmeans_library_path == resolved_path
    ):
        return
    try:
        library = ctypes.CDLL(str(resolved_path))
        function = getattr(library, NATIVE_KMEANS_SYMBOL)
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            "Unable to load the exact FP32 k-means helper "
            f"{resolved_path}: {error}"
        ) from error
    f64_pointer = ctypes.POINTER(ctypes.c_double)
    u32_pointer = ctypes.POINTER(ctypes.c_uint32)
    function.argtypes = [
        f64_pointer,
        f64_pointer,
        ctypes.c_uint32,
        ctypes.c_uint32,
        u32_pointer,
        f64_pointer,
    ]
    function.restype = ctypes.c_int
    _native_kmeans_library = library
    _native_kmeans_function = function
    _native_kmeans_library_path = resolved_path


def run_native_kmeans1d_sorted_unique(
    unique_values: np.ndarray,
    counts: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if _native_kmeans_function is None:
        raise RuntimeError(
            "The exact FP32 native k-means helper is not configured"
        )
    values_f64 = np.ascontiguousarray(unique_values, dtype=np.float64)
    weights_f64 = np.ascontiguousarray(counts, dtype=np.float64)
    if values_f64.ndim != 1 or weights_f64.shape != values_f64.shape:
        raise ValueError(
            "Native exact k-means requires matching one-dimensional values "
            "and weights"
        )
    if values_f64.size > np.iinfo(np.uint32).max:
        raise ValueError(
            "Native exact k-means input exceeds the uint32 index range: "
            f"{values_f64.size}"
        )
    if cluster_count < 1 or cluster_count > values_f64.size:
        raise ValueError(
            f"Invalid native exact k-means cluster count {cluster_count} "
            f"for {values_f64.size} values"
        )

    clusters_u32 = np.empty(values_f64.size, dtype=np.uint32)
    centroids_f64 = np.empty(cluster_count, dtype=np.float64)
    f64_pointer = ctypes.POINTER(ctypes.c_double)
    u32_pointer = ctypes.POINTER(ctypes.c_uint32)
    status = _native_kmeans_function(
        values_f64.ctypes.data_as(f64_pointer),
        weights_f64.ctypes.data_as(f64_pointer),
        values_f64.size,
        cluster_count,
        clusters_u32.ctypes.data_as(u32_pointer),
        centroids_f64.ctypes.data_as(f64_pointer),
    )
    if status:
        meanings = {
            1: "invalid argument",
            2: "allocation failure",
            3: "internal error",
        }
        raise RuntimeError(
            "Exact FP32 native k-means failed with status "
            f"{status} ({meanings.get(status, 'unknown')})"
        )
    return clusters_u32, centroids_f64


def coreml_kmeans1d() -> Any:
    try:
        from coremltools._deps import _HAS_KMEANS1D, _kmeans1d
    except ImportError as error:
        raise RuntimeError(
            "coremltools 9.0 weighted kmeans1d is unavailable"
        ) from error
    if not _HAS_KMEANS1D or _kmeans1d is None:
        raise RuntimeError("coremltools 9.0 weighted kmeans1d is unavailable")
    return _kmeans1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Package directory for a single precision. Defaults to model/files "
            "for fp16 and model/files-fp32 for fp32."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=(*PACKAGE_VARIANTS, "both"),
        default=FP16_PACKAGE_VARIANT.precision,
        help=(
            "Package precision. 'both' writes the two canonical package "
            "directories and cannot be combined with --output-dir."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        choices=tuple(range(MAX_FP32_CONVERSION_JOBS + 1)),
        default=0,
        help=(
            "Concurrent FP32 encoder-layer conversion workers. The default "
            "(0) selects up to eight workers from CPU and physical-memory "
            "capacity. FP16 conversion remains serial."
        ),
    )
    return parser.parse_args()


def package_variants_for_precision(
    precision: str,
) -> tuple[PackageVariant, ...]:
    if precision == "both":
        return (FP16_PACKAGE_VARIANT, FP32_PACKAGE_VARIANT)
    try:
        return (PACKAGE_VARIANTS[precision],)
    except KeyError as error:
        raise ValueError(f"Unsupported package precision {precision}") from error


def package_output_directories(
    precision: str,
    output_dir: Path | None,
    *,
    model_dir: Path | None = None,
) -> tuple[tuple[PackageVariant, Path], ...]:
    variants = package_variants_for_precision(precision)
    if output_dir is not None:
        if len(variants) != 1:
            raise ValueError(
                "--output-dir cannot be combined with --precision both"
            )
        return ((variants[0], output_dir),)
    root = (
        model_dir
        if model_dir is not None
        else Path(__file__).resolve().parent
    )
    return tuple(
        (variant, root / variant.default_output_directory)
        for variant in variants
    )


def physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or physical_pages <= 0:
        return None
    return page_size * physical_pages


def resolve_conversion_jobs(
    requested_jobs: int,
    variant: PackageVariant,
    *,
    cpu_count: int | None = None,
    memory_bytes: int | None = None,
) -> int:
    if requested_jobs < 0 or requested_jobs > MAX_FP32_CONVERSION_JOBS:
        raise ValueError(
            f"--jobs must be between 0 and {MAX_FP32_CONVERSION_JOBS}"
        )
    if variant.precision != FP32_PACKAGE_VARIANT.precision:
        return 1
    available_cpus = cpu_count if cpu_count is not None else os.cpu_count()
    available_cpus = max(1, int(available_cpus or 1))
    if requested_jobs:
        return min(requested_jobs, ENCODER_LAYERS, available_cpus)
    available_memory = (
        memory_bytes if memory_bytes is not None else physical_memory_bytes()
    )
    if available_memory is None:
        return 1
    memory_for_workers = max(
        0,
        available_memory - FP32_CONVERSION_RESERVE_BYTES,
    )
    memory_limited_jobs = max(
        1,
        memory_for_workers // FP32_KMEANS_WORKER_BYTES,
    )
    return min(
        MAX_FP32_CONVERSION_JOBS,
        ENCODER_LAYERS,
        available_cpus,
        memory_limited_jobs,
    )


def main() -> None:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Run the model converter with uv and Python 3.13")

    from huggingface_hub import snapshot_download

    args = parse_args()
    packages = package_output_directories(
        args.precision,
        args.output_dir,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for _, output_dir in packages:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    native_kmeans_library_path = (
        build_native_kmeans_library(args.cache_dir)
        if any(
            variant.precision == FP32_PACKAGE_VARIANT.precision
            for variant, _ in packages
        )
        else None
    )

    snapshot_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=SOURCE_REVISION,
            cache_dir=args.cache_dir / "huggingface",
            local_dir=args.cache_dir / "official",
            allow_patterns=[SOURCE_ARCHIVE_FILENAME],
        )
    )
    archive_path = snapshot_path / SOURCE_ARCHIVE_FILENAME

    with NemoCheckpoint(archive_path, validate_hashes=True) as checkpoint:
        source_specs = official_source_tensor_specs()
        validate_checkpoint_schema(checkpoint, source_specs)
        for variant, output_dir in packages:
            encoder_jobs = resolve_conversion_jobs(args.jobs, variant)
            print(
                f"Converting {variant.precision} package with "
                f"{encoder_jobs} encoder worker"
                f"{'' if encoder_jobs == 1 else 's'}",
                flush=True,
            )
            convert_package_transactionally(
                checkpoint,
                archive_path,
                output_dir,
                source_specs,
                variant,
                encoder_jobs=encoder_jobs,
                native_kmeans_library_path=native_kmeans_library_path,
            )


def convert_package_transactionally(
    checkpoint: Any,
    archive_path: Path,
    output_dir: Path,
    source_specs: dict[str, SourceTensorSpec],
    variant: PackageVariant,
    *,
    encoder_jobs: int = 1,
    native_kmeans_library_path: Path | None = None,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    validate_package_output_directory(output_dir)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".partial",
            dir=output_dir.parent,
        )
    )
    primary_error: BaseException | None = None
    try:
        convert_package(
            checkpoint,
            archive_path,
            staging_dir,
            source_specs,
            variant,
            encoder_jobs=encoder_jobs,
            native_kmeans_library_path=native_kmeans_library_path,
        )
        # Workers reopen the large archive by pathname to keep conversion
        # bounded-memory. Persistent source changes are caught here; the
        # canonical staged-manifest check inside convert_package also rejects
        # a swap-and-restore race whose converted bytes differ from the fixed
        # package identity.
        validate_source(archive_path)
        install_package_directory(staging_dir, output_dir)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if staging_dir.exists():
            try:
                shutil.rmtree(staging_dir)
            except BaseException as error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"Staged-package cleanup at {staging_dir} also failed: "
                    f"{error!r}"
                )


def install_package_directory(
    staging_dir: Path,
    output_dir: Path,
) -> None:
    if not staging_dir.is_dir():
        raise RuntimeError(
            f"Staged package directory does not exist: {staging_dir}"
        )
    if not (staging_dir / "manifest.json").is_file():
        raise RuntimeError(
            f"Staged package has no manifest: {staging_dir}"
        )
    output_mode = validate_package_output_directory(output_dir)
    staging_dir.chmod(output_mode)
    if not output_dir.exists():
        os.replace(staging_dir, output_dir)
        return

    backup_dir = output_dir.with_name(
        f".{output_dir.name}.{os.getpid()}.backup"
    )
    if backup_dir.exists():
        raise RuntimeError(
            f"Package backup path already exists: {backup_dir}"
        )
    os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
    except BaseException as install_error:
        try:
            os.replace(backup_dir, output_dir)
        except BaseException as rollback_error:
            raise BaseExceptionGroup(
                "Installing the staged package failed and restoring the "
                f"previous package at {output_dir} also failed",
                [install_error, rollback_error],
            ) from install_error
        raise
    try:
        shutil.rmtree(backup_dir)
    except BaseException as error:
        error.add_note(
            f"The new package remains installed at {output_dir}; the "
            f"previous-package backup may remain at {backup_dir}"
        )
        raise


def validate_package_output_directory(output_dir: Path) -> int:
    if output_dir.is_symlink():
        raise RuntimeError(
            f"Package output path must not be a symbolic link: {output_dir}"
        )
    if not output_dir.exists():
        return 0o755
    if not output_dir.is_dir():
        raise RuntimeError(
            f"Package output path is not a directory: {output_dir}"
        )
    if any(output_dir.iterdir()) and not (
        output_dir / "manifest.json"
    ).is_file():
        raise RuntimeError(
            "Refusing to replace a non-empty directory without a package "
            f"manifest: {output_dir}"
        )
    return stat.S_IMODE(output_dir.stat().st_mode)


def convert_package(
    checkpoint: Any,
    archive_path: Path,
    output_dir: Path,
    source_specs: dict[str, SourceTensorSpec],
    variant: PackageVariant,
    *,
    encoder_jobs: int = 1,
    native_kmeans_library_path: Path | None = None,
) -> None:
    if variant.precision == FP32_PACKAGE_VARIANT.precision:
        if native_kmeans_library_path is None:
            raise RuntimeError(
                "FP32 conversion requires the exact native k-means helper"
            )
        configure_native_kmeans_library(native_kmeans_library_path)
    available = set(checkpoint.keys())
    tensor_records: dict[str, TensorRecord] = {}
    consumed: set[str] = set()
    relative_positions = build_relative_positions()
    ignored = {
        name for name, spec in source_specs.items() if spec.ignored
    }
    pack_preprocessor(checkpoint, output_dir, tensor_records, consumed)
    if encoder_jobs == 1:
        pack_subsampling(
            checkpoint,
            output_dir,
            tensor_records,
            consumed,
            float_dtype=variant.float_dtype,
        )
        pack_encoder_layers(
            checkpoint,
            archive_path,
            output_dir,
            tensor_records,
            consumed,
            relative_positions,
            float_dtype=variant.float_dtype,
            jobs=encoder_jobs,
            native_kmeans_library_path=native_kmeans_library_path,
        )
    else:
        # Keep the parent process's large palettization allocations out of the
        # encoder-worker peak. Shards and manifest records are independent of
        # conversion order.
        pack_encoder_layers(
            checkpoint,
            archive_path,
            output_dir,
            tensor_records,
            consumed,
            relative_positions,
            float_dtype=variant.float_dtype,
            jobs=encoder_jobs,
            native_kmeans_library_path=native_kmeans_library_path,
        )
        pack_subsampling(
            checkpoint,
            output_dir,
            tensor_records,
            consumed,
            float_dtype=variant.float_dtype,
        )
    pack_decoder(
        checkpoint,
        output_dir,
        tensor_records,
        consumed,
        float_dtype=variant.float_dtype,
    )

    unexpected = available - consumed - ignored
    expected_consumed = set(source_specs) - ignored
    missing = expected_consumed - consumed
    if missing:
        raise RuntimeError(
            f"Converter did not consume expected tensors: {sorted(missing)}"
        )
    if unexpected:
        raise RuntimeError(
            f"Converter did not account for tensors: {sorted(unexpected)}"
        )

    copied_files = write_metadata_files(checkpoint, output_dir)
    shard_names = sorted({record.shard for record in tensor_records.values()})
    files = [file_record(output_dir / name) for name in shard_names]
    files.extend(file_record(path) for path in copied_files)

    manifest = {
        "format": variant.format_version,
        "source": {
            "repository": MODEL_ID,
            "revision": SOURCE_REVISION,
            "archive": {
                "name": SOURCE_ARCHIVE_FILENAME,
                "byteLength": SOURCE_ARCHIVE_BYTES,
                "sha256": SOURCE_ARCHIVE_SHA256,
            },
            "weights": {
                "name": "model_weights.ckpt",
                "byteLength": SOURCE_WEIGHT_BYTES,
                "sha256": SOURCE_WEIGHT_SHA256,
            },
            "license": "CC-BY-4.0",
        },
        "config": {
            "sampleRate": 16_000,
            "windowSamples": 240_000,
            "windowSeconds": 15,
            "windowOverlapSeconds": 2,
            "featureBins": 128,
            "fftSize": 512,
            "windowSize": 400,
            "hopSize": 160,
            "preemphasis": 0.97,
            "logZeroGuard": 2**-24,
            "normalizationEpsilon": 1e-5,
            "subsamplingFactor": 8,
            "encoderLayers": 24,
            "encoderHiddenSize": 1024,
            "intermediateSize": FFN_SOURCE_WIDTH,
            "attentionHeads": 8,
            "attentionHeadSize": 128,
            "convolutionKernelSize": 9,
            "decoderHiddenSize": 640,
            "decoderLayers": 2,
            "vocabularySize": 1025,
            "blankTokenId": 1024,
            "jointOutputSize": 1030,
            "jointStorageOutputSize": 1032,
            "durations": [0, 1, 2, 3, 4],
            "maxSymbolsPerStep": 10,
            "encoderWeightFormat": variant.encoder_weight_format,
        },
        "files": [
            asdict(record)
            for record in sorted(files, key=lambda item: item.name)
        ],
        "tensors": {
            name: asdict(record)
            for name, record in sorted(tensor_records.items())
        },
    }
    # Preserve the established FP16 package byte-for-byte. Its canonical
    # manifest predates the converter's trailing newline, while the separately
    # introduced FP32 package includes one.
    write_json_atomic(
        output_dir / "manifest.json",
        manifest,
        variant.precision == FP32_PACKAGE_VARIANT.precision,
    )
    validate_staged_package_identity(output_dir, variant)
    print(
        f"Wrote {len(tensor_records)} tensors in {len(shard_names)} shards "
        f"({sum(record.byteLength for record in files):,} bytes)"
    )


def validate_staged_package_identity(
    output_dir: Path,
    variant: PackageVariant,
) -> None:
    """Require the complete converted package to reproduce its fixed identity."""
    try:
        expected = CANONICAL_MANIFEST_SHA256[variant.precision]
    except KeyError as error:
        raise RuntimeError(
            f"No canonical manifest identity for precision {variant.precision}"
        ) from error
    manifest_path = output_dir / "manifest.json"
    digest = sha256_file(manifest_path)
    if digest != expected:
        raise RuntimeError(
            f"Converted {variant.precision} manifest SHA-256 mismatch: "
            f"expected {expected}, got {digest}. The staged package does not "
            "reproduce the pinned source and conversion identity."
        )


def validate_source(archive_path: Path) -> None:
    """Validate the outer source identity without parsing its archive members."""
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    size = archive_path.stat().st_size
    if size != SOURCE_ARCHIVE_BYTES:
        raise RuntimeError(
            "Official NeMo archive length changed: expected "
            f"{SOURCE_ARCHIVE_BYTES}, got {size}"
        )
    digest = sha256_file(archive_path)
    if digest != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            "Official NeMo archive SHA-256 mismatch: expected "
            f"{SOURCE_ARCHIVE_SHA256}, got {digest}"
        )


def official_source_tensor_specs() -> dict[str, SourceTensorSpec]:
    """Return the complete canonical schema for the pinned NeMo checkpoint."""
    specs: dict[str, SourceTensorSpec] = {}

    def add(name: str, shape: tuple[int, ...], *, ignored: bool = False) -> None:
        if name in specs:
            raise AssertionError(f"Duplicate source tensor specification {name}")
        specs[name] = SourceTensorSpec(shape=shape, ignored=ignored)

    add("preprocessor.window", (400,))
    add("preprocessor.mel_filters", (1, 128, 257))

    for layer_index, weight_shape in (
        (0, (256, 1, 3, 3)),
        (2, (256, 1, 3, 3)),
        (3, (256, 256, 1, 1)),
        (5, (256, 1, 3, 3)),
        (6, (256, 256, 1, 1)),
    ):
        prefix = f"encoder.subsampling.layers.{layer_index}"
        add(f"{prefix}.weight", weight_shape)
        add(f"{prefix}.bias", (256,))
    add("encoder.subsampling.linear.weight", (1024, 4096))
    add("encoder.subsampling.linear.bias", (1024,))

    for layer_index in range(ENCODER_LAYERS):
        prefix = f"encoder.layers.{layer_index}"
        for feed_forward in ("feed_forward1", "feed_forward2"):
            add(f"{prefix}.{feed_forward}.linear1.weight", (4096, 1024))
            add(f"{prefix}.{feed_forward}.linear2.weight", (1024, 4096))

        for norm in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            add(f"{prefix}.{norm}.weight", (1024,))
            add(f"{prefix}.{norm}.bias", (1024,))

        attention_prefix = f"{prefix}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj", "relative_k_proj", "o_proj"):
            add(f"{attention_prefix}.{projection}.weight", (1024, 1024))
        add(f"{attention_prefix}.bias_u", (8, 128))
        add(f"{attention_prefix}.bias_v", (8, 128))

        convolution_prefix = f"{prefix}.conv"
        add(f"{convolution_prefix}.pointwise_conv1.weight", (2048, 1024, 1))
        add(f"{convolution_prefix}.depthwise_conv.weight", (1024, 1, 9))
        add(f"{convolution_prefix}.pointwise_conv2.weight", (1024, 1024, 1))
        for parameter in ("weight", "bias", "running_mean", "running_var"):
            add(f"{convolution_prefix}.norm.{parameter}", (1024,))
        add(
            f"{convolution_prefix}.norm.num_batches_tracked",
            (),
            ignored=True,
        )

    add("decoder.embedding.weight", (1025, 640))
    for layer_index in range(2):
        for kind in ("ih", "hh"):
            add(f"decoder.lstm.weight_{kind}_l{layer_index}", (2560, 640))
            add(f"decoder.lstm.bias_{kind}_l{layer_index}", (2560,))
    add("decoder.decoder_projector.weight", (640, 640))
    add("decoder.decoder_projector.bias", (640,))
    add("encoder_projector.weight", (640, 1024))
    add("encoder_projector.bias", (640,))
    add("joint.head.weight", (1030, 640))
    add("joint.head.bias", (1030,))
    return specs


def validate_checkpoint_schema(
    checkpoint: Any,
    specs: dict[str, SourceTensorSpec] | None = None,
) -> None:
    """Validate names and shapes without materializing any tensor payload."""
    expected = specs if specs is not None else official_source_tensor_specs()
    available = set(checkpoint.keys())
    expected_names = set(expected)
    missing = sorted(expected_names - available)
    unexpected = sorted(available - expected_names)
    errors: list[str] = []
    if missing:
        errors.append(f"missing names: {missing}")
    if unexpected:
        errors.append(f"unexpected names: {unexpected}")

    for name in sorted(expected_names & available):
        try:
            actual_shape = tuple(int(value) for value in checkpoint.get_slice(name).get_shape())
        except Exception as error:
            raise RuntimeError(f"Unable to inspect source tensor {name}") from error
        expected_shape = expected[name].shape
        if actual_shape != expected_shape:
            errors.append(
                f"{name}: expected shape {expected_shape}, got {actual_shape}"
            )

    if errors:
        raise RuntimeError("Official checkpoint schema mismatch:\n" + "\n".join(errors))


def numpy_float_dtype(float_dtype: str) -> np.dtype[Any]:
    try:
        return {
            "float16": np.dtype("<f2"),
            "float32": np.dtype("<f4"),
        }[float_dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported package float dtype {float_dtype}") from error


def build_relative_positions(
    encoder_frames: int = ENCODER_FRAMES,
    channels: int = ENCODER_HIDDEN_SIZE,
) -> np.ndarray:
    """Build the checkpoint's descending Transformer-XL position table."""
    if encoder_frames < 1:
        raise ValueError("encoder_frames must be positive")
    if channels < 2 or channels % 2:
        raise ValueError("channels must be a positive even number")
    positions = np.arange(
        encoder_frames - 1,
        -encoder_frames,
        -1,
        dtype=np.float32,
    ).reshape(-1, 1)
    channel_indices = np.arange(0, channels, 2, dtype=np.float32)
    exponent = np.float32(-(math.log(10_000.0) / channels))
    div_term = np.exp(channel_indices * exponent)
    angles = positions * div_term.reshape(1, -1)
    table = np.empty((2 * encoder_frames - 1, channels), dtype=np.float32)
    table[:, 0::2] = np.sin(angles)
    table[:, 1::2] = np.cos(angles)
    return table.astype("<f4", copy=False)


def project_relative_positions(
    positions: np.ndarray,
    weight: np.ndarray,
    *,
    float_dtype: str = "float16",
) -> np.ndarray:
    """Apply nn.Linear's source `[out, in]` orientation in float32."""
    if positions.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            "Expected positions [relative_position, channel] and weight [output, input], "
            f"got {positions.shape} and {weight.shape}"
        )
    if positions.shape[1] != weight.shape[1]:
        raise ValueError(
            f"Relative position input width {positions.shape[1]} does not match "
            f"projection width {weight.shape[1]}"
        )
    projected = np.asarray(positions, dtype=np.float32) @ np.asarray(
        weight,
        dtype=np.float32,
    ).T
    return np.ascontiguousarray(
        projected,
        dtype=numpy_float_dtype(float_dtype),
    )


def pack_projected_relative_positions(
    projected: np.ndarray,
    *,
    heads: int = 8,
    vector_width: int = VECTOR_WIDTH,
    float_dtype: str = "float16",
) -> np.ndarray:
    """Pack `[position, channel]` for contiguous relative rows per head vector."""
    values = np.asarray(projected)
    if values.ndim != 2:
        raise ValueError(
            "Expected projected relative positions [position, channel], "
            f"got {values.shape}"
        )
    position_rows, channels = (int(value) for value in values.shape)
    if heads <= 0 or channels % heads:
        raise ValueError(
            f"Projected relative-position channels {channels} must divide "
            f"evenly across {heads} heads"
        )
    head_size = channels // heads
    if vector_width <= 0 or head_size % vector_width:
        raise ValueError(
            f"Relative-position head size {head_size} must be divisible by "
            f"vector width {vector_width}"
        )
    storage_dtype = numpy_float_dtype(float_dtype)
    packed = np.asarray(values, dtype=storage_dtype).reshape(
        position_rows,
        heads,
        head_size // vector_width,
        vector_width,
    ).transpose(1, 2, 0, 3)
    return np.ascontiguousarray(packed, dtype=storage_dtype)


def pack_preprocessor(
    checkpoint: Any,
    output_dir: Path,
    records: dict[str, TensorRecord],
    consumed: set[str],
) -> None:
    window = tensor(checkpoint, "preprocessor.window", consumed)
    mel_filters_source = tensor(
        checkpoint,
        "preprocessor.mel_filters",
        consumed,
    )
    if mel_filters_source.shape != (1, 128, 257):
        raise ValueError(
            "Expected embedded preprocessor mel filters [1, 128, 257], got "
            f"{mel_filters_source.shape}"
        )
    mel_filters = mel_filters_source[0]
    with ShardWriter(output_dir, "preprocessor.bin", records) as shard:
        shard.add(
            "preprocessor.window",
            window,
            dtype="float32",
            layout="hann-periodic-false",
        )
        shard.add(
            "preprocessor.mel_filters",
            mel_filters,
            dtype="float32",
            layout="mel-bin-by-rfft-bin",
        )


def pack_subsampling(
    checkpoint: Any,
    output_dir: Path,
    records: dict[str, TensorRecord],
    consumed: set[str],
    *,
    float_dtype: str = "float16",
) -> None:
    with ShardWriter(output_dir, "encoder-subsampling.bin", records) as shard:
        for layer_index, packer, layout in (
            (0, pack_conv0_kernel, "kh-kw-output-channel-vec4"),
            (2, pack_depthwise_conv2d_kernel, "kh-kw-channel-vec4"),
            (5, pack_depthwise_conv2d_kernel, "kh-kw-channel-vec4"),
        ):
            prefix = f"encoder.subsampling.layers.{layer_index}"
            weight_name = f"{prefix}.weight"
            packed, logical_shape = packer(
                tensor(checkpoint, weight_name, consumed),
                float_dtype=float_dtype,
            )
            shard.add(
                weight_name,
                packed,
                dtype=float_dtype,
                logical_shape=logical_shape,
                layout=layout,
            )
            add_source(
                checkpoint,
                shard,
                consumed,
                f"{prefix}.bias",
                dtype="float32",
                layout="output-channel",
            )
        for layer_index in (3, 6):
            prefix = f"encoder.subsampling.layers.{layer_index}"
            add_matrix(
                checkpoint,
                shard,
                consumed,
                f"{prefix}.weight",
                packed_name=f"{prefix}.weight",
                squeeze=True,
                palette_bits=PALETTE5_BITS,
                float_dtype=float_dtype,
            )
            add_source(
                checkpoint,
                shard,
                consumed,
                f"{prefix}.bias",
                dtype="float32",
                layout="output-channel",
            )
        linear_name = "encoder.subsampling.linear.weight"
        linear_source = tensor(checkpoint, linear_name, consumed)
        linear_packed, logical_shape = pack_subsampling_linear_matrix(
            linear_source,
            float_dtype=float_dtype,
        )
        add_palettized_matrix(
            shard,
            linear_name,
            linear_packed,
            logical_shape=logical_shape,
            bits=PALETTE5_BITS,
            float_dtype=float_dtype,
        )
        add_source(
            checkpoint,
            shard,
            consumed,
            "encoder.subsampling.linear.bias",
            dtype="float32",
            layout="output-channel",
        )


def pack_encoder_layer(
    checkpoint: Any,
    layer_index: int,
    output_dir: Path,
    records: dict[str, TensorRecord],
    consumed: set[str],
    relative_positions: np.ndarray,
    *,
    float_dtype: str = "float16",
) -> None:
    prefix = f"encoder.layers.{layer_index}"
    with ShardWriter(
        output_dir,
        f"encoder-layer-{layer_index:02d}.bin",
        records,
    ) as shard:
        for feed_forward in ("feed_forward1", "feed_forward2"):
            branch_prefix = f"{prefix}.{feed_forward}"
            for linear in ("linear1", "linear2"):
                add_matrix(
                    checkpoint,
                    shard,
                    consumed,
                    f"{branch_prefix}.{linear}.weight",
                    palette_bits=PALETTE5_BITS,
                    float_dtype=float_dtype,
                )

        for norm in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            for parameter in ("weight", "bias"):
                add_source(
                    checkpoint,
                    shard,
                    consumed,
                    f"{prefix}.{norm}.{parameter}",
                    dtype="float32",
                    layout="channel",
                )

        attention_prefix = f"{prefix}.self_attn"
        qkv_sources = [
            f"{attention_prefix}.q_proj.weight",
            f"{attention_prefix}.k_proj.weight",
            f"{attention_prefix}.v_proj.weight",
        ]
        qkv_packed, logical_shape = pack_output_concatenated_matrices(
            checkpoint,
            qkv_sources,
            consumed,
            float_dtype=float_dtype,
        )
        add_palettized_matrix(
            shard,
            f"{attention_prefix}.qkv.weight",
            qkv_packed,
            logical_shape=logical_shape,
            bits=PALETTE5_BITS,
            output_group_width=QKV_OUTPUT_GROUP_WIDTH,
            float_dtype=float_dtype,
        )
        relative_weight_name = f"{attention_prefix}.relative_k_proj.weight"
        projected_positions = project_relative_positions(
            relative_positions,
            tensor(checkpoint, relative_weight_name, consumed),
            float_dtype=float_dtype,
        )
        packed_projected_positions = pack_projected_relative_positions(
            projected_positions,
            float_dtype=float_dtype,
        )
        shard.add(
            f"{attention_prefix}.projected_positions",
            packed_projected_positions,
            dtype=float_dtype,
            logical_shape=projected_positions.shape,
            layout="head-vector-position-vec4",
        )
        add_matrix(
            checkpoint,
            shard,
            consumed,
            f"{attention_prefix}.o_proj.weight",
            palette_bits=PALETTE6_BITS,
            float_dtype=float_dtype,
        )
        for bias in ("bias_u", "bias_v"):
            add_source(
                checkpoint,
                shard,
                consumed,
                f"{attention_prefix}.{bias}",
                dtype="float32",
                layout="head-by-channel",
            )

        convolution_prefix = f"{prefix}.conv"
        add_matrix(
            checkpoint,
            shard,
            consumed,
            f"{convolution_prefix}.pointwise_conv1.weight",
            squeeze=True,
            palette_bits=PALETTE5_BITS,
            float_dtype=float_dtype,
        )
        depthwise_name = f"{convolution_prefix}.depthwise_conv.weight"
        depthwise, logical_shape = pack_conformer_depthwise_kernel(
            tensor(checkpoint, depthwise_name, consumed),
            float_dtype=float_dtype,
        )
        shard.add(
            depthwise_name,
            depthwise,
            dtype=float_dtype,
            logical_shape=logical_shape,
            layout="kernel-channel-vec4",
        )
        add_matrix(
            checkpoint,
            shard,
            consumed,
            f"{convolution_prefix}.pointwise_conv2.weight",
            squeeze=True,
            palette_bits=PALETTE5_BITS,
            float_dtype=float_dtype,
        )
        fold_batch_norm(checkpoint, shard, consumed, f"{convolution_prefix}.norm")


def _pack_encoder_layer_task(
    task: tuple[str, int, str, str, str],
) -> tuple[int, dict[str, TensorRecord], set[str]]:
    (
        archive_path_value,
        layer_index,
        output_dir_value,
        float_dtype,
        native_kmeans_library_path_value,
    ) = task
    if float_dtype == FP32_PACKAGE_VARIANT.float_dtype:
        configure_native_kmeans_library(
            Path(native_kmeans_library_path_value)
        )
    records: dict[str, TensorRecord] = {}
    consumed: set[str] = set()
    with NemoCheckpoint(
        Path(archive_path_value),
        validate_hashes=False,
    ) as checkpoint:
        pack_encoder_layer(
            checkpoint,
            layer_index,
            Path(output_dir_value),
            records,
            consumed,
            build_relative_positions(),
            float_dtype=float_dtype,
        )
    return layer_index, records, consumed


def _merge_encoder_layer_result(
    result: tuple[int, dict[str, TensorRecord], set[str]],
    records: dict[str, TensorRecord],
    consumed: set[str],
) -> int:
    layer_index, layer_records, layer_consumed = result
    duplicate_records = records.keys() & layer_records.keys()
    if duplicate_records:
        raise RuntimeError(
            "Parallel encoder conversion produced duplicate tensors: "
            f"{sorted(duplicate_records)}"
        )
    duplicate_sources = consumed & layer_consumed
    if duplicate_sources:
        raise RuntimeError(
            "Parallel encoder conversion consumed source tensors twice: "
            f"{sorted(duplicate_sources)}"
        )
    records.update(layer_records)
    consumed.update(layer_consumed)
    return layer_index


def pack_encoder_layers(
    checkpoint: Any,
    archive_path: Path,
    output_dir: Path,
    records: dict[str, TensorRecord],
    consumed: set[str],
    relative_positions: np.ndarray,
    *,
    float_dtype: str,
    jobs: int,
    native_kmeans_library_path: Path | None,
) -> None:
    if jobs < 1:
        raise ValueError(f"Encoder conversion jobs must be positive, got {jobs}")
    if (
        float_dtype == FP32_PACKAGE_VARIANT.float_dtype
        and native_kmeans_library_path is None
    ):
        raise RuntimeError(
            "FP32 encoder conversion requires the exact native k-means helper"
        )
    started = perf_counter()
    if jobs == 1:
        for layer_index in range(ENCODER_LAYERS):
            pack_encoder_layer(
                checkpoint,
                layer_index,
                output_dir,
                records,
                consumed,
                relative_positions,
                float_dtype=float_dtype,
            )
            print(
                f"Packed encoder layer {layer_index + 1}/{ENCODER_LAYERS} "
                f"in {perf_counter() - started:.1f}s",
                flush=True,
            )
        return

    tasks = (
        (
            str(archive_path),
            layer_index,
            str(output_dir),
            float_dtype,
            (
                str(native_kmeans_library_path)
                if native_kmeans_library_path is not None
                else ""
            ),
        )
        for layer_index in range(ENCODER_LAYERS)
    )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        completed = 0
        for result in executor.map(_pack_encoder_layer_task, tasks, chunksize=1):
            layer_index = _merge_encoder_layer_result(
                result,
                records,
                consumed,
            )
            completed += 1
            print(
                f"Packed encoder layer {layer_index + 1}/{ENCODER_LAYERS} "
                f"({completed} complete) in {perf_counter() - started:.1f}s",
                flush=True,
            )


def pack_decoder(
    checkpoint: Any,
    output_dir: Path,
    records: dict[str, TensorRecord],
    consumed: set[str],
    *,
    float_dtype: str = "float16",
) -> None:
    with ShardWriter(output_dir, "decoder.bin", records) as shard:
        add_source(
            checkpoint,
            shard,
            consumed,
            "decoder.embedding.weight",
            dtype=float_dtype,
            layout="token-by-channel-row-major",
        )
        for layer_index in range(2):
            input_weight_name = f"decoder.lstm.weight_ih_l{layer_index}"
            recurrent_weight_name = f"decoder.lstm.weight_hh_l{layer_index}"
            packed, logical_shape = pack_input_concatenated_matrices(
                checkpoint,
                [input_weight_name, recurrent_weight_name],
                consumed,
                float_dtype=float_dtype,
            )
            shard.add(
                f"decoder.lstm.weight_l{layer_index}",
                packed,
                dtype=float_dtype,
                logical_shape=logical_shape,
                layout="input-hidden-k-by-ifgo-row-major-vec4",
            )
            input_bias_name = f"decoder.lstm.bias_ih_l{layer_index}"
            recurrent_bias_name = f"decoder.lstm.bias_hh_l{layer_index}"
            combined_bias = np.array(
                tensor(checkpoint, input_bias_name, consumed),
                dtype="<f4",
                copy=True,
            )
            combined_bias += tensor(checkpoint, recurrent_bias_name, consumed)
            shard.add(
                f"decoder.lstm.bias_l{layer_index}",
                combined_bias,
                dtype="float32",
                layout="ifgo-gates",
            )

        add_matrix(
            checkpoint,
            shard,
            consumed,
            "decoder.decoder_projector.weight",
            float_dtype=float_dtype,
        )
        add_source(
            checkpoint,
            shard,
            consumed,
            "decoder.decoder_projector.bias",
            dtype="float32",
            layout="output-channel",
        )
        add_matrix(
            checkpoint,
            shard,
            consumed,
            "encoder_projector.weight",
            palette_bits=PALETTE5_BITS,
            float_dtype=float_dtype,
        )
        add_source(
            checkpoint,
            shard,
            consumed,
            "encoder_projector.bias",
            dtype="float32",
            layout="output-channel",
        )
        add_matrix(
            checkpoint,
            shard,
            consumed,
            "joint.head.weight",
            float_dtype=float_dtype,
        )
        joint_bias = tensor(checkpoint, "joint.head.bias", consumed)
        padded_bias = pad_last_dimension(joint_bias, multiple=VECTOR_WIDTH)
        shard.add(
            "joint.head.bias",
            padded_bias,
            dtype="float32",
            logical_shape=joint_bias.shape,
            layout="output-channel-padded-vec4",
        )


def fold_batch_norm(
    checkpoint: Any,
    shard: ShardWriter,
    consumed: set[str],
    prefix: str,
) -> None:
    variance = tensor(checkpoint, f"{prefix}.running_var", consumed)
    denominator = np.sqrt(
        np.asarray(variance, dtype=np.float32) + np.float32(1e-5)
    )
    del variance
    gamma = tensor(checkpoint, f"{prefix}.weight", consumed)
    scale = np.asarray(gamma, dtype=np.float32) / denominator
    del gamma, denominator
    mean = tensor(checkpoint, f"{prefix}.running_mean", consumed)
    shift = -np.asarray(mean, dtype=np.float32) * scale
    del mean
    shift += tensor(checkpoint, f"{prefix}.bias", consumed)
    shard.add(
        f"{prefix}.scale",
        scale,
        dtype="float32",
        layout="channel",
    )
    shard.add(
        f"{prefix}.shift",
        shift,
        dtype="float32",
        layout="channel",
    )


def add_matrix(
    checkpoint: Any,
    shard: ShardWriter,
    consumed: set[str],
    source_name: str,
    *,
    packed_name: str | None = None,
    squeeze: bool = False,
    palette_bits: int | None = None,
    float_dtype: str = "float16",
) -> None:
    source = tensor(checkpoint, source_name, consumed)
    if squeeze:
        source = np.squeeze(source)
    packed, logical_shape = pack_matrix(
        source,
        float_dtype=float_dtype,
    )
    name = packed_name or source_name
    if palette_bits is not None:
        add_palettized_matrix(
            shard,
            name,
            packed,
            logical_shape=logical_shape,
            bits=palette_bits,
            float_dtype=float_dtype,
        )
    else:
        shard.add(
            name,
            packed,
            dtype=float_dtype,
            logical_shape=logical_shape,
            layout="k-by-output-row-major-vec4",
        )


def add_palettized_matrix(
    shard: ShardWriter,
    name: str,
    packed: np.ndarray,
    *,
    logical_shape: tuple[int, int],
    bits: int,
    output_group_width: int | None = None,
    float_dtype: str = "float16",
) -> None:
    packed_codes, palettes = palettize_matrix(
        packed,
        bits=bits,
        output_group_width=output_group_width,
        float_dtype=float_dtype,
    )
    layout = {
        PALETTE5_BITS: PALETTE5_MATRIX_LAYOUT,
        PALETTE6_BITS: PALETTE6_MATRIX_LAYOUT,
    }[bits]
    shard.add(
        name,
        packed_codes,
        dtype="uint32",
        logical_shape=logical_shape,
        layout=layout,
    )
    shard.add(
        f"{name}{PALETTE_SUFFIX}",
        palettes,
        dtype=float_dtype,
        layout=PALETTE_LAYOUT,
    )


def add_source(
    checkpoint: Any,
    shard: ShardWriter,
    consumed: set[str],
    source_name: str,
    *,
    dtype: str,
    layout: str,
) -> None:
    values = tensor(checkpoint, source_name, consumed)
    shard.add(
        source_name,
        values,
        dtype=dtype,
        layout=layout,
    )


def tensor(checkpoint: Any, name: str, consumed: set[str]) -> np.ndarray:
    if name in consumed:
        raise RuntimeError(f"Source tensor would be loaded more than once: {name}")
    try:
        values = checkpoint.get_tensor(name)
    except Exception as error:
        raise KeyError(f"Unable to load required tensor {name}") from error
    consumed.add(name)
    return values


def _pack_single_input_conv2d_kernel(
    source: np.ndarray,
    *,
    description: str,
    float_dtype: str,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    if source.ndim != 4 or source.shape[1] != 1:
        raise ValueError(
            f"Expected {description} source [channels, 1, kh, kw], got {source.shape}"
        )
    channels, _, kernel_height, kernel_width = (
        int(value) for value in source.shape
    )
    if channels % VECTOR_WIDTH:
        raise ValueError(
            f"{description} channels must be divisible by {VECTOR_WIDTH}, got {channels}"
        )
    kh_kw_channel = np.ascontiguousarray(
        source[:, 0, :, :].transpose(1, 2, 0),
        dtype=numpy_float_dtype(float_dtype),
    )
    packed = kh_kw_channel.reshape(
        kernel_height,
        kernel_width,
        channels // VECTOR_WIDTH,
        VECTOR_WIDTH,
    )
    return packed, (kernel_height, kernel_width, channels)


def pack_conv0_kernel(
    source: np.ndarray,
    *,
    float_dtype: str = "float16",
) -> tuple[np.ndarray, tuple[int, int, int]]:
    return _pack_single_input_conv2d_kernel(
        source,
        description="conv0",
        float_dtype=float_dtype,
    )


def pack_depthwise_conv2d_kernel(
    source: np.ndarray,
    *,
    float_dtype: str = "float16",
) -> tuple[np.ndarray, tuple[int, int, int]]:
    return _pack_single_input_conv2d_kernel(
        source,
        description="depthwise Conv2D",
        float_dtype=float_dtype,
    )


def pack_conformer_depthwise_kernel(
    source: np.ndarray,
    *,
    float_dtype: str = "float16",
) -> tuple[np.ndarray, tuple[int, int]]:
    if source.ndim != 3 or source.shape[1] != 1:
        raise ValueError(
            "Expected Conformer depthwise source [channels, 1, kernel], "
            f"got {source.shape}"
        )
    channels, _, kernel_size = (int(value) for value in source.shape)
    if channels % VECTOR_WIDTH:
        raise ValueError(
            "Conformer depthwise channels must be divisible by "
            f"{VECTOR_WIDTH}, got {channels}"
        )
    kernel_channel = np.ascontiguousarray(
        source[:, 0, :].T,
        dtype=numpy_float_dtype(float_dtype),
    )
    packed = kernel_channel.reshape(
        kernel_size,
        channels // VECTOR_WIDTH,
        VECTOR_WIDTH,
    )
    return packed, (kernel_size, channels)


def _source_matrix_shape(checkpoint: Any, name: str) -> tuple[int, int]:
    try:
        shape = tuple(int(value) for value in checkpoint.get_slice(name).get_shape())
    except Exception as error:
        raise KeyError(f"Unable to inspect required tensor {name}") from error
    if len(shape) != 2:
        raise ValueError(f"Expected source matrix {name}, got {shape}")
    return shape


def pack_output_concatenated_matrices(
    checkpoint: Any,
    source_names: list[str],
    consumed: set[str],
    *,
    float_dtype: str = "float16",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pack `[out, in]` matrices concatenated on output, one source at a time."""
    if not source_names:
        raise ValueError("At least one output-concatenated matrix is required")
    shapes = [_source_matrix_shape(checkpoint, name) for name in source_names]
    input_width = shapes[0][1]
    if any(shape[1] != input_width for shape in shapes):
        raise ValueError(f"Output-concatenated input widths differ: {shapes}")
    output_width = sum(shape[0] for shape in shapes)
    storage_width = (
        (output_width + VECTOR_WIDTH - 1) // VECTOR_WIDTH
    ) * VECTOR_WIDTH
    packed = np.zeros(
        (input_width, storage_width),
        dtype=numpy_float_dtype(float_dtype),
    )
    output_offset = 0
    for name, (source_output, _) in zip(source_names, shapes, strict=True):
        source = tensor(checkpoint, name, consumed)
        packed[:, output_offset : output_offset + source_output] = source.T
        output_offset += source_output
        del source
    return packed, (input_width, output_width)


def pack_input_concatenated_matrices(
    checkpoint: Any,
    source_names: list[str],
    consumed: set[str],
    *,
    float_dtype: str = "float16",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pack `[out, in]` matrices concatenated on input, one source at a time."""
    if not source_names:
        raise ValueError("At least one input-concatenated matrix is required")
    shapes = [_source_matrix_shape(checkpoint, name) for name in source_names]
    output_width = shapes[0][0]
    if any(shape[0] != output_width for shape in shapes):
        raise ValueError(f"Input-concatenated output widths differ: {shapes}")
    input_width = sum(shape[1] for shape in shapes)
    storage_width = (
        (output_width + VECTOR_WIDTH - 1) // VECTOR_WIDTH
    ) * VECTOR_WIDTH
    packed = np.zeros(
        (input_width, storage_width),
        dtype=numpy_float_dtype(float_dtype),
    )
    input_offset = 0
    for name, (_, source_input) in zip(source_names, shapes, strict=True):
        source = tensor(checkpoint, name, consumed)
        packed[input_offset : input_offset + source_input, :output_width] = source.T
        input_offset += source_input
        del source
    return packed, (input_width, output_width)


def pack_matrix(
    source: np.ndarray,
    *,
    float_dtype: str = "float16",
) -> tuple[np.ndarray, tuple[int, int]]:
    if source.ndim != 2:
        raise ValueError(f"Expected a matrix, got {source.shape}")
    logical = (int(source.shape[1]), int(source.shape[0]))
    transposed = np.ascontiguousarray(
        source.T,
        dtype=numpy_float_dtype(float_dtype),
    )
    return pad_last_dimension(transposed, multiple=VECTOR_WIDTH), logical


def fit_fp16_palette(
    values: np.ndarray,
    *,
    palette_entries: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Core ML 9's exact weighted 1D k-means palette to FP16 values.

    The fitter operates on sorted unique FP16 values and their occurrence
    counts, exactly like Core ML's scalar palettization path. Centroids are
    rounded to their runtime FP16 representation after fitting. The globally
    optimal assignments are deliberately retained after that rounding.
    """
    if palette_entries < 1 or palette_entries > 256:
        raise ValueError(
            f"palette_entries must be between 1 and 256, got {palette_entries}"
        )

    flattened = np.ascontiguousarray(values, dtype="<f2").reshape(-1)
    if flattened.size == 0:
        raise ValueError("Cannot fit a palette to an empty tensor")
    if not np.all(np.isfinite(flattened)):
        raise ValueError("Palette source values must all be finite")

    unique_values, inverse, counts = np.unique(
        flattened,
        return_inverse=True,
        return_counts=True,
    )
    cluster_count = min(int(unique_values.size), palette_entries)
    result = coreml_kmeans1d().cluster(
        unique_values,
        cluster_count,
        weights=counts,
    )
    unique_codes = np.asarray(result.clusters, dtype=np.uint8)
    if unique_codes.shape != unique_values.shape:
        raise RuntimeError(
            "coremltools kmeans1d returned an unexpected assignment shape "
            f"{unique_codes.shape} for {unique_values.shape}"
        )

    palette = np.zeros((palette_entries,), dtype="<f2")
    palette[:cluster_count] = np.asarray(result.centroids, dtype="<f2")
    codes = unique_codes[inverse]
    return np.ascontiguousarray(codes), palette


def fit_fp32_palette(
    values: np.ndarray,
    *,
    palette_entries: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an FP16-anchored palette with FP32 least-squares centroids.

    TDT decoding is discontinuous: tiny changes in encoder error direction can
    select a long blank branch even when they slightly improve aggregate
    weight MSE. Keep the established FP16 package's globally optimal cluster
    assignments by fitting the values after FP16 rounding, then refit each
    assigned cluster's centroid against the original FP32 values. This gives
    the FP32 package true FP32 codebooks and the minimum FP32 reconstruction
    error possible under the stable assignments without requiring the FP16
    package to exist.
    """
    if palette_entries < 1 or palette_entries > 256:
        raise ValueError(
            f"palette_entries must be between 1 and 256, got {palette_entries}"
        )

    flattened_fp32 = np.ascontiguousarray(values, dtype="<f4").reshape(-1)
    if flattened_fp32.size == 0:
        raise ValueError("Cannot fit a palette to an empty tensor")
    if not np.all(np.isfinite(flattened_fp32)):
        raise ValueError("Palette source values must all be finite")

    flattened_fp16 = np.asarray(flattened_fp32, dtype="<f2")
    unique_values, inverse, counts = np.unique(
        flattened_fp16,
        return_inverse=True,
        return_counts=True,
    )
    cluster_count = min(int(unique_values.size), palette_entries)
    unique_codes_u32, _ = run_native_kmeans1d_sorted_unique(
        unique_values,
        counts,
        cluster_count,
    )
    unique_codes = np.asarray(unique_codes_u32, dtype=np.uint8)
    if unique_codes.shape != unique_values.shape:
        raise RuntimeError(
            "Native exact kmeans1d returned an unexpected assignment shape "
            f"{unique_codes.shape} for {unique_values.shape}"
        )

    codes = unique_codes[inverse]
    code_counts = np.bincount(codes, minlength=cluster_count)
    code_sums = np.bincount(
        codes,
        weights=np.asarray(flattened_fp32, dtype=np.float64),
        minlength=cluster_count,
    )
    if np.any(code_counts == 0):
        raise RuntimeError("Exact kmeans1d returned an empty palette cluster")

    palette = np.zeros((palette_entries,), dtype="<f4")
    palette[:cluster_count] = np.asarray(
        code_sums / code_counts,
        dtype="<f4",
    )
    return np.ascontiguousarray(codes), palette


def palette_code_group_spec(bits: int) -> tuple[int, int]:
    """Return `(codes_per_group, u32_words_per_group)` for a palette format."""
    try:
        return {
            PALETTE5_BITS: (32, 5),
            PALETTE6_BITS: (16, 3),
        }[bits]
    except KeyError as error:
        raise ValueError(f"Unsupported palette code width {bits}") from error


def pack_palette_codes_lsb(codes: np.ndarray, *, bits: int) -> np.ndarray:
    """Pack unsigned palette indices into a little-endian LSB-first u32 stream."""
    codes_per_group, words_per_group = palette_code_group_spec(bits)
    source = np.asarray(codes)
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("Palette codes must have an integer dtype")
    if source.size and (int(np.min(source)) < 0 or int(np.max(source)) >= 1 << bits):
        raise ValueError(f"{bits}-bit palette codes must be in [0, {1 << bits})")
    flattened = np.ascontiguousarray(source, dtype=np.uint8).reshape(-1)
    if flattened.size % codes_per_group:
        raise ValueError(
            f"{bits}-bit palette code count {flattened.size} is not divisible "
            f"by {codes_per_group}"
        )

    grouped = flattened.reshape(-1, codes_per_group)
    packed = np.zeros((grouped.shape[0], words_per_group), dtype="<u4")
    for code_index in range(codes_per_group):
        bit_offset = code_index * bits
        word_index = bit_offset // 32
        shift = bit_offset % 32
        code = grouped[:, code_index].astype(np.uint32)
        packed[:, word_index] |= code << np.uint32(shift)
        overflow_bits = shift + bits - 32
        if overflow_bits > 0:
            packed[:, word_index + 1] |= code >> np.uint32(bits - overflow_bits)
    return packed


def palettize_matrix(
    packed: np.ndarray,
    *,
    bits: int,
    output_group_width: int | None = None,
    float_dtype: str = "float16",
) -> tuple[np.ndarray, np.ndarray]:
    """Palettize a runtime `[K, N]` matrix and bit-pack its K-major codes."""
    if packed.ndim != 2 or packed.shape[1] % VECTOR_WIDTH:
        raise ValueError(
            f"Expected a vector-aligned runtime matrix, got {packed.shape}"
        )
    palette_code_group_spec(bits)
    palette_entries = 1 << bits
    palette_storage_dtype = numpy_float_dtype(float_dtype)
    fitter = {
        "float16": fit_fp16_palette,
        "float32": fit_fp32_palette,
    }[float_dtype]
    rows, columns = (int(value) for value in packed.shape)
    group_width = columns if output_group_width is None else output_group_width
    if group_width < 1 or columns % group_width:
        raise ValueError(
            f"Output width {columns} is not divisible by palette group width "
            f"{group_width}"
        )

    palette_groups = columns // group_width
    codes = np.empty((rows, columns), dtype=np.uint8)
    palettes = np.empty(
        (palette_groups, palette_entries),
        dtype=palette_storage_dtype,
    )
    for group_index in range(palette_groups):
        start = group_index * group_width
        end = start + group_width
        group_codes, palette = fitter(
            packed[:, start:end],
            palette_entries=palette_entries,
        )
        codes[:, start:end] = group_codes.reshape(rows, group_width)
        palettes[group_index] = palette

    packed_codes = pack_palette_codes_lsb(codes, bits=bits)
    return packed_codes, np.ascontiguousarray(palettes)


def pack_subsampling_linear_matrix(
    source: np.ndarray,
    *,
    channels: int = 256,
    frequencies: int = 16,
    float_dtype: str = "float16",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Repack the checkpoint's channel-major flatten for the runtime tensor.

    The source linear layer expects `[batch, time, channel, frequency]` before
    flattening the last two dimensions. The WebGPU subsampling kernels store
    `[batch, time, frequency, channel]`. Permuting the weight's input columns
    here preserves the checkpoint's contractions without a runtime transpose.
    """
    if source.ndim != 2:
        raise ValueError(f"Expected a matrix, got {source.shape}")
    output_width, input_width = map(int, source.shape)
    expected_input_width = channels * frequencies
    if input_width != expected_input_width:
        raise ValueError(
            "Expected subsampling input width "
            f"{channels} * {frequencies} = {expected_input_width}, "
            f"got {input_width}"
        )
    runtime_order = np.ascontiguousarray(
        source.reshape(output_width, channels, frequencies).transpose(0, 2, 1)
    ).reshape(output_width, input_width)
    return pack_matrix(runtime_order, float_dtype=float_dtype)


def pad_last_dimension(source: np.ndarray, *, multiple: int) -> np.ndarray:
    width = int(source.shape[-1])
    padded_width = ((width + multiple - 1) // multiple) * multiple
    if padded_width == width:
        return np.ascontiguousarray(source)
    shape = (*source.shape[:-1], padded_width)
    padded = np.zeros(shape, dtype=source.dtype)
    padded[..., :width] = source
    return padded


def sentencepiece_vocab_pieces(payload: bytes) -> list[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("Official SentencePiece vocabulary is not UTF-8") from error
    lines = text.splitlines()
    if len(lines) != 1024:
        raise RuntimeError(
            f"Expected 1024 SentencePiece vocabulary entries, got {len(lines)}"
        )
    pieces: list[str] = []
    for token_id, line in enumerate(lines):
        try:
            piece, score_text = line.rsplit("\t", 1)
            score = float(score_text)
        except (ValueError, OverflowError) as error:
            raise RuntimeError(
                f"Invalid SentencePiece vocabulary entry at ID {token_id}"
            ) from error
        if not piece or not math.isfinite(score):
            raise RuntimeError(
                f"Invalid SentencePiece vocabulary entry at ID {token_id}"
            )
        pieces.append(piece)
    if pieces[0] != "<unk>":
        raise RuntimeError("SentencePiece token 0 must be <unk>")
    if len(set(pieces)) != len(pieces):
        raise RuntimeError("SentencePiece vocabulary contains duplicate pieces")
    if "<pad>" in pieces or "<blank>" in pieces:
        raise RuntimeError("V2 SentencePiece vocabulary must not contain pad or blank")
    return pieces


def build_decode_tokenizer_json(vocabulary_payload: bytes) -> dict[str, Any]:
    pieces = sentencepiece_vocab_pieces(vocabulary_payload)
    return {
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": True},
            {"id": 1024, "content": "<blank>", "special": True},
        ],
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
        },
        "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "byte_fallback": False,
            "vocab": {piece: token_id for token_id, piece in enumerate(pieces)},
        },
        "version": "1.0",
    }


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def write_metadata_files(checkpoint: Any, output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for name in (
        "model_config.yaml",
        "tokenizer.model",
        "tokenizer.vocab",
        "vocab.txt",
    ):
        destination = output_dir / name
        write_bytes_atomic(destination, checkpoint.read_asset(name))
        copied.append(destination)

    tokenizer_path = output_dir / "tokenizer.json"
    write_json_atomic(
        tokenizer_path,
        build_decode_tokenizer_json(checkpoint.read_asset("tokenizer.vocab")),
        trailing_newline=False,
    )
    copied.append(tokenizer_path)
    return copied


def write_json_atomic(
    path: Path,
    value: Any,
    trailing_newline: bool = True,
) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if trailing_newline:
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def file_record(path: Path) -> FileRecord:
    return FileRecord(
        name=path.name,
        byteLength=path.stat().st_size,
        sha256=sha256_file(path),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
