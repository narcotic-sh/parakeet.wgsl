"""Restricted, zero-extraction reader for the pinned Parakeet v2 NeMo archive.

The official source is an uncompressed tar containing a PyTorch ZIP checkpoint.
This module intentionally does not import PyTorch or NeMo.  It validates the
fixed archive contract, parses the small pickle metadata with a deny-by-default
unpickler, and exposes tensor payloads as read-only NumPy mmap views into the
original ``.nemo`` file.
"""

from __future__ import annotations

import collections
import hashlib
import io
import math
import os
import pickle
import re
import struct
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


NEMO_ARCHIVE_FILENAME = "parakeet-tdt-0.6b-v2.nemo"
NEMO_ARCHIVE_BYTES = 2_472_222_720
NEMO_ARCHIVE_SHA256 = (
    "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"
)
NEMO_WEIGHTS_BYTES = 2_471_920_514
NEMO_WEIGHTS_SHA256 = (
    "fb1cbe2765fb80c02c0f874b6ea2c2bd24048c3618c092584fd086c6366e48a9"
)


@dataclass(frozen=True)
class NemoMemberSpec:
    archive_name: str
    normalized_name: str
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class NemoArchiveContract:
    archive_filename: str
    archive_byte_length: int
    archive_sha256: str
    directory_names: tuple[str, ...]
    members: tuple[NemoMemberSpec, ...]
    weights_name: str
    weights_byte_length: int
    weights_sha256: str
    zip_prefix: str
    storage_count: int
    tensor_count: int
    storage_location: str
    data_pickle_byte_length: int
    data_pickle_sha256: str
    serialization_id: bytes


OFFICIAL_NEMO_CONTRACT = NemoArchiveContract(
    archive_filename=NEMO_ARCHIVE_FILENAME,
    archive_byte_length=NEMO_ARCHIVE_BYTES,
    archive_sha256=NEMO_ARCHIVE_SHA256,
    directory_names=(".",),
    members=(
        NemoMemberSpec(
            "./4cf78c8ca4ca44fca36c3754478fb188_vocab.txt",
            "vocab.txt",
            5_066,
            "70b25deed256d79365a3c290d6191c31cc2315d952faf18d5c3a334f46e4866e",
        ),
        NemoMemberSpec(
            "./705f11d22dc04b169effc35ce5cd1361_tokenizer.model",
            "tokenizer.model",
            251_396,
            "5a3a82c48998709f9bc9f5db2924d99a899d20bf82550ca52272da7c7557b0a0",
        ),
        NemoMemberSpec(
            "./a4715c7f6b2d4c2bb709306073d0c0a4_tokenizer.vocab",
            "tokenizer.vocab",
            10_393,
            "dbe7ec6bd066c4526080940e1614e6582b9161515e856c2c39a1d49267c88cbc",
        ),
        NemoMemberSpec(
            "./model_config.yaml",
            "model_config.yaml",
            21_015,
            "43d879d29bf1c56a029074d08eb245e7cbd83a3c0f7798004eb2de2a38885c11",
        ),
        NemoMemberSpec(
            "./model_weights.ckpt",
            "model_weights.ckpt",
            NEMO_WEIGHTS_BYTES,
            NEMO_WEIGHTS_SHA256,
        ),
    ),
    weights_name="model_weights.ckpt",
    weights_byte_length=NEMO_WEIGHTS_BYTES,
    weights_sha256=NEMO_WEIGHTS_SHA256,
    zip_prefix="model_weights",
    storage_count=718,
    tensor_count=725,
    storage_location="cuda:0",
    data_pickle_byte_length=145_537,
    data_pickle_sha256=(
        "6f0c85bcade451588dd9362efcfc6b9902f63e2c02dd5368ecc9514f263c693b"
    ),
    serialization_id=b"1675449553011448033212626957860485250840",
)


_CANONICAL_RENAMES = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"encoder\.pre_encode\.conv\.", "encoder.subsampling.layers."),
        (r"encoder\.pre_encode\.out\.", "encoder.subsampling.linear."),
        (r"encoder\.pos_enc\.", "encoder.encode_positions."),
        (
            r"encoder\.layers\.(\d+)\.conv\.batch_norm\.",
            r"encoder.layers.\1.conv.norm.",
        ),
        (r"decoder\.decoder_layers\.0\.(weight|bias)", r"ctc_head.\1"),
        (r"linear_([kv])", r"\1_proj"),
        (r"linear_out", "o_proj"),
        (r"linear_q", "q_proj"),
        (r"pos_bias_([uv])", r"bias_\1"),
        (r"linear_pos", "relative_k_proj"),
        (r"decoder\.prediction\.embed\.", "decoder.embedding."),
        (r"decoder\.prediction\.dec_rnn\.lstm\.", "decoder.lstm."),
        (r"joint\.enc\.", "encoder_projector."),
        (r"joint\.pred\.", "decoder.decoder_projector."),
        (r"joint\.joint_net\.2\.", "joint.head."),
    )
)


def canonical_tensor_name(name: str) -> str:
    if name == "preprocessor.featurizer.window":
        return "preprocessor.window"
    if name == "preprocessor.featurizer.fb":
        return "preprocessor.mel_filters"
    converted = name
    for pattern, replacement in _CANONICAL_RENAMES:
        converted = pattern.sub(replacement, converted)
    return converted


@dataclass(frozen=True)
class _StorageKind:
    name: str
    dtype: np.dtype[Any]


_FLOAT_STORAGE = _StorageKind("FloatStorage", np.dtype("<f4"))
_LONG_STORAGE = _StorageKind("LongStorage", np.dtype("<i8"))


@dataclass(frozen=True)
class _StorageRef:
    kind: _StorageKind
    key: str
    location: str
    element_count: int


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    width = 1
    for dimension in reversed(shape):
        stride.append(width)
        width *= dimension
    return tuple(reversed(stride))


def _rebuild_tensor_v2(
    storage: object,
    storage_offset: object,
    shape: object,
    stride: object,
    requires_grad: object,
    backward_hooks: object,
) -> _TensorRef:
    if not isinstance(storage, _StorageRef):
        raise pickle.UnpicklingError("Tensor references an invalid storage")
    if not isinstance(storage_offset, int) or isinstance(storage_offset, bool):
        raise pickle.UnpicklingError("Tensor storage offset must be an integer")
    if storage_offset < 0:
        raise pickle.UnpicklingError("Tensor storage offset must be nonnegative")
    if not isinstance(shape, tuple) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in shape
    ):
        raise pickle.UnpicklingError("Tensor shape must contain nonnegative integers")
    if not isinstance(stride, tuple) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in stride
    ):
        raise pickle.UnpicklingError("Tensor stride must contain nonnegative integers")
    if len(stride) != len(shape) or stride != _contiguous_stride(shape):
        raise pickle.UnpicklingError("Only contiguous tensor views are supported")
    if requires_grad is not False:
        raise pickle.UnpicklingError("State-dict tensors must not require gradients")
    if not isinstance(backward_hooks, collections.OrderedDict) or backward_hooks:
        raise pickle.UnpicklingError("State-dict tensor hooks must be empty")

    element_count = math.prod(shape)
    if storage_offset + element_count > storage.element_count:
        raise pickle.UnpicklingError("Tensor view exceeds its backing storage")
    return _TensorRef(
        storage=storage,
        storage_offset=storage_offset,
        shape=shape,
        stride=stride,
    )


class _RestrictedStateDictUnpickler(pickle.Unpickler):
    def __init__(self, stream: BinaryIO, *, expected_location: str) -> None:
        super().__init__(stream)
        self.expected_location = expected_location
        self.storages: dict[str, _StorageRef] = {}

    def find_class(self, module: str, name: str) -> object:
        allowed: dict[tuple[str, str], object] = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
            ("torch", "FloatStorage"): _FLOAT_STORAGE,
            ("torch", "LongStorage"): _LONG_STORAGE,
        }
        try:
            return allowed[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"Forbidden pickle global {module}.{name}"
            ) from error

    def persistent_load(self, persistent_id: object) -> _StorageRef:
        if (
            not isinstance(persistent_id, tuple)
            or len(persistent_id) != 5
            or persistent_id[0] != "storage"
        ):
            raise pickle.UnpicklingError("Invalid storage persistent ID")
        _, kind, key, location, element_count = persistent_id
        if kind not in (_FLOAT_STORAGE, _LONG_STORAGE):
            raise pickle.UnpicklingError("Unsupported checkpoint storage dtype")
        if not isinstance(key, str) or not key.isdecimal():
            raise pickle.UnpicklingError("Storage key must be a decimal string")
        if location != self.expected_location:
            raise pickle.UnpicklingError(
                f"Unexpected checkpoint storage location {location!r}"
            )
        if (
            not isinstance(element_count, int)
            or isinstance(element_count, bool)
            or element_count < 0
        ):
            raise pickle.UnpicklingError("Storage element count must be nonnegative")
        storage = _StorageRef(kind, key, location, element_count)
        existing = self.storages.get(key)
        if existing is not None and existing != storage:
            raise pickle.UnpicklingError(f"Conflicting metadata for storage {key}")
        self.storages[key] = storage
        return storage


def restricted_load_state_dict(
    payload: bytes,
    *,
    expected_location: str,
) -> tuple[collections.OrderedDict[str, _TensorRef], dict[str, _StorageRef]]:
    stream = io.BytesIO(payload)
    unpickler = _RestrictedStateDictUnpickler(
        stream,
        expected_location=expected_location,
    )
    value = unpickler.load()
    if stream.read(1):
        raise pickle.UnpicklingError("Trailing data after state-dict pickle")
    if not isinstance(value, collections.OrderedDict):
        raise pickle.UnpicklingError("Checkpoint root must be an OrderedDict")
    if not all(isinstance(name, str) and isinstance(tensor, _TensorRef) for name, tensor in value.items()):
        raise pickle.UnpicklingError("Checkpoint must map string names to tensors")
    attributes = getattr(value, "__dict__", {})
    if set(attributes) - {"_metadata"}:
        raise pickle.UnpicklingError("Checkpoint OrderedDict has unexpected attributes")
    metadata = attributes.get("_metadata")
    if metadata is not None and not isinstance(metadata, collections.OrderedDict):
        raise pickle.UnpicklingError("Checkpoint metadata must be an OrderedDict")
    return value, unpickler.storages


class _BoundedFile(io.RawIOBase):
    def __init__(self, path: Path, offset: int, byte_length: int) -> None:
        super().__init__()
        self._stream = path.open("rb")
        self._offset = offset
        self._byte_length = byte_length
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._byte_length + offset
        else:
            raise ValueError(f"Unsupported seek mode {whence}")
        if position < 0:
            raise ValueError("Cannot seek before the checkpoint member")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._byte_length - self._position
        size = min(size, max(0, self._byte_length - self._position))
        self._stream.seek(self._offset + self._position)
        payload = self._stream.read(size)
        self._position += len(payload)
        return payload

    def close(self) -> None:
        if not self.closed:
            self._stream.close()
        super().close()


@dataclass(frozen=True)
class _MemberLocation:
    spec: NemoMemberSpec
    data_offset: int


@dataclass(frozen=True)
class _TensorSlice:
    shape: tuple[int, ...]

    def get_shape(self) -> tuple[int, ...]:
        return self.shape


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


class NemoCheckpoint:
    """Canonical-name view over the pinned NeMo checkpoint."""

    def __init__(
        self,
        archive_path: Path,
        *,
        contract: NemoArchiveContract = OFFICIAL_NEMO_CONTRACT,
        validate_hashes: bool = True,
    ) -> None:
        self.archive_path = Path(archive_path)
        self.contract = contract
        self._bounded_weights: _BoundedFile | None = None
        self._zip: zipfile.ZipFile | None = None
        self._members: dict[str, _MemberLocation] = {}
        self._storage_offsets: dict[str, int] = {}
        self._storage_refs: dict[str, _StorageRef] = {}
        self._tensors: dict[str, _TensorRef] = {}

        self._validate_archive_identity(validate_hashes=validate_hashes)
        self._inspect_tar(validate_hashes=validate_hashes)
        self._open_checkpoint()

    def __enter__(self) -> NemoCheckpoint:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None
        if self._bounded_weights is not None:
            self._bounded_weights.close()
            self._bounded_weights = None

    def _validate_archive_identity(self, *, validate_hashes: bool) -> None:
        if not self.archive_path.is_file():
            raise FileNotFoundError(self.archive_path)
        if self.archive_path.name != self.contract.archive_filename:
            raise RuntimeError(
                "Official NeMo archive filename changed: expected "
                f"{self.contract.archive_filename}, got {self.archive_path.name}"
            )
        actual_size = self.archive_path.stat().st_size
        if actual_size != self.contract.archive_byte_length:
            raise RuntimeError(
                "Official NeMo archive length changed: expected "
                f"{self.contract.archive_byte_length}, got {actual_size}"
            )
        if validate_hashes:
            digest = _sha256_file(self.archive_path)
            if digest != self.contract.archive_sha256:
                raise RuntimeError(
                    "Official NeMo archive SHA-256 mismatch: expected "
                    f"{self.contract.archive_sha256}, got {digest}"
                )

    def _inspect_tar(self, *, validate_hashes: bool) -> None:
        expected_files = {spec.archive_name: spec for spec in self.contract.members}
        expected_directories = set(self.contract.directory_names)
        with tarfile.open(self.archive_path, mode="r:", encoding="utf-8") as archive:
            members = archive.getmembers()
            member_names = [member.name for member in members]
            if len(member_names) != len(set(member_names)):
                raise RuntimeError("NeMo tar contains duplicate member names")
            actual_files = {member.name for member in members if member.isfile()}
            actual_directories = {member.name for member in members if member.isdir()}
            unsupported = [member.name for member in members if not (member.isfile() or member.isdir())]
            if unsupported:
                raise RuntimeError(f"Unsupported NeMo tar members: {unsupported}")
            if actual_files != set(expected_files) or actual_directories != expected_directories:
                raise RuntimeError(
                    "Official NeMo tar inventory mismatch: expected files "
                    f"{sorted(expected_files)} and directories {sorted(expected_directories)}, "
                    f"got files {sorted(actual_files)} and directories {sorted(actual_directories)}"
                )

            for member in members:
                if not member.isfile():
                    continue
                spec = expected_files[member.name]
                if member.size != spec.byte_length:
                    raise RuntimeError(
                        f"NeMo member {member.name} length changed: expected "
                        f"{spec.byte_length}, got {member.size}"
                    )
                if member.offset_data < 0 or member.offset_data + member.size > self.contract.archive_byte_length:
                    raise RuntimeError(f"NeMo member {member.name} exceeds archive bounds")
                if validate_hashes:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Unable to read NeMo member {member.name}")
                    with extracted:
                        digest = _sha256_stream(extracted)
                    if digest != spec.sha256:
                        raise RuntimeError(
                            f"NeMo member {member.name} SHA-256 mismatch: expected "
                            f"{spec.sha256}, got {digest}"
                        )
                self._members[spec.normalized_name] = _MemberLocation(
                    spec=spec,
                    data_offset=member.offset_data,
                )

    def _open_checkpoint(self) -> None:
        weights = self._members[self.contract.weights_name]
        if (
            weights.spec.byte_length != self.contract.weights_byte_length
            or weights.spec.sha256 != self.contract.weights_sha256
        ):
            raise RuntimeError("NeMo weights member contract is inconsistent")
        self._bounded_weights = _BoundedFile(
            self.archive_path,
            weights.data_offset,
            weights.spec.byte_length,
        )
        try:
            self._zip = zipfile.ZipFile(self._bounded_weights, mode="r")
            self._inspect_zip_and_pickle()
        except BaseException:
            self.close()
            raise

    def _inspect_zip_and_pickle(self) -> None:
        assert self._zip is not None
        prefix = self.contract.zip_prefix
        expected_names = {
            f"{prefix}/data.pkl",
            f"{prefix}/byteorder",
            f"{prefix}/version",
            f"{prefix}/.data/serialization_id",
            *(f"{prefix}/data/{index}" for index in range(self.contract.storage_count)),
        }
        infos = self._zip.infolist()
        actual_names = [info.filename for info in infos]
        if len(actual_names) != len(set(actual_names)):
            raise RuntimeError("PyTorch checkpoint ZIP contains duplicate names")
        if set(actual_names) != expected_names:
            raise RuntimeError(
                "PyTorch checkpoint ZIP inventory mismatch: missing "
                f"{sorted(expected_names - set(actual_names))}, unexpected "
                f"{sorted(set(actual_names) - expected_names)}"
            )
        if self._zip.comment:
            raise RuntimeError("PyTorch checkpoint ZIP comment must be empty")

        spans: list[tuple[int, int, str]] = []
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED or info.file_size != info.compress_size:
                raise RuntimeError(f"Checkpoint ZIP entry {info.filename} must be stored")
            if info.flag_bits & 1:
                raise RuntimeError(f"Checkpoint ZIP entry {info.filename} is encrypted")
            data_offset = self._zip_data_offset(info)
            data_end = data_offset + info.file_size
            if data_offset < 0 or data_end > self.contract.weights_byte_length:
                raise RuntimeError(f"Checkpoint ZIP entry {info.filename} exceeds bounds")
            spans.append((data_offset, data_end, info.filename))
            storage_prefix = f"{prefix}/data/"
            if info.filename.startswith(storage_prefix):
                key = info.filename[len(storage_prefix) :]
                self._storage_offsets[key] = data_offset

        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if current[0] < previous[1]:
                raise RuntimeError(
                    f"Checkpoint ZIP entries {previous[2]} and {current[2]} overlap"
                )

        if self._zip.read(f"{prefix}/byteorder") != b"little":
            raise RuntimeError("Checkpoint byte order must be little-endian")
        if self._zip.read(f"{prefix}/version") != b"3\n":
            raise RuntimeError("Unsupported PyTorch checkpoint version")
        if self._zip.read(f"{prefix}/.data/serialization_id") != self.contract.serialization_id:
            raise RuntimeError("Checkpoint serialization identity mismatch")
        data_pickle = self._zip.read(f"{prefix}/data.pkl")
        if len(data_pickle) != self.contract.data_pickle_byte_length:
            raise RuntimeError("Checkpoint data.pkl length mismatch")
        digest = hashlib.sha256(data_pickle).hexdigest()
        if digest != self.contract.data_pickle_sha256:
            raise RuntimeError("Checkpoint data.pkl SHA-256 mismatch")

        raw_tensors, storages = restricted_load_state_dict(
            data_pickle,
            expected_location=self.contract.storage_location,
        )
        if len(raw_tensors) != self.contract.tensor_count:
            raise RuntimeError(
                f"Checkpoint tensor count mismatch: expected {self.contract.tensor_count}, "
                f"got {len(raw_tensors)}"
            )
        expected_storage_keys = {str(index) for index in range(self.contract.storage_count)}
        if set(storages) != expected_storage_keys or set(self._storage_offsets) != expected_storage_keys:
            raise RuntimeError("Checkpoint storage inventory mismatch")

        info_by_name = {info.filename: info for info in infos}
        for key, storage in storages.items():
            info = info_by_name[f"{prefix}/data/{key}"]
            expected_size = storage.element_count * storage.kind.dtype.itemsize
            if info.file_size != expected_size:
                raise RuntimeError(
                    f"Checkpoint storage {key} length mismatch: expected "
                    f"{expected_size}, got {info.file_size}"
                )
        self._storage_refs = storages

        for raw_name, tensor in raw_tensors.items():
            canonical_name = canonical_tensor_name(raw_name)
            if canonical_name in self._tensors:
                raise RuntimeError(f"Canonical tensor name collision: {canonical_name}")
            self._tensors[canonical_name] = tensor

    def _zip_data_offset(self, info: zipfile.ZipInfo) -> int:
        assert self._bounded_weights is not None
        self._bounded_weights.seek(info.header_offset)
        header = self._bounded_weights.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Invalid local ZIP header for {info.filename}")
        filename_length, extra_length = struct.unpack_from("<HH", header, 26)
        filename = self._bounded_weights.read(filename_length)
        try:
            decoded = filename.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"Invalid ZIP filename encoding for {info.filename}") from error
        if decoded != info.filename:
            raise RuntimeError(f"Local ZIP filename mismatch for {info.filename}")
        return info.header_offset + 30 + filename_length + extra_length

    def keys(self) -> list[str]:
        return list(self._tensors)

    def get_slice(self, name: str) -> _TensorSlice:
        try:
            tensor = self._tensors[name]
        except KeyError as error:
            raise KeyError(name) from error
        return _TensorSlice(tensor.shape)

    def get_tensor(self, name: str) -> np.ndarray:
        try:
            tensor = self._tensors[name]
            relative_storage_offset = self._storage_offsets[tensor.storage.key]
        except KeyError as error:
            raise KeyError(name) from error
        member = self._members[self.contract.weights_name]
        byte_offset = (
            member.data_offset
            + relative_storage_offset
            + tensor.storage_offset * tensor.storage.kind.dtype.itemsize
        )
        element_count = math.prod(tensor.shape)
        mapped = np.memmap(
            self.archive_path,
            dtype=tensor.storage.kind.dtype,
            mode="r",
            offset=byte_offset,
            shape=(element_count,),
        )
        return mapped.reshape(tensor.shape)

    def read_asset(self, normalized_name: str) -> bytes:
        try:
            member = self._members[normalized_name]
        except KeyError as error:
            raise KeyError(normalized_name) from error
        with self.archive_path.open("rb") as stream:
            stream.seek(member.data_offset)
            payload = stream.read(member.spec.byte_length)
        if len(payload) != member.spec.byte_length:
            raise RuntimeError(f"Unable to read complete NeMo member {normalized_name}")
        return payload
