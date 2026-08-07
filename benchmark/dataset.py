"""Pinned LibriSpeech test-set acquisition and lossless audio preparation.

The Hugging Face artifacts below are a Parquet repackaging of the official
LibriSpeech test-clean and test-other splits.  Their revision, byte lengths,
and downloaded-file SHA-256 identities are fixed here so a benchmark cannot
silently change when the Hub repository moves.
"""

from __future__ import annotations

import os
import re
import tempfile
import wave
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

DATASET_REPO_ID = "hf-audio/open-asr-leaderboard"
DATASET_REVISION = "b6bdcd0beb34f8975dc659796176d88f43aff502"
DATASET_REPO_TYPE = "dataset"

LIBRISPEECH_SOURCE_URL = "https://www.openslr.org/12/"
LIBRISPEECH_LICENSE = "CC BY 4.0"

_SAMPLE_ID = re.compile(r"^[0-9]+-[0-9]+-[0-9]+$")
_REQUIRED_COLUMNS = frozenset({"audio", "dataset", "id", "text"})
_HASH_CHUNK_BYTES = 1024 * 1024
_PCM_BLOCK_FRAMES = 64 * 1024


class DatasetError(RuntimeError):
    """Base class for benchmark dataset failures."""


class DatasetIntegrityError(DatasetError):
    """A pinned dataset artifact does not have its declared identity."""


class DatasetSchemaError(DatasetError):
    """A Parquet row or schema does not match the pinned dataset contract."""


class AudioFormatError(DatasetError):
    """Source audio is not losslessly convertible to canonical PCM16 WAV."""


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """One immutable Parquet artifact and its expected dataset shape."""

    name: str
    filename: str
    byte_length: int
    sha256: str
    rows: int


TEST_CLEAN = DatasetSplit(
    name="test-clean",
    filename="librispeech/test.clean-00000-of-00001.parquet",
    byte_length=350_475_095,
    sha256="e576cf31312bf67d5ab714ac1dd93fb52d60d62659b76402ba1f97484464ddc3",
    rows=2_620,
)

TEST_OTHER = DatasetSplit(
    name="test-other",
    filename="librispeech/test.other-00000-of-00001.parquet",
    byte_length=332_937_634,
    sha256="5a5e64cd25df8e094cba0b5b1cc8e68c031ae4483bc7f7c0a191ebe3aa582f66",
    rows=2_939,
)

LIBRISPEECH_SPLITS = (TEST_CLEAN, TEST_OTHER)
SPLITS_BY_NAME: Mapping[str, DatasetSplit] = MappingProxyType(
    {split.name: split for split in LIBRISPEECH_SPLITS}
)


@dataclass(frozen=True, slots=True)
class LibriSpeechSample:
    """One stable benchmark row in the order stored by the pinned Parquet."""

    split: str
    id: str
    reference: str
    flac_bytes: bytes

    @property
    def name(self) -> str:
        """Canonical filename sent to the browser benchmark bridge."""

        return f"{self.id}.wav"

    def wav_bytes(self) -> bytes:
        """Return exact canonical 16 kHz mono signed PCM16 WAV bytes."""

        return canonical_wav_bytes(self.flac_bytes)


def get_split(name: str) -> DatasetSplit:
    """Resolve an exact public split name."""

    try:
        return SPLITS_BY_NAME[name]
    except KeyError as error:
        choices = ", ".join(SPLITS_BY_NAME)
        raise ValueError(
            f"unknown dataset split {name!r}; choose one of: {choices}"
        ) from error


def verify_artifact(path: Path | str, split: DatasetSplit) -> Path:
    """Verify a downloaded Parquet's exact byte length and SHA-256."""

    artifact = Path(path)
    try:
        actual_length = artifact.stat().st_size
    except OSError as error:
        raise DatasetIntegrityError(
            f"cannot inspect dataset artifact {artifact}: {error}"
        ) from error

    if actual_length != split.byte_length:
        raise DatasetIntegrityError(
            f"{split.name} Parquet has {actual_length} bytes; expected "
            f"{split.byte_length}"
        )

    digest = sha256()
    try:
        with artifact.open("rb") as source:
            while chunk := source.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise DatasetIntegrityError(
            f"cannot read dataset artifact {artifact}: {error}"
        ) from error

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != split.sha256:
        raise DatasetIntegrityError(
            f"{split.name} Parquet SHA-256 is {actual_sha256}; expected {split.sha256}"
        )
    return artifact


def download_split(split: DatasetSplit, cache_dir: Path | str | None = None) -> Path:
    """Download one pinned split through the Hugging Face cache and verify it."""

    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        repo_type=DATASET_REPO_TYPE,
        revision=DATASET_REVISION,
        filename=split.filename,
        cache_dir=os.fspath(cache_dir) if cache_dir is not None else None,
        token=os.environ.get("HF_TOKEN") or None,
    )
    return verify_artifact(path, split)


def iter_samples(
    parquet_path: Path | str,
    split: DatasetSplit,
    *,
    batch_size: int = 32,
) -> Iterator[LibriSpeechSample]:
    """Yield pinned rows in Parquet order without decoding audio through floats."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    try:
        parquet = pq.ParquetFile(parquet_path)
    except Exception as error:
        raise DatasetSchemaError(
            f"cannot open {split.name} Parquet: {error}"
        ) from error

    missing_columns = _REQUIRED_COLUMNS.difference(parquet.schema_arrow.names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DatasetSchemaError(f"{split.name} Parquet is missing columns: {missing}")
    if parquet.metadata.num_rows != split.rows:
        raise DatasetSchemaError(
            f"{split.name} Parquet has {parquet.metadata.num_rows} rows; "
            f"expected {split.rows}"
        )

    seen_ids: set[str] = set()
    try:
        batches = parquet.iter_batches(
            batch_size=batch_size,
            columns=["audio", "dataset", "id", "text"],
            use_threads=False,
        )
        for batch in batches:
            for row in batch.to_pylist():
                sample_id = row["id"]
                reference = row["text"]
                source_dataset = row["dataset"]
                audio = row["audio"]

                if not isinstance(sample_id, str) or not _SAMPLE_ID.fullmatch(
                    sample_id
                ):
                    raise DatasetSchemaError(
                        f"invalid LibriSpeech sample id: {sample_id!r}"
                    )
                if sample_id in seen_ids:
                    raise DatasetSchemaError(
                        f"duplicate LibriSpeech sample id: {sample_id}"
                    )
                seen_ids.add(sample_id)

                if source_dataset != "librispeech":
                    raise DatasetSchemaError(
                        f"sample {sample_id} declares dataset {source_dataset!r}, "
                        "not 'librispeech'"
                    )
                if not isinstance(reference, str) or not reference:
                    raise DatasetSchemaError(
                        f"sample {sample_id} has no reference transcript"
                    )
                if not isinstance(audio, dict):
                    raise DatasetSchemaError(
                        f"sample {sample_id} has an invalid audio value"
                    )

                source_name = audio.get("path")
                source_bytes = audio.get("bytes")
                if source_name != f"{sample_id}.flac":
                    raise DatasetSchemaError(
                        f"sample {sample_id} has unexpected audio path {source_name!r}"
                    )
                if not isinstance(source_bytes, bytes) or not source_bytes:
                    raise DatasetSchemaError(
                        f"sample {sample_id} has no embedded FLAC bytes"
                    )

                yield LibriSpeechSample(
                    split=split.name,
                    id=sample_id,
                    reference=reference,
                    flac_bytes=source_bytes,
                )
    except DatasetSchemaError:
        raise
    except Exception as error:
        raise DatasetSchemaError(
            f"cannot read {split.name} Parquet rows: {error}"
        ) from error

    if len(seen_ids) != split.rows:
        raise DatasetSchemaError(
            f"{split.name} yielded {len(seen_ids)} rows; expected {split.rows}"
        )


def _write_canonical_wav(source_bytes: bytes, destination: BinaryIO) -> None:
    """Decode PCM16 FLAC into a WAV container through int16 samples only."""

    if not source_bytes:
        raise AudioFormatError("source FLAC is empty")

    try:
        source = sf.SoundFile(BytesIO(source_bytes), mode="r")
    except (OSError, RuntimeError, TypeError) as error:
        raise AudioFormatError(f"cannot open source FLAC: {error}") from error

    with source:
        if source.format != "FLAC":
            raise AudioFormatError(
                f"source audio format is {source.format}, expected FLAC"
            )
        if source.subtype != "PCM_16":
            raise AudioFormatError(
                f"source audio subtype is {source.subtype}, expected PCM_16"
            )
        if source.samplerate != 16_000:
            raise AudioFormatError(
                f"source audio sample rate is {source.samplerate} Hz, expected 16000 Hz"
            )
        if source.channels != 1:
            raise AudioFormatError(
                f"source audio has {source.channels} channels, expected mono"
            )

        frames_written = 0
        with wave.open(destination, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.setnframes(source.frames)

            while True:
                pcm = source.read(
                    frames=_PCM_BLOCK_FRAMES,
                    dtype="int16",
                    always_2d=False,
                )
                if pcm.size == 0:
                    break
                frames_written += int(pcm.size)
                output.writeframesraw(pcm.astype("<i2", copy=False).tobytes(order="C"))

        if frames_written != source.frames:
            raise AudioFormatError(
                f"decoded {frames_written} PCM frames; expected {source.frames}"
            )


def canonical_wav_bytes(flac_bytes: bytes) -> bytes:
    """Return a deterministic canonical WAV without resampling or float conversion."""

    destination = BytesIO()
    _write_canonical_wav(flac_bytes, destination)
    return destination.getvalue()


def write_canonical_wav(flac_bytes: bytes, destination: Path | str) -> Path:
    """Atomically write a deterministic canonical WAV file."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            _write_canonical_wav(flac_bytes, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output


__all__ = [
    "DATASET_REPO_ID",
    "DATASET_REVISION",
    "LIBRISPEECH_LICENSE",
    "LIBRISPEECH_SOURCE_URL",
    "LIBRISPEECH_SPLITS",
    "SPLITS_BY_NAME",
    "TEST_CLEAN",
    "TEST_OTHER",
    "AudioFormatError",
    "DatasetError",
    "DatasetIntegrityError",
    "DatasetSchemaError",
    "DatasetSplit",
    "LibriSpeechSample",
    "canonical_wav_bytes",
    "download_split",
    "get_split",
    "iter_samples",
    "verify_artifact",
    "write_canonical_wav",
]
