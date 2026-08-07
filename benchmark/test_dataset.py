from __future__ import annotations

import wave
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

import dataset


def _flac_bytes(
    samples: np.ndarray,
    *,
    samplerate: int = 16_000,
    subtype: str = "PCM_16",
) -> bytes:
    encoded = BytesIO()
    sf.write(encoded, samples, samplerate, format="FLAC", subtype=subtype)
    return encoded.getvalue()


def _fixture_parquet(path: Path, rows: list[dict]) -> dataset.DatasetSplit:
    schema = pa.schema(
        [
            pa.field(
                "audio",
                pa.struct(
                    [
                        pa.field("bytes", pa.binary()),
                        pa.field("path", pa.string()),
                    ]
                ),
            ),
            pa.field("dataset", pa.string()),
            pa.field("text", pa.string()),
            pa.field("id", pa.string()),
            pa.field("audio_length_s", pa.float64()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    contents = path.read_bytes()
    return dataset.DatasetSplit(
        name="fixture",
        filename="fixture.parquet",
        byte_length=len(contents),
        sha256=sha256(contents).hexdigest(),
        rows=len(rows),
    )


def test_pinned_dataset_artifacts() -> None:
    assert dataset.DATASET_REVISION == "b6bdcd0beb34f8975dc659796176d88f43aff502"
    assert dataset.TEST_CLEAN.byte_length == 350_475_095
    assert dataset.TEST_CLEAN.sha256 == (
        "e576cf31312bf67d5ab714ac1dd93fb52d60d62659b76402ba1f97484464ddc3"
    )
    assert dataset.TEST_OTHER.byte_length == 332_937_634
    assert dataset.TEST_OTHER.sha256 == (
        "5a5e64cd25df8e094cba0b5b1cc8e68c031ae4483bc7f7c0a191ebe3aa582f66"
    )


def test_verify_artifact_checks_length_and_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "split.parquet"
    artifact.write_bytes(b"pinned bytes")
    split = dataset.DatasetSplit(
        name="fixture",
        filename="fixture.parquet",
        byte_length=12,
        sha256=sha256(b"pinned bytes").hexdigest(),
        rows=0,
    )

    assert dataset.verify_artifact(artifact, split) == artifact

    artifact.write_bytes(b"wrong length!")
    with pytest.raises(dataset.DatasetIntegrityError, match="bytes; expected"):
        dataset.verify_artifact(artifact, split)

    artifact.write_bytes(b"wrong bytes!")
    with pytest.raises(dataset.DatasetIntegrityError, match="SHA-256"):
        dataset.verify_artifact(artifact, split)


def test_download_split_pins_hub_request_and_honors_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contents = b"downloaded parquet"
    artifact = tmp_path / "artifact.parquet"
    artifact.write_bytes(contents)
    split = dataset.DatasetSplit(
        name="fixture",
        filename="librispeech/fixture.parquet",
        byte_length=len(contents),
        sha256=sha256(contents).hexdigest(),
        rows=0,
    )
    calls: list[dict] = []

    def fake_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(artifact)

    monkeypatch.setenv("HF_TOKEN", "secret-token")
    monkeypatch.setattr(dataset, "hf_hub_download", fake_download)

    assert dataset.download_split(split, cache_dir=tmp_path / "cache") == artifact
    assert calls == [
        {
            "repo_id": dataset.DATASET_REPO_ID,
            "repo_type": "dataset",
            "revision": dataset.DATASET_REVISION,
            "filename": split.filename,
            "cache_dir": str(tmp_path / "cache"),
            "token": "secret-token",
        }
    ]


def test_iter_samples_preserves_pinned_row_order_and_source_bytes(
    tmp_path: Path,
) -> None:
    first_audio = _flac_bytes(np.array([-32768, -1, 0, 1, 32767], dtype=np.int16))
    second_audio = _flac_bytes(np.array([7, 8, 9], dtype=np.int16))
    path = tmp_path / "fixture.parquet"
    split = _fixture_parquet(
        path,
        [
            {
                "audio": {"bytes": first_audio, "path": "1-2-0003.flac"},
                "dataset": "librispeech",
                "text": "first reference",
                "id": "1-2-0003",
                "audio_length_s": 5 / 16_000,
            },
            {
                "audio": {"bytes": second_audio, "path": "4-5-0006.flac"},
                "dataset": "librispeech",
                "text": "second reference",
                "id": "4-5-0006",
                "audio_length_s": 3 / 16_000,
            },
        ],
    )

    samples = list(dataset.iter_samples(path, split, batch_size=1))

    assert [sample.id for sample in samples] == ["1-2-0003", "4-5-0006"]
    assert [sample.reference for sample in samples] == [
        "first reference",
        "second reference",
    ]
    assert [sample.flac_bytes for sample in samples] == [first_audio, second_audio]
    assert samples[0].name == "1-2-0003.wav"


def test_iter_samples_rejects_duplicate_ids(tmp_path: Path) -> None:
    audio = _flac_bytes(np.array([1, 2, 3], dtype=np.int16))
    path = tmp_path / "duplicate.parquet"
    split = _fixture_parquet(
        path,
        [
            {
                "audio": {"bytes": audio, "path": "1-2-0003.flac"},
                "dataset": "librispeech",
                "text": "first",
                "id": "1-2-0003",
                "audio_length_s": 3 / 16_000,
            },
            {
                "audio": {"bytes": audio, "path": "1-2-0003.flac"},
                "dataset": "librispeech",
                "text": "second",
                "id": "1-2-0003",
                "audio_length_s": 3 / 16_000,
            },
        ],
    )

    with pytest.raises(dataset.DatasetSchemaError, match="duplicate"):
        list(dataset.iter_samples(path, split))


def test_canonical_wav_conversion_preserves_every_pcm16_sample(tmp_path: Path) -> None:
    samples = np.array(
        [-32768, -30000, -1, 0, 1, 30000, 32767] * 20_000,
        dtype=np.int16,
    )
    source = _flac_bytes(samples)

    wav_bytes = dataset.canonical_wav_bytes(source)
    with wave.open(BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == samples.size
        assert (
            wav.readframes(wav.getnframes())
            == samples.astype("<i2", copy=False).tobytes()
        )

    output = tmp_path / "sample.wav"
    assert dataset.write_canonical_wav(source, output) == output
    assert output.read_bytes() == wav_bytes


@pytest.mark.parametrize(
    ("samples", "samplerate", "subtype", "message"),
    [
        (np.array([1, 2, 3], dtype=np.int16), 8_000, "PCM_16", "sample rate"),
        (np.array([[1, 2], [3, 4]], dtype=np.int16), 16_000, "PCM_16", "channels"),
        (np.array([1, 2, 3], dtype=np.int32), 16_000, "PCM_24", "subtype"),
    ],
)
def test_canonical_wav_conversion_rejects_noncanonical_source_audio(
    samples: np.ndarray,
    samplerate: int,
    subtype: str,
    message: str,
) -> None:
    source = _flac_bytes(samples, samplerate=samplerate, subtype=subtype)

    with pytest.raises(dataset.AudioFormatError, match=message):
        dataset.canonical_wav_bytes(source)
