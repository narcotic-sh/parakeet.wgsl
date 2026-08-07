from __future__ import annotations

import collections
import contextlib
import hashlib
import io
import pickle
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nemo_checkpoint  # noqa: E402


class _StorageToken:
    def __init__(self, kind: type[object], key: str, count: int) -> None:
        self.kind = kind
        self.key = key
        self.count = count


class _TensorToken:
    def __init__(
        self,
        rebuild: object,
        storage: _StorageToken,
        offset: int,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
    ) -> None:
        self.rebuild = rebuild
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.stride = stride

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            self.rebuild,
            (
                self.storage,
                self.offset,
                self.shape,
                self.stride,
                False,
                collections.OrderedDict(),
            ),
        )


@contextlib.contextmanager
def _fake_torch_pickle_globals():
    previous_torch = sys.modules.get("torch")
    previous_utils = sys.modules.get("torch._utils")
    torch_module = types.ModuleType("torch")
    utils_module = types.ModuleType("torch._utils")
    float_storage = type("FloatStorage", (), {"__module__": "torch"})
    long_storage = type("LongStorage", (), {"__module__": "torch"})

    def rebuild_tensor(*_args: object) -> None:
        return None

    rebuild_tensor.__module__ = "torch._utils"
    rebuild_tensor.__name__ = "_rebuild_tensor_v2"
    rebuild_tensor.__qualname__ = "_rebuild_tensor_v2"
    torch_module.FloatStorage = float_storage  # type: ignore[attr-defined]
    torch_module.LongStorage = long_storage  # type: ignore[attr-defined]
    utils_module._rebuild_tensor_v2 = rebuild_tensor  # type: ignore[attr-defined]
    torch_module._utils = utils_module  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_module
    sys.modules["torch._utils"] = utils_module
    try:
        yield float_storage, long_storage, rebuild_tensor
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
        if previous_utils is None:
            sys.modules.pop("torch._utils", None)
        else:
            sys.modules["torch._utils"] = previous_utils


def _state_dict_pickle(
    tensors: collections.OrderedDict[str, _TensorToken],
    *,
    location: str = "cuda:0",
) -> bytes:
    class StoragePickler(pickle.Pickler):
        def persistent_id(self, value: object) -> object:
            if isinstance(value, _StorageToken):
                return (
                    "storage",
                    value.kind,
                    value.key,
                    location,
                    value.count,
                )
            return None

    output = io.BytesIO()
    StoragePickler(output, protocol=2).dump(tensors)
    return output.getvalue()


def _member(name: str, normalized: str, payload: bytes) -> nemo_checkpoint.NemoMemberSpec:
    return nemo_checkpoint.NemoMemberSpec(
        archive_name=name,
        normalized_name=normalized,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _build_archive(
    *,
    data_pickle: bytes,
    storages: tuple[bytes, ...],
    compression: int = zipfile.ZIP_STORED,
    zip_comment: bytes = b"",
) -> tuple[bytes, nemo_checkpoint.NemoArchiveContract]:
    prefix = "model_weights"
    serialization_id = b"synthetic-checkpoint"
    checkpoint_stream = io.BytesIO()
    with zipfile.ZipFile(checkpoint_stream, "w", compression=compression) as checkpoint:
        checkpoint.writestr(f"{prefix}/data.pkl", data_pickle)
        checkpoint.writestr(f"{prefix}/byteorder", b"little")
        for index, payload in enumerate(storages):
            checkpoint.writestr(f"{prefix}/data/{index}", payload)
        checkpoint.writestr(f"{prefix}/version", b"3\n")
        checkpoint.writestr(
            f"{prefix}/.data/serialization_id",
            serialization_id,
        )
        checkpoint.comment = zip_comment
    checkpoint_payload = checkpoint_stream.getvalue()

    assets = {
        "vocab.txt": b"piece\n",
        "tokenizer.model": b"synthetic sentencepiece model",
        "tokenizer.vocab": b"<unk>\t0\n",
        "model_config.yaml": b"sample_rate: 16000\n",
        "model_weights.ckpt": checkpoint_payload,
    }
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in assets.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    archive_payload = tar_stream.getvalue()
    members = tuple(_member(name, name, payload) for name, payload in assets.items())
    contract = nemo_checkpoint.NemoArchiveContract(
        archive_filename="synthetic.nemo",
        archive_byte_length=len(archive_payload),
        archive_sha256=hashlib.sha256(archive_payload).hexdigest(),
        directory_names=(),
        members=members,
        weights_name="model_weights.ckpt",
        weights_byte_length=len(checkpoint_payload),
        weights_sha256=hashlib.sha256(checkpoint_payload).hexdigest(),
        zip_prefix=prefix,
        storage_count=len(storages),
        tensor_count=3,
        storage_location="cuda:0",
        data_pickle_byte_length=len(data_pickle),
        data_pickle_sha256=hashlib.sha256(data_pickle).hexdigest(),
        serialization_id=serialization_id,
    )
    return archive_payload, contract


def _valid_pickle_and_storages() -> tuple[bytes, tuple[bytes, ...]]:
    with _fake_torch_pickle_globals() as (float_storage, _long_storage, rebuild):
        shared = _StorageToken(float_storage, "0", 8)
        bias = _StorageToken(float_storage, "1", 3)
        tensors = collections.OrderedDict(
            (
                (
                    "decoder.prediction.dec_rnn.lstm.weight_ih_l0",
                    _TensorToken(rebuild, shared, 0, (2, 2), (2, 1)),
                ),
                (
                    "decoder.prediction.dec_rnn.lstm.weight_hh_l0",
                    _TensorToken(rebuild, shared, 4, (2, 2), (2, 1)),
                ),
                (
                    "joint.joint_net.2.bias",
                    _TensorToken(rebuild, bias, 0, (3,), (1,)),
                ),
            )
        )
        data_pickle = _state_dict_pickle(tensors)
    storages = (
        np.arange(1, 9, dtype="<f4").tobytes(),
        np.arange(9, 12, dtype="<f4").tobytes(),
    )
    return data_pickle, storages


class RestrictedPickleTests(unittest.TestCase):
    def test_forbidden_global_is_rejected(self) -> None:
        payload = pickle.dumps(eval, protocol=2)
        with self.assertRaisesRegex(pickle.UnpicklingError, "Forbidden pickle global"):
            nemo_checkpoint.restricted_load_state_dict(
                payload,
                expected_location="cuda:0",
            )

    def test_out_of_bounds_tensor_view_is_rejected(self) -> None:
        with _fake_torch_pickle_globals() as (float_storage, _long, rebuild):
            storage = _StorageToken(float_storage, "0", 4)
            tensors = collections.OrderedDict(
                bad=_TensorToken(rebuild, storage, 2, (2, 2), (2, 1))
            )
            payload = _state_dict_pickle(tensors)
        with self.assertRaisesRegex(pickle.UnpicklingError, "exceeds"):
            nemo_checkpoint.restricted_load_state_dict(
                payload,
                expected_location="cuda:0",
            )

    def test_noncontiguous_stride_is_rejected(self) -> None:
        with _fake_torch_pickle_globals() as (float_storage, _long, rebuild):
            storage = _StorageToken(float_storage, "0", 4)
            tensors = collections.OrderedDict(
                bad=_TensorToken(rebuild, storage, 0, (2, 2), (1, 2))
            )
            payload = _state_dict_pickle(tensors)
        with self.assertRaisesRegex(pickle.UnpicklingError, "contiguous"):
            nemo_checkpoint.restricted_load_state_dict(
                payload,
                expected_location="cuda:0",
            )

    def test_unexpected_storage_location_is_rejected(self) -> None:
        with _fake_torch_pickle_globals() as (float_storage, _long, rebuild):
            storage = _StorageToken(float_storage, "0", 1)
            tensors = collections.OrderedDict(
                value=_TensorToken(rebuild, storage, 0, (1,), (1,))
            )
            payload = _state_dict_pickle(tensors, location="cpu")

        with self.assertRaisesRegex(
            pickle.UnpicklingError,
            "Unexpected checkpoint storage location",
        ):
            nemo_checkpoint.restricted_load_state_dict(
                payload,
                expected_location="cuda:0",
            )

    def test_conflicting_storage_metadata_is_rejected(self) -> None:
        with _fake_torch_pickle_globals() as (float_storage, _long, rebuild):
            first = _StorageToken(float_storage, "0", 1)
            conflicting = _StorageToken(float_storage, "0", 2)
            tensors = collections.OrderedDict(
                (
                    ("first", _TensorToken(rebuild, first, 0, (1,), (1,))),
                    (
                        "conflicting",
                        _TensorToken(rebuild, conflicting, 0, (1,), (1,)),
                    ),
                )
            )
            payload = _state_dict_pickle(tensors)

        with self.assertRaisesRegex(
            pickle.UnpicklingError,
            "Conflicting metadata for storage 0",
        ):
            nemo_checkpoint.restricted_load_state_dict(
                payload,
                expected_location="cuda:0",
            )

    def test_trailing_pickle_data_is_rejected(self) -> None:
        data_pickle, _storages = _valid_pickle_and_storages()

        with self.assertRaisesRegex(
            pickle.UnpicklingError,
            "Trailing data after state-dict pickle",
        ):
            nemo_checkpoint.restricted_load_state_dict(
                data_pickle + b"unexpected",
                expected_location="cuda:0",
            )


class NemoCheckpointTests(unittest.TestCase):
    def test_archive_filename_is_part_of_the_pinned_identity(self) -> None:
        data_pickle, storages = _valid_pickle_and_storages()
        archive_payload, contract = _build_archive(
            data_pickle=data_pickle,
            storages=storages,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "renamed.nemo"
            path.write_bytes(archive_payload)
            with self.assertRaisesRegex(RuntimeError, "filename changed"):
                nemo_checkpoint.NemoCheckpoint(
                    path,
                    contract=contract,
                    validate_hashes=True,
                )

    def test_canonical_mapping_and_shared_storage_offsets(self) -> None:
        data_pickle, storages = _valid_pickle_and_storages()
        archive_payload, contract = _build_archive(
            data_pickle=data_pickle,
            storages=storages,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.nemo"
            path.write_bytes(archive_payload)
            with nemo_checkpoint.NemoCheckpoint(
                path,
                contract=contract,
                validate_hashes=True,
            ) as checkpoint:
                self.assertEqual(
                    set(checkpoint.keys()),
                    {
                        "decoder.lstm.weight_ih_l0",
                        "decoder.lstm.weight_hh_l0",
                        "joint.head.bias",
                    },
                )
                np.testing.assert_array_equal(
                    checkpoint.get_tensor("decoder.lstm.weight_ih_l0"),
                    [[1, 2], [3, 4]],
                )
                np.testing.assert_array_equal(
                    checkpoint.get_tensor("decoder.lstm.weight_hh_l0"),
                    [[5, 6], [7, 8]],
                )
                np.testing.assert_array_equal(
                    checkpoint.get_tensor("joint.head.bias"),
                    [9, 10, 11],
                )

    def test_compressed_checkpoint_entry_is_rejected(self) -> None:
        data_pickle, storages = _valid_pickle_and_storages()
        archive_payload, contract = _build_archive(
            data_pickle=data_pickle,
            storages=storages,
            compression=zipfile.ZIP_DEFLATED,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.nemo"
            path.write_bytes(archive_payload)
            with self.assertRaisesRegex(RuntimeError, "must be stored"):
                nemo_checkpoint.NemoCheckpoint(
                    path,
                    contract=contract,
                    validate_hashes=True,
                )

    def test_checkpoint_zip_comment_is_rejected(self) -> None:
        data_pickle, storages = _valid_pickle_and_storages()
        archive_payload, contract = _build_archive(
            data_pickle=data_pickle,
            storages=storages,
            zip_comment=b"unexpected metadata",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.nemo"
            path.write_bytes(archive_payload)
            with self.assertRaisesRegex(RuntimeError, "ZIP comment must be empty"):
                nemo_checkpoint.NemoCheckpoint(
                    path,
                    contract=contract,
                    validate_hashes=True,
                )

    def test_checkpoint_storage_payload_length_is_validated(self) -> None:
        data_pickle, storages = _valid_pickle_and_storages()
        truncated_storages = (storages[0][:-4], storages[1])
        archive_payload, contract = _build_archive(
            data_pickle=data_pickle,
            storages=truncated_storages,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.nemo"
            path.write_bytes(archive_payload)
            with self.assertRaisesRegex(
                RuntimeError,
                "Checkpoint storage 0 length mismatch",
            ):
                nemo_checkpoint.NemoCheckpoint(
                    path,
                    contract=contract,
                    validate_hashes=True,
                )

    def test_canonical_tensor_name_collision_is_rejected(self) -> None:
        with _fake_torch_pickle_globals() as (float_storage, _long, rebuild):
            shared = _StorageToken(float_storage, "0", 2)
            bias = _StorageToken(float_storage, "1", 1)
            tensors = collections.OrderedDict(
                (
                    (
                        "decoder.prediction.embed.weight",
                        _TensorToken(rebuild, shared, 0, (1,), (1,)),
                    ),
                    (
                        "decoder.embedding.weight",
                        _TensorToken(rebuild, shared, 1, (1,), (1,)),
                    ),
                    (
                        "joint.joint_net.2.bias",
                        _TensorToken(rebuild, bias, 0, (1,), (1,)),
                    ),
                )
            )
            data_pickle = _state_dict_pickle(tensors)
        storages = (
            np.arange(2, dtype="<f4").tobytes(),
            np.arange(1, dtype="<f4").tobytes(),
        )
        archive_payload, contract = _build_archive(
            data_pickle=data_pickle,
            storages=storages,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.nemo"
            path.write_bytes(archive_payload)
            with self.assertRaisesRegex(
                RuntimeError,
                "Canonical tensor name collision: decoder.embedding.weight",
            ):
                nemo_checkpoint.NemoCheckpoint(
                    path,
                    contract=contract,
                    validate_hashes=True,
                )

    def test_archive_hash_is_checked_before_parsing(self) -> None:
        data_pickle, storages = _valid_pickle_and_storages()
        archive_payload, contract = _build_archive(
            data_pickle=data_pickle,
            storages=storages,
        )
        corrupted = bytearray(archive_payload)
        corrupted[-1] ^= 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.nemo"
            path.write_bytes(corrupted)
            with self.assertRaisesRegex(RuntimeError, "archive SHA-256 mismatch"):
                nemo_checkpoint.NemoCheckpoint(
                    path,
                    contract=contract,
                    validate_hashes=True,
                )

    def test_member_hash_is_checked_after_outer_archive_identity(self) -> None:
        data_pickle, storages = _valid_pickle_and_storages()
        archive_payload, contract = _build_archive(
            data_pickle=data_pickle,
            storages=storages,
        )
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as archive:
            vocab_offset = archive.getmember("vocab.txt").offset_data
        corrupted = bytearray(archive_payload)
        corrupted[vocab_offset] ^= 1
        corrupted_contract = replace(
            contract,
            archive_sha256=hashlib.sha256(corrupted).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.nemo"
            path.write_bytes(corrupted)
            with self.assertRaisesRegex(
                RuntimeError,
                "NeMo member vocab.txt SHA-256 mismatch",
            ):
                nemo_checkpoint.NemoCheckpoint(
                    path,
                    contract=corrupted_contract,
                    validate_hashes=True,
                )

    def test_official_cached_source_validates_full_identity_and_maps_to_schema(
        self,
    ) -> None:
        model_dir = Path(__file__).resolve().parents[1]
        archive_path = next(
            (
                candidate
                for candidate in (
                    model_dir
                    / "cache"
                    / "official"
                    / nemo_checkpoint.NEMO_ARCHIVE_FILENAME,
                    model_dir / "cache" / nemo_checkpoint.NEMO_ARCHIVE_FILENAME,
                )
                if candidate.is_file()
            ),
            None,
        )
        if archive_path is None:
            self.skipTest("official v2 NeMo archive is not cached")
        import convert

        with nemo_checkpoint.NemoCheckpoint(
            archive_path,
            validate_hashes=True,
        ) as checkpoint:
            specs = convert.official_source_tensor_specs()
            self.assertEqual(set(checkpoint.keys()), set(specs))
            convert.validate_checkpoint_schema(checkpoint, specs)
            self.assertEqual(
                hashlib.sha256(
                    checkpoint.get_tensor("preprocessor.window").tobytes()
                ).hexdigest(),
                "7f26d47d05919cf2a1a323072ff6c7da0ef35fd3472c5147d100e47ef84112cd",
            )
            self.assertEqual(
                hashlib.sha256(
                    checkpoint.get_tensor("preprocessor.mel_filters").tobytes()
                ).hexdigest(),
                "621d722a4bb3d53cbece1297b7c2058a359d37b52ae6bd45d902fe4427a415d4",
            )
            tokenizer = convert.build_decode_tokenizer_json(
                checkpoint.read_asset("tokenizer.vocab")
            )
            self.assertEqual(len(tokenizer["model"]["vocab"]), 1024)
            self.assertEqual(tokenizer["model"]["vocab"]["<unk>"], 0)
            self.assertEqual(tokenizer["model"]["vocab"]["▁the"], 5)
            self.assertEqual(tokenizer["model"]["vocab"]["ό"], 1023)
            self.assertNotIn("<pad>", tokenizer["model"]["vocab"])
            self.assertEqual(
                tokenizer["added_tokens"],
                [
                    {"id": 0, "content": "<unk>", "special": True},
                    {"id": 1024, "content": "<blank>", "special": True},
                ],
            )


if __name__ == "__main__":
    unittest.main()
