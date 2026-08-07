from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1]
FILES_DIR = MODEL_DIR / "files"
MANIFEST_PATH = FILES_DIR / "manifest.json"
FP32_FILES_DIR = MODEL_DIR / "files-fp32"
FP32_MANIFEST_PATH = FP32_FILES_DIR / "manifest.json"
SOURCE_PATH_CANDIDATES = (
    MODEL_DIR / "cache" / "official" / "parakeet-tdt-0.6b-v2.nemo",
    MODEL_DIR / "cache" / "parakeet-tdt-0.6b-v2.nemo",
)
ALIGNMENT = 256
FP16_MANIFEST_SHA256 = (
    "11a359db3d050fd82b002c745b24a5280f3ff13a76834b548df671c95c786c65"
)
FP16_MANIFEST_LISTED_BYTES = 405_363_502
FP16_BINARY_BYTES = 405_063_936
FP16_TENSOR_BYTES = 405_030_496
FP32_MANIFEST_SHA256 = (
    "28dee836aefc2bfb01236fda6d10e1df7447724d2489040168549999ea267b1b"
)
FP32_MANIFEST_LISTED_BYTES = 440_817_198
FP32_BINARY_BYTES = 440_517_632
FP32_TENSOR_BYTES = 440_495_200

sys.path.insert(0, str(MODEL_DIR))

import convert  # noqa: E402


def official_source_path() -> Path:
    for path in SOURCE_PATH_CANDIDATES:
        if path.is_file():
            return path
    locations = ", ".join(str(path) for path in SOURCE_PATH_CANDIDATES)
    raise AssertionError(
        "PARAKEET_VERIFY_SOURCE_PACKING=1 requires the pinned v2 NeMo "
        f"archive at one of: {locations}"
    )


@dataclass(frozen=True)
class PackedSpec:
    dtype: str
    logical_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]
    layout: str
    shard: str
    runtime_required: bool = True


def expected_packed_specs(
    *,
    vocabulary_size: int = 1025,
    joint_output_size: int = 1030,
    joint_storage_output_size: int = 1032,
) -> dict[str, PackedSpec]:
    specs: dict[str, PackedSpec] = {}

    def add(
        name: str,
        dtype: str,
        logical_shape: tuple[int, ...],
        storage_shape: tuple[int, ...],
        layout: str,
        shard: str,
        *,
        runtime_required: bool = True,
    ) -> None:
        if name in specs:
            raise AssertionError(f"Duplicate packed specification {name}")
        specs[name] = PackedSpec(
            dtype,
            logical_shape,
            storage_shape,
            layout,
            shard,
            runtime_required,
        )

    def add_palettized_matrix(
        name: str,
        logical_shape: tuple[int, int],
        runtime_matrix_shape: tuple[int, int],
        shard: str,
        *,
        bits: int = 5,
        palette_groups: int = 1,
    ) -> None:
        codes_per_group, words_per_group = convert.palette_code_group_spec(bits)
        code_count = runtime_matrix_shape[0] * runtime_matrix_shape[1]
        if code_count % codes_per_group:
            raise AssertionError(
                f"{name} has {code_count} codes, not divisible by "
                f"{codes_per_group}"
            )
        add(
            name,
            "uint32",
            logical_shape,
            (code_count // codes_per_group, words_per_group),
            {
                5: "k-by-output-row-major-palette5-lsb-u32",
                6: "k-by-output-row-major-palette6-lsb-u32",
            }[bits],
            shard,
        )
        add(
            f"{name}.palette",
            "float16",
            (palette_groups, 1 << bits),
            (palette_groups, 1 << bits),
            "output-group-by-palette-lut",
            shard,
        )

    add(
        "preprocessor.window",
        "float32",
        (400,),
        (400,),
        "hann-periodic-false",
        "preprocessor.bin",
        runtime_required=False,
    )
    add(
        "preprocessor.mel_filters",
        "float32",
        (128, 257),
        (128, 257),
        "mel-bin-by-rfft-bin",
        "preprocessor.bin",
        runtime_required=False,
    )

    for layer, layout in (
        (0, "kh-kw-output-channel-vec4"),
        (2, "kh-kw-channel-vec4"),
        (5, "kh-kw-channel-vec4"),
    ):
        add(
            f"encoder.subsampling.layers.{layer}.weight",
            "float16",
            (3, 3, 256),
            (3, 3, 64, 4),
            layout,
            "encoder-subsampling.bin",
        )
        add(
            f"encoder.subsampling.layers.{layer}.bias",
            "float32",
            (256,),
            (256,),
            "output-channel",
            "encoder-subsampling.bin",
        )
    for layer in (3, 6):
        add_palettized_matrix(
            f"encoder.subsampling.layers.{layer}.weight",
            (256, 256),
            (256, 256),
            "encoder-subsampling.bin",
        )
        add(
            f"encoder.subsampling.layers.{layer}.bias",
            "float32",
            (256,),
            (256,),
            "output-channel",
            "encoder-subsampling.bin",
        )
    add_palettized_matrix(
        "encoder.subsampling.linear.weight",
        (4096, 1024),
        (4096, 1024),
        "encoder-subsampling.bin",
    )
    add(
        "encoder.subsampling.linear.bias",
        "float32",
        (1024,),
        (1024,),
        "output-channel",
        "encoder-subsampling.bin",
    )

    for layer in range(24):
        prefix = f"encoder.layers.{layer}"
        shard = f"encoder-layer-{layer:02d}.bin"
        for feed_forward in ("feed_forward1", "feed_forward2"):
            add_palettized_matrix(
                f"{prefix}.{feed_forward}.linear1.weight",
                (1024, 4096),
                (1024, 4096),
                shard,
            )
            add_palettized_matrix(
                f"{prefix}.{feed_forward}.linear2.weight",
                (4096, 1024),
                (4096, 1024),
                shard,
            )
        for norm in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            for parameter in ("weight", "bias"):
                add(
                    f"{prefix}.{norm}.{parameter}",
                    "float32",
                    (1024,),
                    (1024,),
                    "channel",
                    shard,
                )
        attention = f"{prefix}.self_attn"
        add_palettized_matrix(
            f"{attention}.qkv.weight",
            (1024, 3072),
            (1024, 3072),
            shard,
            palette_groups=3,
        )
        add(
            f"{attention}.projected_positions",
            "float16",
            (375, 1024),
            (8, 32, 375, 4),
            "head-vector-position-vec4",
            shard,
        )
        add_palettized_matrix(
            f"{attention}.o_proj.weight",
            (1024, 1024),
            (1024, 1024),
            shard,
            bits=6,
        )
        for bias in ("bias_u", "bias_v"):
            add(
                f"{attention}.{bias}",
                "float32",
                (8, 128),
                (8, 128),
                "head-by-channel",
                shard,
            )
        convolution = f"{prefix}.conv"
        add_palettized_matrix(
            f"{convolution}.pointwise_conv1.weight",
            (1024, 2048),
            (1024, 2048),
            shard,
        )
        add(
            f"{convolution}.depthwise_conv.weight",
            "float16",
            (9, 1024),
            (9, 256, 4),
            "kernel-channel-vec4",
            shard,
        )
        add_palettized_matrix(
            f"{convolution}.pointwise_conv2.weight",
            (1024, 1024),
            (1024, 1024),
            shard,
        )
        for parameter in ("scale", "shift"):
            add(
                f"{convolution}.norm.{parameter}",
                "float32",
                (1024,),
                (1024,),
                "channel",
                shard,
            )

    add(
        "decoder.embedding.weight",
        "float16",
        (vocabulary_size, 640),
        (vocabulary_size, 640),
        "token-by-channel-row-major",
        "decoder.bin",
    )
    for layer in range(2):
        add(
            f"decoder.lstm.weight_l{layer}",
            "float16",
            (1280, 2560),
            (1280, 2560),
            "input-hidden-k-by-ifgo-row-major-vec4",
            "decoder.bin",
        )
        add(
            f"decoder.lstm.bias_l{layer}",
            "float32",
            (2560,),
            (2560,),
            "ifgo-gates",
            "decoder.bin",
        )
    for name, logical, storage in (
        ("decoder.decoder_projector.weight", (640, 640), (640, 640)),
        (
            "joint.head.weight",
            (640, joint_output_size),
            (640, joint_storage_output_size),
        ),
    ):
        add(
            name,
            "float16",
            logical,
            storage,
            "k-by-output-row-major-vec4",
            "decoder.bin",
        )
    add_palettized_matrix(
        "encoder_projector.weight",
        (1024, 640),
        (1024, 640),
        "decoder.bin",
    )
    for name, shape in (
        ("decoder.decoder_projector.bias", (640,)),
        ("encoder_projector.bias", (640,)),
    ):
        add(
            name,
            "float32",
            shape,
            shape,
            "output-channel",
            "decoder.bin",
        )
    add(
        "joint.head.bias",
        "float32",
        (joint_output_size,),
        (joint_storage_output_size,),
        "output-channel-padded-vec4",
        "decoder.bin",
    )
    return specs


def expected_fp32_packed_specs(
    *,
    vocabulary_size: int = 1025,
    joint_output_size: int = 1030,
    joint_storage_output_size: int = 1032,
) -> dict[str, PackedSpec]:
    return {
        name: PackedSpec(
            dtype="float32" if spec.dtype == "float16" else spec.dtype,
            logical_shape=spec.logical_shape,
            storage_shape=spec.storage_shape,
            layout=spec.layout,
            shard=spec.shard,
            runtime_required=spec.runtime_required,
        )
        for name, spec in expected_packed_specs(
            vocabulary_size=vocabulary_size,
            joint_output_size=joint_output_size,
            joint_storage_output_size=joint_storage_output_size,
        ).items()
    }


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise unittest.SkipTest("model/files/manifest.json has not been generated")
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AssertionError("Manifest root must be an object")
    return value


def load_fp32_manifest() -> dict[str, Any]:
    if not FP32_MANIFEST_PATH.is_file():
        raise unittest.SkipTest(
            "model/files-fp32/manifest.json has not been generated"
        )
    with FP32_MANIFEST_PATH.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AssertionError("FP32 manifest root must be an object")
    return value


def packed_array(manifest: dict[str, Any], name: str) -> np.ndarray:
    record = manifest["tensors"][name]
    dtype = {
        "float16": "<f2",
        "float32": "<f4",
        "uint32": "<u4",
    }[record["dtype"]]
    return np.memmap(
        FILES_DIR / record["shard"],
        mode="r",
        dtype=dtype,
        offset=record["byteOffset"],
        shape=tuple(record["storageShape"]),
    )


def fp32_packed_array(manifest: dict[str, Any], name: str) -> np.ndarray:
    record = manifest["tensors"][name]
    dtype = {
        "float32": "<f4",
        "uint32": "<u4",
    }[record["dtype"]]
    return np.memmap(
        FP32_FILES_DIR / record["shard"],
        mode="r",
        dtype=dtype,
        offset=record["byteOffset"],
        shape=tuple(record["storageShape"]),
    )


class V2PackedContractTests(unittest.TestCase):
    def test_joint_output_is_logical_1030_with_vec4_storage_1032(self) -> None:
        specs = expected_packed_specs()
        joint_weight = specs["joint.head.weight"]
        joint_bias = specs["joint.head.bias"]

        self.assertEqual(joint_weight.logical_shape, (640, 1030))
        self.assertEqual(joint_weight.storage_shape, (640, 1032))
        self.assertEqual(joint_bias.logical_shape, (1030,))
        self.assertEqual(joint_bias.storage_shape, (1032,))


class GeneratedPackageTests(unittest.TestCase):
    def test_manifest_exactly_satisfies_runtime_tensor_contract(self) -> None:
        manifest = load_manifest()
        records = manifest["tensors"]
        specs = expected_packed_specs()

        self.assertEqual(manifest["format"], "parakeet-webgpu-v1")
        self.assertEqual(
            hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            FP16_MANIFEST_SHA256,
        )
        self.assertEqual(
            manifest["source"],
            {
                "archive": {
                    "byteLength": convert.SOURCE_ARCHIVE_BYTES,
                    "name": convert.SOURCE_ARCHIVE_FILENAME,
                    "sha256": convert.SOURCE_ARCHIVE_SHA256,
                },
                "license": "CC-BY-4.0",
                "repository": convert.MODEL_ID,
                "revision": convert.SOURCE_REVISION,
                "weights": {
                    "byteLength": convert.SOURCE_WEIGHT_BYTES,
                    "name": "model_weights.ckpt",
                    "sha256": convert.SOURCE_WEIGHT_SHA256,
                },
            },
        )
        self.assertEqual(
            manifest["config"]["encoderWeightFormat"],
            "mixed-palette5-o-proj-palette6-fp16-lut",
        )
        self.assertEqual(len(specs), 797)
        self.assertEqual(set(records), set(specs))
        self.assertEqual(
            sum(spec.runtime_required for spec in specs.values()),
            795,
        )

        for name, spec in specs.items():
            with self.subTest(name=name):
                record = records[name]
                self.assertEqual(record["dtype"], spec.dtype)
                self.assertEqual(tuple(record["logicalShape"]), spec.logical_shape)
                self.assertEqual(tuple(record["storageShape"]), spec.storage_shape)
                self.assertEqual(record["layout"], spec.layout)
                self.assertEqual(record["shard"], spec.shard)
                element_bytes = {
                    "float16": 2,
                    "float32": 4,
                    "uint32": 4,
                }[spec.dtype]
                expected_bytes = element_bytes
                for dimension in spec.storage_shape:
                    expected_bytes *= dimension
                self.assertEqual(record["byteLength"], expected_bytes)

    def test_file_sizes_tensor_ranges_alignment_and_zero_padding(self) -> None:
        manifest = load_manifest()
        file_records = manifest["files"]
        names = [record["name"] for record in file_records]
        self.assertEqual(len(file_records), 32)
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            {path.name for path in FILES_DIR.iterdir() if path.is_file()},
            set(names) | {"manifest.json"},
        )
        lengths = {record["name"]: record["byteLength"] for record in file_records}
        for name, byte_length in lengths.items():
            self.assertEqual((FILES_DIR / name).stat().st_size, byte_length)

        self.assertEqual(
            sum(record["byteLength"] for record in file_records),
            FP16_MANIFEST_LISTED_BYTES,
        )
        binary_records = [
            record for record in file_records if record["name"].endswith(".bin")
        ]
        self.assertEqual(len(binary_records), 27)
        self.assertEqual(
            sum(record["byteLength"] for record in binary_records),
            FP16_BINARY_BYTES,
        )
        self.assertEqual(
            sum(record["byteLength"] for record in manifest["tensors"].values()),
            FP16_TENSOR_BYTES,
        )

        ranges_by_shard: dict[str, list[tuple[int, int, str]]] = {}
        for name, record in manifest["tensors"].items():
            shard = record["shard"]
            self.assertTrue(shard.endswith(".bin"))
            self.assertIn(shard, lengths)
            offset = record["byteOffset"]
            end = offset + record["byteLength"]
            self.assertEqual(offset % ALIGNMENT, 0)
            self.assertEqual(record["byteLength"] % 4, 0)
            self.assertLessEqual(end, lengths[shard])
            ranges_by_shard.setdefault(shard, []).append((offset, end, name))

        binary_files = {name for name in names if name.endswith(".bin")}
        self.assertEqual(set(ranges_by_shard), binary_files)
        for shard, ranges in ranges_by_shard.items():
            ranges.sort()
            previous_end = 0
            with (FILES_DIR / shard).open("rb") as stream:
                for offset, end, name in ranges:
                    with self.subTest(shard=shard, tensor=name):
                        self.assertGreaterEqual(offset, previous_end)
                        self.assertLess(offset - previous_end, ALIGNMENT)
                        if offset > previous_end:
                            stream.seek(previous_end)
                            self.assertEqual(
                                stream.read(offset - previous_end),
                                b"\0" * (offset - previous_end),
                            )
                        previous_end = end
                trailing = lengths[shard] - previous_end
                self.assertLess(trailing, ALIGNMENT)
                if trailing:
                    stream.seek(previous_end)
                    self.assertEqual(stream.read(trailing), b"\0" * trailing)

    @unittest.skipUnless(
        os.environ.get("PARAKEET_VERIFY_PACKAGE_HASHES") == "1",
        "set PARAKEET_VERIFY_PACKAGE_HASHES=1 for the full package hash pass",
    )
    def test_all_manifest_file_hashes(self) -> None:
        manifest = load_manifest()
        for record in manifest["files"]:
            path = FILES_DIR / record["name"]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
            self.assertEqual(digest.hexdigest(), record["sha256"], path.name)

    @unittest.skipUnless(
        os.environ.get("PARAKEET_VERIFY_SOURCE_PACKING") == "1",
        "set PARAKEET_VERIFY_SOURCE_PACKING=1 for source-to-package checks",
    )
    def test_high_risk_packing_against_official_source(self) -> None:
        source_path = official_source_path()
        manifest = load_manifest()
        self.assertEqual(manifest["source"]["repository"], convert.MODEL_ID)

        def assert_palettized_matrix(
            name: str,
            expected: np.ndarray,
            *,
            bits: int,
            output_group_width: int | None = None,
        ) -> None:
            expected_words, expected_palettes = convert.palettize_matrix(
                expected,
                bits=bits,
                output_group_width=output_group_width,
            )
            np.testing.assert_array_equal(
                packed_array(manifest, name),
                expected_words,
            )
            np.testing.assert_array_equal(
                packed_array(manifest, f"{name}.palette"),
                expected_palettes,
            )

        with convert.NemoCheckpoint(
            source_path,
            validate_hashes=False,
        ) as checkpoint:
            np.testing.assert_array_equal(
                packed_array(manifest, "preprocessor.window"),
                checkpoint.get_tensor("preprocessor.window"),
            )
            np.testing.assert_array_equal(
                packed_array(manifest, "preprocessor.mel_filters"),
                checkpoint.get_tensor("preprocessor.mel_filters")[0],
            )

            for layer in (0, 2, 5):
                name = f"encoder.subsampling.layers.{layer}.weight"
                source = checkpoint.get_tensor(name)
                packer = (
                    convert.pack_conv0_kernel
                    if layer == 0
                    else convert.pack_depthwise_conv2d_kernel
                )
                expected, _ = packer(source)
                np.testing.assert_array_equal(packed_array(manifest, name), expected)

            for layer in (0, 23):
                prefix = f"encoder.layers.{layer}"
                depthwise_name = f"{prefix}.conv.depthwise_conv.weight"
                expected_depthwise, _ = convert.pack_conformer_depthwise_kernel(
                    checkpoint.get_tensor(depthwise_name)
                )
                np.testing.assert_array_equal(
                    packed_array(manifest, depthwise_name),
                    expected_depthwise,
                )

                relative_name = f"{prefix}.self_attn.relative_k_proj.weight"
                expected_positions = convert.project_relative_positions(
                    convert.build_relative_positions(),
                    checkpoint.get_tensor(relative_name),
                )
                expected_positions = convert.pack_projected_relative_positions(
                    expected_positions,
                )
                np.testing.assert_array_equal(
                    packed_array(
                        manifest,
                        f"{prefix}.self_attn.projected_positions",
                    ),
                    expected_positions,
                )

            for source_name in (
                "encoder.subsampling.layers.3.weight",
                "encoder.layers.0.feed_forward1.linear1.weight",
                "encoder.layers.0.conv.pointwise_conv1.weight",
                "encoder.layers.0.self_attn.o_proj.weight",
                "decoder.decoder_projector.weight",
                "encoder_projector.weight",
                "joint.head.weight",
            ):
                source = checkpoint.get_tensor(source_name)
                if source.ndim > 2:
                    source = np.squeeze(source)
                expected, _ = convert.pack_matrix(source)
                if manifest["tensors"][source_name]["dtype"] == "uint32":
                    assert_palettized_matrix(
                        source_name,
                        expected,
                        bits=(
                            6
                            if source_name.endswith("self_attn.o_proj.weight")
                            else 5
                        ),
                    )
                else:
                    np.testing.assert_array_equal(
                        packed_array(manifest, source_name),
                        expected,
                    )

            subsampling_linear_name = "encoder.subsampling.linear.weight"
            expected_subsampling_linear, _ = (
                convert.pack_subsampling_linear_matrix(
                    checkpoint.get_tensor(subsampling_linear_name)
                )
            )
            assert_palettized_matrix(
                subsampling_linear_name,
                expected_subsampling_linear,
                bits=5,
            )

            attention = "encoder.layers.0.self_attn"
            expected_qkv, _ = convert.pack_output_concatenated_matrices(
                checkpoint,
                [
                    f"{attention}.q_proj.weight",
                    f"{attention}.k_proj.weight",
                    f"{attention}.v_proj.weight",
                ],
                set(),
            )
            assert_palettized_matrix(
                f"{attention}.qkv.weight",
                expected_qkv,
                bits=5,
                output_group_width=1024,
            )

            convolution = "encoder.layers.0.conv"
            variance = checkpoint.get_tensor(f"{convolution}.norm.running_var")
            denominator = np.sqrt(
                np.asarray(variance, dtype=np.float32) + np.float32(1e-5)
            )
            scale = (
                np.asarray(
                    checkpoint.get_tensor(f"{convolution}.norm.weight"),
                    dtype=np.float32,
                )
                / denominator
            )
            shift = (
                -np.asarray(
                    checkpoint.get_tensor(f"{convolution}.norm.running_mean"),
                    dtype=np.float32,
                )
                * scale
            )
            shift += checkpoint.get_tensor(f"{convolution}.norm.bias")
            np.testing.assert_array_equal(
                packed_array(manifest, f"{convolution}.norm.scale"),
                scale,
            )
            np.testing.assert_array_equal(
                packed_array(manifest, f"{convolution}.norm.shift"),
                shift,
            )

            expected_lstm, _ = convert.pack_input_concatenated_matrices(
                checkpoint,
                [
                    "decoder.lstm.weight_ih_l0",
                    "decoder.lstm.weight_hh_l0",
                ],
                set(),
            )
            np.testing.assert_array_equal(
                packed_array(manifest, "decoder.lstm.weight_l0"),
                expected_lstm,
            )
            expected_lstm_bias = np.array(
                checkpoint.get_tensor("decoder.lstm.bias_ih_l0"),
                dtype="<f4",
                copy=True,
            )
            expected_lstm_bias += checkpoint.get_tensor("decoder.lstm.bias_hh_l0")
            np.testing.assert_array_equal(
                packed_array(manifest, "decoder.lstm.bias_l0"),
                expected_lstm_bias,
            )

            np.testing.assert_array_equal(
                packed_array(manifest, "decoder.embedding.weight"),
                checkpoint.get_tensor("decoder.embedding.weight").astype("<f2"),
            )
            joint_bias = checkpoint.get_tensor("joint.head.bias")
            np.testing.assert_array_equal(
                packed_array(manifest, "joint.head.bias"),
                convert.pad_last_dimension(
                    joint_bias,
                    multiple=convert.VECTOR_WIDTH,
                ),
            )


class GeneratedFp32PackageTests(unittest.TestCase):
    def test_manifest_exactly_satisfies_fp32_runtime_tensor_contract(self) -> None:
        manifest = load_fp32_manifest()
        fp16_manifest = load_manifest()
        records = manifest["tensors"]
        specs = expected_fp32_packed_specs()

        self.assertEqual(
            set(manifest),
            {"format", "source", "config", "files", "tensors"},
        )
        self.assertEqual(manifest["format"], convert.FP32_FORMAT_VERSION)
        self.assertEqual(manifest["source"], fp16_manifest["source"])
        expected_config = dict(fp16_manifest["config"])
        expected_config["encoderWeightFormat"] = (
            convert.FP32_ENCODER_WEIGHT_FORMAT
        )
        self.assertEqual(manifest["config"], expected_config)
        self.assertEqual(
            hashlib.sha256(FP32_MANIFEST_PATH.read_bytes()).hexdigest(),
            FP32_MANIFEST_SHA256,
        )

        self.assertEqual(len(specs), 797)
        self.assertEqual(set(records), set(specs))
        self.assertEqual(
            sum(spec.runtime_required for spec in specs.values()),
            795,
        )
        self.assertEqual(
            sum(spec.dtype == "float32" for spec in specs.values()),
            601,
        )
        self.assertEqual(
            sum(spec.dtype == "uint32" for spec in specs.values()),
            196,
        )
        self.assertNotIn("float16", {spec.dtype for spec in specs.values()})

        for name, spec in specs.items():
            with self.subTest(name=name):
                record = records[name]
                self.assertEqual(
                    set(record),
                    {
                        "shard",
                        "byteOffset",
                        "byteLength",
                        "dtype",
                        "logicalShape",
                        "storageShape",
                        "layout",
                    },
                )
                self.assertEqual(record["dtype"], spec.dtype)
                self.assertEqual(tuple(record["logicalShape"]), spec.logical_shape)
                self.assertEqual(tuple(record["storageShape"]), spec.storage_shape)
                self.assertEqual(record["layout"], spec.layout)
                self.assertEqual(record["shard"], spec.shard)
                element_bytes = {
                    "float32": 4,
                    "uint32": 4,
                }[spec.dtype]
                expected_bytes = element_bytes
                for dimension in spec.storage_shape:
                    expected_bytes *= dimension
                self.assertEqual(record["byteLength"], expected_bytes)

    def test_all_palette_assignments_match_fp16_package(self) -> None:
        manifest = load_fp32_manifest()
        fp16_manifest = load_manifest()
        palette_matrix_names = [
            name
            for name, record in manifest["tensors"].items()
            if record["dtype"] == "uint32"
        ]
        self.assertEqual(len(palette_matrix_names), 196)

        for name in palette_matrix_names:
            with self.subTest(name=name):
                record = manifest["tensors"][name]
                fp16_record = fp16_manifest["tensors"][name]
                self.assertEqual(fp16_record["dtype"], "uint32")
                self.assertEqual(record["layout"], fp16_record["layout"])
                self.assertEqual(
                    record["storageShape"],
                    fp16_record["storageShape"],
                )
                np.testing.assert_array_equal(
                    fp32_packed_array(manifest, name),
                    packed_array(fp16_manifest, name),
                )

    def test_file_sizes_tensor_ranges_alignment_and_zero_padding(self) -> None:
        manifest = load_fp32_manifest()
        fp16_manifest = load_manifest()
        file_records = manifest["files"]
        names = [record["name"] for record in file_records]

        self.assertEqual(len(file_records), 32)
        self.assertEqual(names, sorted(names))
        self.assertEqual(
            names,
            [record["name"] for record in fp16_manifest["files"]],
        )
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            {path.name for path in FP32_FILES_DIR.iterdir() if path.is_file()},
            set(names) | {"manifest.json"},
        )
        for record in file_records:
            with self.subTest(file=record["name"]):
                self.assertEqual(
                    set(record),
                    {"name", "byteLength", "sha256"},
                )
                self.assertEqual(len(record["sha256"]), 64)
                self.assertTrue(
                    all(character in "0123456789abcdef" for character in record["sha256"])
                )
                self.assertEqual(
                    (FP32_FILES_DIR / record["name"]).stat().st_size,
                    record["byteLength"],
                )

        self.assertEqual(
            sum(record["byteLength"] for record in file_records),
            FP32_MANIFEST_LISTED_BYTES,
        )
        binary_records = [
            record for record in file_records if record["name"].endswith(".bin")
        ]
        self.assertEqual(len(binary_records), 27)
        self.assertEqual(
            sum(record["byteLength"] for record in binary_records),
            FP32_BINARY_BYTES,
        )
        self.assertEqual(
            sum(record["byteLength"] for record in manifest["tensors"].values()),
            FP32_TENSOR_BYTES,
        )

        lengths = {
            record["name"]: record["byteLength"] for record in file_records
        }
        ranges_by_shard: dict[str, list[tuple[int, int, str]]] = {}
        for name, record in manifest["tensors"].items():
            shard = record["shard"]
            self.assertTrue(shard.endswith(".bin"))
            self.assertIn(shard, lengths)
            offset = record["byteOffset"]
            end = offset + record["byteLength"]
            self.assertEqual(offset % ALIGNMENT, 0)
            self.assertEqual(record["byteLength"] % 4, 0)
            self.assertLessEqual(end, lengths[shard])
            ranges_by_shard.setdefault(shard, []).append((offset, end, name))

        binary_files = {record["name"] for record in binary_records}
        self.assertEqual(set(ranges_by_shard), binary_files)
        for shard, ranges in ranges_by_shard.items():
            ranges.sort()
            previous_end = 0
            with (FP32_FILES_DIR / shard).open("rb") as stream:
                for offset, end, name in ranges:
                    with self.subTest(shard=shard, tensor=name):
                        self.assertGreaterEqual(offset, previous_end)
                        self.assertLess(offset - previous_end, ALIGNMENT)
                        if offset > previous_end:
                            stream.seek(previous_end)
                            self.assertEqual(
                                stream.read(offset - previous_end),
                                b"\0" * (offset - previous_end),
                            )
                        previous_end = end
                trailing = lengths[shard] - previous_end
                self.assertLess(trailing, ALIGNMENT)
                if trailing:
                    stream.seek(previous_end)
                    self.assertEqual(stream.read(trailing), b"\0" * trailing)

    @unittest.skipUnless(
        os.environ.get("PARAKEET_VERIFY_PACKAGE_HASHES") == "1",
        "set PARAKEET_VERIFY_PACKAGE_HASHES=1 for the full package hash pass",
    )
    def test_all_fp32_manifest_file_hashes(self) -> None:
        manifest = load_fp32_manifest()
        for record in manifest["files"]:
            path = FP32_FILES_DIR / record["name"]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
            self.assertEqual(digest.hexdigest(), record["sha256"], path.name)

    @unittest.skipUnless(
        os.environ.get("PARAKEET_VERIFY_SOURCE_PACKING") == "1",
        "set PARAKEET_VERIFY_SOURCE_PACKING=1 for source-to-package checks",
    )
    def test_representative_fp32_packing_against_official_source(self) -> None:
        source_path = official_source_path()
        manifest = load_fp32_manifest()
        self.assertEqual(manifest["source"]["repository"], convert.MODEL_ID)
        native_library_path = convert.build_native_kmeans_library(
            MODEL_DIR / "cache"
        )
        convert.configure_native_kmeans_library(native_library_path)

        with convert.NemoCheckpoint(
            source_path,
            validate_hashes=False,
        ) as checkpoint:
            raw_name = "decoder.embedding.weight"
            np.testing.assert_array_equal(
                fp32_packed_array(manifest, raw_name),
                np.asarray(checkpoint.get_tensor(raw_name), dtype="<f4"),
            )

            palette_name = "encoder.subsampling.layers.3.weight"
            palette_source = np.squeeze(checkpoint.get_tensor(palette_name))
            expected_matrix, _ = convert.pack_matrix(
                palette_source,
                float_dtype="float32",
            )
            expected_words, expected_palettes = convert.palettize_matrix(
                expected_matrix,
                bits=5,
                float_dtype="float32",
            )
            np.testing.assert_array_equal(
                fp32_packed_array(manifest, palette_name),
                expected_words,
            )
            np.testing.assert_array_equal(
                fp32_packed_array(manifest, f"{palette_name}.palette"),
                expected_palettes,
            )


if __name__ == "__main__":
    unittest.main()
