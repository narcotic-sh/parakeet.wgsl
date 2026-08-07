from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert  # noqa: E402


class _FakeSlice:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

    def get_shape(self) -> tuple[int, ...]:
        return self._shape


class _FakeCheckpoint:
    def __init__(
        self,
        *,
        shapes: dict[str, tuple[int, ...]] | None = None,
        tensors: dict[str, np.ndarray] | None = None,
        assets: dict[str, bytes] | None = None,
    ) -> None:
        self._tensors = tensors or {}
        self._assets = assets or {}
        self._shapes = shapes or {
            name: tuple(values.shape) for name, values in self._tensors.items()
        }
        self.loads: list[str] = []

    def keys(self) -> list[str]:
        return list(self._shapes)

    def get_slice(self, name: str) -> _FakeSlice:
        return _FakeSlice(self._shapes[name])

    def get_tensor(self, name: str) -> np.ndarray:
        self.loads.append(name)
        return self._tensors[name]

    def read_asset(self, name: str) -> bytes:
        return self._assets[name]


class ConverterPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._native_cache = tempfile.TemporaryDirectory()
        library_path = convert.build_native_kmeans_library(
            Path(cls._native_cache.name)
        )
        convert.configure_native_kmeans_library(library_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._native_cache.cleanup()

    def test_manifest_writer_preserves_precision_specific_newline_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fp16_path = root / "fp16.json"
            fp32_path = root / "fp32.json"
            convert.write_json_atomic(
                fp16_path,
                {"format": convert.FORMAT_VERSION},
                False,
            )
            convert.write_json_atomic(
                fp32_path,
                {"format": convert.FP32_FORMAT_VERSION},
                True,
            )

            self.assertFalse(fp16_path.read_bytes().endswith(b"\n"))
            self.assertTrue(fp32_path.read_bytes().endswith(b"\n"))

    @staticmethod
    def _unpack_palette_codes(
        packed: np.ndarray,
        *,
        bits: int,
    ) -> np.ndarray:
        codes_per_group, words_per_group = convert.palette_code_group_spec(bits)
        mask = (1 << bits) - 1
        output = np.empty((packed.shape[0], codes_per_group), dtype=np.uint8)
        for code_index in range(codes_per_group):
            bit_offset = code_index * bits
            word_index = bit_offset // 32
            shift = bit_offset % 32
            value = packed[:, word_index].astype(np.uint64) >> np.uint64(shift)
            if shift + bits > 32:
                value |= (
                    packed[:, word_index + 1].astype(np.uint64)
                    << np.uint64(32 - shift)
                )
            output[:, code_index] = (value & np.uint64(mask)).astype(np.uint8)
        word_count = packed.shape[1]
        if word_count != words_per_group:
            raise AssertionError(
                f"Expected {words_per_group} words per group, got {word_count}"
            )
        return output.reshape(-1)

    def test_conv0_is_kh_kw_output_channel_vec4(self) -> None:
        source = np.arange(256 * 3 * 3, dtype=np.float32).reshape(256, 1, 3, 3)
        packed, logical = convert.pack_conv0_kernel(source)

        self.assertEqual(logical, (3, 3, 256))
        self.assertEqual(packed.shape, (3, 3, 64, 4))
        self.assertEqual(packed.dtype, np.dtype("<f2"))
        for channel in (0, 3, 4, 127, 255):
            for kh, kw in ((0, 0), (1, 2), (2, 1)):
                self.assertEqual(
                    packed[kh, kw, channel // 4, channel % 4],
                    np.float16(source[channel, 0, kh, kw]),
                )

    def test_subsampling_depthwise_is_kh_kw_channel_vec4(self) -> None:
        source = np.arange(256 * 3 * 3, dtype=np.float32).reshape(256, 1, 3, 3)
        packed, logical = convert.pack_depthwise_conv2d_kernel(source)

        self.assertEqual(logical, (3, 3, 256))
        self.assertEqual(packed.shape, (3, 3, 64, 4))
        for channel in (0, 4, 129, 255):
            self.assertEqual(
                packed[2, 1, channel // 4, channel % 4],
                np.float16(source[channel, 0, 2, 1]),
            )

    def test_subsampling_linear_reorders_channel_frequency_columns(self) -> None:
        source = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2 * 4)
        packed, logical = convert.pack_subsampling_linear_matrix(
            source,
            channels=2,
            frequencies=4,
        )

        self.assertEqual(logical, (8, 3))
        self.assertEqual(packed.shape, (8, 4))
        for frequency in range(4):
            for channel in range(2):
                runtime_k = frequency * 2 + channel
                source_k = channel * 4 + frequency
                np.testing.assert_array_equal(
                    packed[runtime_k, :3],
                    source[:, source_k].astype(np.float16),
                )
        np.testing.assert_array_equal(packed[:, 3], 0)

    def test_conformer_depthwise_is_kernel_channel_vec4(self) -> None:
        source = np.arange(1024 * 9, dtype=np.float32).reshape(1024, 1, 9)
        packed, logical = convert.pack_conformer_depthwise_kernel(source)

        self.assertEqual(logical, (9, 1024))
        self.assertEqual(packed.shape, (9, 256, 4))
        for kernel in (0, 4, 8):
            for channel in (0, 3, 4, 511, 1023):
                self.assertEqual(
                    packed[kernel, channel // 4, channel % 4],
                    np.float16(source[channel, 0, kernel]),
                )

    def test_output_concatenation_packs_qkv_orientation(self) -> None:
        tensors = {
            "q": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "k": np.array([[5, 6]], dtype=np.float32),
            "v": np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32),
        }
        checkpoint = _FakeCheckpoint(tensors=tensors)
        consumed: set[str] = set()
        packed, logical = convert.pack_output_concatenated_matrices(
            checkpoint,
            ["q", "k", "v"],
            consumed,
        )

        expected_source = np.concatenate([tensors["q"], tensors["k"], tensors["v"]])
        self.assertEqual(logical, (2, 6))
        self.assertEqual(packed.shape, (2, 8))
        np.testing.assert_array_equal(packed[:, :6], expected_source.T.astype(np.float16))
        np.testing.assert_array_equal(packed[:, 6:], 0)
        self.assertEqual(checkpoint.loads, ["q", "k", "v"])
        self.assertEqual(consumed, {"q", "k", "v"})

    def test_input_concatenation_packs_lstm_orientation(self) -> None:
        tensors = {
            "input": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            "hidden": np.array([[7], [8], [9]], dtype=np.float32),
        }
        checkpoint = _FakeCheckpoint(tensors=tensors)
        consumed: set[str] = set()
        packed, logical = convert.pack_input_concatenated_matrices(
            checkpoint,
            ["input", "hidden"],
            consumed,
        )

        expected_source = np.concatenate(
            [tensors["input"], tensors["hidden"]],
            axis=1,
        )
        self.assertEqual(logical, (3, 3))
        self.assertEqual(packed.shape, (3, 4))
        np.testing.assert_array_equal(packed[:, :3], expected_source.T.astype(np.float16))
        np.testing.assert_array_equal(packed[:, 3], 0)
        self.assertEqual(checkpoint.loads, ["input", "hidden"])

    def test_weighted_kmeans_matches_coreml_global_optimum(self) -> None:
        values = np.array(
            [0.0, 0.0, 1.0, 1.0, 9.0, 10.0],
            dtype="<f2",
        )
        codes, palette = convert.fit_fp16_palette(values, palette_entries=2)

        self.assertEqual(codes.dtype, np.dtype("u1"))
        self.assertEqual(palette.dtype, np.dtype("<f2"))
        np.testing.assert_array_equal(codes, [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(palette, np.array([0.5, 9.5], dtype="<f2"))

    def test_fp16_centroid_rounding_does_not_reassign_codes(self) -> None:
        result = SimpleNamespace(
            clusters=[0, 1],
            centroids=[1.0001, 1.0002],
        )
        kmeans = SimpleNamespace(cluster=mock.Mock(return_value=result))
        with mock.patch.object(
            convert,
            "coreml_kmeans1d",
            return_value=kmeans,
        ):
            codes, palette = convert.fit_fp16_palette(
                np.array([0.0, 1.0], dtype="<f2"),
                palette_entries=2,
            )

        np.testing.assert_array_equal(codes, [0, 1])
        np.testing.assert_array_equal(palette, np.array([1.0, 1.0], dtype="<f2"))

    def test_fp32_palette_refits_original_values_under_fp16_assignments(
        self,
    ) -> None:
        result = SimpleNamespace(
            clusters=[0, 1],
            centroids=[1.0, 9.0],
        )
        with mock.patch.object(
            convert,
            "run_native_kmeans1d_sorted_unique",
            return_value=(
                np.asarray(result.clusters, dtype=np.uint32),
                np.asarray(result.centroids, dtype=np.float64),
            ),
        ) as cluster:
            codes, palette = convert.fit_fp32_palette(
                np.array(
                    [1.0001, 1.0002, 9.0001, 9.0002],
                    dtype="<f4",
                ),
                palette_entries=2,
            )

        self.assertEqual(palette.dtype, np.dtype("<f4"))
        self.assertNotEqual(palette[0], palette[1])
        np.testing.assert_array_equal(codes, [0, 0, 1, 1])
        np.testing.assert_allclose(
            palette,
            np.array([1.00015, 9.00015], dtype="<f4"),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            cluster.call_args.args[0],
            np.array([1.0, 9.0], dtype="<f2"),
        )
        self.assertEqual(cluster.call_args.args[0].dtype, np.dtype("<f2"))
        self.assertEqual(cluster.call_args.args[1].dtype, np.dtype(np.intp))
        self.assertEqual(cluster.call_args.args[2], 2)

    def test_weighted_fp32_palette_matches_fp16_assignments(self) -> None:
        values = np.array(
            [0.0, 0.0, 1.0, 1.0, 9.0, 10.0],
            dtype="<f4",
        )
        codes, palette = convert.fit_fp32_palette(values, palette_entries=2)
        fp16_codes, _ = convert.fit_fp16_palette(
            values,
            palette_entries=2,
        )

        self.assertEqual(codes.dtype, np.dtype("u1"))
        self.assertEqual(palette.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(codes, fp16_codes)
        np.testing.assert_array_equal(codes, [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(palette, np.array([0.5, 9.5], dtype="<f4"))

    def test_fp32_palette_randomized_assignments_match_fp16(self) -> None:
        random = np.random.default_rng(0xF032)
        values = random.normal(size=8_192).astype("<f4")
        values += random.uniform(-0.0002, 0.0002, values.size).astype("<f4")

        fp16_codes, _ = convert.fit_fp16_palette(
            values,
            palette_entries=32,
        )
        fp32_codes, fp32_palette = convert.fit_fp32_palette(
            values,
            palette_entries=32,
        )

        np.testing.assert_array_equal(fp32_codes, fp16_codes)
        for code in range(32):
            expected = np.mean(
                values[fp32_codes == code],
                dtype=np.float64,
            )
            self.assertEqual(fp32_palette[code], np.float32(expected))

    def test_fp32_packers_preserve_source_values(self) -> None:
        matrix_source = (
            np.arange(6, dtype=np.float32).reshape(3, 2)
            + np.float32(0.0001)
        )
        packed_matrix, matrix_logical = convert.pack_matrix(
            matrix_source,
            float_dtype="float32",
        )
        self.assertEqual(matrix_logical, (2, 3))
        self.assertEqual(packed_matrix.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(packed_matrix[:, :3], matrix_source.T)
        np.testing.assert_array_equal(packed_matrix[:, 3], 0)

        conv_source = (
            np.arange(4 * 1 * 2 * 2, dtype=np.float32).reshape(4, 1, 2, 2)
            + np.float32(0.0001)
        )
        packed_conv, conv_logical = convert.pack_conv0_kernel(
            conv_source,
            float_dtype="float32",
        )
        self.assertEqual(conv_logical, (2, 2, 4))
        self.assertEqual(packed_conv.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(
            packed_conv[:, :, 0, :],
            conv_source[:, 0, :, :].transpose(1, 2, 0),
        )

        depthwise_source = (
            np.arange(4 * 3, dtype=np.float32).reshape(4, 1, 3)
            + np.float32(0.0001)
        )
        packed_depthwise, depthwise_logical = (
            convert.pack_conformer_depthwise_kernel(
                depthwise_source,
                float_dtype="float32",
            )
        )
        self.assertEqual(depthwise_logical, (3, 4))
        self.assertEqual(packed_depthwise.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(
            packed_depthwise[:, 0, :],
            depthwise_source[:, 0, :].T,
        )

    def test_fp32_concatenation_and_subsampling_reorders_are_exact(self) -> None:
        tensors = {
            "q": np.array([[1.0001, 2.0002], [3.0003, 4.0004]], dtype="<f4"),
            "k": np.array([[5.0005, 6.0006]], dtype="<f4"),
        }
        checkpoint = _FakeCheckpoint(tensors=tensors)
        packed, logical = convert.pack_output_concatenated_matrices(
            checkpoint,
            ["q", "k"],
            set(),
            float_dtype="float32",
        )
        expected = np.concatenate([tensors["q"], tensors["k"]]).T
        self.assertEqual(logical, (2, 3))
        self.assertEqual(packed.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(packed[:, :3], expected)
        np.testing.assert_array_equal(packed[:, 3], 0)

        linear_source = (
            np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2 * 4)
            + np.float32(0.0001)
        )
        linear_packed, linear_logical = (
            convert.pack_subsampling_linear_matrix(
                linear_source,
                channels=2,
                frequencies=4,
                float_dtype="float32",
            )
        )
        self.assertEqual(linear_logical, (8, 3))
        self.assertEqual(linear_packed.dtype, np.dtype("<f4"))
        for frequency in range(4):
            for channel in range(2):
                runtime_k = frequency * 2 + channel
                source_k = channel * 4 + frequency
                np.testing.assert_array_equal(
                    linear_packed[runtime_k, :3],
                    linear_source[:, source_k],
                )

    def test_fp32_relative_projection_and_pack_stay_fp32(self) -> None:
        positions = np.array([[1.0001, 2.0002], [3.0003, 4.0004]], dtype="<f4")
        weight = np.array([[1.0005, 3.0006], [2.0007, 4.0008]], dtype="<f4")
        projected = convert.project_relative_positions(
            positions,
            weight,
            float_dtype="float32",
        )
        expected = positions @ weight.T
        self.assertEqual(projected.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(projected, expected)

        projected_for_pack = np.arange(16, dtype=np.float32).reshape(2, 8)
        projected_for_pack += np.float32(0.0001)
        packed = convert.pack_projected_relative_positions(
            projected_for_pack,
            heads=2,
            float_dtype="float32",
        )
        self.assertEqual(packed.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(
            packed[0, 0, 0],
            projected_for_pack[0, 0:4],
        )
        np.testing.assert_array_equal(
            packed[1, 0, 1],
            projected_for_pack[1, 4:8],
        )

    def test_palette_codes_have_exact_lsb_first_words(self) -> None:
        cases = (
            (
                5,
                32,
                [
                    0x8A418820,
                    0xC5A92839,
                    0xCA307B9A,
                    0x38BDAB49,
                    0xFFBBCDEB,
                ],
            ),
            (6, 16, [0x440C2040, 0xA2481C61, 0x3CE34C2C]),
        )
        for bits, code_count, expected_words in cases:
            with self.subTest(bits=bits):
                codes = np.arange(code_count, dtype=np.uint8)
                packed = convert.pack_palette_codes_lsb(codes, bits=bits)
                self.assertEqual(packed.dtype, np.dtype("<u4"))
                self.assertEqual(packed.shape, (1, len(expected_words)))
                np.testing.assert_array_equal(
                    packed[0],
                    np.array(expected_words, dtype="<u4"),
                )
                self.assertEqual(
                    packed[0, 0].tobytes(),
                    expected_words[0].to_bytes(4, "little"),
                )
                np.testing.assert_array_equal(
                    self._unpack_palette_codes(packed, bits=bits),
                    codes,
                )

    def test_palettized_output_groups_keep_k_major_code_order(self) -> None:
        packed = np.empty((4, 8), dtype="<f2")
        packed[:, :4] = np.arange(16, dtype=np.float16).reshape(4, 4)
        packed[:, 4:] = 100 + np.arange(16, dtype=np.float16).reshape(4, 4)

        words, palettes = convert.palettize_matrix(
            packed,
            bits=5,
            output_group_width=4,
        )

        self.assertEqual(words.shape, (1, 5))
        self.assertEqual(palettes.shape, (2, 32))
        np.testing.assert_array_equal(palettes[0, :16], np.arange(16))
        np.testing.assert_array_equal(palettes[1, :16], 100 + np.arange(16))
        np.testing.assert_array_equal(palettes[:, 16:], 0)
        expected_codes = np.concatenate(
            [
                np.concatenate(
                    [
                        np.arange(row * 4, row * 4 + 4),
                        np.arange(row * 4, row * 4 + 4),
                    ]
                )
                for row in range(4)
            ]
        ).astype(np.uint8)
        np.testing.assert_array_equal(
            self._unpack_palette_codes(words, bits=5),
            expected_codes,
        )

    def test_shard_writer_records_uint32_codes_and_2d_fp16_palette(self) -> None:
        records: dict[str, convert.TensorRecord] = {}
        words = np.array([[0x01234567, 0x89ABCDEF, 0, 1, 2]], dtype="<u4")
        palette = np.arange(32, dtype=np.float16).reshape(1, 32)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with convert.ShardWriter(output_dir, "test.bin", records) as shard:
                shard.add(
                    "matrix",
                    words,
                    dtype="uint32",
                    logical_shape=(4, 8),
                    layout=convert.PALETTE5_MATRIX_LAYOUT,
                )
                shard.add(
                    "matrix.palette",
                    palette,
                    dtype="float16",
                    layout=convert.PALETTE_LAYOUT,
                )

            matrix_record = records["matrix"]
            palette_record = records["matrix.palette"]
            self.assertEqual(matrix_record.storageShape, [1, 5])
            self.assertEqual(matrix_record.byteLength, 20)
            self.assertEqual(palette_record.storageShape, [1, 32])
            self.assertEqual(palette_record.byteLength, 64)
            self.assertEqual(palette_record.byteOffset, convert.ALIGNMENT)
            with (output_dir / "test.bin").open("rb") as stream:
                self.assertEqual(stream.read(4), b"\x67\x45\x23\x01")
                stream.seek(matrix_record.byteLength)
                self.assertEqual(
                    stream.read(palette_record.byteOffset - matrix_record.byteLength),
                    b"\0" * (palette_record.byteOffset - matrix_record.byteLength),
                )

    def test_shard_writer_removes_partial_output_after_failure(self) -> None:
        records: dict[str, convert.TensorRecord] = {}
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "packing failed"):
                with convert.ShardWriter(
                    output_dir,
                    "test.bin",
                    records,
                ) as shard:
                    shard.add(
                        "matrix",
                        np.arange(8, dtype="<f4"),
                        dtype="float32",
                        layout="flat",
                    )
                    raise RuntimeError("packing failed")

            self.assertFalse((output_dir / "test.bin").exists())
            self.assertFalse((output_dir / "test.bin.partial").exists())

    def test_shard_writer_removes_partial_output_when_install_fails(
        self,
    ) -> None:
        records: dict[str, convert.TensorRecord] = {}
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with (
                mock.patch.object(
                    convert.os,
                    "replace",
                    side_effect=OSError("install failed"),
                ),
                self.assertRaisesRegex(OSError, "install failed"),
            ):
                with convert.ShardWriter(
                    output_dir,
                    "test.bin",
                    records,
                ) as shard:
                    shard.add(
                        "matrix",
                        np.arange(8, dtype="<f4"),
                        dtype="float32",
                        layout="flat",
                    )

            self.assertFalse((output_dir / "test.bin").exists())
            self.assertFalse((output_dir / "test.bin.partial").exists())

    def test_shard_writer_closes_before_cleaning_up_flush_failure(
        self,
    ) -> None:
        records: dict[str, convert.TensorRecord] = {}
        stream: object | None = None
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with (
                mock.patch.object(
                    convert.os,
                    "fsync",
                    side_effect=OSError("flush failed"),
                ),
                self.assertRaisesRegex(OSError, "flush failed"),
            ):
                with convert.ShardWriter(
                    output_dir,
                    "test.bin",
                    records,
                ) as shard:
                    stream = shard.stream
                    shard.add(
                        "matrix",
                        np.arange(8, dtype="<f4"),
                        dtype="float32",
                        layout="flat",
                    )

            self.assertIsNotNone(stream)
            self.assertTrue(getattr(stream, "closed"))
            self.assertFalse((output_dir / "test.bin").exists())
            self.assertFalse((output_dir / "test.bin.partial").exists())

    def test_fp32_decoder_uses_fp32_for_every_float_weight(self) -> None:
        tensors: dict[str, np.ndarray] = {
            "decoder.embedding.weight": np.arange(15, dtype="<f4").reshape(5, 3),
            "decoder.decoder_projector.weight": np.arange(
                6,
                dtype="<f4",
            ).reshape(3, 2),
            "decoder.decoder_projector.bias": np.arange(3, dtype="<f4"),
            "encoder_projector.weight": np.arange(
                32,
                dtype="<f4",
            ).reshape(8, 4),
            "encoder_projector.bias": np.arange(8, dtype="<f4"),
            "joint.head.weight": np.arange(12, dtype="<f4").reshape(6, 2),
            "joint.head.bias": np.arange(6, dtype="<f4"),
        }
        for layer_index in range(2):
            tensors[f"decoder.lstm.weight_ih_l{layer_index}"] = np.arange(
                8,
                dtype="<f4",
            ).reshape(4, 2)
            tensors[f"decoder.lstm.weight_hh_l{layer_index}"] = np.arange(
                8,
                16,
                dtype="<f4",
            ).reshape(4, 2)
            tensors[f"decoder.lstm.bias_ih_l{layer_index}"] = np.arange(
                4,
                dtype="<f4",
            )
            tensors[f"decoder.lstm.bias_hh_l{layer_index}"] = np.arange(
                4,
                8,
                dtype="<f4",
            )

        checkpoint = _FakeCheckpoint(tensors=tensors)
        records: dict[str, convert.TensorRecord] = {}
        with tempfile.TemporaryDirectory() as temporary:
            convert.pack_decoder(
                checkpoint,
                Path(temporary),
                records,
                set(),
                float_dtype="float32",
            )

        float_records = [
            record for record in records.values() if record.dtype != "uint32"
        ]
        self.assertTrue(float_records)
        self.assertTrue(
            all(record.dtype == "float32" for record in float_records)
        )
        palette = records["encoder_projector.weight.palette"]
        self.assertEqual(palette.dtype, "float32")
        self.assertEqual(palette.storageShape, [1, 32])
        self.assertEqual(palette.byteLength, 32 * np.dtype("<f4").itemsize)
        self.assertEqual(set(checkpoint.loads), set(tensors))

    def test_preprocessor_packs_embedded_source_values_exactly(self) -> None:
        window = np.arange(400, dtype="<f4")
        mel = np.arange(128 * 257, dtype="<f4").reshape(1, 128, 257)
        checkpoint = _FakeCheckpoint(
            tensors={
                "preprocessor.window": window,
                "preprocessor.mel_filters": mel,
            }
        )
        records: dict[str, convert.TensorRecord] = {}
        consumed: set[str] = set()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            convert.pack_preprocessor(
                checkpoint,
                output_dir,
                records,
                consumed,
            )
            payload = (output_dir / "preprocessor.bin").read_bytes()

        window_record = records["preprocessor.window"]
        mel_record = records["preprocessor.mel_filters"]
        np.testing.assert_array_equal(
            np.frombuffer(
                payload,
                dtype="<f4",
                count=400,
                offset=window_record.byteOffset,
            ),
            window,
        )
        np.testing.assert_array_equal(
            np.frombuffer(
                payload,
                dtype="<f4",
                count=128 * 257,
                offset=mel_record.byteOffset,
            ).reshape(128, 257),
            mel[0],
        )
        self.assertEqual(
            consumed,
            {"preprocessor.window", "preprocessor.mel_filters"},
        )

    def test_decode_tokenizer_has_exact_v2_blank_contract(self) -> None:
        lines = [
            "<unk>\t0",
            *(f"piece-{index}\t-{index}" for index in range(1, 1024)),
        ]
        tokenizer = convert.build_decode_tokenizer_json(
            ("\n".join(lines) + "\n").encode()
        )
        self.assertEqual(tokenizer["model"]["type"], "BPE")
        self.assertFalse(tokenizer["model"]["byte_fallback"])
        self.assertEqual(len(tokenizer["model"]["vocab"]), 1024)
        self.assertEqual(
            tokenizer["added_tokens"],
            [
                {"id": 0, "content": "<unk>", "special": True},
                {"id": 1024, "content": "<blank>", "special": True},
            ],
        )
        self.assertNotIn("<pad>", tokenizer["model"]["vocab"])
        self.assertEqual(tokenizer["decoder"]["replacement"], "▁")

    def test_metadata_files_copy_v2_assets_and_build_decode_tokenizer(self) -> None:
        vocabulary = (
            "\n".join(
                [
                    "<unk>\t0",
                    *(f"piece-{index}\t-{index}" for index in range(1, 1024)),
                ]
            )
            + "\n"
        ).encode()
        assets = {
            "model_config.yaml": b"sample_rate: 16000\n",
            "tokenizer.model": b"embedded sentencepiece bytes",
            "tokenizer.vocab": vocabulary,
            "vocab.txt": b"<unk>\npiece-1\n",
        }
        checkpoint = _FakeCheckpoint(assets=assets)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            written = convert.write_metadata_files(checkpoint, output_dir)

            self.assertEqual(
                {path.name for path in written},
                {*assets, "tokenizer.json"},
            )
            for name, expected in assets.items():
                self.assertEqual((output_dir / name).read_bytes(), expected)
            tokenizer_payload = (output_dir / "tokenizer.json").read_bytes()
            self.assertFalse(tokenizer_payload.endswith(b"\n"))
            tokenizer = convert.build_decode_tokenizer_json(vocabulary)
            self.assertEqual(
                tokenizer_payload,
                json.dumps(
                    tokenizer,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            )

    def test_relative_positions_match_checkpoint_formula(self) -> None:
        table = convert.build_relative_positions(encoder_frames=3, channels=4)
        positions = np.array([2, 1, 0, -1, -2], dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(
            np.arange(0, 4, 2, dtype=np.float32)
            * np.float32(-(np.log(10_000.0) / 4))
        )
        expected = np.empty((5, 4), dtype=np.float32)
        expected[:, 0::2] = np.sin(positions * div_term)
        expected[:, 1::2] = np.cos(positions * div_term)

        self.assertEqual(table.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(table, expected)
        full_table = convert.build_relative_positions()
        self.assertEqual(full_table.shape, (375, 1024))
        np.testing.assert_array_equal(full_table[187, 0::2], 0)
        np.testing.assert_array_equal(full_table[187, 1::2], 1)

    def test_relative_projection_uses_weight_transpose(self) -> None:
        positions = np.array([[1, 2], [3, 4]], dtype=np.float32)
        weight = np.array([[1, 3], [2, 4]], dtype=np.float32)
        projected = convert.project_relative_positions(positions, weight)

        self.assertEqual(projected.dtype, np.dtype("<f2"))
        np.testing.assert_array_equal(
            projected,
            np.array([[7, 10], [15, 22]], dtype=np.float16),
        )

    def test_projected_relative_positions_pack_rows_inside_head_vectors(self) -> None:
        projected = np.arange(16, dtype=np.float16).reshape(2, 8)
        packed = convert.pack_projected_relative_positions(
            projected,
            heads=2,
        )

        self.assertEqual(packed.dtype, np.dtype("<f2"))
        self.assertEqual(packed.shape, (2, 1, 2, 4))
        np.testing.assert_array_equal(packed[0, 0, 0], projected[0, 0:4])
        np.testing.assert_array_equal(packed[0, 0, 1], projected[1, 0:4])
        np.testing.assert_array_equal(packed[1, 0, 0], projected[0, 4:8])
        np.testing.assert_array_equal(packed[1, 0, 1], projected[1, 4:8])

    def test_projected_relative_position_pack_rejects_invalid_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "position, channel"):
            convert.pack_projected_relative_positions(
                np.zeros((8,), dtype=np.float16)
            )
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            convert.pack_projected_relative_positions(
                np.zeros((3, 10), dtype=np.float16),
                heads=4,
            )


class CheckpointSchemaTests(unittest.TestCase):
    def test_schema_exactly_covers_pinned_official_header(self) -> None:
        specs = convert.official_source_tensor_specs()
        ignored = {name for name, spec in specs.items() if spec.ignored}

        self.assertEqual(len(specs), 725)
        self.assertEqual(len(ignored), 24)
        self.assertEqual(
            specs["encoder.subsampling.layers.0.weight"].shape,
            (256, 1, 3, 3),
        )
        self.assertEqual(
            specs["encoder.layers.23.conv.depthwise_conv.weight"].shape,
            (1024, 1, 9),
        )
        self.assertEqual(specs["preprocessor.window"].shape, (400,))
        self.assertEqual(
            specs["preprocessor.mel_filters"].shape,
            (1, 128, 257),
        )
        self.assertEqual(specs["decoder.embedding.weight"].shape, (1025, 640))
        self.assertEqual(specs["joint.head.weight"].shape, (1030, 640))
        self.assertEqual(
            ignored,
            {
                f"encoder.layers.{layer}.conv.norm.num_batches_tracked"
                for layer in range(24)
            },
        )
        checkpoint = _FakeCheckpoint(
            shapes={name: spec.shape for name, spec in specs.items()}
        )
        convert.validate_checkpoint_schema(checkpoint, specs)
        self.assertEqual(checkpoint.loads, [])

    def test_schema_rejects_wrong_shape_and_unexpected_name(self) -> None:
        specs = convert.official_source_tensor_specs()
        shapes = {name: spec.shape for name, spec in specs.items()}
        shapes["joint.head.weight"] = (1031, 640)
        shapes["not.an.official.tensor"] = (1,)
        checkpoint = _FakeCheckpoint(shapes=shapes)

        with self.assertRaisesRegex(RuntimeError, "unexpected names"):
            convert.validate_checkpoint_schema(checkpoint, specs)
        with self.assertRaisesRegex(RuntimeError, "expected shape"):
            convert.validate_checkpoint_schema(checkpoint, specs)


class PackageVariantTests(unittest.TestCase):
    def test_default_cli_remains_fp16(self) -> None:
        with mock.patch.object(sys, "argv", ["convert.py"]):
            arguments = convert.parse_args()

        self.assertEqual(arguments.precision, "fp16")
        self.assertIsNone(arguments.output_dir)
        self.assertEqual(arguments.jobs, 0)

    def test_canonical_output_directories_are_distinct(self) -> None:
        model_dir = Path("/model")
        fp16 = convert.package_output_directories(
            "fp16",
            None,
            model_dir=model_dir,
        )
        fp32 = convert.package_output_directories(
            "fp32",
            None,
            model_dir=model_dir,
        )
        both = convert.package_output_directories(
            "both",
            None,
            model_dir=model_dir,
        )

        self.assertEqual(fp16, ((convert.FP16_PACKAGE_VARIANT, model_dir / "files"),))
        self.assertEqual(
            fp32,
            ((convert.FP32_PACKAGE_VARIANT, model_dir / "files-fp32"),),
        )
        self.assertEqual(both, (*fp16, *fp32))
        with self.assertRaisesRegex(ValueError, "--output-dir"):
            convert.package_output_directories(
                "both",
                model_dir / "custom",
                model_dir=model_dir,
            )

    def test_fp32_manifest_identity_and_precision_propagation(self) -> None:
        checkpoint = _FakeCheckpoint(shapes={})
        captured_manifest: dict[str, object] = {}

        def capture_manifest(
            path: Path,
            value: dict[str, object],
            trailing_newline: bool,
        ) -> None:
            self.assertEqual(path.name, "manifest.json")
            self.assertTrue(trailing_newline)
            captured_manifest.update(value)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / convert.SOURCE_ARCHIVE_FILENAME
            archive_path.write_bytes(b"x")
            with (
                mock.patch.object(convert, "ENCODER_LAYERS", 0),
                mock.patch.object(convert, "pack_preprocessor"),
                mock.patch.object(convert, "pack_subsampling") as subsampling,
                mock.patch.object(convert, "pack_decoder") as decoder,
                mock.patch.object(convert, "write_metadata_files", return_value=[]),
                mock.patch.object(convert, "write_json_atomic", capture_manifest),
                mock.patch.object(
                    convert,
                    "validate_staged_package_identity",
                ) as validate_identity,
                mock.patch.object(convert, "configure_native_kmeans_library"),
                mock.patch("builtins.print"),
            ):
                convert.convert_package(
                    checkpoint,
                    archive_path,
                    root / "files-fp32",
                    {},
                    convert.FP32_PACKAGE_VARIANT,
                    native_kmeans_library_path=root / "native.dylib",
                )

            validate_identity.assert_called_once_with(
                root / "files-fp32",
                convert.FP32_PACKAGE_VARIANT,
            )

        self.assertEqual(captured_manifest["format"], convert.FP32_FORMAT_VERSION)
        self.assertEqual(
            captured_manifest["source"],
            {
                "repository": "nvidia/parakeet-tdt-0.6b-v2",
                "revision": convert.SOURCE_REVISION,
                "archive": {
                    "name": convert.SOURCE_ARCHIVE_FILENAME,
                    "byteLength": convert.SOURCE_ARCHIVE_BYTES,
                    "sha256": convert.SOURCE_ARCHIVE_SHA256,
                },
                "weights": {
                    "name": "model_weights.ckpt",
                    "byteLength": convert.SOURCE_WEIGHT_BYTES,
                    "sha256": convert.SOURCE_WEIGHT_SHA256,
                },
                "license": "CC-BY-4.0",
            },
        )
        config = captured_manifest["config"]
        self.assertIsInstance(config, dict)
        self.assertEqual(
            config["encoderWeightFormat"],  # type: ignore[index]
            convert.FP32_ENCODER_WEIGHT_FORMAT,
        )
        self.assertEqual(config["vocabularySize"], 1025)  # type: ignore[index]
        self.assertEqual(config["blankTokenId"], 1024)  # type: ignore[index]
        self.assertEqual(config["jointOutputSize"], 1030)  # type: ignore[index]
        self.assertEqual(config["jointStorageOutputSize"], 1032)  # type: ignore[index]
        self.assertEqual(
            subsampling.call_args.kwargs["float_dtype"],
            "float32",
        )
        self.assertEqual(decoder.call_args.kwargs["float_dtype"], "float32")

    def test_float_dtype_validation_is_explicit(self) -> None:
        self.assertEqual(convert.numpy_float_dtype("float16"), np.dtype("<f2"))
        self.assertEqual(convert.numpy_float_dtype("float32"), np.dtype("<f4"))
        with self.assertRaisesRegex(ValueError, "Unsupported package float dtype"):
            convert.numpy_float_dtype("float64")

    def test_fp32_worker_auto_selection_is_memory_bounded(self) -> None:
        self.assertEqual(
            convert.resolve_conversion_jobs(
                0,
                convert.FP16_PACKAGE_VARIANT,
                cpu_count=8,
                memory_bytes=16 * 1024**3,
            ),
            1,
        )
        self.assertEqual(
            convert.resolve_conversion_jobs(
                0,
                convert.FP32_PACKAGE_VARIANT,
                cpu_count=8,
                memory_bytes=16 * 1024**3,
            ),
            8,
        )
        self.assertEqual(
            convert.resolve_conversion_jobs(
                0,
                convert.FP32_PACKAGE_VARIANT,
                cpu_count=8,
                memory_bytes=8 * 1024**3,
            ),
            2,
        )
        self.assertEqual(
            convert.resolve_conversion_jobs(
                8,
                convert.FP32_PACKAGE_VARIANT,
                cpu_count=1,
                memory_bytes=16 * 1024**3,
            ),
            1,
        )
        with self.assertRaisesRegex(ValueError, "--jobs"):
            convert.resolve_conversion_jobs(
                9,
                convert.FP32_PACKAGE_VARIANT,
            )

    def test_native_cache_name_and_compile_flags_are_deterministic(self) -> None:
        cache_dir = Path("/model/cache")
        compiler_identity = "/usr/bin/clang++\nclang version 20.0.0"
        first = convert.native_kmeans_cache_path(
            cache_dir,
            compiler_identity,
        )
        second = convert.native_kmeans_cache_path(
            cache_dir,
            compiler_identity,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.parent, cache_dir / "native")
        self.assertIn(
            convert.native_kmeans_build_sha256(compiler_identity)[:16],
            first.name,
        )
        self.assertIn(convert.native_kmeans_platform_tag(), first.name)
        self.assertNotEqual(
            first,
            convert.native_kmeans_cache_path(
                cache_dir,
                "/usr/bin/clang++\nclang version 21.0.0",
            ),
        )
        with mock.patch.object(
            convert,
            "NATIVE_KMEANS_BUILD_RECIPE_VERSION",
            convert.NATIVE_KMEANS_BUILD_RECIPE_VERSION + 1,
        ):
            self.assertNotEqual(
                first,
                convert.native_kmeans_cache_path(
                    cache_dir,
                    compiler_identity,
                ),
            )
        command = convert.native_kmeans_compile_command(
            "/usr/bin/clang++",
            Path("/tmp/exact-kmeans.dylib"),
        )
        self.assertEqual(command[0], "/usr/bin/clang++")
        self.assertEqual(
            command[1:5],
            ("-O3", "-DNDEBUG", "-std=c++17", "-fvisibility=hidden"),
        )
        self.assertNotIn("-ffast-math", command)
        self.assertEqual(command[-3:], (
            str(convert.NATIVE_KMEANS_SOURCE),
            "-o",
            "/tmp/exact-kmeans.dylib",
        ))

    def test_native_compiler_identity_uses_resolved_path_and_version(
        self,
    ) -> None:
        with mock.patch.object(
            convert.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="  ",
                stderr="clang version 20.0.0\nTarget: arm64-apple-darwin",
            ),
        ):
            identity = convert.native_kmeans_compiler_identity(
                "/toolchain/clang++"
            )

        self.assertTrue(
            identity.startswith(
                f"{Path('/toolchain/clang++').resolve()}\n"
            )
        )
        self.assertIn("clang version 20.0.0", identity)
        self.assertIn("Target: arm64-apple-darwin", identity)

    def test_package_conversion_replaces_the_directory_only_after_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "files"
            output_dir.mkdir()
            output_dir.chmod(0o750)
            (output_dir / "manifest.json").write_text("old", encoding="utf-8")
            (output_dir / "stale.bin").write_bytes(b"stale")

            def write_staged_package(
                _checkpoint: object,
                _archive_path: Path,
                staging_dir: Path,
                _source_specs: dict[str, convert.SourceTensorSpec],
                _variant: convert.PackageVariant,
                **_kwargs: object,
            ) -> None:
                (staging_dir / "manifest.json").write_text(
                    "new",
                    encoding="utf-8",
                )
                (staging_dir / "new.bin").write_bytes(b"new")

            with mock.patch.object(
                convert,
                "convert_package",
                side_effect=write_staged_package,
            ), mock.patch.object(convert, "validate_source") as validate_source:
                convert.convert_package_transactionally(
                    object(),
                    root / convert.SOURCE_ARCHIVE_FILENAME,
                    output_dir,
                    {},
                    convert.FP16_PACKAGE_VARIANT,
                )

            validate_source.assert_called_once_with(
                root / convert.SOURCE_ARCHIVE_FILENAME
            )

            self.assertEqual(
                (output_dir / "manifest.json").read_text(encoding="utf-8"),
                "new",
            )
            self.assertEqual((output_dir / "new.bin").read_bytes(), b"new")
            self.assertFalse((output_dir / "stale.bin").exists())
            self.assertEqual(output_dir.stat().st_mode & 0o777, 0o750)
            self.assertEqual(
                [path for path in root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_staged_package_identity_rejects_noncanonical_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "manifest.json").write_bytes(b"not canonical")
            with self.assertRaisesRegex(
                RuntimeError,
                "Converted fp16 manifest SHA-256 mismatch",
            ):
                convert.validate_staged_package_identity(
                    output_dir,
                    convert.FP16_PACKAGE_VARIANT,
                )

    def test_staged_package_identity_accepts_pinned_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "manifest.json").write_bytes(b"manifest")
            with mock.patch.object(
                convert,
                "sha256_file",
                return_value=convert.FP32_MANIFEST_SHA256,
            ):
                convert.validate_staged_package_identity(
                    output_dir,
                    convert.FP32_PACKAGE_VARIANT,
                )

    def test_failed_package_conversion_preserves_the_previous_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "files"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("old", encoding="utf-8")

            def fail_conversion(
                _checkpoint: object,
                _archive_path: Path,
                staging_dir: Path,
                _source_specs: dict[str, convert.SourceTensorSpec],
                _variant: convert.PackageVariant,
                **_kwargs: object,
            ) -> None:
                (staging_dir / "partial.bin").write_bytes(b"partial")
                raise RuntimeError("conversion failed")

            with (
                mock.patch.object(
                    convert,
                    "convert_package",
                    side_effect=fail_conversion,
                ),
                self.assertRaisesRegex(RuntimeError, "conversion failed"),
            ):
                convert.convert_package_transactionally(
                    object(),
                    root / convert.SOURCE_ARCHIVE_FILENAME,
                    output_dir,
                    {},
                    convert.FP16_PACKAGE_VARIANT,
                )

            self.assertEqual(
                (output_dir / "manifest.json").read_text(encoding="utf-8"),
                "old",
            )
            self.assertEqual(
                [path for path in root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_changed_source_after_conversion_preserves_previous_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / convert.SOURCE_ARCHIVE_FILENAME
            output_dir = root / "files"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("old", encoding="utf-8")

            def write_staged_package(
                _checkpoint: object,
                _archive_path: Path,
                staging_dir: Path,
                _source_specs: dict[str, convert.SourceTensorSpec],
                _variant: convert.PackageVariant,
                **_kwargs: object,
            ) -> None:
                (staging_dir / "manifest.json").write_text(
                    "new",
                    encoding="utf-8",
                )

            with (
                mock.patch.object(
                    convert,
                    "convert_package",
                    side_effect=write_staged_package,
                ),
                mock.patch.object(
                    convert,
                    "validate_source",
                    side_effect=RuntimeError("source changed"),
                ),
                self.assertRaisesRegex(RuntimeError, "source changed"),
            ):
                convert.convert_package_transactionally(
                    object(),
                    archive_path,
                    output_dir,
                    {},
                    convert.FP16_PACKAGE_VARIANT,
                )

            self.assertEqual(
                (output_dir / "manifest.json").read_text(encoding="utf-8"),
                "old",
            )
            self.assertEqual(
                [path for path in root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_package_conversion_refuses_unowned_nonempty_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "files"
            output_dir.mkdir()
            sentinel = output_dir / "unrelated.txt"
            sentinel.write_text("keep", encoding="utf-8")

            def write_staged_package(
                _checkpoint: object,
                _archive_path: Path,
                staging_dir: Path,
                _source_specs: dict[str, convert.SourceTensorSpec],
                _variant: convert.PackageVariant,
                **_kwargs: object,
            ) -> None:
                (staging_dir / "manifest.json").write_text(
                    "new",
                    encoding="utf-8",
                )

            with (
                mock.patch.object(
                    convert,
                    "convert_package",
                    side_effect=write_staged_package,
                ) as conversion,
                self.assertRaisesRegex(
                    RuntimeError,
                    "non-empty directory without a package manifest",
                ),
            ):
                convert.convert_package_transactionally(
                    object(),
                    root / convert.SOURCE_ARCHIVE_FILENAME,
                    output_dir,
                    {},
                    convert.FP16_PACKAGE_VARIANT,
                )

            conversion.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(
                [path for path in root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_failed_package_install_restores_the_previous_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "files"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("old", encoding="utf-8")
            staging_directories: list[Path] = []

            def write_staged_package(
                _checkpoint: object,
                _archive_path: Path,
                staging_dir: Path,
                _source_specs: dict[str, convert.SourceTensorSpec],
                _variant: convert.PackageVariant,
                **_kwargs: object,
            ) -> None:
                staging_directories.append(staging_dir)
                (staging_dir / "manifest.json").write_text(
                    "new",
                    encoding="utf-8",
                )

            real_replace = convert.os.replace

            def fail_staged_install(source: object, destination: object) -> None:
                if (
                    staging_directories
                    and Path(source) == staging_directories[0]
                    and Path(destination) == output_dir
                ):
                    raise OSError("installation failed")
                real_replace(source, destination)

            with (
                mock.patch.object(
                    convert,
                    "convert_package",
                    side_effect=write_staged_package,
                ),
                mock.patch.object(convert, "validate_source"),
                mock.patch.object(
                    convert.os,
                    "replace",
                    side_effect=fail_staged_install,
                ),
                self.assertRaisesRegex(OSError, "installation failed"),
            ):
                convert.convert_package_transactionally(
                    object(),
                    root / convert.SOURCE_ARCHIVE_FILENAME,
                    output_dir,
                    {},
                    convert.FP16_PACKAGE_VARIANT,
                )

            self.assertEqual(
                (output_dir / "manifest.json").read_text(encoding="utf-8"),
                "old",
            )
            self.assertEqual(
                [path for path in root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_backup_cleanup_failure_reports_installed_package_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "files"
            output_dir.mkdir()
            (output_dir / "manifest.json").write_text("old", encoding="utf-8")

            def write_staged_package(
                _checkpoint: object,
                _archive_path: Path,
                staging_dir: Path,
                _source_specs: dict[str, convert.SourceTensorSpec],
                _variant: convert.PackageVariant,
                **_kwargs: object,
            ) -> None:
                (staging_dir / "manifest.json").write_text(
                    "new",
                    encoding="utf-8",
                )

            backup_dir = output_dir.with_name(
                f".{output_dir.name}.{convert.os.getpid()}.backup"
            )
            real_rmtree = convert.shutil.rmtree

            def fail_backup_cleanup(path: object) -> None:
                if Path(path) == backup_dir:
                    raise OSError("backup cleanup failed")
                real_rmtree(path)

            with (
                mock.patch.object(
                    convert,
                    "convert_package",
                    side_effect=write_staged_package,
                ),
                mock.patch.object(convert, "validate_source"),
                mock.patch.object(
                    convert.shutil,
                    "rmtree",
                    side_effect=fail_backup_cleanup,
                ),
                self.assertRaisesRegex(
                    OSError,
                    "backup cleanup failed",
                ) as raised,
            ):
                convert.convert_package_transactionally(
                    object(),
                    root / convert.SOURCE_ARCHIVE_FILENAME,
                    output_dir,
                    {},
                    convert.FP16_PACKAGE_VARIANT,
                )

            self.assertEqual(
                (output_dir / "manifest.json").read_text(encoding="utf-8"),
                "new",
            )
            self.assertTrue(backup_dir.is_dir())
            self.assertIn(
                "new package remains installed",
                "\n".join(raised.exception.__notes__),
            )
            real_rmtree(backup_dir)

    def test_spawned_fp32_layer_receives_and_loads_native_library(self) -> None:
        library_path = Path("/model/cache/native/exact.dylib")
        checkpoint = _FakeCheckpoint()
        checkpoint_context = mock.MagicMock()
        checkpoint_context.__enter__.return_value = checkpoint
        with (
            mock.patch.object(
                convert,
                "configure_native_kmeans_library",
            ) as configure,
            mock.patch.object(
                convert,
                "NemoCheckpoint",
                return_value=checkpoint_context,
            ) as open_checkpoint,
            mock.patch.object(convert, "pack_encoder_layer") as pack_layer,
            mock.patch.object(
                convert,
                "build_relative_positions",
                return_value=np.empty((0,), dtype=np.float32),
            ),
        ):
            result = convert._pack_encoder_layer_task(
                (
                    "/model/cache/official/parakeet-tdt-0.6b-v2.nemo",
                    7,
                    "/model/files-fp32",
                    "float32",
                    str(library_path),
                )
            )

        configure.assert_called_once_with(library_path)
        open_checkpoint.assert_called_once_with(
            Path("/model/cache/official/parakeet-tdt-0.6b-v2.nemo"),
            validate_hashes=False,
        )
        self.assertEqual(pack_layer.call_args.args[1], 7)
        self.assertEqual(pack_layer.call_args.kwargs["float_dtype"], "float32")
        self.assertEqual(result, (7, {}, set()))

    def test_parallel_encoder_metadata_is_order_independent(self) -> None:
        def layer_result(
            layer_index: int,
            tensor_names: tuple[str, ...],
        ) -> tuple[int, dict[str, convert.TensorRecord], set[str]]:
            shard = f"encoder-layer-{layer_index:02d}.bin"
            records = {
                name: convert.TensorRecord(
                    shard=shard,
                    byteOffset=offset * convert.ALIGNMENT,
                    byteLength=4,
                    dtype="float32",
                    logicalShape=[1],
                    storageShape=[1],
                    layout="channel",
                )
                for offset, name in enumerate(tensor_names)
            }
            return layer_index, records, set(tensor_names)

        results = (
            layer_result(
                0,
                (
                    "encoder.layers.0.z",
                    "encoder.layers.0.a",
                ),
            ),
            layer_result(
                1,
                (
                    "encoder.layers.1.y",
                    "encoder.layers.1.b",
                ),
            ),
        )

        def normalized_merge(
            merge_order: tuple[
                tuple[int, dict[str, convert.TensorRecord], set[str]],
                ...,
            ],
        ) -> tuple[
            tuple[tuple[str, convert.TensorRecord], ...],
            tuple[str, ...],
            tuple[str, ...],
        ]:
            records: dict[str, convert.TensorRecord] = {}
            consumed: set[str] = set()
            for result in merge_order:
                convert._merge_encoder_layer_result(
                    result,
                    records,
                    consumed,
                )
            return (
                tuple(sorted(records.items())),
                tuple(sorted({record.shard for record in records.values()})),
                tuple(sorted(consumed)),
            )

        forward = normalized_merge(results)
        reverse = normalized_merge(tuple(reversed(results)))
        self.assertEqual(forward, reverse)
        self.assertEqual(
            tuple(name for name, _ in forward[0]),
            (
                "encoder.layers.0.a",
                "encoder.layers.0.z",
                "encoder.layers.1.b",
                "encoder.layers.1.y",
            ),
        )


if __name__ == "__main__":
    unittest.main()
