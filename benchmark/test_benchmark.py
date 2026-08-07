from __future__ import annotations

import json

import pytest

from benchmark import (
    OPEN_ASR_SCORE_KEY,
    STANDARD_SCORE_KEY,
    BenchmarkError,
    append_jsonl,
    initialize_run_manifest,
    load_jsonl,
    split_summary,
)
from dataset import TEST_CLEAN
from scoring import CorpusScore


def test_jsonl_append_and_load_round_trip(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    records = [{"id": "first", "text": "hello"}, {"id": "second", "text": ""}]
    for record in records:
        append_jsonl(path, record)

    assert load_jsonl(path) == records
    assert path.read_bytes().endswith(b"\n")


def test_load_jsonl_recovers_only_an_incomplete_trailing_record(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    committed = {"id": "committed", "text": "hello"}
    encoded = json.dumps(committed, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(encoded + b'{"id":"interrupted"')

    assert load_jsonl(path) == [committed]
    assert path.read_bytes() == encoded


def test_load_jsonl_rejects_a_corrupt_committed_record(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"id":}\n')

    with pytest.raises(BenchmarkError, match="invalid JSON"):
        load_jsonl(path)


def test_manifest_refuses_orphaned_result_files(tmp_path) -> None:
    (tmp_path / "test-clean.jsonl").write_text("{}\n")

    with pytest.raises(BenchmarkError, match=r"result files exist without run\.json"):
        initialize_run_manifest(tmp_path, {"contract": "new"})


def test_manifest_round_trip_requires_the_exact_contract(tmp_path) -> None:
    contract = {"contract": "original"}
    manifest = initialize_run_manifest(tmp_path, contract)
    assert initialize_run_manifest(tmp_path, contract) == manifest

    with pytest.raises(BenchmarkError, match="benchmark provenance changed"):
        initialize_run_manifest(tmp_path, {"contract": "changed"})


def test_split_summary_keeps_standard_and_compound_scores_distinct() -> None:
    standard = CorpusScore(2, 3, 2, 1, 100)
    compound_aware = CorpusScore(2, 1, 1, 1, 100)
    summary = split_summary(
        TEST_CLEAN,
        {
            STANDARD_SCORE_KEY: standard,
            OPEN_ASR_SCORE_KEY: compound_aware,
        },
        complete=False,
    )

    assert summary[STANDARD_SCORE_KEY]["errors"] == 6
    assert summary[STANDARD_SCORE_KEY]["werPercent"] == 6.0
    assert summary[OPEN_ASR_SCORE_KEY]["errors"] == 3
    assert summary[OPEN_ASR_SCORE_KEY]["werPercent"] == 3.0
