"""Pinned Open ASR normalization and independently audited corpus WER."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import jiwer
from kaldialign import batch_error_rate, edit_distance

from normalizer.normalizer import EnglishTextNormalizer

OPEN_ASR_REPOSITORY = "https://github.com/huggingface/open_asr_leaderboard"
OPEN_ASR_COMMIT = "9585fc39bff55697a2ec1c5f13921b18812bfde8"
KALDIALIGN_VERSION = "0.12.0"
KALDIALIGN_COMMIT = "06ac40f03c3d368932adf8536965a088d54189b1"
JIWER_VERSION = "3.1.0"
JIWER_COMMIT = "f2f13e0231d2de54dbd7abeea951de676159a715"

_NORMALIZER_FILES = {
    "normalizer.py": "5cfcaab0e772ac331e9904495ad7c1e07e3eace072f7399d68369303d6927cc4",
    "english_abbreviations.py": (
        "d3b2c326bdb700b954b2c7125cd0c0b502deb7e22dbb57f707f914646098eaab"
    ),
}
_NORMALIZER = EnglishTextNormalizer()


@dataclass(frozen=True, slots=True)
class UtteranceScore:
    normalized_reference: str
    normalized_hypothesis: str
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True, slots=True)
class CorpusScore:
    utterances: int
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            if self.errors == 0:
                return 0.0
            return float("inf")
        return self.errors / self.reference_words

    @property
    def wer_percent(self) -> float:
        return 100.0 * self.wer


def verify_vendored_normalizer() -> None:
    """Fail closed if either pinned upstream normalizer file changes."""
    normalizer_directory = Path(__file__).with_name("normalizer")
    for filename, expected in _NORMALIZER_FILES.items():
        actual = sha256((normalizer_directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"vendored Open ASR {filename} SHA-256 is {actual}; expected "
                f"{expected}",
            )


def verify_kaldialign_version() -> None:
    """Fail closed if the imported scorer is not the pinned release."""
    actual = importlib.metadata.version("kaldialign")
    if actual != KALDIALIGN_VERSION:
        raise RuntimeError(
            f"installed Kaldialign version is {actual}; expected {KALDIALIGN_VERSION}",
        )


def verify_jiwer_version() -> None:
    """Fail closed if the standard WER scorer is not the pinned release."""
    actual = importlib.metadata.version("jiwer")
    if actual != JIWER_VERSION:
        raise RuntimeError(
            f"installed JiWER version is {actual}; expected {JIWER_VERSION}",
        )


def normalize_text(text: str) -> str:
    """Apply the pinned Open ASR English normalizer and canonicalize spacing."""
    if not isinstance(text, str):
        raise TypeError("transcripts must be strings")
    return " ".join(_NORMALIZER(text).split())


def score_utterance(
    reference: str,
    hypothesis: str,
    *,
    merge_compounds: bool,
) -> UtteranceScore:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_words = tuple(normalized_reference.split())
    hypothesis_words = tuple(normalized_hypothesis.split())
    if merge_compounds:
        result = edit_distance(
            reference_words,
            hypothesis_words,
            merge_compounds=True,
        )
        _verify_kaldialign_result(
            result,
            reference_words,
            hypothesis_words,
            merge_compounds=True,
        )
        substitutions = int(result["sub"])
        deletions = int(result["del"])
        insertions = int(result["ins"])
    else:
        result = jiwer.process_words(
            [normalized_reference],
            [normalized_hypothesis],
        )
        _verify_standard_result(result, reference_words, hypothesis_words)
        substitutions = result.substitutions
        deletions = result.deletions
        insertions = result.insertions
    return UtteranceScore(
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(reference_words),
    )


def score_corpus(
    records: Iterable[Mapping[str, str]],
    *,
    merge_compounds: bool,
) -> CorpusScore:
    normalized = [
        (
            tuple(normalize_text(record["reference"]).split()),
            tuple(normalize_text(record["hypothesis"]).split()),
        )
        for record in records
    ]
    references = [reference for reference, _ in normalized]
    hypotheses = [hypothesis for _, hypothesis in normalized]
    independent_errors = sum(
        word_edit_distance(
            reference,
            hypothesis,
            merge_compounds=merge_compounds,
        )
        for reference, hypothesis in normalized
    )
    reference_words = sum(len(reference) for reference in references)
    kaldialign_result = batch_error_rate(
        references,
        hypotheses,
        merge_compounds=merge_compounds,
    )
    _verify_kaldialign_corpus_result(
        kaldialign_result,
        independent_errors,
        reference_words,
        merge_compounds=merge_compounds,
    )
    if merge_compounds:
        substitutions = int(kaldialign_result["sub"])
        deletions = int(kaldialign_result["del"])
        insertions = int(kaldialign_result["ins"])
    else:
        result = jiwer.process_words(
            [" ".join(reference) for reference in references],
            [" ".join(hypothesis) for hypothesis in hypotheses],
        )
        reported_errors = result.substitutions + result.deletions + result.insertions
        if reported_errors != independent_errors:
            raise AssertionError(
                "JiWER and the independent word edit-distance totals disagree: "
                f"{reported_errors} != {independent_errors}",
            )
        jiwer_reference_words = result.hits + result.substitutions + result.deletions
        if jiwer_reference_words != reference_words:
            raise AssertionError(
                f"JiWER reference length is {jiwer_reference_words}; expected "
                f"{reference_words}",
            )
        substitutions = result.substitutions
        deletions = result.deletions
        insertions = result.insertions
    return CorpusScore(
        utterances=len(normalized),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=reference_words,
    )


def word_edit_distance(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    *,
    merge_compounds: bool,
) -> int:
    """Independent word DP, optionally with Kaldialign compound matches."""
    rows = len(reference)
    columns = len(hypothesis)
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        distance[row][0] = row
    for column in range(columns + 1):
        distance[0][column] = column

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            best = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1]
                + (reference[row - 1] != hypothesis[column - 1]),
            )

            if merge_compounds:
                merged_reference = reference[row - 1]
                for count in range(2, row + 1):
                    merged_reference = reference[row - count] + merged_reference
                    if len(merged_reference) > len(hypothesis[column - 1]):
                        break
                    if merged_reference == hypothesis[column - 1]:
                        best = min(best, distance[row - count][column - 1])

                merged_hypothesis = hypothesis[column - 1]
                for count in range(2, column + 1):
                    merged_hypothesis = hypothesis[column - count] + merged_hypothesis
                    if len(merged_hypothesis) > len(reference[row - 1]):
                        break
                    if merged_hypothesis == reference[row - 1]:
                        best = min(best, distance[row - 1][column - count])

            distance[row][column] = best
    return distance[rows][columns]


def _verify_standard_result(
    result: jiwer.WordOutput,
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> None:
    reported_errors = result.insertions + result.deletions + result.substitutions
    independent_errors = word_edit_distance(
        reference,
        hypothesis,
        merge_compounds=False,
    )
    if reported_errors != independent_errors:
        raise AssertionError(
            "JiWER and the independent word edit-distance totals disagree: "
            f"{reported_errors} != {independent_errors}",
        )
    reference_words = result.hits + result.substitutions + result.deletions
    if reference_words != len(reference):
        raise AssertionError(
            f"JiWER reference length is {reference_words}; expected {len(reference)}",
        )
    kaldialign_result = edit_distance(
        reference,
        hypothesis,
        merge_compounds=False,
    )
    _verify_kaldialign_result(
        kaldialign_result,
        reference,
        hypothesis,
        merge_compounds=False,
    )


def _verify_kaldialign_result(
    result: Mapping[str, int | float],
    reference: Sequence[str],
    hypothesis: Sequence[str],
    *,
    merge_compounds: bool,
) -> None:
    reported_errors = int(result["ins"] + result["del"] + result["sub"])
    independent_errors = word_edit_distance(
        reference,
        hypothesis,
        merge_compounds=merge_compounds,
    )
    if (
        reported_errors != independent_errors
        or int(result["total"]) != independent_errors
    ):
        raise AssertionError(
            "Kaldialign and the independent word edit-distance totals disagree: "
            f"{reported_errors}/{result['total']} != {independent_errors}",
        )
    if int(result["ref_len"]) != len(reference):
        raise AssertionError(
            f"Kaldialign reference length is {result['ref_len']}; expected "
            f"{len(reference)}",
        )


def _verify_kaldialign_corpus_result(
    result: Mapping[str, int | float],
    independent_errors: int,
    reference_words: int,
    *,
    merge_compounds: bool,
) -> None:
    reported_errors = int(result["ins"] + result["del"] + result["sub"])
    if (
        reported_errors != independent_errors
        or int(result["total"]) != independent_errors
    ):
        mode = "compound-aware" if merge_compounds else "standard"
        raise AssertionError(
            f"Kaldialign {mode} and independent word edit-distance totals "
            f"disagree: {reported_errors}/{result['total']} != "
            f"{independent_errors}",
        )
    if int(result["ref_len"]) != reference_words:
        raise AssertionError(
            f"Kaldialign reference length is {result['ref_len']}; expected "
            f"{reference_words}",
        )
