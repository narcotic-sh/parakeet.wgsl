from __future__ import annotations

import random
from math import isclose, isinf

import jiwer
import pytest
from kaldialign import edit_distance

from scoring import (
    JIWER_COMMIT,
    JIWER_VERSION,
    KALDIALIGN_COMMIT,
    KALDIALIGN_VERSION,
    OPEN_ASR_COMMIT,
    normalize_text,
    score_corpus,
    score_utterance,
    verify_jiwer_version,
    verify_kaldialign_version,
    verify_vendored_normalizer,
    word_edit_distance,
)


def test_pinned_source_identities() -> None:
    assert OPEN_ASR_COMMIT == "9585fc39bff55697a2ec1c5f13921b18812bfde8"
    assert KALDIALIGN_VERSION == "0.12.0"
    assert KALDIALIGN_COMMIT == "06ac40f03c3d368932adf8536965a088d54189b1"
    assert JIWER_VERSION == "3.1.0"
    assert JIWER_COMMIT == "f2f13e0231d2de54dbd7abeea951de676159a715"
    verify_vendored_normalizer()
    verify_kaldialign_version()
    verify_jiwer_version()


def test_kaldialign_runtime_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "scoring.importlib.metadata.version",
        lambda _distribution: "999.0.0",
    )
    with pytest.raises(
        RuntimeError,
        match=r"installed Kaldialign version is 999\.0\.0",
    ):
        verify_kaldialign_version()


def test_jiwer_runtime_version_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scoring.importlib.metadata.version",
        lambda _distribution: "999.0.0",
    )
    with pytest.raises(
        RuntimeError,
        match=r"installed JiWER version is 999\.0\.0",
    ):
        verify_jiwer_version()


@pytest.mark.parametrize(
    ("source", "normalized"),
    [
        ("HELLO, world!", "hello world"),
        ("I can't pay twenty-one dollars.", "i can not pay $21"),
        ("The colour of café.", "the color of cafe"),
        ("b b c and 5 g", "bbc and 5g"),
        ("wi fi", "wifi"),
        ("[noise] (aside) uh", ""),
    ],
)
def test_open_asr_normalizer_oracles(source: str, normalized: str) -> None:
    assert normalize_text(source) == normalized


@pytest.mark.parametrize(
    ("reference", "hypothesis", "insertions", "deletions", "substitutions", "words"),
    [
        ("red blue green", "red blue green", 0, 0, 0, 3),
        ("red blue green", "red black green", 0, 0, 1, 3),
        ("red blue green", "red bright blue green", 1, 0, 0, 3),
        ("red blue green", "red green", 0, 1, 0, 3),
        ("white paper", "whitepaper", 0, 1, 1, 2),
        ("whitepaper", "white paper", 1, 0, 1, 1),
        ("red blue blue", "blue blue red", 0, 0, 2, 3),
        ("red blue", "", 0, 2, 0, 2),
    ],
)
def test_authoritative_alignment_vectors(
    reference: str,
    hypothesis: str,
    insertions: int,
    deletions: int,
    substitutions: int,
    words: int,
) -> None:
    score = score_utterance(
        reference,
        hypothesis,
        merge_compounds=False,
    )
    assert (
        score.insertions,
        score.deletions,
        score.substitutions,
        score.reference_words,
    ) == (insertions, deletions, substitutions, words)


def test_compound_aware_score_is_explicitly_distinct() -> None:
    standard = score_utterance(
        "white paper",
        "whitepaper",
        merge_compounds=False,
    )
    compound_aware = score_utterance(
        "white paper",
        "whitepaper",
        merge_compounds=True,
    )
    assert standard.errors == 2
    assert compound_aware.errors == 0


def test_corpus_wer_preserves_boundaries_and_micro_averages() -> None:
    boundary_score = score_corpus(
        [
            {"reference": "red", "hypothesis": "red blue"},
            {"reference": "blue", "hypothesis": ""},
        ],
        merge_compounds=False,
    )
    assert boundary_score.errors == 2
    assert boundary_score.reference_words == 2
    assert boundary_score.wer_percent == 100.0

    micro_score = score_corpus(
        [
            {"reference": "cat", "hypothesis": "wrong"},
            {
                "reference": "a quick brown fox jumps over the lazy dog",
                "hypothesis": "a quick brown fox jumps over the lazy dog",
            },
        ],
        merge_compounds=False,
    )
    assert micro_score.errors == 1
    assert micro_score.reference_words == 10
    assert isclose(micro_score.wer_percent, 10.0)


def test_empty_reference_semantics_are_explicit() -> None:
    with pytest.raises(ValueError, match="references are empty"):
        score_corpus(
            [{"reference": "", "hypothesis": ""}],
            merge_compounds=False,
        )

    empty = score_corpus(
        [{"reference": "", "hypothesis": ""}],
        merge_compounds=True,
    )
    assert empty.wer == 0.0
    insertion = score_corpus(
        [{"reference": "", "hypothesis": "ghost"}],
        merge_compounds=True,
    )
    assert isinf(insertion.wer)


def test_independent_dp_matches_both_scorers_randomized() -> None:
    randomizer = random.Random(0)
    vocabulary = ["a", "b", "c", "ab", "bc", "abc", "red", "blue"]
    for _ in range(1_000):
        reference = tuple(
            randomizer.choice(vocabulary) for _ in range(randomizer.randrange(7))
        )
        hypothesis = tuple(
            randomizer.choice(vocabulary) for _ in range(randomizer.randrange(7))
        )
        for merge_compounds in (False, True):
            authoritative = edit_distance(
                reference,
                hypothesis,
                merge_compounds=merge_compounds,
            )
            assert (
                word_edit_distance(
                    reference,
                    hypothesis,
                    merge_compounds=merge_compounds,
                )
                == authoritative["total"]
            )
        if reference:
            standard = jiwer.process_words(
                [" ".join(reference)],
                [" ".join(hypothesis)],
            )
            assert (
                word_edit_distance(
                    reference,
                    hypothesis,
                    merge_compounds=False,
                )
                == standard.substitutions + standard.deletions + standard.insertions
            )
