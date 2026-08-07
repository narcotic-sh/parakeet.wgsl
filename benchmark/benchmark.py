#!/usr/bin/env python3
"""Run the pinned parakeet.wgsl LibriSpeech accuracy benchmark."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.server
import importlib.metadata
import json
import os
import platform
import shutil
import socketserver
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from playwright.sync_api import BrowserContext, Page, sync_playwright

from dataset import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    LIBRISPEECH_SPLITS,
    DatasetSplit,
    LibriSpeechSample,
    download_split,
    get_split,
    iter_samples,
)
from scoring import (
    JIWER_COMMIT,
    JIWER_VERSION,
    KALDIALIGN_COMMIT,
    KALDIALIGN_VERSION,
    OPEN_ASR_COMMIT,
    OPEN_ASR_REPOSITORY,
    CorpusScore,
    score_corpus,
    score_utterance,
    verify_jiwer_version,
    verify_kaldialign_version,
    verify_vendored_normalizer,
)

SCHEMA_VERSION = 2
DEFAULT_PORT = 4179
BENCHMARK_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = BENCHMARK_ROOT.parent
DEFAULT_CACHE_ROOT = BENCHMARK_ROOT / ".cache"
DEFAULT_RESULTS_ROOT = BENCHMARK_ROOT / "results"
CHROME_ARGUMENTS = ("--no-first-run", "--no-default-browser-check")
STANDARD_SCORE_KEY = "standardWer"
OPEN_ASR_SCORE_KEY = "openAsrLeaderboardWer"
SCORE_MODES = {
    STANDARD_SCORE_KEY: False,
    OPEN_ASR_SCORE_KEY: True,
}


class BenchmarkError(RuntimeError):
    """The benchmark cannot proceed without compromising its contract."""


class AudioStore:
    """One-origin handoff for WAV bytes without serializing them through CDP."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, bytes] = {}

    def put(self, contents: bytes) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._items[token] = contents
        return token

    def get(self, token: str) -> bytes | None:
        with self._lock:
            return self._items.get(token)

    def remove(self, token: str) -> None:
        with self._lock:
            self._items.pop(token, None)


class BenchmarkHttpServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BenchmarkRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        audio_store: AudioStore,
        directory: str,
        **kwargs: Any,
    ) -> None:
        self.audio_store = audio_store
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        prefix = "/__parakeet_benchmark_audio__/"
        if not path.startswith(prefix):
            super().do_GET()
            return

        token = unquote(path.removeprefix(prefix)).removesuffix(".wav")
        if not token or "/" in token or "\\" in token:
            self.send_error(404)
            return
        contents = self.audio_store.get(token)
        if contents is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(contents)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(contents)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, _format: str, *args: object) -> None:
        return


@contextlib.contextmanager
def serve_repository(port: int, audio_store: AudioStore) -> Iterator[str]:
    handler = partial(
        BenchmarkRequestHandler,
        audio_store=audio_store,
        directory=os.fspath(REPOSITORY_ROOT),
    )
    try:
        server = BenchmarkHttpServer(("127.0.0.1", port), handler)
    except OSError as error:
        raise BenchmarkError(
            f"cannot bind benchmark origin http://127.0.0.1:{port}: {error}",
        ) from error
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate parakeet.wgsl accuracy on pinned LibriSpeech test-clean "
            "and test-other data."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("both", "test-clean", "test-other"),
        default="both",
        help="dataset split to run (default: both standard test splits)",
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        help="evaluate only the first N utterances of each selected split",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="show the Chrome window instead of using Chrome's new headless mode",
    )
    parser.add_argument(
        "--chrome-path",
        type=Path,
        help=(
            "use this Chrome executable instead of Playwright's installed "
            "Chrome channel"
        ),
    )
    parser.add_argument(
        "--port",
        type=valid_port,
        default=DEFAULT_PORT,
        help=(
            "fixed localhost port used for the persistent model cache "
            f"(default: {DEFAULT_PORT})"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="dataset and persistent Chrome cache directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help=(
            "output directory; defaults to benchmark/results/full or a "
            "smoke-test directory"
        ),
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="use the existing dist build (the build identity is still recorded)",
    )
    return parser.parse_args()


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def valid_port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def selected_splits(name: str) -> tuple[DatasetSplit, ...]:
    if name == "both":
        return LIBRISPEECH_SPLITS
    return (get_split(name),)


def run_command(command: list[str], *, capture: bool = False) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as error:
        raise BenchmarkError(
            f"required command is unavailable: {command[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        details = "\n".join(part for part in (error.stdout, error.stderr) if part)
        raise BenchmarkError(
            f"command failed ({' '.join(command)}):\n{details}".rstrip(),
        ) from error
    return result.stdout.strip() if capture and result.stdout is not None else ""


def build_package(pnpm_executable: Path) -> None:
    print("Preparing the locked JavaScript workspace…", flush=True)
    run_command([os.fspath(pnpm_executable), "install", "--frozen-lockfile"])
    print("Building the public package…", flush=True)
    run_command([os.fspath(pnpm_executable), "build"])


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BenchmarkError(f"executable does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": os.fspath(resolved),
        "byteLength": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def resolve_command(name: str) -> Path:
    executable = shutil.which(name)
    if executable is None:
        raise BenchmarkError(f"required command is unavailable: {name}")
    return Path(executable).resolve()


def javascript_toolchain() -> tuple[dict[str, object], Path]:
    package = json.loads((REPOSITORY_ROOT / "package.json").read_text())
    declaration = package.get("packageManager")
    if not isinstance(declaration, str) or not declaration.startswith("pnpm@"):
        raise BenchmarkError("package.json must declare an exact pnpm version")
    expected_pnpm = declaration.removeprefix("pnpm@")
    if not expected_pnpm or any(character in expected_pnpm for character in "<>=~^"):
        raise BenchmarkError("package.json must declare an exact pnpm version")

    node = resolve_command("node")
    pnpm = resolve_command("pnpm")
    actual_pnpm = run_command([os.fspath(pnpm), "--version"], capture=True)
    if actual_pnpm != expected_pnpm:
        raise BenchmarkError(
            f"installed pnpm version is {actual_pnpm}; expected {expected_pnpm}",
        )
    return (
        {
            "declaredPackageManager": declaration,
            "node": {
                "version": run_command([os.fspath(node), "--version"], capture=True),
                "executable": file_identity(node),
            },
            "pnpm": {
                "version": actual_pnpm,
                "executable": file_identity(pnpm),
            },
        },
        pnpm,
    )


def git_provenance() -> dict[str, object]:
    commit = run_command(["git", "rev-parse", "HEAD"], capture=True)
    status = run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        capture=True,
    ).splitlines()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status,
        "trackedDiffSha256": hashlib.sha256(diff).hexdigest(),
    }


def hash_tree(root: Path) -> str:
    if not root.is_dir():
        raise BenchmarkError(f"package build directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise BenchmarkError(f"package build directory is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def benchmark_source_hash() -> str:
    files = [
        BENCHMARK_ROOT / "benchmark.py",
        BENCHMARK_ROOT / "dataset.py",
        BENCHMARK_ROOT / "scoring.py",
        BENCHMARK_ROOT / "pyproject.toml",
        BENCHMARK_ROOT / "uv.lock",
        BENCHMARK_ROOT / "browser" / "index.html",
        BENCHMARK_ROOT / "browser" / "runner.js",
        BENCHMARK_ROOT / "normalizer" / "normalizer.py",
        BENCHMARK_ROOT / "normalizer" / "english_abbreviations.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            raise BenchmarkError(f"benchmark source file is missing: {path}")
        relative = path.relative_to(BENCHMARK_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    names = (
        "huggingface-hub",
        "jiwer",
        "kaldialign",
        "playwright",
        "pyarrow",
        "regex",
        "soundfile",
    )
    return {name: importlib.metadata.version(name) for name in names}


def package_version() -> str:
    package = json.loads((REPOSITORY_ROOT / "package.json").read_text())
    version = package.get("version")
    if not isinstance(version, str) or not version:
        raise BenchmarkError("package.json does not contain a valid version")
    return version


def resolve_chrome_executable(requested: Path | None) -> Path:
    if requested is not None:
        executable = requested.expanduser().resolve()
        if not executable.is_file():
            raise BenchmarkError(f"Chrome executable does not exist: {executable}")
        return executable

    candidates: list[Path] = []
    system = platform.system()
    if system == "Darwin":
        candidates.extend(
            [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path.home()
                / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ],
        )
    elif system == "Windows":
        for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            root = os.environ.get(variable)
            if root:
                candidates.append(
                    Path(root) / "Google/Chrome/Application/chrome.exe",
                )
    else:
        for name in ("google-chrome-stable", "google-chrome", "chrome"):
            executable = shutil.which(name)
            if executable is not None:
                candidates.append(Path(executable))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise BenchmarkError(
        "could not locate installed Google Chrome; pass --chrome-path",
    )


def launch_context(
    playwright: Any,
    args: argparse.Namespace,
    chrome_executable: Path,
) -> BrowserContext:
    profile = args.cache_dir / "chrome-profile"
    profile.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "user_data_dir": os.fspath(profile.resolve()),
        "headless": not args.headed,
        "args": list(CHROME_ARGUMENTS),
        "executable_path": os.fspath(chrome_executable),
    }
    try:
        return playwright.chromium.launch_persistent_context(**options)
    except Exception as error:
        raise BenchmarkError(
            "could not launch installed Google Chrome; install Chrome or pass "
            "--chrome-path",
        ) from error


def open_benchmark_page(context: BrowserContext, origin: str) -> tuple[Page, dict]:
    page = context.pages[0] if context.pages else context.new_page()
    page.set_default_timeout(30 * 60 * 1000)
    page.on(
        "console",
        lambda message: (
            print(
                f"Browser console {message.type}: {message.text}",
                file=sys.stderr,
                flush=True,
            )
            if message.type in {"error", "warning"}
            else None
        ),
    )
    page.on(
        "pageerror",
        lambda error: print(
            f"Browser page error: {error}", file=sys.stderr, flush=True
        ),
    )
    page.goto(f"{origin}/benchmark/browser/index.html", wait_until="load")
    report = page.evaluate(
        """async () => {
          try {
            return { ok: true, value: await window.parakeetBenchmark.ready };
          } catch (error) {
            const serialize = window.parakeetBenchmark?.serializeError;
            return {
              ok: false,
              error: typeof serialize === "function"
                ? serialize(error)
                : { message: String(error) },
            };
          }
        }""",
    )
    if not report.get("ok"):
        raise BenchmarkError(
            f"browser support initialization failed: {json.dumps(report.get('error'))}",
        )
    value = report.get("value")
    if not isinstance(value, dict) or not isinstance(value.get("support"), dict):
        raise BenchmarkError("browser benchmark returned invalid support diagnostics")
    return page, value


def browser_identity(
    context: BrowserContext,
    page: Page,
    headed: bool,
    chrome_executable: Path,
) -> dict:
    browser = context.browser
    return {
        "productVersion": browser.version if browser is not None else None,
        "userAgent": page.evaluate("() => navigator.userAgent"),
        "headless": not headed,
        "executable": file_identity(chrome_executable),
        "launchArguments": list(CHROME_ARGUMENTS),
    }


def transcribe_sample(
    page: Page,
    origin: str,
    audio_store: AudioStore,
    sample: LibriSpeechSample,
    wav_bytes: bytes,
) -> dict:
    token = audio_store.put(wav_bytes)
    try:
        report = page.evaluate(
            """async ({ id, name, audioUrl }) => {
              try {
                const response = await fetch(audioUrl, { cache: "no-store" });
                if (!response.ok) {
                  throw new Error(`Audio request failed with HTTP ${response.status}`);
                }
                const bytes = await response.arrayBuffer();
                const value = await window.parakeetBenchmark.transcribe({
                  id,
                  name,
                  bytes,
                });
                return { ok: true, value };
              } catch (error) {
                return {
                  ok: false,
                  error: window.parakeetBenchmark.serializeError(error),
                };
              }
            }""",
            {
                "id": sample.id,
                "name": sample.name,
                "audioUrl": f"{origin}/__parakeet_benchmark_audio__/{token}.wav",
            },
        )
    finally:
        audio_store.remove(token)
    if not report.get("ok"):
        raise BenchmarkError(
            f"transcription failed for {sample.split}/{sample.id}: "
            f"{json.dumps(report.get('error'), ensure_ascii=False)}",
        )
    value = report.get("value")
    if (
        not isinstance(value, dict)
        or value.get("id") != sample.id
        or value.get("name") != sample.name
        or not isinstance(value.get("text"), str)
    ):
        raise BenchmarkError(
            f"browser returned an invalid result for {sample.split}/{sample.id}",
        )
    return value


def result_directory(args: argparse.Namespace) -> Path:
    if args.results_dir is not None:
        return args.results_dir.expanduser().resolve()
    suffix = "full" if args.limit is None else f"smoke-{args.limit}"
    return DEFAULT_RESULTS_ROOT / suffix


def create_contract(
    args: argparse.Namespace,
    splits: tuple[DatasetSplit, ...],
    support: Mapping[str, object],
    browser: Mapping[str, object],
    toolchain: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "package": {
            "name": "parakeet.wgsl",
            "version": package_version(),
            "source": git_provenance(),
            "distSha256": hash_tree(REPOSITORY_ROOT / "dist"),
        },
        "dataset": {
            "repository": DATASET_REPO_ID,
            "revision": DATASET_REVISION,
            "splits": [asdict(split) for split in splits],
            "limitPerSplit": args.limit,
        },
        "scoring": {
            "repository": OPEN_ASR_REPOSITORY,
            "revision": OPEN_ASR_COMMIT,
            "normalizer": "EnglishTextNormalizer",
            "jiwerVersion": JIWER_VERSION,
            "jiwerCommit": JIWER_COMMIT,
            "kaldialignVersion": KALDIALIGN_VERSION,
            "kaldialignCommit": KALDIALIGN_COMMIT,
            "metrics": {
                STANDARD_SCORE_KEY: {
                    "alignment": "jiwer-word-levenshtein",
                    "mergeCompounds": False,
                    "aggregation": "corpus-micro",
                },
                OPEN_ASR_SCORE_KEY: {
                    "alignment": "kaldialign-word-levenshtein",
                    "mergeCompounds": True,
                    "aggregation": "corpus-micro",
                },
            },
            "benchmarkSourceSha256": benchmark_source_hash(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": dependency_versions(),
            "browser": browser,
            "javascriptToolchain": toolchain,
            "support": support,
            "origin": f"http://127.0.0.1:{args.port}",
        },
    }


def contract_digest(contract: Mapping[str, object]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def initialize_run_manifest(directory: Path, contract: dict[str, object]) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    expected_digest = contract_digest(contract)
    if path.exists():
        manifest = read_json(path)
        if (
            manifest.get("contract") != contract
            or manifest.get("contractSha256") != expected_digest
        ):
            raise BenchmarkError(
                f"cannot resume {directory}: benchmark provenance changed; "
                "use a new --results-dir",
            )
        return manifest
    existing_entries = sorted(entry.name for entry in directory.iterdir())
    if existing_entries:
        raise BenchmarkError(
            f"cannot start {directory}: result files exist without run.json; "
            "use a new --results-dir or remove the orphaned files",
        )
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "contractSha256": expected_digest,
        "contract": contract,
        "runtimeDiagnostics": None,
    }
    write_json_atomic(path, manifest)
    return manifest


def set_runtime_diagnostics(
    directory: Path, manifest: dict, diagnostics: object
) -> None:
    if not isinstance(diagnostics, dict):
        raise BenchmarkError("transcriber runtime diagnostics are unavailable")
    existing = manifest.get("runtimeDiagnostics")
    if existing is not None and existing != diagnostics:
        raise BenchmarkError(
            "cannot resume: transcriber runtime diagnostics differ from the "
            "original run",
        )
    if existing is None:
        manifest["runtimeDiagnostics"] = diagnostics
        write_json_atomic(directory / "run.json", manifest)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r+b") as source:
        source.seek(0, os.SEEK_END)
        length = source.tell()
        if length:
            source.seek(length - 1)
            if source.read(1) != b"\n":
                source.seek(0)
                contents = source.read()
                committed_length = contents.rfind(b"\n") + 1
                source.seek(committed_length)
                source.truncate()
                source.flush()
                os.fsync(source.fileno())
                print(
                    f"Recovered incomplete trailing record from {path}",
                    file=sys.stderr,
                    flush=True,
                )
    records: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.endswith("\n"):
                raise BenchmarkError(
                    f"{path}:{line_number} is not a complete newline-terminated record",
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BenchmarkError(
                    f"invalid JSON in {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise BenchmarkError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise BenchmarkError(f"could not append result record to {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def make_result_record(
    split: DatasetSplit,
    index: int,
    sample: LibriSpeechSample,
    hypothesis: str,
    source_sha256: str,
    wav_sha256: str,
) -> dict[str, object]:
    scores = {
        key: score_utterance(
            sample.reference,
            hypothesis,
            merge_compounds=merge_compounds,
        )
        for key, merge_compounds in SCORE_MODES.items()
    }
    standard_score = scores[STANDARD_SCORE_KEY]
    if any(
        score.normalized_reference != standard_score.normalized_reference
        or score.normalized_hypothesis != standard_score.normalized_hypothesis
        or score.reference_words != standard_score.reference_words
        for score in scores.values()
    ):
        raise BenchmarkError("scoring modes produced inconsistent normalization")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "split": split.name,
        "index": index,
        "id": sample.id,
        "sourceFlacSha256": source_sha256,
        "canonicalWavSha256": wav_sha256,
        "reference": sample.reference,
        "hypothesis": hypothesis,
        "normalizedReference": standard_score.normalized_reference,
        "normalizedHypothesis": standard_score.normalized_hypothesis,
        "scores": {
            key: {
                "substitutions": score.substitutions,
                "deletions": score.deletions,
                "insertions": score.insertions,
                "referenceWords": score.reference_words,
            }
            for key, score in scores.items()
        },
    }


def validate_existing_record(
    record: Mapping[str, object],
    split: DatasetSplit,
    index: int,
    sample: LibriSpeechSample,
    source_sha256: str,
    wav_sha256: str,
) -> None:
    hypothesis = record.get("hypothesis")
    if not isinstance(hypothesis, str):
        raise BenchmarkError(
            f"stored result {split.name}/{sample.id} has no hypothesis"
        )
    expected = make_result_record(
        split,
        index,
        sample,
        hypothesis,
        source_sha256,
        wav_sha256,
    )
    if dict(record) != expected:
        raise BenchmarkError(
            f"stored result {split.name}/{sample.id} does not match pinned "
            "input/scoring",
        )


def run_split(
    split: DatasetSplit,
    parquet_path: Path,
    limit: int | None,
    results_directory: Path,
    page: Page,
    origin: str,
    audio_store: AudioStore,
    run_manifest: dict,
    runtime_checked: bool,
) -> tuple[list[dict], dict[str, CorpusScore], bool]:
    output_path = results_directory / f"{split.name}.jsonl"
    existing = load_jsonl(output_path)
    expected_count = min(split.rows, limit) if limit is not None else split.rows
    if len(existing) > expected_count:
        raise BenchmarkError(
            f"{output_path} contains {len(existing)} records; expected at most "
            f"{expected_count}",
        )

    records: list[dict] = []
    for index, sample in enumerate(iter_samples(parquet_path, split)):
        if index >= expected_count:
            break
        source_sha256 = hashlib.sha256(sample.flac_bytes).hexdigest()
        wav_bytes = sample.wav_bytes()
        wav_sha256 = hashlib.sha256(wav_bytes).hexdigest()
        if index < len(existing):
            record = existing[index]
            validate_existing_record(
                record,
                split,
                index,
                sample,
                source_sha256,
                wav_sha256,
            )
            records.append(record)
            continue

        result = transcribe_sample(page, origin, audio_store, sample, wav_bytes)
        record = make_result_record(
            split,
            index,
            sample,
            result["text"],
            source_sha256,
            wav_sha256,
        )
        if not runtime_checked:
            diagnostics = page.evaluate(
                "() => window.parakeetBenchmark.diagnostics().runtime",
            )
            set_runtime_diagnostics(results_directory, run_manifest, diagnostics)
            runtime_checked = True

        append_jsonl(output_path, record)
        records.append(record)

        completed = index + 1
        if completed == 1 or completed % 25 == 0 or completed == expected_count:
            print(f"{split.name}: {completed}/{expected_count}", flush=True)

    if len(records) != expected_count:
        raise BenchmarkError(
            f"{split.name} produced {len(records)} records; expected {expected_count}",
        )

    corpus_records = [
        {
            "reference": str(record["reference"]),
            "hypothesis": str(record["hypothesis"]),
        }
        for record in records
    ]
    scores = {
        key: score_corpus(
            corpus_records,
            merge_compounds=merge_compounds,
        )
        for key, merge_compounds in SCORE_MODES.items()
    }
    for key, score in scores.items():
        stored_scores = [_stored_score(record, key) for record in records]
        per_utterance = {
            "substitutions": sum(int(item["substitutions"]) for item in stored_scores),
            "deletions": sum(int(item["deletions"]) for item in stored_scores),
            "insertions": sum(int(item["insertions"]) for item in stored_scores),
            "reference_words": sum(
                int(item["referenceWords"]) for item in stored_scores
            ),
        }
        if per_utterance != {
            "substitutions": score.substitutions,
            "deletions": score.deletions,
            "insertions": score.insertions,
            "reference_words": score.reference_words,
        }:
            raise BenchmarkError(
                f"{split.name} {key} per-utterance and corpus totals disagree",
            )
    return records, scores, runtime_checked


def _stored_score(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    scores = record.get("scores")
    if not isinstance(scores, Mapping):
        raise BenchmarkError(f"stored result has no scores object: {record.get('id')}")
    score = scores.get(key)
    if not isinstance(score, Mapping):
        raise BenchmarkError(f"stored result has no {key} score: {record.get('id')}")
    return score


def score_summary(score: CorpusScore) -> dict[str, object]:
    return {
        "substitutions": score.substitutions,
        "deletions": score.deletions,
        "insertions": score.insertions,
        "errors": score.errors,
        "referenceWords": score.reference_words,
        "wer": score.wer,
        "werPercent": score.wer_percent,
        "werPercentRoundedTwoDecimals": round(score.wer_percent, 2),
    }


def split_summary(
    split: DatasetSplit,
    scores: Mapping[str, CorpusScore],
    complete: bool,
) -> dict[str, object]:
    return {
        "split": split.name,
        "complete": complete,
        "utterances": scores[STANDARD_SCORE_KEY].utterances,
        STANDARD_SCORE_KEY: score_summary(scores[STANDARD_SCORE_KEY]),
        OPEN_ASR_SCORE_KEY: score_summary(scores[OPEN_ASR_SCORE_KEY]),
    }


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_arguments()
    verify_vendored_normalizer()
    verify_jiwer_version()
    verify_kaldialign_version()
    splits = selected_splits(args.split)
    toolchain, pnpm_executable = javascript_toolchain()
    if not args.skip_build:
        build_package(pnpm_executable)
    elif not (REPOSITORY_ROOT / "dist" / "index.js").is_file():
        raise BenchmarkError("--skip-build requires an existing dist/index.js")

    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    results_directory = result_directory(args)
    chrome_executable = resolve_chrome_executable(args.chrome_path)

    parquet_paths: dict[str, Path] = {}
    for split in splits:
        print(f"Downloading/verifying {split.name}…", flush=True)
        parquet_paths[split.name] = download_split(
            split,
            cache_dir=args.cache_dir / "huggingface",
        )

    audio_store = AudioStore()
    with (
        serve_repository(args.port, audio_store) as origin,
        sync_playwright() as playwright,
    ):
        context = launch_context(playwright, args, chrome_executable)
        try:
            page, ready = open_benchmark_page(context, origin)
            support = ready["support"]
            browser = browser_identity(
                context,
                page,
                args.headed,
                chrome_executable,
            )
            contract = create_contract(args, splits, support, browser, toolchain)
            run_manifest = initialize_run_manifest(results_directory, contract)
            print(
                "Execution profile: "
                f"{json.dumps(support.get('executionProfile'), sort_keys=True)}",
                flush=True,
            )
            print(
                "The first utterance may download and initialize the selected model.",
                flush=True,
            )

            summaries = []
            runtime_checked = False
            for split in splits:
                _, scores, runtime_checked = run_split(
                    split,
                    parquet_paths[split.name],
                    args.limit,
                    results_directory,
                    page,
                    origin,
                    audio_store,
                    run_manifest,
                    runtime_checked,
                )
                summary = split_summary(
                    split,
                    scores,
                    args.limit is None or args.limit >= split.rows,
                )
                summaries.append(summary)
                standard = summary[STANDARD_SCORE_KEY]
                open_asr = summary[OPEN_ASR_SCORE_KEY]
                print(
                    f"{split.name}: "
                    f"{standard['werPercentRoundedTwoDecimals']:.2f}% standard WER "
                    f"({standard['errors']}/{standard['referenceWords']}); "
                    f"{open_asr['werPercentRoundedTwoDecimals']:.2f}% "
                    "Open ASR leaderboard WER "
                    f"({open_asr['errors']}/{open_asr['referenceWords']})",
                    flush=True,
                )

            summary_document = {
                "schemaVersion": SCHEMA_VERSION,
                "contractSha256": run_manifest["contractSha256"],
                "scoring": run_manifest["contract"]["scoring"],
                "results": summaries,
            }
            write_json_atomic(results_directory / "summary.json", summary_document)
            page.evaluate("() => window.parakeetBenchmark.dispose()")
        finally:
            context.close()

    print(f"Results written to {results_directory}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(
            "Benchmark interrupted; rerun the same command to resume."
        ) from None
    except BenchmarkError as error:
        raise SystemExit(f"Benchmark failed: {error}") from None
