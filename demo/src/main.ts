import {
  PacedTranscriptDomRenderer,
  checkSupport,
  createTranscriber,
  deleteCachedModels,
  getModelCacheInfo,
  requiresAudioDecoding,
  type ModelLoadProgress,
  type PacedTranscriptDomWordHit,
  type PacedTranscriptUpdate,
  type PacedTranscriptWord,
  type TranscriptionProgress,
  type TranscriptionSnapshot,
} from "parakeet.wgsl";
import lightModeIcon from "./assets/light-mode.png";
import moonIcon from "./assets/moon.png";
import { performanceWarningForBrowser } from "./browser-performance-warning.js";
import {
  isTranscriptTailReached,
  transcriptWordAtTime,
} from "./transcript-interactions.js";
import { formatProgressElapsedMilliseconds } from "./progress-timing.js";
import {
  SAMPLE_AUDIO_MAX_DOWNLOAD_ATTEMPTS,
  parseRetryAfterMilliseconds,
  retryableSampleStatus,
  sampleRetryDelayMilliseconds,
} from "./sample-download-policy.js";
import "./style.css";

interface PointerLocation {
  readonly clientX: number;
  readonly clientY: number;
}

type DemoTheme = "light" | "dark";

const THEME_STORAGE_KEY = "parakeet-wgsl-demo-theme";
const PLAYBACK_WORD_HIGHLIGHT_NAME = "parakeet-demo-playback-word";
const SAMPLE_AUDIO_BYTE_LENGTH = 3_091_941;
const SAMPLE_AUDIO_SHA256 =
  "dc12e3163c0da5f7fdfbfa4b4dfd0219d0d1c6c432d98c3b4a173829419f6f3f";
const SAMPLE_AUDIO_URL =
  "https://parakeet-wgsl-models.narcotic.sh/samples/v1/" +
  `${SAMPLE_AUDIO_SHA256}/kennedy.mp3`;

const themeToggle = requiredElement<HTMLButtonElement>("theme-toggle");
const themeIcon = requiredElement<HTMLImageElement>("theme-icon");
const runtimeToggle = requiredElement<HTMLButtonElement>("runtime-toggle");
const runtimeDialog = requiredElement<HTMLDialogElement>("runtime-dialog");
const runtimeDialogTitle = requiredElement<HTMLHeadingElement>(
  "runtime-dialog-title",
);
const runtimeClose = requiredElement<HTMLButtonElement>("runtime-close");
const status = requiredElement<HTMLParagraphElement>("status");
const input = requiredElement<HTMLInputElement>("audio");
const sampleAudioButton = requiredElement<HTMLButtonElement>("sample-audio");
const sampleStatus = requiredElement<HTMLParagraphElement>("sample-status");
const transcribeButton = requiredElement<HTMLButtonElement>("transcribe");
const cancelButton = requiredElement<HTMLButtonElement>("cancel");
const summary = requiredElement<HTMLElement>("summary");
const summaryDuration = requiredElement<HTMLElement>("summary-duration");
const summaryProcessingTime = requiredElement<HTMLElement>(
  "summary-processing-time",
);
const summarySpeed = requiredElement<HTMLElement>("summary-speed");
const progressPanel = requiredElement<HTMLElement>("progress-panel");
const progressTitleLabel = requiredElement<HTMLElement>(
  "progress-title-label",
);
const progressTitleDetail = requiredElement<HTMLElement>(
  "progress-title-detail",
);
const progressElapsed = requiredElement<HTMLTimeElement>("progress-elapsed");
const progressPercent = requiredElement<HTMLElement>("progress-percent");
const progressElement = requiredElement<HTMLProgressElement>("progress");
const transcriptPanel = requiredElement<HTMLElement>("transcript-panel");
const outputState = requiredElement<HTMLParagraphElement>("output-state");
const copyTranscriptButton = requiredElement<HTMLButtonElement>(
  "copy-transcript",
);
const copyTranscriptStatus = requiredElement<HTMLSpanElement>(
  "copy-transcript-status",
);
const playbackRow = requiredElement<HTMLElement>("playback-row");
const audioPlayer = requiredElement<HTMLAudioElement>("audio-player");
const playbackUnavailable = requiredElement<HTMLParagraphElement>(
  "playback-unavailable",
);
const output = requiredElement<HTMLDivElement>("output");
const wordTooltip = requiredElement<HTMLDivElement>("word-tooltip");
const metrics = requiredElement<HTMLPreElement>("metrics");
const cacheStatus = requiredElement<HTMLParagraphElement>("cache-status");
const deleteModelsButton = requiredElement<HTMLButtonElement>("delete-models");
const browserPerformanceWarning = requiredElement<HTMLParagraphElement>(
  "browser-performance-warning",
);

const transcriptRenderer = new PacedTranscriptDomRenderer(output, {
  autoFollow: false,
});
const playbackWordHighlight = new Highlight();
CSS.highlights.set(PLAYBACK_WORD_HIGHLIGHT_NAME, playbackWordHighlight);

let transcriber: ReturnType<typeof createTranscriber> | undefined;
let activeAbortController: AbortController | undefined;
let selectedFileIsValid = false;
let isBusy = false;
let sampleAudioDownloadPending = false;
let sampleAudioRequestRevision = 0;
let sampleAudioAbortController: AbortController | undefined;
let downloadedModelsCanBeDeleted = false;
let supportDetails: unknown;
let downloadedModelDetails: unknown;
let transcriptionDetails: unknown;
let initializationFailure: string | undefined;
let fileValidationRevision = 0;
let displayedTranscriptionProgressKey: string | undefined;
let modelDownloadFraction = 0;
let cachedWarmupTranscriber:
  | ReturnType<typeof createTranscriber>
  | undefined;
let progressElapsedAnchorMilliseconds: number | undefined;
let progressElapsedAnchorNow = 0;
let progressElapsedInterval: number | undefined;
let displayedProgressElapsedText: string | undefined;
let pendingTooltipPoint: PointerLocation | undefined;
let tooltipRenderFrame: number | undefined;
let activeWordHit: PacedTranscriptDomWordHit | undefined;
let tooltipWidth = 0;
let tooltipHeight = 0;
let liveTailFollow = false;
let playbackObjectUrl: string | undefined;
let playbackSyncFrame: number | undefined;
let pendingPlaybackSeekSeconds: number | undefined;
let activePlaybackWord: PacedTranscriptWord | undefined;
let followedPlaybackWord: PacedTranscriptWord | undefined;
let transcriptCopyFeedbackTimeout: number | undefined;

configureTheme();
configureBrowserPerformanceWarning();
wireEvents();
void initialize();

function configureTheme(): void {
  const initialTheme =
    document.documentElement.dataset.parakeetDemoTheme === "dark"
      ? "dark"
      : "light";
  applyTheme(initialTheme);
}

function configureBrowserPerformanceWarning(): void {
  const warning = performanceWarningForBrowser(navigator);
  if (warning === undefined) return;
  browserPerformanceWarning.textContent = warning;
  browserPerformanceWarning.hidden = false;
}

function applyTheme(theme: DemoTheme): void {
  document.documentElement.dataset.parakeetDemoTheme = theme;
  const dark = theme === "dark";
  const label = dark ? "Switch to light theme" : "Switch to dark theme";
  themeToggle.setAttribute("aria-pressed", String(dark));
  themeToggle.setAttribute("aria-label", label);
  themeToggle.title = label;
  themeIcon.src = dark ? lightModeIcon : moonIcon;
  themeIcon.classList.toggle("is-sun", dark);
}

function wireEvents(): void {
  themeToggle.addEventListener("click", () => {
    const current =
      document.documentElement.dataset.parakeetDemoTheme === "dark"
        ? "dark"
        : "light";
    const next: DemoTheme = current === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // The theme still applies for this visit when storage is unavailable.
    }
    applyTheme(next);
  });

  wireModalDialog(
    runtimeToggle,
    runtimeDialog,
    runtimeDialogTitle,
    runtimeClose,
  );

  input.addEventListener("change", () => {
    cancelSampleAudioDownload();
    void validateSelectedFile();
  });

  sampleAudioButton.addEventListener("click", () => {
    void selectSampleAudio();
  });

  transcribeButton.addEventListener("click", () => {
    void transcribeSelectedFile();
  });

  cancelButton.addEventListener("click", () => {
    activeAbortController?.abort(
      new DOMException("Transcription cancelled", "AbortError"),
    );
  });

  deleteModelsButton.addEventListener("click", () => {
    void deleteDownloadedModels();
  });

  copyTranscriptButton.addEventListener("click", () => {
    void copyFinalTranscript();
  });

  output.addEventListener("pointermove", (event) => {
    pendingTooltipPoint = {
      clientX: event.clientX,
      clientY: event.clientY,
    };
    if (tooltipRenderFrame !== undefined) return;
    tooltipRenderFrame = requestAnimationFrame(renderWordTooltip);
  });

  output.addEventListener("pointerleave", hideWordTooltip);
  output.addEventListener("scroll", handleTranscriptScroll, { passive: true });
  output.addEventListener("click", seekPlaybackToClickedWord);

  audioPlayer.addEventListener("play", startPlaybackSync);
  audioPlayer.addEventListener("pause", () => {
    stopPlaybackSync();
    syncPlaybackWord(false);
  });
  audioPlayer.addEventListener("timeupdate", () => {
    syncPlaybackWord(!audioPlayer.paused && !audioPlayer.ended);
  });
  audioPlayer.addEventListener("seeking", () => {
    followedPlaybackWord = undefined;
    syncPlaybackWord(!audioPlayer.paused && !audioPlayer.ended);
  });
  audioPlayer.addEventListener("seeked", () => {
    followedPlaybackWord = undefined;
    syncPlaybackWord(!audioPlayer.paused && !audioPlayer.ended);
  });
  audioPlayer.addEventListener("loadedmetadata", () => {
    applyPendingPlaybackSeek();
    syncPlaybackWord(false);
  });
  audioPlayer.addEventListener("ended", () => {
    stopPlaybackSync();
    syncPlaybackWord(false);
  });
  audioPlayer.addEventListener("error", handlePlaybackError);
  audioPlayer.addEventListener("emptied", clearPlaybackWord);

  window.addEventListener("pagehide", (event) => {
    if (!event.persisted) {
      cancelSampleAudioDownload();
      releaseAudioPlayback();
    }
  });
}

function wireModalDialog(
  trigger: HTMLButtonElement,
  dialog: HTMLDialogElement,
  title: HTMLElement,
  closeButton: HTMLButtonElement,
): void {
  trigger.addEventListener("click", () => openModalDialog(dialog, title));
  closeButton.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    dialog.close();
  });
  dialog.addEventListener("close", () => {
    document.documentElement.classList.remove("has-modal-dialog");
    trigger.focus({ preventScroll: true });
  });
}

function openModalDialog(
  dialog: HTMLDialogElement,
  title: HTMLElement,
): void {
  if (dialog.open) return;
  dialog.showModal();
  document.documentElement.classList.add("has-modal-dialog");
  title.focus({ preventScroll: true });
}

async function initialize(): Promise<void> {
  void refreshDownloadedModelInfo();
  try {
    setStatus("Checking browser support…", "busy");
    const support = await checkSupport();
    supportDetails = support;
    updateRuntimeDetails();

    const supportRecord = recordValue(support);
    if (supportRecord?.supported !== true) {
      throw new Error(unsupportedMessage(supportRecord));
    }

    transcriber = makeTranscriber();
    warmDownloadedModel(transcriber);
    initializationFailure = undefined;
    if (input.files?.[0] === undefined) {
      setStatus("", "ready");
    } else {
      await validateSelectedFile();
    }
    updateActionAvailability();
  } catch (error) {
    initializationFailure = errorMessage(error);
    setStatus(initializationFailure, "error");
    metrics.textContent = printableJson({
      support: supportDetails,
      error: serializeError(error),
    });
  }
}

function makeTranscriber(): ReturnType<typeof createTranscriber> {
  return createTranscriber({ onLoadProgress: updateModelLoadProgress });
}

function warmDownloadedModel(
  candidate: ReturnType<typeof createTranscriber>,
): void {
  cachedWarmupTranscriber = candidate;
  void candidate.loadCachedModel().then(
    () => {
      if (transcriber === candidate) updateRuntimeDetails();
    },
    () => {
      // A later transcribe() call may retry initialization normally. Cached
      // warmup remains deliberately silent in the page UI.
    },
  ).finally(() => {
    if (cachedWarmupTranscriber === candidate) {
      cachedWarmupTranscriber = undefined;
    }
  });
}

async function selectSampleAudio(): Promise<void> {
  if (
    isBusy ||
    sampleAudioDownloadPending ||
    transcriber === undefined
  ) {
    return;
  }

  const revision = ++sampleAudioRequestRevision;
  const abortController = new AbortController();
  sampleAudioAbortController = abortController;
  sampleAudioDownloadPending = true;
  sampleAudioButton.setAttribute("aria-busy", "true");
  setStatus("", "ready");
  setSampleStatus("Downloading sample audio…", "busy");
  updateActionAvailability();

  try {
    const bytes = await downloadSampleAudio(abortController.signal);
    if (
      revision !== sampleAudioRequestRevision ||
      sampleAudioAbortController !== abortController
    ) {
      return;
    }

    const file = new File([bytes], "kennedy.mp3", { type: "audio/mpeg" });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    setSampleStatus("", "ready");
    await validateSelectedFile();
  } catch (error) {
    if (
      abortController.signal.aborted ||
      revision !== sampleAudioRequestRevision
    ) {
      return;
    }
    setSampleStatus(
      `Could not download sample audio: ${errorMessage(error)}`,
      "error",
    );
  } finally {
    if (
      revision === sampleAudioRequestRevision &&
      sampleAudioAbortController === abortController
    ) {
      sampleAudioAbortController = undefined;
      sampleAudioDownloadPending = false;
      sampleAudioButton.setAttribute("aria-busy", "false");
      updateActionAvailability();
    }
  }
}

function cancelSampleAudioDownload(): void {
  setSampleStatus("", "ready");
  const abortController = sampleAudioAbortController;
  if (abortController === undefined) return;
  sampleAudioRequestRevision += 1;
  sampleAudioAbortController = undefined;
  sampleAudioDownloadPending = false;
  sampleAudioButton.setAttribute("aria-busy", "false");
  abortController.abort();
  updateActionAvailability();
}

async function downloadSampleAudio(signal: AbortSignal): Promise<ArrayBuffer> {
  for (
    let attempt = 1;
    attempt <= SAMPLE_AUDIO_MAX_DOWNLOAD_ATTEMPTS;
    attempt += 1
  ) {
    try {
      return await downloadSampleAudioAttempt(signal);
    } catch (error) {
      signal.throwIfAborted();
      if (
        !(error instanceof RetryableSampleAudioError) ||
        attempt === SAMPLE_AUDIO_MAX_DOWNLOAD_ATTEMPTS
      ) {
        throw error;
      }
      await abortableDelay(
        sampleRetryDelayMilliseconds(attempt, error.retryAfterMs),
        signal,
      );
    }
  }
  throw new Error("Sample audio download exhausted its retry policy");
}

async function downloadSampleAudioAttempt(
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await fetch(SAMPLE_AUDIO_URL, { signal });
  } catch (error) {
    signal.throwIfAborted();
    throw new RetryableSampleAudioError(
      "Sample audio request was interrupted",
      error,
    );
  }

  if (!response.ok) {
    const status = response.status;
    const retryAfterMs = retryableSampleStatus(status)
      ? parseRetryAfterMilliseconds(response.headers.get("retry-after"))
      : undefined;
    await response.body?.cancel().catch(() => undefined);
    const detail =
      response.statusText.length > 0 ? ` ${response.statusText}` : "";
    const error = new Error(
      `Sample audio request failed (HTTP ${status}${detail})`,
    );
    if (retryAfterMs !== undefined) {
      throw new RetryableSampleAudioError(
        error.message,
        error,
        retryAfterMs,
      );
    }
    throw error;
  }

  const declaredLength = response.headers.get("content-length");
  if (
    declaredLength !== null &&
    Number(declaredLength) !== SAMPLE_AUDIO_BYTE_LENGTH
  ) {
    await response.body?.cancel().catch(() => undefined);
    throw new Error("Sample audio response had an unexpected byte length");
  }

  let bytes: ArrayBuffer;
  try {
    bytes = await response.arrayBuffer();
  } catch (error) {
    signal.throwIfAborted();
    throw new RetryableSampleAudioError(
      "Sample audio response ended before it was complete",
      error,
    );
  }
  signal.throwIfAborted();
  if (bytes.byteLength < SAMPLE_AUDIO_BYTE_LENGTH) {
    throw new RetryableSampleAudioError(
      "Sample audio response ended before it was complete",
    );
  }
  if (bytes.byteLength > SAMPLE_AUDIO_BYTE_LENGTH) {
    throw new Error("Sample audio download had an unexpected byte length");
  }

  const digest = await crypto.subtle.digest("SHA-256", bytes);
  signal.throwIfAborted();
  if (hexadecimalBytes(digest) !== SAMPLE_AUDIO_SHA256) {
    throw new Error("Sample audio integrity validation failed");
  }
  return bytes;
}

class RetryableSampleAudioError extends Error {
  readonly retryAfterMs: number;

  constructor(message: string, cause?: unknown, retryAfterMs = 0) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = "RetryableSampleAudioError";
    this.retryAfterMs = retryAfterMs;
  }
}

async function abortableDelay(
  milliseconds: number,
  signal: AbortSignal,
): Promise<void> {
  signal.throwIfAborted();
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(finish, milliseconds);
    const abort = (): void => {
      window.clearTimeout(timeout);
      signal.removeEventListener("abort", abort);
      reject(signal.reason);
    };
    function finish(): void {
      signal.removeEventListener("abort", abort);
      resolve();
    }
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
}

function hexadecimalBytes(bytes: ArrayBuffer): string {
  return Array.from(
    new Uint8Array(bytes),
    (value) => value.toString(16).padStart(2, "0"),
  ).join("");
}

async function validateSelectedFile(): Promise<void> {
  const revision = ++fileValidationRevision;
  selectedFileIsValid = false;
  updateActionAvailability();
  const file = input.files?.[0];
  if (file === undefined) {
    if (!isBusy) {
      if (transcriber === undefined) {
        setStatus(
          initializationFailure ?? "Parakeet is unavailable in this browser.",
          "error",
        );
      } else {
        setStatus("", "ready");
      }
    }
    return;
  }

  if (file.size === 0) {
    if (!isBusy) setStatus("Choose a non-empty audio file.", "error");
    updateActionAvailability();
    return;
  }

  if (!isBusy) setStatus("Inspecting the audio format…", "busy");
  try {
    const needsDecoding = await requiresAudioDecoding(file);
    if (revision !== fileValidationRevision) return;
    const decodingUnavailable = needsDecoding
      ? audioDecodingUnavailableMessage()
      : undefined;
    if (decodingUnavailable !== undefined) {
      if (!isBusy) setStatus(decodingUnavailable, "error");
      return;
    }
    selectedFileIsValid = true;
    if (!isBusy) {
      if (transcriber === undefined) {
        setStatus(
          initializationFailure ?? "Parakeet is unavailable in this browser.",
          "error",
        );
      } else {
        setStatus("", "ready");
      }
    }
  } catch (error) {
    if (revision !== fileValidationRevision) return;
    if (!isBusy) setStatus(errorMessage(error), "error");
  } finally {
    if (revision === fileValidationRevision) updateActionAvailability();
  }
}

async function transcribeSelectedFile(): Promise<void> {
  const file = input.files?.[0];
  if (file === undefined || !selectedFileIsValid) {
    setStatus("Choose a non-empty audio file first.", "error");
    return;
  }
  if (transcriber === undefined) {
    setStatus("Parakeet is unavailable in this browser.", "error");
    return;
  }

  const abortController = new AbortController();
  activeAbortController = abortController;
  setBusy(true);
  resetProgressElapsedTimer();
  displayedTranscriptionProgressKey = undefined;
  modelDownloadFraction = 0;
  transcriptionDetails = undefined;
  liveTailFollow = false;
  releaseAudioPlayback();
  summary.hidden = true;
  transcriptPanel.hidden = true;
  output.hidden = true;
  output.scrollTop = 0;
  output.dataset.state = "provisional";
  output.dataset.hasTimestamps = "false";
  output.setAttribute("aria-busy", "true");
  hideTranscriptSubtitle();
  setTranscriptCopyAvailable(false);
  transcriptRenderer.reset();
  hideWordTooltip();
  setStatus("", "ready");
  setIndeterminateProgress("Transcribing", true);
  updateRuntimeDetails();

  try {
    const result = await transcriber.transcribe(file, {
      signal: abortController.signal,
      onProgress: updateTranscriptionProgress,
      onPartialResult(snapshot: TranscriptionSnapshot) {
        if (transcriptPanel.hidden && snapshot.text.length > 0) {
          transcriptPanel.hidden = false;
          output.hidden = false;
        }
      },
      onPacedTranscript: renderPacedTranscriptUpdate,
    });

    if (
      !transcriptRenderer.isFinal ||
      transcriptRenderer.text !== result.text
    ) {
      throw new Error("The paced transcript did not match the final result.");
    }

    transcriptPanel.hidden = false;
    output.hidden = false;
    output.dataset.state = "final";
    output.setAttribute("aria-busy", "false");
    showFinalTranscriptSubtitle();
    setTranscriptCopyAvailable(result.text.length > 0);
    prepareAudioPlayback(file);
    transcriptionDetails = result.metrics;
    summaryDuration.textContent = formatDuration(
      result.metrics.audioDurationSeconds,
    );
    summaryProcessingTime.textContent = formatElapsedMilliseconds(
      result.metrics.totalMs,
    );
    summarySpeed.textContent =
      `${result.metrics.speedFactor.toFixed(1)}x real-time`;
    summary.hidden = false;
    setStatus("", "success");
    hideProgressPanel();
    await refreshDownloadedModelInfo();
  } catch (error) {
    output.setAttribute("aria-busy", "false");
    if (abortController.signal.aborted || isAbortError(error)) {
      output.dataset.state = "cancelled";
      hideTranscriptSubtitle();
      setStatus("Cancelled.", "ready");
      setDeterminateProgress(clampFraction(progressElement.value), "Cancelled");
    } else {
      output.dataset.state = "error";
      hideTranscriptSubtitle();
      setStatus(errorMessage(error), "error");
      setDeterminateProgress(
        clampFraction(progressElement.value),
        "Transcription failed",
      );
      transcriptionDetails = { error: serializeError(error) };

      if (transcriber.diagnostics === null) {
        transcriber.dispose();
        transcriber = makeTranscriber();
      }
    }
  } finally {
    if (activeAbortController === abortController) {
      activeAbortController = undefined;
    }
    resetProgressElapsedTimer();
    setBusy(false);
    updateRuntimeDetails();
  }
}

function updateTranscriptionProgress(progress: TranscriptionProgress): void {
  if (progress.phase === "done") {
    hideProgressPanel();
    return;
  }

  syncProgressElapsedTimer(progress.metrics.elapsedMs);
  const fraction = clampFraction(progress.fraction);
  const completed = progress.metrics.completedWindows;
  const total = progress.metrics.totalWindows;
  if (fraction === 0 && total === 0) {
    displayedTranscriptionProgressKey = "indeterminate";
    setIndeterminateProgress("Transcribing", true);
    return;
  }

  const percentage = Math.min(99, Math.round(fraction * 100));
  const displayKey = `${percentage}:${completed}:${total}`;
  if (displayKey === displayedTranscriptionProgressKey) {
    progressElement.value = fraction;
    return;
  }
  displayedTranscriptionProgressKey = displayKey;
  setDeterminateProgress(fraction, "Transcribing", true);
}

function updateModelLoadProgress(progress: ModelLoadProgress): void {
  // A transcription may join the silent cache-only warmup. Those local reads
  // must not be presented as a model download.
  if (!isBusy || cachedWarmupTranscriber === transcriber) {
    return;
  }

  switch (progress.phase) {
    case "manifest":
      setModelDownloadProgress(modelDownloadFraction);
      return;
    case "weights":
      modelDownloadFraction = clampFraction(progress.fraction);
      setModelDownloadProgress(modelDownloadFraction);
      return;
    case "tokenizer":
      setModelDownloadProgress(modelDownloadFraction);
      return;
    case "cache":
    case "retry":
      modelDownloadFraction = 0;
      setModelDownloadProgress(modelDownloadFraction);
      return;
    case "ready":
      modelDownloadFraction = 1;
      setModelDownloadProgress(modelDownloadFraction);
      return;
    case "pipelines":
      setIndeterminateProgress("Transcribing", true);
      return;
    case "webgpu":
    case "wasm":
      resetProgressElapsedTimer();
      return;
  }
}

function renderPacedTranscriptUpdate(update: PacedTranscriptUpdate): void {
  transcriptRenderer.applyUpdate(update);
  if (liveTailFollow) transcriptRenderer.jumpToLatest();
  if (
    update.textSplice.deleteCount > 0 ||
    update.wordSplice.deleteCount > 0
  ) {
    hideWordTooltip();
  }

  const hasTimestamps =
    transcriptRenderer.wordRanges.length > 0 ? "true" : "false";
  if (output.dataset.hasTimestamps !== hasTimestamps) {
    output.dataset.hasTimestamps = hasTimestamps;
  }

  if (update.isFinal) {
    output.dataset.state = "final";
    output.setAttribute("aria-busy", "false");
    showFinalTranscriptSubtitle();
    return;
  }
}

function hideTranscriptSubtitle(): void {
  outputState.hidden = true;
  outputState.textContent = "";
}

function showFinalTranscriptSubtitle(): void {
  outputState.hidden = false;
  outputState.textContent = "Hover for timestamps. Click a word to seek.";
}

function setTranscriptCopyAvailable(available: boolean): void {
  resetTranscriptCopyFeedback();
  copyTranscriptButton.disabled = !available;
  copyTranscriptButton.hidden = !available;
}

async function copyFinalTranscript(): Promise<void> {
  if (!transcriptRenderer.isFinal || transcriptRenderer.text.length === 0) {
    return;
  }

  try {
    await navigator.clipboard.writeText(transcriptRenderer.text);
    setTranscriptCopyFeedback("copied", "Copied", "Transcript copied.");
  } catch {
    setTranscriptCopyFeedback(
      "error",
      "Copy failed",
      "Could not copy transcript.",
    );
  }
}

function setTranscriptCopyFeedback(
  state: "copied" | "error",
  label: string,
  statusText: string,
): void {
  resetTranscriptCopyFeedback();
  copyTranscriptButton.dataset.copyState = state;
  copyTranscriptButton.setAttribute("aria-label", label);
  copyTranscriptButton.title = label;
  copyTranscriptStatus.textContent = statusText;
  transcriptCopyFeedbackTimeout = window.setTimeout(
    resetTranscriptCopyFeedback,
    1_600,
  );
}

function resetTranscriptCopyFeedback(): void {
  if (transcriptCopyFeedbackTimeout !== undefined) {
    clearTimeout(transcriptCopyFeedbackTimeout);
    transcriptCopyFeedbackTimeout = undefined;
  }
  copyTranscriptButton.dataset.copyState = "idle";
  copyTranscriptButton.setAttribute("aria-label", "Copy transcript");
  copyTranscriptButton.title = "Copy transcript";
  copyTranscriptStatus.textContent = "";
}

function handleTranscriptScroll(): void {
  hideWordTooltip();
  if (output.dataset.state !== "provisional") return;
  liveTailFollow = isTranscriptTailReached(output);
}

function seekPlaybackToClickedWord(event: MouseEvent): void {
  if (playbackObjectUrl === undefined || !transcriptRenderer.isFinal) return;
  const hit = transcriptRenderer.wordHitAtPoint(event.clientX, event.clientY);
  if (hit === undefined) return;

  pendingPlaybackSeekSeconds = hit.word.startSeconds;
  followedPlaybackWord = undefined;
  applyPendingPlaybackSeek();
}

function prepareAudioPlayback(file: File): void {
  releaseAudioPlayback();
  playbackObjectUrl = URL.createObjectURL(file);
  playbackUnavailable.hidden = true;
  audioPlayer.hidden = false;
  audioPlayer.src = playbackObjectUrl;
  playbackRow.hidden = false;
  audioPlayer.load();
}

function releaseAudioPlayback(): void {
  const objectUrl = playbackObjectUrl;
  playbackObjectUrl = undefined;
  pendingPlaybackSeekSeconds = undefined;
  stopPlaybackSync();
  clearPlaybackWord();
  audioPlayer.pause();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  audioPlayer.hidden = false;
  playbackUnavailable.hidden = true;
  playbackRow.hidden = true;
  if (objectUrl !== undefined) URL.revokeObjectURL(objectUrl);
}

function handlePlaybackError(): void {
  const objectUrl = playbackObjectUrl;
  if (objectUrl === undefined) return;

  playbackObjectUrl = undefined;
  pendingPlaybackSeekSeconds = undefined;
  stopPlaybackSync();
  clearPlaybackWord();
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  audioPlayer.hidden = true;
  playbackUnavailable.hidden = false;
  URL.revokeObjectURL(objectUrl);
  if (transcriptRenderer.wordRanges.length > 0) {
    outputState.textContent = "Hover for timestamps.";
  }
}

function applyPendingPlaybackSeek(): void {
  const seconds = pendingPlaybackSeekSeconds;
  if (
    seconds === undefined ||
    audioPlayer.readyState === HTMLMediaElement.HAVE_NOTHING
  ) {
    return;
  }

  const duration = audioPlayer.duration;
  const target =
    Number.isFinite(duration) && duration > 0
      ? Math.min(seconds, duration)
      : seconds;
  try {
    audioPlayer.currentTime = target;
  } catch {
    return;
  }
  pendingPlaybackSeekSeconds = undefined;
  syncPlaybackWord(!audioPlayer.paused && !audioPlayer.ended);
}

function startPlaybackSync(): void {
  stopPlaybackSync();
  followedPlaybackWord = undefined;

  const renderFrame = (): void => {
    playbackSyncFrame = undefined;
    syncPlaybackWord(true);
    if (!audioPlayer.paused && !audioPlayer.ended) {
      playbackSyncFrame = requestAnimationFrame(renderFrame);
    }
  };
  renderFrame();
}

function stopPlaybackSync(): void {
  if (playbackSyncFrame === undefined) return;
  cancelAnimationFrame(playbackSyncFrame);
  playbackSyncFrame = undefined;
}

function syncPlaybackWord(followPlayback: boolean): void {
  if (playbackObjectUrl === undefined || !transcriptRenderer.isFinal) {
    clearPlaybackWord();
    return;
  }

  const word = transcriptWordAtTime(
    transcriptRenderer.wordRanges,
    audioPlayer.currentTime,
  );
  let range: Range | undefined;
  if (word !== activePlaybackWord) {
    playbackWordHighlight.clear();
    activePlaybackWord = word;
    followedPlaybackWord = undefined;
    if (word !== undefined) {
      range = transcriptRenderer.domRangeForWord(word);
      if (range !== undefined) playbackWordHighlight.add(range);
    }
  }

  if (
    followPlayback &&
    word !== undefined &&
    word !== followedPlaybackWord
  ) {
    range ??= transcriptRenderer.domRangeForWord(word);
    if (range !== undefined) {
      revealPlaybackRange(range);
      followedPlaybackWord = word;
    }
  }
}

function clearPlaybackWord(): void {
  playbackWordHighlight.clear();
  activePlaybackWord = undefined;
  followedPlaybackWord = undefined;
}

function revealPlaybackRange(range: Range): void {
  const rectangles = Array.from(range.getClientRects());
  const first = rectangles[0];
  const last = rectangles.at(-1);
  if (first === undefined || last === undefined) return;

  const viewport = output.getBoundingClientRect();
  const margin = 18;
  if (
    first.top >= viewport.top + margin &&
    last.bottom <= viewport.bottom - margin
  ) {
    return;
  }

  const wordCenter = (first.top + last.bottom) / 2;
  const viewportCenter = (viewport.top + viewport.bottom) / 2;
  output.scrollTop += wordCenter - viewportCenter;
}

async function refreshDownloadedModelInfo(): Promise<void> {
  try {
    const info = await getModelCacheInfo();
    downloadedModelDetails = info;
    const record = recordValue(info);
    const supported = record?.supported === true;
    const assetCount = numericValue(record?.assetCount);
    const sizeBytes = numericValue(record?.sizeBytes);
    downloadedModelsCanBeDeleted = supported && assetCount > 0;

    if (!supported) {
      cacheStatus.textContent =
        "Downloaded model storage is unavailable in this context.";
    } else {
      cacheStatus.textContent = formatMebibytes(sizeBytes);
    }
  } catch (error) {
    downloadedModelsCanBeDeleted = false;
    downloadedModelDetails = { error: serializeError(error) };
    cacheStatus.textContent =
      `Could not inspect downloaded model: ${errorMessage(error)}`;
  } finally {
    updateActionAvailability();
    updateRuntimeDetails();
  }
}

async function deleteDownloadedModels(): Promise<void> {
  if (isBusy || !downloadedModelsCanBeDeleted) return;
  deleteModelsButton.disabled = true;
  cacheStatus.textContent = "Deleting downloaded model…";
  try {
    await deleteCachedModels();
    await refreshDownloadedModelInfo();
  } catch (error) {
    cacheStatus.textContent =
      `Could not delete downloaded model: ${errorMessage(error)}`;
  } finally {
    updateActionAvailability();
  }
}

function setBusy(busy: boolean): void {
  isBusy = busy;
  input.disabled = busy;
  cancelButton.disabled = !busy;
  updateActionAvailability();
}

function updateActionAvailability(): void {
  transcribeButton.disabled =
    isBusy ||
    sampleAudioDownloadPending ||
    transcriber === undefined ||
    !selectedFileIsValid;
  sampleAudioButton.disabled =
    isBusy || sampleAudioDownloadPending || transcriber === undefined;
  deleteModelsButton.disabled = isBusy || !downloadedModelsCanBeDeleted;
}

function setStatus(
  message: string,
  state: "busy" | "ready" | "success" | "error",
): void {
  setStatusElement(status, message, state);
}

function setSampleStatus(
  message: string,
  state: "busy" | "ready" | "success" | "error",
): void {
  setStatusElement(sampleStatus, message, state);
}

function setStatusElement(
  element: HTMLElement,
  message: string,
  state: "busy" | "ready" | "success" | "error",
): void {
  element.textContent = message;
  element.classList.toggle("is-busy", state === "busy");
  element.classList.toggle("is-success", state === "success");
  element.classList.toggle("is-error", state === "error");
}

function setDeterminateProgress(
  value: number,
  title: string,
  showElapsed = false,
): void {
  const fraction = clampFraction(value);
  progressPanel.hidden = false;
  progressElement.max = 1;
  progressElement.value = fraction;
  setProgressTitle(title);
  setProgressElapsedVisible(showElapsed);
  progressPercent.textContent = `${Math.round(fraction * 100)}%`;
}

function setIndeterminateProgress(
  title: string,
  showElapsed = false,
): void {
  progressPanel.hidden = false;
  progressElement.removeAttribute("value");
  setProgressTitle(title);
  setProgressElapsedVisible(showElapsed);
  progressPercent.textContent = "";
}

function setModelDownloadProgress(value: number): void {
  const fraction = clampFraction(value);
  resetProgressElapsedTimer();
  progressPanel.hidden = false;
  progressElement.max = 1;
  progressElement.value = fraction;
  setProgressTitle(
    "Downloading model",
    `(${(fraction * 100).toFixed(1)}%)`,
  );
  progressPercent.textContent = "";
}

function setProgressTitle(title: string, detail?: string): void {
  progressTitleLabel.textContent = title;
  progressTitleDetail.textContent = detail ?? "";
  progressTitleDetail.hidden = detail === undefined;
}

function hideProgressPanel(): void {
  resetProgressElapsedTimer();
  progressPanel.hidden = true;
}

function syncProgressElapsedTimer(reportedMilliseconds: number): void {
  const now = performance.now();
  const reported =
    Number.isFinite(reportedMilliseconds) && reportedMilliseconds > 0
      ? reportedMilliseconds
      : 0;
  const current = currentProgressElapsedMilliseconds(now);
  progressElapsedAnchorMilliseconds = Math.max(current ?? 0, reported);
  progressElapsedAnchorNow = now;
  if (progressElapsedInterval === undefined) {
    progressElapsedInterval = window.setInterval(
      renderProgressElapsedTimer,
      250,
    );
  }
  renderProgressElapsedTimer();
}

function currentProgressElapsedMilliseconds(
  now = performance.now(),
): number | undefined {
  const anchor = progressElapsedAnchorMilliseconds;
  if (anchor === undefined) return undefined;
  return anchor + Math.max(0, now - progressElapsedAnchorNow);
}

function renderProgressElapsedTimer(): void {
  const milliseconds = currentProgressElapsedMilliseconds();
  if (milliseconds === undefined) return;
  const text = formatProgressElapsedMilliseconds(milliseconds);
  if (text === displayedProgressElapsedText) return;
  displayedProgressElapsedText = text;
  progressElapsed.textContent = text;
  progressElapsed.dateTime = `PT${Math.floor(milliseconds / 1_000)}S`;
}

function setProgressElapsedVisible(visible: boolean): void {
  const hasElapsedTime = progressElapsedAnchorMilliseconds !== undefined;
  progressElapsed.hidden = !visible || !hasElapsedTime;
}

function resetProgressElapsedTimer(): void {
  if (progressElapsedInterval !== undefined) {
    window.clearInterval(progressElapsedInterval);
    progressElapsedInterval = undefined;
  }
  progressElapsedAnchorMilliseconds = undefined;
  progressElapsedAnchorNow = 0;
  displayedProgressElapsedText = undefined;
  progressElapsed.textContent = "";
  progressElapsed.removeAttribute("datetime");
  progressElapsed.hidden = true;
}

function renderWordTooltip(): void {
  tooltipRenderFrame = undefined;
  const point = pendingTooltipPoint;
  pendingTooltipPoint = undefined;
  if (point === undefined) return;

  let hit = activeWordHit;
  if (
    hit === undefined ||
    !pointIntersectsRectangles(hit.clientRects, point.clientX, point.clientY)
  ) {
    hit = transcriptRenderer.wordHitAtPoint(point.clientX, point.clientY);
  }

  if (hit === undefined) {
    hideWordTooltip();
    return;
  }

  if (activeWordHit?.word !== hit.word) {
    activeWordHit = hit;
    wordTooltip.textContent =
      `Start ${formatTimestamp(hit.word.startSeconds)} · ` +
      `End ${formatTimestamp(hit.word.endSeconds)}`;
    wordTooltip.hidden = false;
    tooltipWidth = wordTooltip.offsetWidth;
    tooltipHeight = wordTooltip.offsetHeight;
  }
  positionWordTooltip(
    point.clientX,
    point.clientY,
    tooltipWidth,
    tooltipHeight,
  );
}

function pointIntersectsRectangles(
  rectangles: readonly DOMRect[],
  clientX: number,
  clientY: number,
): boolean {
  for (const rectangle of rectangles) {
    if (
      clientX >= rectangle.left - 1 &&
      clientX <= rectangle.right + 1 &&
      clientY >= rectangle.top - 1 &&
      clientY <= rectangle.bottom + 1
    ) {
      return true;
    }
  }
  return false;
}

function positionWordTooltip(
  clientX: number,
  clientY: number,
  width: number,
  height: number,
): void {
  const margin = 8;
  const gap = 12;
  const left = Math.min(
    window.innerWidth - width - margin,
    Math.max(margin, clientX + gap),
  );
  const above = clientY - height - gap;
  const top =
    above >= margin
      ? above
      : Math.min(window.innerHeight - height - margin, clientY + gap);
  wordTooltip.style.transform =
    `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
}

function hideWordTooltip(): void {
  pendingTooltipPoint = undefined;
  if (tooltipRenderFrame !== undefined) {
    cancelAnimationFrame(tooltipRenderFrame);
    tooltipRenderFrame = undefined;
  }
  activeWordHit = undefined;
  wordTooltip.hidden = true;
}

function updateRuntimeDetails(): void {
  metrics.textContent = printableJson({
    support: supportDetails,
    runtime: transcriber?.diagnostics ?? null,
    transcription: transcriptionDetails ?? null,
    downloadedModel: downloadedModelDetails,
  });
}

function unsupportedMessage(
  support: Record<string, unknown> | undefined,
): string {
  const errors = support?.errors;
  if (Array.isArray(errors)) {
    const messages = errors.filter(
      (value): value is string => typeof value === "string",
    );
    if (messages.length > 0) return messages.join(" ");
  }
  const reason = support?.reason;
  return typeof reason === "string" && reason.length > 0
    ? reason
    : "This browser or GPU does not meet parakeet.wgsl's WebGPU requirements.";
}

function audioDecodingUnavailableMessage(): string | undefined {
  const support = recordValue(supportDetails);
  const decoding = recordValue(support?.audioDecoding);
  if (decoding?.available !== false) return undefined;
  const reason = decoding.reason;
  return typeof reason === "string" && reason.length > 0
    ? reason
    : "This browser can transcribe canonical WAV files, but it cannot decode this audio format.";
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : undefined;
}

function numericValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : 0;
}

function clampFraction(value: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
}

function formatMebibytes(bytes: number): string {
  const value = Number.isFinite(bytes) && bytes > 0 ? bytes : 0;
  return `${(value / 1_048_576).toFixed(1)} MiB`;
}

function formatDuration(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const roundedSeconds = Math.floor(safeSeconds);
  const hours = Math.floor(roundedSeconds / 3_600);
  const minutes = Math.floor((roundedSeconds % 3_600) / 60);
  const remainder = roundedSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m ${remainder}s`;
  if (minutes > 0) return `${minutes}m ${remainder}s`;
  return `${remainder}s`;
}

function formatElapsedMilliseconds(milliseconds: number): string {
  const safeMilliseconds =
    Number.isFinite(milliseconds) && milliseconds > 0 ? milliseconds : 0;
  const seconds = safeMilliseconds / 1_000;
  if (seconds < 60) return `${seconds.toFixed(2)}s`;

  const wholeMinutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds - wholeMinutes * 60;
  const formattedSeconds = remainingSeconds.toFixed(2).padStart(5, "0");
  const hours = Math.floor(wholeMinutes / 60);
  const minutes = wholeMinutes % 60;
  return hours > 0
    ? `${hours}h ${minutes}m ${formattedSeconds}s`
    : `${minutes}m ${formattedSeconds}s`;
}

function formatTimestamp(seconds: number): string {
  const safeMilliseconds = Math.max(
    0,
    Math.round((Number.isFinite(seconds) ? seconds : 0) * 1_000),
  );
  const milliseconds = safeMilliseconds % 1_000;
  const totalSeconds = Math.floor(safeMilliseconds / 1_000);
  const second = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minute = totalMinutes % 60;
  const hour = Math.floor(totalMinutes / 60);
  const clock =
    hour > 0
      ? `${hour}:${pad2(minute)}:${pad2(second)}`
      : `${pad2(minute)}:${pad2(second)}`;
  return `${clock}.${milliseconds.toString().padStart(3, "0")}`;
}

function pad2(value: number): string {
  return value.toString().padStart(2, "0");
}

function printableJson(value: unknown): string {
  try {
    return (
      JSON.stringify(
        value,
        (key, nestedValue) =>
          key === "audioDecoding" ? undefined : nestedValue,
        2,
      ) ?? "{}"
    );
  } catch {
    return "Runtime details are not serializable.";
  }
}

function serializeError(error: unknown): Record<string, unknown> {
  if (!(error instanceof Error)) return { message: String(error) };
  const code = "code" in error ? (error as { code?: unknown }).code : undefined;
  return {
    name: error.name,
    message: error.message,
    ...(typeof code === "string" ? { code } : {}),
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing #${id}`);
  return element as T;
}
