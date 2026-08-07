const PUBLIC_PACKAGE_URL = "/dist/index.js";

class BenchmarkBrowserError extends Error {
  constructor(code, message, details = {}, options = undefined) {
    super(`[${code}] ${message}`, options);
    this.name = "BenchmarkBrowserError";
    this.code = code;
    this.details = details;
  }
}

const statusElement = document.querySelector("#status");
if (!(statusElement instanceof HTMLParagraphElement)) {
  throw new Error("Benchmark page is missing its status element");
}

let phase = "starting";
let support = null;
let publicPackage;
let transcriber;
let activeJobId = null;
let completedJobs = 0;
let acceptingJobs = true;
let queueTail = Promise.resolve();
let disposePromise;

const ready = bootstrap();

const automationApi = Object.freeze({
  version: 1,
  ready,
  transcribe,
  diagnostics: diagnosticsSnapshot,
  state: stateSnapshot,
  dispose,
  serializeError,
});

Object.defineProperty(window, "parakeetBenchmark", {
  value: automationApi,
  enumerable: true,
  configurable: false,
  writable: false,
});

window.addEventListener("pagehide", (event) => {
  if (!event.persisted) {
    acceptingJobs = false;
    transcriber?.dispose();
    transcriber = undefined;
    phase = "disposed";
  }
});

async function bootstrap() {
  try {
    publicPackage = await import(PUBLIC_PACKAGE_URL);
    assertPublicPackage(publicPackage);

    support = await publicPackage.checkSupport();
    if (!isSupportResult(support)) {
      throw new BenchmarkBrowserError(
        "INVALID_SUPPORT_RESULT",
        "checkSupport() returned an invalid result",
      );
    }
    if (!support.supported) {
      throw new BenchmarkBrowserError(
        "UNSUPPORTED_BROWSER",
        support.errors.join(" ") || "This browser cannot run parakeet.wgsl",
        { support },
      );
    }

    transcriber = publicPackage.createTranscriber();
    phase = "ready";
    statusElement.textContent = "Ready for benchmark automation.";
    document.documentElement.dataset.benchmarkState = phase;
    return diagnosticsSnapshot();
  } catch (error) {
    phase = error instanceof BenchmarkBrowserError &&
        error.code === "UNSUPPORTED_BROWSER"
      ? "unsupported"
      : "failed";
    document.documentElement.dataset.benchmarkState = phase;
    const serialized = serializeError(error);
    statusElement.textContent = serialized.message;
    throw error;
  }
}

function transcribe(request) {
  if (!acceptingJobs) {
    return Promise.reject(disposedError());
  }

  let input;
  try {
    input = validateRequest(request);
  } catch (error) {
    return Promise.reject(error);
  }

  return enqueue(async () => {
    await ready;
    if (transcriber === undefined || publicPackage === undefined) {
      throw disposedError();
    }

    const file = new File([input.bytes], input.name, {
      type: "audio/wav",
      lastModified: 0,
    });
    let requiresDecoding;
    try {
      requiresDecoding = await publicPackage.requiresAudioDecoding(file);
    } catch (error) {
      throw new BenchmarkBrowserError(
        "AUDIO_VALIDATION_FAILED",
        `Could not validate benchmark input ${input.id}`,
        {
          id: input.id,
          name: input.name,
          cause: serializeError(error),
        },
        { cause: error },
      );
    }
    if (requiresDecoding) {
      throw new BenchmarkBrowserError(
        "NONCANONICAL_AUDIO",
        "Benchmark input must be a 16 kHz mono signed PCM16 WAV",
        { id: input.id, name: input.name, byteLength: file.size },
      );
    }

    activeJobId = input.id;
    phase = "transcribing";
    document.documentElement.dataset.benchmarkState = phase;
    statusElement.textContent = `Transcribing ${input.id}…`;

    try {
      const result = await transcriber.transcribe(file);
      if (typeof result?.text !== "string") {
        throw new BenchmarkBrowserError(
          "INVALID_TRANSCRIPTION_RESULT",
          "parakeet.wgsl returned a result without final text",
          { id: input.id, name: input.name },
        );
      }
      completedJobs += 1;
      return {
        id: input.id,
        name: input.name,
        text: result.text,
      };
    } catch (error) {
      if (error instanceof BenchmarkBrowserError) {
        throw error;
      }
      throw new BenchmarkBrowserError(
        "TRANSCRIPTION_FAILED",
        `Transcription failed for ${input.id}`,
        {
          id: input.id,
          name: input.name,
          cause: serializeError(error),
        },
        { cause: error },
      );
    } finally {
      activeJobId = null;
      if (acceptingJobs) {
        phase = "ready";
        document.documentElement.dataset.benchmarkState = phase;
        statusElement.textContent = "Ready for benchmark automation.";
      }
    }
  });
}

function dispose() {
  if (disposePromise !== undefined) {
    return disposePromise;
  }

  acceptingJobs = false;
  phase = "disposing";
  document.documentElement.dataset.benchmarkState = phase;
  statusElement.textContent = "Disposing benchmark runtime…";
  disposePromise = enqueue(async () => {
    try {
      await ready;
    } catch {
      // A failed bootstrap has no usable runtime, but disposal still succeeds.
    }
    transcriber?.dispose();
    transcriber = undefined;
    activeJobId = null;
    phase = "disposed";
    document.documentElement.dataset.benchmarkState = phase;
    statusElement.textContent = "Benchmark runtime disposed.";
  });
  return disposePromise;
}

function enqueue(operation) {
  const job = queueTail.then(operation);
  queueTail = job.then(
    () => undefined,
    () => undefined,
  );
  return job;
}

function validateRequest(request) {
  if (typeof request !== "object" || request === null) {
    throw new BenchmarkBrowserError(
      "INVALID_REQUEST",
      "transcribe() expects an object containing id, name, and bytes",
    );
  }

  const id = validateNonEmptyString(request.id, "id");
  const name = validateNonEmptyString(request.name, "name");
  if (name.includes("\0")) {
    throw new BenchmarkBrowserError(
      "INVALID_NAME",
      "Benchmark input name must not contain a null character",
      { id },
    );
  }

  const bytes = normalizeBytes(request.bytes);
  if (bytes.byteLength === 0) {
    throw new BenchmarkBrowserError(
      "EMPTY_AUDIO",
      "Benchmark input WAV is empty",
      { id, name },
    );
  }
  return { id, name, bytes };
}

function validateNonEmptyString(value, field) {
  if (typeof value !== "string" || value.length === 0 || value.trim() === "") {
    throw new BenchmarkBrowserError(
      "INVALID_REQUEST",
      `Benchmark input ${field} must be a non-empty string`,
    );
  }
  return value;
}

function normalizeBytes(value) {
  if (value instanceof ArrayBuffer) {
    return new Uint8Array(value).slice();
  }
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(
      value.buffer,
      value.byteOffset,
      value.byteLength,
    ).slice();
  }
  throw new BenchmarkBrowserError(
    "INVALID_AUDIO_BYTES",
    "Benchmark input bytes must be an ArrayBuffer or typed-array view",
  );
}

function diagnosticsSnapshot() {
  return {
    support,
    runtime: transcriber?.diagnostics ?? null,
  };
}

function stateSnapshot() {
  return {
    phase,
    acceptingJobs,
    activeJobId,
    completedJobs,
  };
}

function serializeError(error) {
  if (error instanceof BenchmarkBrowserError) {
    return {
      name: error.name,
      code: error.code,
      message: error.message,
      details: error.details,
      ...(typeof error.stack === "string" ? { stack: error.stack } : {}),
    };
  }
  if (error instanceof Error) {
    return {
      name: error.name,
      code: typeof error.code === "string" ? error.code : "UNEXPECTED_ERROR",
      message: error.message,
      ...(typeof error.stack === "string" ? { stack: error.stack } : {}),
    };
  }
  return {
    name: "Error",
    code: "UNEXPECTED_ERROR",
    message: String(error),
  };
}

function assertPublicPackage(value) {
  if (
    typeof value?.checkSupport !== "function" ||
    typeof value?.createTranscriber !== "function" ||
    typeof value?.requiresAudioDecoding !== "function"
  ) {
    throw new BenchmarkBrowserError(
      "INVALID_PACKAGE_BUILD",
      `${PUBLIC_PACKAGE_URL} does not expose the required public API`,
    );
  }
}

function isSupportResult(value) {
  return typeof value === "object" &&
    value !== null &&
    typeof value.supported === "boolean" &&
    Array.isArray(value.errors) &&
    Array.isArray(value.warnings);
}

function disposedError() {
  return new BenchmarkBrowserError(
    "BENCHMARK_DISPOSED",
    "Benchmark runtime has been disposed",
  );
}
