import {
  assertAudioDecodingOpfsAvailable,
  removeAudioDecodingScratchDirectory,
} from "./audio-scratch";
import type { AudioDecodingRuntimeAssets } from "./audio-decoding-assets";
import {
  isStreamingDecoderControlResponse,
  type StartStreamingDecodeRequest,
} from "./streaming-decoder-protocol";

export interface StreamingAudioDecoderSession {
  readonly inputByteLength: number;
  /** Advisory only; the inference worker reconciles against exact PCM EOF. */
  readonly estimatedSampleCount: number | null;
  /** Transfer exactly once to the inference worker's prepare-stream message. */
  takePort(): MessagePort;
  /** Idempotently terminate decode, close the local port, and remove OPFS data. */
  cleanup(): Promise<void>;
}

/**
 * Open and validate a compressed/container source before model initialization.
 * Once ready, the isolated decoder immediately spools immutable PCM segments;
 * its data port can queue disk-backed File snapshots while the model loads.
 */
export async function openStreamingAudioDecoder(
  audio: Blob,
  sourceName: string,
  assets: AudioDecodingRuntimeAssets,
  signal: AbortSignal,
  onWorkerFailure?: (error: unknown) => void,
): Promise<StreamingAudioDecoderSession> {
  signal.throwIfAborted();
  assertAudioDecodingOpfsAvailable();

  const scratchDirectory = `${Date.now()}-${crypto.randomUUID()}`;
  const requestId = crypto.randomUUID();
  const channel = new MessageChannel();
  // Keep the standards-based constructor shape intact so package and
  // consuming bundlers retain this lazy graph as a dedicated module worker.
  const worker = new Worker(
    new URL("./streaming-decoder.worker.ts", import.meta.url),
    {
      name: "parakeet-wgsl-streaming-audio-decoder",
      type: "module",
    },
  );
  let ready = false;
  let estimatedSampleCount: number | null = null;
  let portTaken = false;
  let decoderPortTransferred = false;
  let cleanupPromise: Promise<void> | undefined;

  const cleanup = (): Promise<void> => {
    cleanupPromise ??= (async () => {
      signal.removeEventListener("abort", handleAbort);
      worker.removeEventListener("error", handleLateWorkerError);
      worker.terminate();
      if (!portTaken) channel.port1.close();
      if (!decoderPortTransferred) channel.port2.close();
      await removeAudioDecodingScratchDirectory(scratchDirectory, true);
    })();
    return cleanupPromise;
  };
  const handleAbort = (): void => {
    void cleanup();
  };
  const handleLateWorkerError = (event: ErrorEvent): void => {
    const error = new Error(
      event.message || "The streaming audio decoder worker crashed.",
    );
    if (ready) onWorkerFailure?.(error);
  };
  signal.addEventListener("abort", handleAbort, { once: true });
  worker.addEventListener("error", handleLateWorkerError);

  try {
    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const finish = (action: () => void): void => {
        if (settled) return;
        settled = true;
        worker.removeEventListener("message", handleMessage);
        worker.removeEventListener("error", handleOpeningWorkerError);
        signal.removeEventListener("abort", handleOpeningAbort);
        action();
      };
      const handleOpeningAbort = (): void => {
        finish(() => reject(abortReason(signal)));
      };
      const handleMessage = (event: MessageEvent<unknown>): void => {
        const response = event.data;
        if (!isStreamingDecoderControlResponse(response, requestId)) return;
        if (response.type === "streaming-decoder-open-failed") {
          finish(() => reject(new Error(response.message)));
        } else {
          ready = true;
          estimatedSampleCount = response.estimatedSampleCount;
          finish(resolve);
        }
      };
      const handleOpeningWorkerError = (event: ErrorEvent): void => {
        finish(() =>
          reject(
            event.error instanceof Error
              ? event.error
              : new Error(
                  event.message ||
                    "The streaming audio decoder worker crashed while opening.",
                ),
          ),
        );
      };

      worker.addEventListener("message", handleMessage);
      worker.addEventListener("error", handleOpeningWorkerError);
      signal.addEventListener("abort", handleOpeningAbort, { once: true });
      const request: StartStreamingDecodeRequest = {
        type: "start-streaming-decode",
        requestId,
        audio,
        fileName: sourceName,
        ffmpegCoreWasm: assets.ffmpegCoreWasm,
        scratchDirectory,
        port: channel.port2,
      };
      try {
        worker.postMessage(request, [channel.port2]);
        decoderPortTransferred = true;
      } catch (error) {
        finish(() => reject(error));
      }
      if (signal.aborted) handleOpeningAbort();
    });
    signal.throwIfAborted();
    return {
      inputByteLength: audio.size,
      estimatedSampleCount,
      takePort: () => {
        if (portTaken) {
          throw new Error("Streaming audio decoder port was already transferred");
        }
        if (cleanupPromise !== undefined) {
          throw new Error("Streaming audio decoder session was cleaned up");
        }
        portTaken = true;
        return channel.port1;
      },
      cleanup,
    };
  } catch (error) {
    await cleanup();
    throw error;
  }
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted", "AbortError");
}
