/**
 * Fixed first pull: strict lookahead for forty stable non-final primary
 * windows. An EOF before this bound resolves a short file in the same pull.
 */
export const STREAMING_DECODER_INITIAL_SAMPLES =
  39 * 206_080 + 238_080 + 1;

/**
 * Subsequent pulls cover forty primary strides. This keeps segment metadata
 * small while bounding the decoder's live output to about 16.5 MB.
 */
export const STREAMING_DECODER_STEADY_SAMPLES = 40 * 206_080;

export interface StartStreamingDecodeRequest {
  readonly type: "start-streaming-decode";
  readonly requestId: string;
  readonly audio: Blob;
  readonly fileName: string;
  readonly ffmpegCoreWasm: string;
  readonly scratchDirectory: string;
  readonly port: MessagePort;
}

export interface StreamingDecoderReadyResponse {
  readonly type: "streaming-decoder-ready";
  readonly requestId: string;
  /**
   * Best-effort selected-audio duration rescaled to 16 kHz. Codec trimming,
   * timestamp edits, and final resampler drain can change the exact EOF count.
   */
  readonly estimatedSampleCount: number | null;
}

export interface StreamingDecoderOpenFailedResponse {
  readonly type: "streaming-decoder-open-failed";
  readonly requestId: string;
  readonly message: string;
}

export type StreamingDecoderControlResponse =
  | StreamingDecoderReadyResponse
  | StreamingDecoderOpenFailedResponse;

export interface StreamingPcmSegmentMessage {
  readonly type: "pcm-segment";
  readonly segmentIndex: number;
  readonly startSample: number;
  readonly sampleCount: number;
  /** Immutable OPFS snapshot obtained after its sync handle was closed. */
  readonly file: File;
}

export interface StreamingPcmEndMessage {
  readonly type: "pcm-end";
  readonly totalSamples: number;
  readonly outputByteLength: number;
  /** Full worker session, including lazy core startup and OPFS writes. */
  readonly decoderWallMs: number;
  /** Sum of time spent inside bounded native decode steps. */
  readonly decoderActiveMs: number;
  /** Comparable monotonic epoch timestamps across same-origin workers. */
  readonly decoderStartedAtMs: number;
  readonly decoderEndedAtMs: number;
}

export interface StreamingPcmFailureMessage {
  readonly type: "pcm-failed";
  readonly message: string;
}

export type StreamingPcmProducerMessage =
  | StreamingPcmSegmentMessage
  | StreamingPcmEndMessage
  | StreamingPcmFailureMessage;

export function isStreamingDecoderControlResponse(
  value: unknown,
  requestId: string,
): value is StreamingDecoderControlResponse {
  if (!isRecord(value) || value.requestId !== requestId) return false;
  if (value.type === "streaming-decoder-ready") {
    return (
      value.estimatedSampleCount === null ||
      isNonNegativeInteger(value.estimatedSampleCount)
    );
  }
  return (
    value.type === "streaming-decoder-open-failed" &&
    typeof value.message === "string"
  );
}

export function isStreamingPcmProducerMessage(
  value: unknown,
): value is StreamingPcmProducerMessage {
  if (!isRecord(value) || typeof value.type !== "string") return false;
  switch (value.type) {
    case "pcm-segment":
      return (
        isNonNegativeInteger(value.segmentIndex) &&
        isNonNegativeInteger(value.startSample) &&
        isNonNegativeInteger(value.sampleCount) &&
        value.sampleCount > 0 &&
        value.file instanceof File &&
        value.file.size === value.sampleCount * 2
      );
    case "pcm-end":
      return (
        isNonNegativeInteger(value.totalSamples) &&
        isNonNegativeInteger(value.outputByteLength) &&
        value.outputByteLength === value.totalSamples * 2 &&
        isNonNegativeFinite(value.decoderWallMs) &&
        isNonNegativeFinite(value.decoderActiveMs) &&
        isNonNegativeFinite(value.decoderStartedAtMs) &&
        isNonNegativeFinite(value.decoderEndedAtMs) &&
        value.decoderEndedAtMs >= value.decoderStartedAtMs
      );
    case "pcm-failed":
      return typeof value.message === "string";
    default:
      return false;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNonNegativeInteger(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isSafeInteger(value) &&
    value >= 0
  );
}

function isNonNegativeFinite(value: unknown): value is number {
  return (
    typeof value === "number" && Number.isFinite(value) && value >= 0
  );
}
