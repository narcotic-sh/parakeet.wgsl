/**
 * Bounded-memory random access for Parakeet's canonical input format.
 *
 * A Blob/File is never materialized as one ArrayBuffer. Header fields are read
 * through small slices and PCM is decoded through one reusable BYOB backing
 * store. `consume` callbacks are synchronous because Chrome transfers the BYOB
 * backing ArrayBuffer on every read.
 */

export const PARAKEET_SAMPLE_RATE = 16_000 as const;
export const PARAKEET_CHANNELS = 1 as const;
export const PARAKEET_BITS_PER_SAMPLE = 16 as const;
export const WAV_REUSABLE_READ_BUFFER_BYTES = 320_000;
export const DEFAULT_PCM_STREAM_SAMPLES = 160_000;

const BYTES_PER_SAMPLE = PARAKEET_BITS_PER_SAMPLE / 8;

export interface RandomAccessByteSource {
  readonly size: number;
  read(offset: number, length: number): Promise<ArrayBuffer>;
}

export interface ReusableChunkByteSource extends RandomAccessByteSource {
  readonly reusableReadBufferBytes: number;
  consume(
    offset: number,
    length: number,
    consumer: (bytes: Uint8Array) => void,
  ): Promise<void>;
}

export interface Pcm16WavInfo {
  readonly sampleRate: typeof PARAKEET_SAMPLE_RATE;
  readonly channels: typeof PARAKEET_CHANNELS;
  readonly bitsPerSample: typeof PARAKEET_BITS_PER_SAMPLE;
  readonly sampleCount: number;
  readonly dataOffset: number;
  readonly dataByteLength: number;
}

interface ParsedWaveFormat {
  readonly audioFormat: number;
  readonly channels: number;
  readonly sampleRate: number;
  readonly byteRate: number;
  readonly blockAlign: number;
  readonly bitsPerSample: number;
}

export class WavFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WavFormatError";
  }
}

/**
 * Return whether Parakeet will need to decode an audio Blob with FFmpeg.
 *
 * This uses the same strict parser as the transcription fast path, so a
 * canonical 16 kHz mono PCM16 WAV is guaranteed to bypass FFmpeg and OPFS.
 */
export async function requiresAudioDecoding(
  audio: Blob,
): Promise<boolean> {
  try {
    await Pcm16WavReader.open(audio);
    return false;
  } catch (error) {
    if (error instanceof WavFormatError) return true;
    throw error;
  }
}

export class BlobByteSource implements ReusableChunkByteSource {
  readonly size: number;
  readonly reusableReadBufferBytes = WAV_REUSABLE_READ_BUFFER_BYTES;

  private reusable = new Uint8Array(WAV_REUSABLE_READ_BUFFER_BYTES);
  private consumeTail: Promise<void> = Promise.resolve();

  constructor(private readonly blob: Blob) {
    this.size = blob.size;
  }

  async read(offset: number, length: number): Promise<ArrayBuffer> {
    validateRange(offset, length, this.size, "byte");
    return this.blob.slice(offset, offset + length).arrayBuffer();
  }

  consume(
    offset: number,
    length: number,
    consumer: (bytes: Uint8Array) => void,
  ): Promise<void> {
    validateRange(offset, length, this.size, "byte");
    const operation = this.consumeTail.then(() =>
      this.consumeUnlocked(offset, length, consumer),
    );
    // Random-access consumers may overlap logically, but exactly one operation
    // owns the persistent BYOB allocation at a time.
    this.consumeTail = operation.catch(() => undefined);
    return operation;
  }

  private async consumeUnlocked(
    offset: number,
    length: number,
    consumer: (bytes: Uint8Array) => void,
  ): Promise<void> {
    if (length === 0) return;

    const reader = this.blob
      .slice(offset, offset + length)
      .stream()
      .getReader({ mode: "byob" });
    let consumed = 0;
    let failure: { readonly error: unknown } | undefined;
    try {
      while (consumed < length) {
        const result = await reader.read(this.reusable);
        const bytes = result.value;
        if (result.done || bytes === undefined || bytes.byteLength === 0) {
          throw new WavFormatError(
            `short Blob read: expected ${length} bytes, got ${consumed}`,
          );
        }
        if (
          bytes.byteOffset !== 0 ||
          bytes.buffer.byteLength !== this.reusableReadBufferBytes ||
          bytes.byteLength > length - consumed
        ) {
          throw new WavFormatError(
            "unexpected Blob BYOB backing-store behavior",
          );
        }

        consumer(bytes);
        consumed += bytes.byteLength;
        // The supplied view is detached after read(). The returned view owns
        // the same allocation under a new ArrayBuffer wrapper.
        this.reusable = new Uint8Array(bytes.buffer);
      }
    } catch (error) {
      failure = { error };
    }
    try {
      if (failure === undefined) await reader.cancel();
      else await reader.cancel(failure.error);
    } catch (error) {
      failure ??= { error };
    }
    try {
      reader.releaseLock();
    } catch (error) {
      failure ??= { error };
    }
    if (this.reusable.byteLength === 0) {
      this.reusable = new Uint8Array(WAV_REUSABLE_READ_BUFFER_BYTES);
    }
    if (failure !== undefined) throw failure.error;
  }
}

/**
 * Bounded random-access reader for 16 kHz, mono PCM16 WAV.
 *
 * RIFF chunks are walked by declared size and odd-byte padding, so metadata
 * chunks such as `LIST`, `JUNK`, `bext`, and `iXML` never need to be loaded.
 */
export class Pcm16WavReader {
  readonly info: Pcm16WavInfo;

  private constructor(
    private readonly source: RandomAccessByteSource,
    info: Pcm16WavInfo,
  ) {
    this.info = info;
  }

  static async open(
    input: Blob | RandomAccessByteSource,
  ): Promise<Pcm16WavReader> {
    const source = isRandomAccessByteSource(input)
      ? input
      : new BlobByteSource(input);
    return new Pcm16WavReader(source, await parsePcm16Wav(source));
  }

  get sampleRate(): typeof PARAKEET_SAMPLE_RATE {
    return this.info.sampleRate;
  }

  get sampleCount(): number {
    return this.info.sampleCount;
  }

  get reusableReadBufferBytes(): number {
    return isReusableChunkByteSource(this.source)
      ? this.source.reusableReadBufferBytes
      : 0;
  }

  /**
   * Decode a requested range into caller-owned storage. The request is clipped
   * at EOF and the returned number is the exact count written.
   */
  async readSamplesInto(
    startSample: number,
    target: Float32Array,
    targetOffset = 0,
    sampleCount = target.length - targetOffset,
  ): Promise<number> {
    validateSampleRequest(startSample, sampleCount);
    if (!Number.isSafeInteger(targetOffset) || targetOffset < 0) {
      throw new RangeError("targetOffset must be a non-negative safe integer");
    }
    if (targetOffset + sampleCount > target.length) {
      throw new RangeError("target range exceeds the destination array");
    }

    const count = availableSamples(
      startSample,
      sampleCount,
      this.info.sampleCount,
    );
    if (count === 0) return 0;

    const byteOffset =
      this.info.dataOffset + startSample * BYTES_PER_SAMPLE;
    if (isReusableChunkByteSource(this.source)) {
      await decodeReusablePcm16Range(
        this.source,
        byteOffset,
        count,
        target,
        targetOffset,
      );
      return count;
    }

    let decoded = 0;
    while (decoded < count) {
      const chunkSamples = Math.min(
        DEFAULT_PCM_STREAM_SAMPLES,
        count - decoded,
      );
      const byteLength = chunkSamples * BYTES_PER_SAMPLE;
      const bytes = await this.source.read(
        byteOffset + decoded * BYTES_PER_SAMPLE,
        byteLength,
      );
      if (bytes.byteLength !== byteLength) {
        throw new WavFormatError(
          `short WAV read: expected ${byteLength} bytes, got ${bytes.byteLength}`,
        );
      }
      decodePcm16(
        new Uint8Array(bytes),
        target,
        targetOffset + decoded,
      );
      decoded += chunkSamples;
    }
    return decoded;
  }
}

async function decodeReusablePcm16Range(
  source: ReusableChunkByteSource,
  byteOffset: number,
  sampleCount: number,
  target: Float32Array,
  targetOffset: number,
): Promise<void> {
  let decoded = 0;
  let lowByte: number | undefined;

  await source.consume(
    byteOffset,
    sampleCount * BYTES_PER_SAMPLE,
    (bytes) => {
      let byteIndex = 0;
      if (lowByte !== undefined) {
        const unsigned = lowByte | (bytes[0]! << 8);
        target[targetOffset + decoded] =
          (unsigned >= 0x8000 ? unsigned - 0x1_0000 : unsigned) / 32_768;
        decoded += 1;
        lowByte = undefined;
        byteIndex = 1;
      }

      const pairBytes = bytes.byteLength - ((bytes.byteLength - byteIndex) & 1);
      const view = new DataView(
        bytes.buffer,
        bytes.byteOffset,
        bytes.byteLength,
      );
      for (; byteIndex < pairBytes; byteIndex += BYTES_PER_SAMPLE) {
        target[targetOffset + decoded] =
          view.getInt16(byteIndex, true) / 32_768;
        decoded += 1;
      }
      if (byteIndex < bytes.byteLength) lowByte = bytes[byteIndex];
    },
  );

  if (lowByte !== undefined || decoded !== sampleCount) {
    throw new WavFormatError(
      `reusable WAV read decoded ${decoded}/${sampleCount} samples`,
    );
  }
}

function decodePcm16(
  bytes: Uint8Array,
  target: Float32Array,
  targetOffset: number,
): void {
  if ((bytes.byteLength & 1) !== 0) {
    throw new WavFormatError("PCM16 byte range has an odd length");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < bytes.byteLength; i += BYTES_PER_SAMPLE) {
    target[targetOffset + i / BYTES_PER_SAMPLE] =
      view.getInt16(i, true) / 32_768;
  }
}

async function parsePcm16Wav(
  source: RandomAccessByteSource,
): Promise<Pcm16WavInfo> {
  if (!Number.isSafeInteger(source.size) || source.size < 12) {
    throw new WavFormatError("WAV is too small to contain a RIFF header");
  }

  const riff = await readExactly(source, 0, 12);
  if (fourCc(riff, 0) !== "RIFF" || fourCc(riff, 8) !== "WAVE") {
    throw new WavFormatError("expected a RIFF/WAVE file");
  }

  const declaredRiffBytes = riff.getUint32(4, true) + 8;
  if (declaredRiffBytes > source.size) {
    throw new WavFormatError("RIFF container extends past EOF");
  }
  // Some writers append trailing bytes. They are not part of the RIFF chunk.
  const riffEnd = Math.min(source.size, declaredRiffBytes);

  let format: ParsedWaveFormat | undefined;
  let dataOffset: number | undefined;
  let dataByteLength: number | undefined;
  let offset = 12;
  let chunks = 0;

  while (offset + 8 <= riffEnd) {
    chunks += 1;
    if (chunks > 1_000_000) {
      throw new WavFormatError("unreasonable number of RIFF chunks");
    }

    const header = await readExactly(source, offset, 8);
    const chunkId = fourCc(header, 0);
    const declaredSize = header.getUint32(4, true);
    const chunkDataOffset = offset + 8;
    const remaining = riffEnd - chunkDataOffset;
    if (declaredSize > remaining) {
      throw new WavFormatError(`WAV chunk ${chunkId} extends past EOF`);
    }

    if (chunkId === "fmt ") {
      if (declaredSize < 16) {
        throw new WavFormatError("WAV fmt chunk is shorter than 16 bytes");
      }
      const fmt = await readExactly(source, chunkDataOffset, 16);
      format = {
        audioFormat: fmt.getUint16(0, true),
        channels: fmt.getUint16(2, true),
        sampleRate: fmt.getUint32(4, true),
        byteRate: fmt.getUint32(8, true),
        blockAlign: fmt.getUint16(12, true),
        bitsPerSample: fmt.getUint16(14, true),
      };
    } else if (chunkId === "data" && dataOffset === undefined) {
      dataOffset = chunkDataOffset;
      dataByteLength = declaredSize;
    }

    if (format !== undefined && dataOffset !== undefined) break;

    const paddedSize = declaredSize + (declaredSize & 1);
    const nextOffset = chunkDataOffset + paddedSize;
    if (
      !Number.isSafeInteger(nextOffset) ||
      nextOffset <= offset ||
      nextOffset > riffEnd
    ) {
      throw new WavFormatError("invalid RIFF chunk size or padding");
    }
    offset = nextOffset;
  }

  if (
    format === undefined ||
    dataOffset === undefined ||
    dataByteLength === undefined
  ) {
    throw new WavFormatError("WAV must contain fmt and data chunks");
  }

  validateParakeetFormat(format);
  if ((dataByteLength & 1) !== 0) {
    throw new WavFormatError("PCM16 data chunk has an odd byte length");
  }

  const sampleCount = dataByteLength / BYTES_PER_SAMPLE;
  return {
    sampleRate: PARAKEET_SAMPLE_RATE,
    channels: PARAKEET_CHANNELS,
    bitsPerSample: PARAKEET_BITS_PER_SAMPLE,
    sampleCount,
    dataOffset,
    dataByteLength,
  };
}

function validateParakeetFormat(format: ParsedWaveFormat): void {
  if (format.audioFormat !== 1) {
    throw new WavFormatError(
      `unsupported WAV encoding ${format.audioFormat}; expected integer PCM`,
    );
  }
  if (format.channels !== PARAKEET_CHANNELS) {
    throw new WavFormatError(
      `unsupported channel count ${format.channels}; expected mono`,
    );
  }
  if (format.sampleRate !== PARAKEET_SAMPLE_RATE) {
    throw new WavFormatError(
      `unsupported sample rate ${format.sampleRate}; expected 16000 Hz`,
    );
  }
  if (format.bitsPerSample !== PARAKEET_BITS_PER_SAMPLE) {
    throw new WavFormatError(
      `unsupported bit depth ${format.bitsPerSample}; expected PCM16`,
    );
  }
  if (format.blockAlign !== BYTES_PER_SAMPLE) {
    throw new WavFormatError(
      `invalid PCM16 block alignment ${format.blockAlign}`,
    );
  }
  if (format.byteRate !== PARAKEET_SAMPLE_RATE * BYTES_PER_SAMPLE) {
    throw new WavFormatError(`invalid PCM16 byte rate ${format.byteRate}`);
  }
}

async function readExactly(
  source: RandomAccessByteSource,
  offset: number,
  length: number,
): Promise<DataView> {
  validateRange(offset, length, source.size, "byte");
  const bytes = await source.read(offset, length);
  if (bytes.byteLength !== length) {
    throw new WavFormatError(
      `short WAV read: expected ${length} bytes, got ${bytes.byteLength}`,
    );
  }
  return new DataView(bytes);
}

function fourCc(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

function isRandomAccessByteSource(
  input: Blob | RandomAccessByteSource,
): input is RandomAccessByteSource {
  return "read" in input && typeof input.read === "function";
}

function isReusableChunkByteSource(
  source: RandomAccessByteSource,
): source is ReusableChunkByteSource {
  return (
    "consume" in source &&
    typeof source.consume === "function" &&
    "reusableReadBufferBytes" in source &&
    source.reusableReadBufferBytes === WAV_REUSABLE_READ_BUFFER_BYTES
  );
}

function validateRange(
  offset: number,
  length: number,
  sourceSize: number,
  unit: string,
): void {
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new RangeError(`${unit} offset must be a non-negative safe integer`);
  }
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new RangeError(`${unit} length must be a non-negative safe integer`);
  }
  if (offset + length > sourceSize) {
    throw new RangeError(`${unit} range exceeds source size`);
  }
}

function validateSampleRequest(
  startSample: number,
  sampleCount: number,
): void {
  if (!Number.isSafeInteger(startSample) || startSample < 0) {
    throw new RangeError("startSample must be a non-negative safe integer");
  }
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError("sampleCount must be a non-negative safe integer");
  }
}

function availableSamples(
  startSample: number,
  requestedSamples: number,
  totalSamples: number,
): number {
  return startSample >= totalSamples
    ? 0
    : Math.min(requestedSamples, totalSamples - startSample);
}
