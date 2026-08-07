import { PARAKEET_SAMPLE_RATE } from "./wav-stream";

export const PARAKEET_FBANK_BINS = 128 as const;
export const PARAKEET_FBANK_HOP_SAMPLES = 160 as const;
export const PARAKEET_FBANK_MAX_INPUT_SAMPLES = 240_000 as const;
export const PARAKEET_FBANK_MAX_FRAMES = 1_501 as const;
export const PARAKEET_FBANK_MAX_BATCH = 40;
export const PARAKEET_FBANK_RMS_FRAME_SAMPLES =
  8 * PARAKEET_FBANK_HOP_SAMPLES;
export const PARAKEET_FBANK_MAX_RMS_FRAMES = Math.floor(
  PARAKEET_FBANK_MAX_INPUT_SAMPLES /
    PARAKEET_FBANK_RMS_FRAME_SAMPLES,
);

const compiledModules = new Map<string, Promise<WebAssembly.Module>>();

interface ParakeetFbankWasmExports extends WebAssembly.Exports {
  readonly memory: WebAssembly.Memory;
  readonly _initialize: () => void;
  readonly parakeet_fbank_init: () => number;
  readonly parakeet_fbank_input_ptr: () => number;
  readonly parakeet_fbank_output_ptr: () => number;
  readonly parakeet_fbank_rms_bits_ptr: () => number;
  readonly parakeet_fbank_compute_rms_bits: (
    sampleOffset: number,
    frameCount: number,
  ) => number;
  readonly parakeet_fbank_begin_batch: (
    batchCount: number,
    sampleCount: number,
  ) => number;
  readonly parakeet_fbank_compute_item: (
    batchIndex: number,
    validSampleCount: number,
    reusableFrameShift: number,
  ) => number;
  readonly parakeet_fbank_finish_batch: () => number;
  readonly parakeet_fbank_reset_reuse: () => void;
  readonly parakeet_fbank_dispose: () => void;
  readonly parakeet_fbank_last_reused_frames: () => number;
  readonly parakeet_fbank_max_batch: () => number;
  readonly parakeet_fbank_max_samples: () => number;
  readonly parakeet_fbank_max_frames: () => number;
  readonly parakeet_fbank_bins: () => number;
  readonly parakeet_fbank_sample_rate: () => number;
  readonly parakeet_fbank_hop_length: () => number;
  readonly parakeet_fbank_rms_frame_samples: () => number;
  readonly parakeet_fbank_max_rms_frames: () => number;
}

export interface ParakeetFbankMemoryStats {
  readonly heapBytes: number;
  readonly inputCapacitySamples: number;
  readonly outputCapacityValues: number;
  readonly maxBatchSize: number;
  readonly maxInputSamples: number;
  readonly maxFramesPerWindow: number;
  readonly rmsFrameSamples: number;
  readonly maxRmsFramesPerWindow: number;
}

export interface ParakeetFbankItem {
  /**
   * Feature-major `[128, frameCount]` data. This aliases one fixed Wasm output
   * item and must be uploaded or copied before computing the next item.
   */
  readonly data: Float32Array<ArrayBuffer>;
  readonly batchIndex: number;
  readonly binCount: typeof PARAKEET_FBANK_BINS;
  readonly frameCount: number;
  /** Valid lengths before the three stride-2 subsampling convolutions. */
  readonly validFrameCount: number;
  readonly reusedRawFrames: number;
  readonly layout: "feature-frame";
}

export interface FbankItemComputeOptions {
  /**
   * Per-item advance, in fixed-hop frames, from the immediately
   * preceding window. Item zero may refer to the final window of the previous
   * call.
   *
   * Set zero unless the PCM overlap is known to be identical. The native
   * kernel still recomputes three frames at both changed boundaries.
   */
  readonly reusableFrameShift?: number;
  /**
   * Real (non-right-padded) samples for each fixed-size model input. The model
   * normalizes over floor(validSamples / hop) frames and zeros the
   * remaining physical centered-STFT frames.
   */
  readonly validSampleCount?: number;
}

export interface FbankWindowDescriptor {
  readonly startSample: number;
  readonly sampleCount: number;
  /** Real source samples before right padding. */
  readonly validSampleCount?: number;
}

/**
 * Derive a safe native reuse hint for two equal-length sliding windows.
 *
 * A zero result means the windows must be computed independently. The native
 * kernel applies the additional boundary-frame exclusion.
 */
export function reusableFrameShiftForWindows(
  previous: FbankWindowDescriptor | undefined,
  current: FbankWindowDescriptor,
): number {
  validateWindowDescriptor(current);
  if (previous === undefined) return 0;
  validateWindowDescriptor(previous);
  if (previous.sampleCount !== current.sampleCount) return 0;

  const sampleShift = current.startSample - previous.startSample;
  if (
    sampleShift <= 0 ||
    sampleShift >= current.sampleCount ||
    sampleShift % PARAKEET_FBANK_HOP_SAMPLES !== 0
  ) {
    return 0;
  }
  const frameShift = sampleShift / PARAKEET_FBANK_HOP_SAMPLES;
  const currentValidFrames = parakeetValidFeatureFrameCount(
    current.validSampleCount ?? current.sampleCount,
  );
  return frameShift < currentValidFrames ? frameShift : 0;
}

export function parakeetFeatureFrameCount(
  sampleCount: number,
): number {
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError("sampleCount must be a non-negative safe integer");
  }
  return Math.floor(sampleCount / PARAKEET_FBANK_HOP_SAMPLES) + 1;
}

export function parakeetValidFeatureFrameCount(
  sampleCount: number,
): number {
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError("sampleCount must be a non-negative safe integer");
  }
  return Math.floor(sampleCount / PARAKEET_FBANK_HOP_SAMPLES);
}

/**
 * Standalone SIMD Wasm frontend for the fixed 240,000-sample, hop-160 source
 * timeline, which produces the model's 1,501-frame physical input.
 *
 * The module has no Emscripten JS glue, allocator, filesystem, or growing
 * memory. One input/output item is reused across a logical batch; callers
 * decode directly into the input and upload each output before advancing.
 */
export class ParakeetFbankWasm {
  readonly memoryStats: ParakeetFbankMemoryStats;
  readonly hopSamples: number;

  private exports: ParakeetFbankWasmExports | undefined;
  private input: Float32Array<ArrayBuffer> | undefined;
  private output: Float32Array<ArrayBuffer> | undefined;
  private rmsBits: Uint32Array<ArrayBuffer> | undefined;
  private activeBatch:
    | {
        readonly batchSize: number;
        readonly sampleCount: number;
        readonly frameCount: number;
        nextBatchIndex: number;
        lastReusedRawFrames: number;
      }
    | undefined;

  private constructor(wasm: ParakeetFbankWasmExports) {
    wasm._initialize();
    if (wasm.parakeet_fbank_init() !== 1) {
      throw new Error("failed to initialize Parakeet FBank Wasm");
    }

    const maxBatchSize = wasm.parakeet_fbank_max_batch();
    const maxInputSamples = wasm.parakeet_fbank_max_samples();
    const maxFramesPerWindow = wasm.parakeet_fbank_max_frames();
    const binCount = wasm.parakeet_fbank_bins();
    const sampleRate = wasm.parakeet_fbank_sample_rate();
    const actualHopSamples = wasm.parakeet_fbank_hop_length();
    const rmsFrameSamples =
      wasm.parakeet_fbank_rms_frame_samples();
    const maxRmsFrames = wasm.parakeet_fbank_max_rms_frames();

    if (
      maxBatchSize !== PARAKEET_FBANK_MAX_BATCH ||
      maxInputSamples !== PARAKEET_FBANK_MAX_INPUT_SAMPLES ||
      maxFramesPerWindow !== PARAKEET_FBANK_MAX_FRAMES ||
      binCount !== PARAKEET_FBANK_BINS ||
      sampleRate !== PARAKEET_SAMPLE_RATE ||
      actualHopSamples !== PARAKEET_FBANK_HOP_SAMPLES ||
      rmsFrameSamples !== PARAKEET_FBANK_RMS_FRAME_SAMPLES ||
      maxRmsFrames !== PARAKEET_FBANK_MAX_RMS_FRAMES
    ) {
      throw new Error(
        "Parakeet FBank Wasm constants do not match the TypeScript runtime",
      );
    }

    const inputCapacitySamples = maxInputSamples;
    const outputCapacityValues = maxFramesPerWindow * binCount;
    this.input = checkedFloat32View(
      wasm.memory,
      wasm.parakeet_fbank_input_ptr(),
      inputCapacitySamples,
      "input",
    );
    this.output = checkedFloat32View(
      wasm.memory,
      wasm.parakeet_fbank_output_ptr(),
      outputCapacityValues,
      "output",
    );
    this.rmsBits = checkedUint32View(
      wasm.memory,
      wasm.parakeet_fbank_rms_bits_ptr(),
      maxRmsFrames,
      "RMS-bit",
    );
    this.exports = wasm;
    this.hopSamples = actualHopSamples;
    this.memoryStats = {
      heapBytes: wasm.memory.buffer.byteLength,
      inputCapacitySamples,
      outputCapacityValues,
      maxBatchSize,
      maxInputSamples,
      maxFramesPerWindow,
      rmsFrameSamples,
      maxRmsFramesPerWindow: maxRmsFrames,
    };
  }

  static async create(
    wasmUrl: string | URL,
    signal?: AbortSignal,
  ): Promise<ParakeetFbankWasm> {
    const module = await loadCompiledModule(wasmUrl, signal);
    signal?.throwIfAborted();
    return ParakeetFbankWasm.fromInstance(
      await WebAssembly.instantiate(module, {}),
    );
  }

  /** Instantiate caller-supplied bytes, primarily for Node/Vitest validation. */
  static async fromBytes(bytes: BufferSource): Promise<ParakeetFbankWasm> {
    const instantiated = await WebAssembly.instantiate(bytes, {});
    const instance =
      instantiated instanceof WebAssembly.Instance
        ? instantiated
        : instantiated.instance;
    return ParakeetFbankWasm.fromInstance(instance);
  }

  static fromInstance(instance: WebAssembly.Instance): ParakeetFbankWasm {
    return new ParakeetFbankWasm(
      instance.exports as ParakeetFbankWasmExports,
    );
  }

  /**
   * Begin a logical batch and return its single reusable PCM input item.
   * Decode one window, call `computePreparedItem`, upload/copy that output,
   * then reuse this same view for the next item.
   */
  beginPreparedBatch(
    batchSize: number,
    sampleCount: number,
  ): Float32Array<ArrayBuffer> {
    this.validateShape(batchSize, sampleCount);
    if (this.activeBatch !== undefined) {
      throw new Error("Parakeet FBank batch is already active");
    }
    const expectedFrameCount = parakeetFeatureFrameCount(sampleCount);
    const actualFrameCount = this.requireExports().parakeet_fbank_begin_batch(
      batchSize,
      sampleCount,
    );
    if (actualFrameCount !== expectedFrameCount) {
      throw new Error(
        `Parakeet FBank Wasm failed to begin batch (${actualFrameCount}); ` +
          `expected ${expectedFrameCount} frames`,
      );
    }
    this.activeBatch = {
      batchSize,
      sampleCount,
      frameCount: actualFrameCount,
      nextBatchIndex: 0,
      lastReusedRawFrames: 0,
    };
    return this.requireInput().subarray(0, sampleCount);
  }

  /**
   * Transform the window already written to the reusable input item.
   *
   * The returned feature view aliases Wasm memory and must be consumed before
   * the next item is computed.
   */
  computePreparedItem(
    batchIndex: number,
    options: FbankItemComputeOptions = {},
  ): ParakeetFbankItem {
    const active = this.requireActiveBatch();
    if (
      !Number.isSafeInteger(batchIndex) ||
      batchIndex !== active.nextBatchIndex
    ) {
      throw new RangeError(
        `FBank item index must be ${active.nextBatchIndex}`,
      );
    }
    const validSampleCount =
      options.validSampleCount ?? active.sampleCount;
    if (
      !Number.isSafeInteger(validSampleCount) ||
      validSampleCount < 0 ||
      validSampleCount > active.sampleCount
    ) {
      throw new RangeError(
        "validSampleCount must be in [0, sampleCount]",
      );
    }
    const validFrameCount =
      parakeetValidFeatureFrameCount(validSampleCount);
    const reusableFrameShift = options.reusableFrameShift ?? 0;
    if (
      !Number.isSafeInteger(reusableFrameShift) ||
      reusableFrameShift < 0 ||
      (reusableFrameShift !== 0 &&
        reusableFrameShift >= validFrameCount)
    ) {
      throw new RangeError(
        "reusableFrameShift must be zero or below the valid frame count",
      );
    }

    const wasm = this.requireExports();
    const actualFrameCount = wasm.parakeet_fbank_compute_item(
      batchIndex,
      validSampleCount,
      reusableFrameShift,
    );
    if (actualFrameCount !== active.frameCount) {
      throw new Error(
        `Parakeet FBank Wasm failed item ${batchIndex} ` +
          `(${actualFrameCount}); expected ${active.frameCount} frames`,
      );
    }
    const totalReused = wasm.parakeet_fbank_last_reused_frames();
    const reusedRawFrames = totalReused - active.lastReusedRawFrames;
    active.lastReusedRawFrames = totalReused;
    active.nextBatchIndex += 1;
    const valueCount = PARAKEET_FBANK_BINS * actualFrameCount;
    return {
      data: this.requireOutput().subarray(0, valueCount),
      batchIndex,
      binCount: PARAKEET_FBANK_BINS,
      frameCount: actualFrameCount,
      validFrameCount,
      reusedRawFrames,
      layout: "feature-frame",
    };
  }

  /**
   * Compute exact ordered-Float32 RMS bit patterns for complete 80 ms frames
   * in the PCM item currently resident in the fixed input arena.
   *
   * The returned view aliases one small Wasm scratch and must be consumed
   * before the next call.
   */
  computeRmsBits(
    sampleOffset: number,
    frameCount: number,
  ): Uint32Array<ArrayBuffer> {
    const active = this.requireActiveBatch();
    if (
      !Number.isSafeInteger(sampleOffset) ||
      sampleOffset < 0 ||
      !Number.isSafeInteger(frameCount) ||
      frameCount < 0 ||
      frameCount > PARAKEET_FBANK_MAX_RMS_FRAMES ||
      sampleOffset +
          frameCount * PARAKEET_FBANK_RMS_FRAME_SAMPLES >
        active.sampleCount
    ) {
      throw new RangeError(
        "RMS frame range must fit the active FBank PCM item",
      );
    }
    const actual = this.requireExports().parakeet_fbank_compute_rms_bits(
      sampleOffset,
      frameCount,
    );
    if (actual !== frameCount) {
      throw new Error(
        `Parakeet FBank Wasm failed RMS range (${actual}); ` +
          `expected ${frameCount} frames`,
      );
    }
    return this.requireRmsBits().subarray(0, frameCount);
  }

  finishPreparedBatch(): void {
    const active = this.requireActiveBatch();
    if (active.nextBatchIndex !== active.batchSize) {
      throw new Error(
        `FBank batch has ${active.nextBatchIndex}/${active.batchSize} items`,
      );
    }
    const actualFrameCount =
      this.requireExports().parakeet_fbank_finish_batch();
    if (actualFrameCount !== active.frameCount) {
      throw new Error(
        `Parakeet FBank Wasm failed to finish batch (${actualFrameCount})`,
      );
    }
    this.activeBatch = undefined;
  }

  compute(
    samples: Float32Array,
    reusableFrameShift = 0,
  ): ParakeetFbankItem {
    const input = this.beginPreparedBatch(1, samples.length);
    input.set(samples);
    const result = this.computePreparedItem(0, { reusableFrameShift });
    this.finishPreparedBatch();
    return result;
  }

  resetReuse(): void {
    this.requireExports().parakeet_fbank_reset_reuse();
    this.activeBatch = undefined;
  }

  dispose(): void {
    if (this.exports === undefined) return;
    this.exports.parakeet_fbank_dispose();
    this.exports = undefined;
    this.input = undefined;
    this.output = undefined;
    this.rmsBits = undefined;
    this.activeBatch = undefined;
  }

  private validateShape(batchSize: number, sampleCount: number): void {
    if (
      !Number.isSafeInteger(batchSize) ||
      batchSize < 1 ||
      batchSize > this.memoryStats.maxBatchSize
    ) {
      throw new RangeError(
        `batchSize must be in [1, ${this.memoryStats.maxBatchSize}]`,
      );
    }
    if (
      !Number.isSafeInteger(sampleCount) ||
      sampleCount < this.hopSamples ||
      sampleCount > this.memoryStats.maxInputSamples
    ) {
      throw new RangeError(
        `sampleCount must be in [${this.hopSamples}, ${this.memoryStats.maxInputSamples}]`,
      );
    }
    if (
      parakeetFeatureFrameCount(sampleCount) >
      this.memoryStats.maxFramesPerWindow
    ) {
      throw new RangeError(
        `sampleCount produces more than ${this.memoryStats.maxFramesPerWindow} frames`,
      );
    }
  }

  private requireExports(): ParakeetFbankWasmExports {
    if (this.exports === undefined) {
      throw new Error("ParakeetFbankWasm has been disposed");
    }
    return this.exports;
  }

  private requireInput(): Float32Array<ArrayBuffer> {
    if (this.input === undefined) {
      throw new Error("ParakeetFbankWasm has been disposed");
    }
    return this.input;
  }

  private requireOutput(): Float32Array<ArrayBuffer> {
    if (this.output === undefined) {
      throw new Error("ParakeetFbankWasm has been disposed");
    }
    return this.output;
  }

  private requireRmsBits(): Uint32Array<ArrayBuffer> {
    if (this.rmsBits === undefined) {
      throw new Error("ParakeetFbankWasm has been disposed");
    }
    return this.rmsBits;
  }

  private requireActiveBatch(): NonNullable<
    ParakeetFbankWasm["activeBatch"]
  > {
    this.requireExports();
    if (this.activeBatch === undefined) {
      throw new Error("Parakeet FBank batch is not active");
    }
    return this.activeBatch;
  }
}

function loadCompiledModule(
  wasmUrl: string | URL,
  signal?: AbortSignal,
): Promise<WebAssembly.Module> {
  const url = String(wasmUrl);
  const cached = compiledModules.get(url);
  if (cached !== undefined) return cached;

  const loading = fetch(
    url,
    signal === undefined ? undefined : { signal },
  ).then(async (response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} fetching Parakeet FBank Wasm`);
    }
    try {
      return await WebAssembly.compileStreaming(response.clone());
    } catch {
      signal?.throwIfAborted();
      return WebAssembly.compile(await response.arrayBuffer());
    }
  });
  compiledModules.set(url, loading);
  void loading.catch(() => {
    if (compiledModules.get(url) === loading) compiledModules.delete(url);
  });
  return loading;
}

function checkedFloat32View(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
  label: string,
): Float32Array<ArrayBuffer> {
  checkedViewRange(
    memory,
    pointer,
    length,
    Float32Array.BYTES_PER_ELEMENT,
    label,
  );
  return new Float32Array(memory.buffer, pointer, length);
}

function checkedUint32View(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
  label: string,
): Uint32Array<ArrayBuffer> {
  checkedViewRange(
    memory,
    pointer,
    length,
    Uint32Array.BYTES_PER_ELEMENT,
    label,
  );
  return new Uint32Array(memory.buffer, pointer, length);
}

function checkedViewRange(
  memory: WebAssembly.Memory,
  pointer: number,
  length: number,
  elementBytes: number,
  label: string,
): void {
  if (!Number.isSafeInteger(pointer) || pointer < 0 || pointer % elementBytes) {
    throw new Error(`unaligned Parakeet FBank Wasm ${label} pointer`);
  }
  const byteEnd = pointer + length * elementBytes;
  if (
    !Number.isSafeInteger(length) ||
    length < 0 ||
    byteEnd > memory.buffer.byteLength
  ) {
    throw new Error(`Parakeet FBank Wasm ${label} arena is outside memory`);
  }
}

function validateWindowDescriptor(window: FbankWindowDescriptor): void {
  if (!Number.isSafeInteger(window.startSample) || window.startSample < 0) {
    throw new RangeError(
      "FBank window startSample must be a non-negative safe integer",
    );
  }
  if (!Number.isSafeInteger(window.sampleCount) || window.sampleCount < 1) {
    throw new RangeError(
      "FBank window sampleCount must be a positive safe integer",
    );
  }
  if (
    window.validSampleCount !== undefined &&
    (
      !Number.isSafeInteger(window.validSampleCount) ||
      window.validSampleCount < 0 ||
      window.validSampleCount > window.sampleCount
    )
  ) {
    throw new RangeError(
      "FBank window validSampleCount must be in [0, sampleCount]",
    );
  }
}
