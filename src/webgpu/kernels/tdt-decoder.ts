/// <reference types="@webgpu/types" />

import type { ParakeetGpuPackage, GpuTensor } from "../../model/package";
import type { ArenaSlice } from "../arena";
import { GpuActivationArena } from "../arena";
import {
  PARAKEET_FP16_EXECUTION_PROFILE,
  type ParakeetExecutionProfile,
} from "../capabilities";

const HIDDEN_SIZE = 640;
const GATE_SIZE = HIDDEN_SIZE * 4;
const LSTM_INPUT_SIZE = HIDDEN_SIZE * 2;
const TOKEN_OUTPUTS = 1025;
const BLANK_TOKEN = 1024;
const HEAD_STORAGE_OUTPUTS = 1032;
const WORKGROUP_SIZE = 256;
const SUBGROUP_SIZE = 32;
const SUBGROUPS_PER_WORKGROUP = WORKGROUP_SIZE / SUBGROUP_SIZE;
const LSTM_OUTPUT_VECTORS = GATE_SIZE / 4;
const LSTM_FULL_VECTORS_PER_LANE = Math.floor(
  LSTM_OUTPUT_VECTORS / WORKGROUP_SIZE,
);
const LSTM_TAIL_VECTORS =
  LSTM_OUTPUT_VECTORS - LSTM_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE;
const HEAD_FULL_VECTORS_PER_LANE =
  Math.floor(TOKEN_OUTPUTS / 4 / WORKGROUP_SIZE);
const PREDICTOR_SCRATCH_SLOTS = 5;
const UNIFORM_WORDS = 32;
const UNIFORM_BYTES = UNIFORM_WORDS * 4;

export const TDT_DECODER_OVERFLOW = 1;
export const TDT_PAIRED_WORKGROUP_STORAGE_BYTES = 25_808;
export const TDT_PORTABLE_PAIRED_WORKGROUP_STORAGE_BYTES = 29_776;

export interface TdtDecoderExecutionPlan {
  readonly workgroupCount: number;
  readonly predictorScratchByteLength: number;
}

export function planTdtDecoderExecution(
  batchSize: number,
): TdtDecoderExecutionPlan {
  if (!Number.isSafeInteger(batchSize) || batchSize <= 0) {
    throw new RangeError("TDT decoder batchSize must be a positive integer");
  }
  return {
    workgroupCount: Math.ceil(batchSize / 2),
    predictorScratchByteLength:
      batchSize * PREDICTOR_SCRATCH_SLOTS * HIDDEN_SIZE * 4,
  };
}

export interface TdtDecoderDescriptor {
  readonly label?: string;
  /** Profile scalar [batch, frames, 640], after encoder_projector. */
  readonly encoderProjected: ArenaSlice;
  /** u32 [batch]. */
  readonly encodedLengths: ArenaSlice;
  /** u32 [batch], first encoder frame consumed by each cold decoder. */
  readonly decodeStarts: ArenaSlice;
  /** u32 [batch], exclusive encoder-frame boundary for each decoder. */
  readonly decodeEnds: ArenaSlice;
  /** u32 [batch], nonzero runs the shared bounded end-of-stream flush. */
  readonly flushFinals: ArenaSlice;
  /** f32 [batch, 4, 640], used for committed h0/c0/h1/c1. */
  readonly state: ArenaSlice;
  /** f32 [batch, 5, 640], candidate h0/c0/h1/c1 and projection. */
  readonly predictorScratch: ArenaSlice;
  /** u32 [batch, maxTokens]. */
  readonly tokens: ArenaSlice;
  /** u32 [batch, maxTokens], local encoder-frame emission timestamps. */
  readonly tokenFrames: ArenaSlice;
  /** u32 [batch, maxTokens], selected TDT durations. */
  readonly tokenDurations: ArenaSlice;
  /** u32 [batch * 2]: counts followed by status flags. */
  readonly metadata: ArenaSlice;
  readonly batchSize: number;
  readonly frames: number;
  readonly maxTokens: number;
}

export class TdtDecoderDispatch {
  private destroyed = false;
  private activeBatchSize: number;

  constructor(
    readonly label: string,
    private readonly device: GPUDevice,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly uniformBuffer: GPUBuffer,
    private readonly batchCapacity: number,
  ) {
    this.activeBatchSize = batchCapacity;
  }

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
    activeBatchSize = this.batchCapacity,
  ): void {
    if (this.destroyed) throw new Error(`${this.label} was destroyed`);
    if (
      !Number.isSafeInteger(activeBatchSize) ||
      activeBatchSize <= 0 ||
      activeBatchSize > this.batchCapacity
    ) {
      throw new RangeError(
        `${this.label} active batch must be in [1, ${this.batchCapacity}]`,
      );
    }
    if (activeBatchSize !== this.activeBatchSize) {
      this.device.queue.writeBuffer(
        this.uniformBuffer,
        0,
        new Uint32Array([activeBatchSize]),
      );
      this.activeBatchSize = activeBatchSize;
    }
    const pass = encoder.beginComputePass(
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites },
    );
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(
      planTdtDecoderExecution(activeBatchSize).workgroupCount,
      1,
      1,
    );
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniformBuffer.destroy();
  }
}

/**
 * Greedy TDT decoding in one persistent workgroup per pair of audio windows.
 *
 * The two sequences share each vocabulary-head weight load while preserving
 * independent profile-native accumulator chains and decoder control. FP16
 * retains the established shader exactly; FP32 uses raw FP32 weights,
 * projection inputs, LSTM accumulators, and vocabulary-head accumulators. A
 * single predictor workspace is reused sequentially; candidate states and
 * projections remain in bounded f32 arena scratch while blanks advance
 * through encoder frames. Keeping the complete autoregressive loop on the GPU
 * avoids a command submission and CPU readback for every token decision.
 */
export class PersistentTdtDecoderKernel {
  private constructor(
    private readonly device: GPUDevice,
    private readonly arena: GpuActivationArena,
    private readonly decoderWeights: GPUBuffer,
    private readonly pipeline: GPUComputePipeline,
    private readonly layout: GPUBindGroupLayout,
    private readonly weights: DecoderWeightOffsets,
    private readonly executionProfile: ParakeetExecutionProfile,
  ) {}

  static async create(
    device: GPUDevice,
    arena: GpuActivationArena,
    model: ParakeetGpuPackage,
    executionProfile: ParakeetExecutionProfile =
      PARAKEET_FP16_EXECUTION_PROFILE,
  ): Promise<PersistentTdtDecoderKernel> {
    if (device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE) {
      throw new Error(`TDT decoder requires ${WORKGROUP_SIZE} workgroup lanes`);
    }
    const requiredWorkgroupStorageBytes =
      executionProfile.kernelBackend === "portable"
        ? TDT_PORTABLE_PAIRED_WORKGROUP_STORAGE_BYTES
        : TDT_PAIRED_WORKGROUP_STORAGE_BYTES;
    if (
      device.limits.maxComputeWorkgroupStorageSize <
      requiredWorkgroupStorageBytes
    ) {
      throw new Error(
        `TDT decoder requires ${requiredWorkgroupStorageBytes} ` +
          "bytes of workgroup storage",
      );
    }
    if (
      executionProfile.kernelBackend === "subgroups" &&
      !device.features.has("subgroups")
    ) {
      throw new Error("TDT decoder requires WebGPU subgroups");
    }
    if (
      executionProfile.precision === "fp16" &&
      !device.features.has("shader-f16")
    ) {
      throw new Error("TDT decoder requires WebGPU shader-f16");
    }
    if (model.precision !== executionProfile.precision) {
      throw new Error(
        `TDT decoder ${executionProfile.precision} profile cannot use ` +
          `${model.precision} decoder weights`,
      );
    }

    const residentWeights = requireDecoderWeights(
      model,
      executionProfile,
    );
    const pipelineLabel = executionProfile.kernelBackend === "portable"
      ? executionProfile.precision === "fp16"
        ? "parakeet-persistent-tdt-decoder-portable"
        : "parakeet-persistent-tdt-decoder-fp32-portable"
      : executionProfile.precision === "fp16"
        ? "parakeet-persistent-tdt-decoder"
        : "parakeet-persistent-tdt-decoder-fp32";
    const layout = device.createBindGroupLayout({
      label: "parakeet-tdt-decoder-bindings",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
      ],
    });
    const module = device.createShaderModule({
      label: pipelineLabel,
      code: tdtDecoderWgslForProfile(executionProfile),
    });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `TDT decoder WGSL failed: ${errors.map((error) => error.message).join("; ")}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label: pipelineLabel,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
    });
    return new PersistentTdtDecoderKernel(
      device,
      arena,
      residentWeights.buffer,
      pipeline,
      layout,
      residentWeights.offsets,
      executionProfile,
    );
  }

  createDispatch(descriptor: TdtDecoderDescriptor): TdtDecoderDispatch {
    validateDescriptor(descriptor, this.executionProfile);
    const words = new Uint32Array(UNIFORM_WORDS);
    words[0] = descriptor.batchSize;
    words[1] = descriptor.frames;
    words[2] = descriptor.maxTokens;
    words[3] = wordOffset(descriptor.encoderProjected);
    words[4] = wordOffset(descriptor.encodedLengths);
    words[5] = wordOffset(descriptor.state);
    words[6] = wordOffset(descriptor.tokens);
    words[7] = wordOffset(descriptor.tokenFrames);
    words[8] = wordOffset(descriptor.tokenDurations);
    words[9] = wordOffset(descriptor.metadata);
    words[10] = wordOffset(descriptor.metadata) + descriptor.batchSize;
    words[11] = this.weights.embedding;
    words[12] = this.weights.lstm0;
    words[13] = this.weights.lstmBias0;
    words[14] = this.weights.lstm1;
    words[15] = this.weights.lstmBias1;
    words[16] = this.weights.decoderProjector;
    words[17] = this.weights.decoderProjectorBias;
    words[18] = this.weights.head;
    words[19] = this.weights.headBias;
    words[20] = wordOffset(descriptor.predictorScratch);
    words[21] = wordOffset(descriptor.decodeStarts);
    words[22] = wordOffset(descriptor.decodeEnds);
    words[23] = wordOffset(descriptor.flushFinals);

    const label = descriptor.label ?? "parakeet-persistent-tdt-decode";
    const uniformBuffer = this.device.createBuffer({
      label: `${label}-params`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, words);
    const arenaBinding = decoderArenaBinding(this.arena, descriptor);
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: this.layout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.decoderWeights,
            offset: 0,
            size: this.decoderWeights.size,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: arenaBinding.buffer,
            offset: 0,
            size: arenaBinding.byteLength,
          },
        },
        { binding: 2, resource: { buffer: uniformBuffer } },
        {
          binding: 3,
          resource: {
            buffer: this.decoderWeights,
            offset: 0,
            size: this.decoderWeights.size,
          },
        },
      ],
    });
    return new TdtDecoderDispatch(
      label,
      this.device,
      this.pipeline,
      bindGroup,
      uniformBuffer,
      descriptor.batchSize,
    );
  }
}

interface DecoderWeightOffsets {
  readonly embedding: number;
  readonly lstm0: number;
  readonly lstmBias0: number;
  readonly lstm1: number;
  readonly lstmBias1: number;
  readonly decoderProjector: number;
  readonly decoderProjectorBias: number;
  readonly head: number;
  readonly headBias: number;
}

interface ResidentDecoderWeights {
  readonly buffer: GPUBuffer;
  readonly offsets: DecoderWeightOffsets;
}

function requireDecoderWeights(
  model: ParakeetGpuPackage,
  executionProfile: ParakeetExecutionProfile,
): ResidentDecoderWeights {
  const weightDtype =
    executionProfile.precision === "fp16"
      ? "float16"
      : "float32";
  const embedding = requireTensor(
    model.tensor("decoder.embedding.weight"),
    weightDtype,
    [TOKEN_OUTPUTS, HIDDEN_SIZE],
    "token-by-channel-row-major",
  );
  const lstm0 = requireTensor(
    model.tensor("decoder.lstm.weight_l0"),
    weightDtype,
    [LSTM_INPUT_SIZE, GATE_SIZE],
    "input-hidden-k-by-ifgo-row-major-vec4",
  );
  const lstm1 = requireTensor(
    model.tensor("decoder.lstm.weight_l1"),
    weightDtype,
    [LSTM_INPUT_SIZE, GATE_SIZE],
    "input-hidden-k-by-ifgo-row-major-vec4",
  );
  const decoderProjector = requireTensor(
    model.tensor("decoder.decoder_projector.weight"),
    weightDtype,
    [HIDDEN_SIZE, HIDDEN_SIZE],
    "k-by-output-row-major-vec4",
  );
  const head = requireTensor(
    model.tensor("joint.head.weight"),
    weightDtype,
    [HIDDEN_SIZE, HEAD_STORAGE_OUTPUTS],
    "k-by-output-row-major-vec4",
  );
  const tensors = {
    embedding,
    lstm0,
    lstmBias0: requireTensor(
      model.tensor("decoder.lstm.bias_l0"),
      "float32",
      [GATE_SIZE],
      "ifgo-gates",
    ),
    lstm1,
    lstmBias1: requireTensor(
      model.tensor("decoder.lstm.bias_l1"),
      "float32",
      [GATE_SIZE],
      "ifgo-gates",
    ),
    decoderProjector,
    decoderProjectorBias: requireTensor(
      model.tensor("decoder.decoder_projector.bias"),
      "float32",
      [HIDDEN_SIZE],
      "output-channel",
    ),
    head,
    headBias: requireTensor(
      model.tensor("joint.head.bias"),
      "float32",
      [HEAD_STORAGE_OUTPUTS],
      "output-channel-padded-vec4",
    ),
  };
  const buffer = embedding.buffer;
  for (const tensor of Object.values(tensors)) {
    if (
      tensor.sourceRecord.shard !== "decoder.bin" ||
      tensor.runtimeRecord.bufferId !== "source-shard-decoder.bin" ||
      tensor.buffer !== buffer
    ) {
      throw new Error(
        "Persistent TDT decoder requires one resident decoder source shard",
      );
    }
  }
  return {
    buffer,
    offsets: Object.fromEntries(
      Object.entries(tensors).map(([name, tensor]) => [
        name,
        tensor.runtimeRecord.byteOffset / 4,
      ]),
    ) as unknown as DecoderWeightOffsets,
  };
}

function requireTensor(
  tensor: GpuTensor,
  dtype: "float16" | "float32",
  storageShape: readonly number[],
  layout: string,
): GpuTensor {
  const source = tensor.sourceRecord;
  const runtime = tensor.runtimeRecord;
  if (
    source.dtype !== dtype ||
    source.storageShape.join(",") !== storageShape.join(",") ||
    source.layout !== layout ||
    runtime.dtype !== dtype ||
    runtime.storageShape.join(",") !== storageShape.join(",") ||
    runtime.layout !== layout ||
    runtime.byteLength !== source.byteLength ||
    runtime.byteOffset % 4 !== 0
  ) {
    throw new Error(
      `Unexpected decoder tensor ${source.layout} ` +
        `[${source.storageShape.join(",")}]`,
    );
  }
  return tensor;
}

interface DecoderArenaBinding {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
}

function decoderArenaBinding(
  arena: GpuActivationArena,
  descriptor: TdtDecoderDescriptor,
): DecoderArenaBinding {
  const slices = [
    descriptor.encoderProjected,
    descriptor.encodedLengths,
    descriptor.decodeStarts,
    descriptor.decodeEnds,
    descriptor.flushFinals,
    descriptor.state,
    descriptor.predictorScratch,
    descriptor.tokens,
    descriptor.tokenFrames,
    descriptor.tokenDurations,
    descriptor.metadata,
  ] as const;
  const buffer = arena.bufferFor(slices[0]);
  for (const slice of slices.slice(1)) {
    if (arena.bufferFor(slice) !== buffer) {
      throw new Error(
        "TDT decoder slices must share one physical arena buffer",
      );
    }
  }
  return {
    buffer,
    byteLength: arena.bufferByteLengthFor(slices[0]),
  };
}

function validateDescriptor(
  descriptor: TdtDecoderDescriptor,
  executionProfile: ParakeetExecutionProfile,
): void {
  for (const [name, value] of [
    ["batchSize", descriptor.batchSize],
    ["frames", descriptor.frames],
    ["maxTokens", descriptor.maxTokens],
  ] as const) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`Invalid TDT decoder ${name}`);
    }
  }
  requireSliceBytes(
    descriptor.encoderProjected,
    descriptor.batchSize *
      descriptor.frames *
      HIDDEN_SIZE *
      (executionProfile.precision === "fp16" ? 2 : 4),
  );
  requireSliceBytes(descriptor.encodedLengths, descriptor.batchSize * 4);
  requireSliceBytes(descriptor.decodeStarts, descriptor.batchSize * 4);
  requireSliceBytes(descriptor.decodeEnds, descriptor.batchSize * 4);
  requireSliceBytes(descriptor.flushFinals, descriptor.batchSize * 4);
  requireSliceBytes(
    descriptor.state,
    descriptor.batchSize * 4 * HIDDEN_SIZE * 4,
  );
  requireSliceBytes(
    descriptor.predictorScratch,
    planTdtDecoderExecution(descriptor.batchSize).predictorScratchByteLength,
  );
  const outputBytes = descriptor.batchSize * descriptor.maxTokens * 4;
  requireSliceBytes(descriptor.tokens, outputBytes);
  requireSliceBytes(descriptor.tokenFrames, outputBytes);
  requireSliceBytes(descriptor.tokenDurations, outputBytes);
  requireSliceBytes(descriptor.metadata, descriptor.batchSize * 2 * 4);
}

function requireSliceBytes(slice: ArenaSlice, minimum: number): void {
  if (slice.byteLength < minimum) {
    throw new RangeError(`${slice.label} is too small for TDT decoding`);
  }
}

function wordOffset(slice: ArenaSlice): number {
  if (slice.byteOffset % 4 !== 0) {
    throw new RangeError(`${slice.label} is not word aligned`);
  }
  return slice.byteOffset / 4;
}

function wgslLstmAccumulatorDeclarations(
  prefix: string,
  biasOffset: string,
): string {
  const declarations = Array.from(
    { length: LSTM_FULL_VECTORS_PER_LANE },
    (_, slot) =>
      `      var ${prefix}${slot} = vec4<f16>(read_weight_f32x4(${biasOffset}, (lane + ${slot * WORKGROUP_SIZE}u) * 4u));
      var ${prefix}${slot}_odd = vec4<f16>(0.0);`,
  );
  declarations.push(`      var ${prefix}${LSTM_FULL_VECTORS_PER_LANE} = vec4<f16>(0.0);
      var ${prefix}${LSTM_FULL_VECTORS_PER_LANE}_odd = vec4<f16>(0.0);
      if (lane < ${LSTM_TAIL_VECTORS}u) {
        ${prefix}${LSTM_FULL_VECTORS_PER_LANE} = vec4<f16>(
          read_weight_f32x4(
            ${biasOffset},
            (lane + ${LSTM_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u
          )
        );
      }`);
  return declarations.join("\n");
}

function wgslFp32LstmAccumulatorDeclarations(
  prefix: string,
  biasOffset: string,
): string {
  const declarations = Array.from(
    { length: LSTM_FULL_VECTORS_PER_LANE },
    (_, slot) =>
      `      var ${prefix}${slot} = read_weight_f32x4(${biasOffset}, (lane + ${slot * WORKGROUP_SIZE}u) * 4u);
      var ${prefix}${slot}_odd = vec4<f32>(0.0);`,
  );
  declarations.push(`      var ${prefix}${LSTM_FULL_VECTORS_PER_LANE} = vec4<f32>(0.0);
      var ${prefix}${LSTM_FULL_VECTORS_PER_LANE}_odd = vec4<f32>(0.0);
      if (lane < ${LSTM_TAIL_VECTORS}u) {
        ${prefix}${LSTM_FULL_VECTORS_PER_LANE} = read_weight_f32x4(
          ${biasOffset},
          (lane + ${LSTM_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u
        );
      }`);
  return declarations.join("\n");
}

function wgslLstmAccumulate(
  prefix: string,
  weightOffset: string,
  inputValue: string,
  inputIndex: string,
  accumulatorSuffix: string,
): string {
  const statements = Array.from(
    { length: LSTM_FULL_VECTORS_PER_LANE },
    (_, slot) =>
      `        ${prefix}${slot}${accumulatorSuffix} = fma(vec4<f16>(f16(${inputValue})), read_weight_vec4_f16(${weightOffset}, ${inputIndex} * ${GATE_SIZE}u + (lane + ${slot * WORKGROUP_SIZE}u) * 4u), ${prefix}${slot}${accumulatorSuffix});`,
  );
  statements.push(`        if (lane < ${LSTM_TAIL_VECTORS}u) {
          ${prefix}${LSTM_FULL_VECTORS_PER_LANE}${accumulatorSuffix} = fma(
            vec4<f16>(f16(${inputValue})),
            read_weight_vec4_f16(
              ${weightOffset},
              ${inputIndex} * ${GATE_SIZE}u +
                (lane + ${LSTM_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u
            ),
            ${prefix}${LSTM_FULL_VECTORS_PER_LANE}${accumulatorSuffix}
          );
        }`);
  return statements.join("\n");
}

function wgslFp32LstmAccumulate(
  prefix: string,
  weightOffset: string,
  inputValue: string,
  inputIndex: string,
  accumulatorSuffix: string,
): string {
  const statements = Array.from(
    { length: LSTM_FULL_VECTORS_PER_LANE },
    (_, slot) =>
      `        ${prefix}${slot}${accumulatorSuffix} = fma(vec4<f32>(${inputValue}), read_weight_vec4(${weightOffset}, ${inputIndex} * ${GATE_SIZE}u + (lane + ${slot * WORKGROUP_SIZE}u) * 4u), ${prefix}${slot}${accumulatorSuffix});`,
  );
  statements.push(`        if (lane < ${LSTM_TAIL_VECTORS}u) {
          ${prefix}${LSTM_FULL_VECTORS_PER_LANE}${accumulatorSuffix} = fma(
            vec4<f32>(${inputValue}),
            read_weight_vec4(
              ${weightOffset},
              ${inputIndex} * ${GATE_SIZE}u +
                (lane + ${LSTM_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u
            ),
            ${prefix}${LSTM_FULL_VECTORS_PER_LANE}${accumulatorSuffix}
          );
        }`);
  return statements.join("\n");
}

function wgslLstmStore(prefix: string): string {
  const statements: string[] = [];
  for (let slot = 0; slot < LSTM_FULL_VECTORS_PER_LANE; slot += 1) {
    for (let component = 0; component < 4; component += 1) {
      statements.push(
        `      gate_values[(lane + ${slot * WORKGROUP_SIZE}u) * 4u + ${component}u] = f32((${prefix}${slot} + ${prefix}${slot}_odd)[${component}u]);`,
      );
    }
  }
  statements.push(`      if (lane < ${LSTM_TAIL_VECTORS}u) {`);
  for (let component = 0; component < 4; component += 1) {
    statements.push(
      `        gate_values[(lane + ${LSTM_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u + ${component}u] = f32((${prefix}${LSTM_FULL_VECTORS_PER_LANE} + ${prefix}${LSTM_FULL_VECTORS_PER_LANE}_odd)[${component}u]);`,
    );
  }
  statements.push("      }");
  return statements.join("\n");
}

function wgslPairedHeadAccumulatorDeclarations(): string {
  return Array.from(
    { length: HEAD_FULL_VECTORS_PER_LANE },
    (_, slot) =>
      `  let head_bias${slot} = vec4<f16>(read_weight_f32x4(
    params.head_bias_offset,
    (lane + ${slot * WORKGROUP_SIZE}u) * 4u
  ));
  var head_a${slot} = head_bias${slot};
  var head_b${slot} = head_bias${slot};`,
  ).join("\n");
}

function wgslFp32PairedHeadAccumulatorDeclarations(): string {
  return Array.from(
    { length: HEAD_FULL_VECTORS_PER_LANE },
    (_, slot) => slot,
  )
    .map(
      (slot) =>
        `    let head_bias${slot} = read_weight_f32x4(
      params.head_bias_offset,
      (lane + ${slot * WORKGROUP_SIZE}u) * 4u
    );
    var head_a${slot} = head_bias${slot};
    var head_b${slot} = head_bias${slot};`,
    )
    .join("\n");
}

function wgslPairedHeadAccumulate(): string {
  return Array.from(
    { length: HEAD_FULL_VECTORS_PER_LANE },
    (_, slot) =>
      `      let head_weight${slot} = read_weight_vec4_f16(
        params.head_offset,
        k * ${HEAD_STORAGE_OUTPUTS}u +
          (lane + ${slot * WORKGROUP_SIZE}u) * 4u
      );
      if (!skip_a) {
        head_a${slot} = fma(
          vec4<f16>(f16(input_a)),
          head_weight${slot},
          head_a${slot}
        );
      }
      if (!skip_b) {
        head_b${slot} = fma(
          vec4<f16>(f16(input_b)),
          head_weight${slot},
          head_b${slot}
        );
      }`,
  ).join("\n");
}

function wgslFp32PairedHeadAccumulate(): string {
  return Array.from(
    { length: HEAD_FULL_VECTORS_PER_LANE },
    (_, slot) => slot,
  )
    .map(
      (slot) =>
        `        let head_weight${slot} = read_weight_vec4(
          params.head_offset,
          k * ${HEAD_STORAGE_OUTPUTS}u +
            (lane + ${slot * WORKGROUP_SIZE}u) * 4u
        );
        if (!skip_a) {
          head_a${slot} = fma(
            vec4<f32>(input_a),
            head_weight${slot},
            head_a${slot}
          );
        }
        if (!skip_b) {
          head_b${slot} = fma(
            vec4<f32>(input_b),
            head_weight${slot},
            head_b${slot}
          );
        }`,
    )
    .join("\n");
}

function wgslHeadTokenArgmax(prefix: "a" | "b"): string {
  const statements: string[] = [];
  for (let slot = 0; slot < HEAD_FULL_VECTORS_PER_LANE; slot += 1) {
    for (let component = 0; component < 4; component += 1) {
      statements.push(`  {
    let output = (lane + ${slot * WORKGROUP_SIZE}u) * 4u + ${component}u;
    let score = sanitize_score(f32(head_${prefix}${slot}[${component}u]));
    if (choose_better(
      local_token_value_${prefix},
      local_token_id_${prefix},
      score,
      output
    )) {
      local_token_value_${prefix} = score;
      local_token_id_${prefix} = output;
    }
  }`);
    }
  }
  return statements.join("\n");
}

function wgslHeadArgmaxAndReduce(prefix: "a" | "b"): string {
  const pairSlot = prefix === "a" ? 0 : 1;
  return /* wgsl */ `
  var local_token_value_${prefix} = -3.402823466e+38;
  var local_token_id_${prefix} = 0xffffffffu;
  var local_duration_value_${prefix} = -3.402823466e+38;
  var local_duration_id_${prefix} = 0xffffffffu;
${wgslHeadTokenArgmax(prefix)}
  if (lane == 0u) {
    let blank_score = sanitize_score(f32(tail_${prefix}[0u]));
    if (choose_better(
      local_token_value_${prefix},
      local_token_id_${prefix},
      blank_score,
      ${BLANK_TOKEN}u
    )) {
      local_token_value_${prefix} = blank_score;
      local_token_id_${prefix} = ${BLANK_TOKEN}u;
    }
    for (var component = 1u; component < 4u; component += 1u) {
      let duration = component - 1u;
      let score = sanitize_score(f32(tail_${prefix}[component]));
      if (choose_better(
        local_duration_value_${prefix},
        local_duration_id_${prefix},
        score,
        duration
      )) {
        local_duration_value_${prefix} = score;
        local_duration_id_${prefix} = duration;
      }
    }
  } else if (lane == 1u) {
    for (var component = 0u; component < 2u; component += 1u) {
      let duration = component + 3u;
      let score = sanitize_score(f32(tail_${prefix}[component]));
      if (choose_better(
        local_duration_value_${prefix},
        local_duration_id_${prefix},
        score,
        duration
      )) {
        local_duration_value_${prefix} = score;
        local_duration_id_${prefix} = duration;
      }
    }
  }
  reduce_joint_result(
    ${pairSlot}u,
    lane,
    subgroup_lane,
    subgroup_slot,
    local_token_value_${prefix},
    local_token_id_${prefix},
    local_duration_value_${prefix},
    local_duration_id_${prefix}
  );`;
}

function wgslFp32HeadTokenArgmax(
  prefix: "a" | "b",
): string {
  const statements: string[] = [];
  for (let slot = 0; slot < HEAD_FULL_VECTORS_PER_LANE; slot += 1) {
    for (let component = 0; component < 4; component += 1) {
      statements.push(`    {
      let output = (lane + ${slot * WORKGROUP_SIZE}u) * 4u + ${component}u;
      let score = sanitize_score(f32(head_${prefix}${slot}[${component}u]));
      if (choose_better(
        local_token_value_${prefix},
        local_token_id_${prefix},
        score,
        output
      )) {
        local_token_value_${prefix} = score;
        local_token_id_${prefix} = output;
      }
    }`);
    }
  }
  return statements.join("\n");
}

function wgslFp32PairedHeadBlock(): string {
  return `  {
    // V2's complete 1024-token block and joint tail share one increasing-K pass.
${wgslFp32PairedHeadAccumulatorDeclarations()}
    var zero_skip_safe_a = true;
    var zero_skip_safe_b = true;
    for (var k = 0u; k < ${HIDDEN_SIZE}u; k += 1u) {
      let input_a = joint_input_a[k];
      let input_b = joint_input_b[k];
      let input_a_bits = bitcast<u32>(input_a);
      let input_b_bits = bitcast<u32>(input_b);
      let input_a_is_nan =
        (input_a_bits & 0x7fffffffu) > 0x7f800000u;
      let input_b_is_nan =
        (input_b_bits & 0x7fffffffu) > 0x7f800000u;
      let exceptional_a =
        input_a_bits == 0x80000000u || input_a_is_nan;
      let exceptional_b =
        input_b_bits == 0x80000000u || input_b_is_nan;
      zero_skip_safe_a = zero_skip_safe_a && !exceptional_a;
      zero_skip_safe_b = zero_skip_safe_b && !exceptional_b;
      let skip_a = zero_skip_safe_a && input_a_bits == 0u;
      let skip_b = zero_skip_safe_b && input_b_bits == 0u;
      // These shared inputs make the outer branch workgroup-uniform for each k.
      if (!skip_a || !skip_b) {
${wgslFp32PairedHeadAccumulate()}
        if (lane < 2u) {
          let tail_weight = read_weight_vec4(
            params.head_offset,
            k * ${HEAD_STORAGE_OUTPUTS}u +
              (lane + ${HEAD_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u
          );
          if (!skip_a) {
            tail_a = fma(vec4<f32>(input_a), tail_weight, tail_a);
          }
          if (!skip_b) {
            tail_b = fma(vec4<f32>(input_b), tail_weight, tail_b);
          }
        }
      }
    }
${wgslFp32HeadTokenArgmax("a")}
${wgslFp32HeadTokenArgmax("b")}
  }`;
}

function wgslFp32HeadTailAndReduce(prefix: "a" | "b"): string {
  const pairSlot = prefix === "a" ? 0 : 1;
  return `  if (lane == 0u) {
    let blank_score = sanitize_score(f32(tail_${prefix}[0u]));
    if (choose_better(
      local_token_value_${prefix},
      local_token_id_${prefix},
      blank_score,
      ${BLANK_TOKEN}u
    )) {
      local_token_value_${prefix} = blank_score;
      local_token_id_${prefix} = ${BLANK_TOKEN}u;
    }
    for (var component = 1u; component < 4u; component += 1u) {
      let duration = component - 1u;
      let score = sanitize_score(f32(tail_${prefix}[component]));
      if (choose_better(
        local_duration_value_${prefix},
        local_duration_id_${prefix},
        score,
        duration
      )) {
        local_duration_value_${prefix} = score;
        local_duration_id_${prefix} = duration;
      }
    }
  } else if (lane == 1u) {
    for (var component = 0u; component < 2u; component += 1u) {
      let duration = component + 3u;
      let score = sanitize_score(f32(tail_${prefix}[component]));
      if (choose_better(
        local_duration_value_${prefix},
        local_duration_id_${prefix},
        score,
        duration
      )) {
        local_duration_value_${prefix} = score;
        local_duration_id_${prefix} = duration;
      }
    }
  }
  reduce_joint_result(
    ${pairSlot}u,
    lane,
    subgroup_lane,
    subgroup_slot,
    local_token_value_${prefix},
    local_token_id_${prefix},
    local_duration_value_${prefix},
    local_duration_id_${prefix}
  );`;
}

function wgslFp16PairedHeadBody(): string {
  return `${wgslPairedHeadAccumulatorDeclarations()}
  var tail_a = vec4<f16>(-65504.0);
  var tail_b = vec4<f16>(-65504.0);
  if (lane < 2u) {
    let tail_bias = vec4<f16>(
      read_weight_f32x4(
        params.head_bias_offset,
        (lane + ${HEAD_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u
      )
    );
    tail_a = tail_bias;
    tail_b = tail_bias;
  }
  var zero_skip_safe_a = true;
  var zero_skip_safe_b = true;
  for (var k = 0u; k < ${HIDDEN_SIZE}u; k += 1u) {
    let input_a = joint_input_a[k];
    let input_b = joint_input_b[k];
    let input_a_bits = bitcast<u32>(input_a);
    let input_b_bits = bitcast<u32>(input_b);
    let input_a_is_nan =
      (input_a_bits & 0x7fffffffu) > 0x7f800000u;
    let input_b_is_nan =
      (input_b_bits & 0x7fffffffu) > 0x7f800000u;
    let exceptional_a =
      input_a_bits == 0x80000000u || input_a_is_nan;
    let exceptional_b =
      input_b_bits == 0x80000000u || input_b_is_nan;
    zero_skip_safe_a = zero_skip_safe_a && !exceptional_a;
    zero_skip_safe_b = zero_skip_safe_b && !exceptional_b;
    let skip_a = zero_skip_safe_a && input_a_bits == 0u;
    let skip_b = zero_skip_safe_b && input_b_bits == 0u;
    // These shared inputs make the outer branch workgroup-uniform for each k.
    if (!skip_a || !skip_b) {
${wgslPairedHeadAccumulate()}
      if (lane < 2u) {
        let tail_weight = read_weight_vec4_f16(
          params.head_offset,
          k * ${HEAD_STORAGE_OUTPUTS}u +
            (lane + ${HEAD_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u
        );
        if (!skip_a) {
          tail_a = fma(vec4<f16>(f16(input_a)), tail_weight, tail_a);
        }
        if (!skip_b) {
          tail_b = fma(vec4<f16>(f16(input_b)), tail_weight, tail_b);
        }
      }
    }
  }

${wgslHeadArgmaxAndReduce("a")}
${wgslHeadArgmaxAndReduce("b")}`;
}

function wgslFp32PairedHeadBody(): string {
  return `  var local_token_value_a = -3.402823466e+38;
  var local_token_id_a = 0xffffffffu;
  var local_duration_value_a = -3.402823466e+38;
  var local_duration_id_a = 0xffffffffu;
  var local_token_value_b = -3.402823466e+38;
  var local_token_id_b = 0xffffffffu;
  var local_duration_value_b = -3.402823466e+38;
  var local_duration_id_b = 0xffffffffu;

  var tail_a = vec4<f32>(-3.402823466e+38);
  var tail_b = vec4<f32>(-3.402823466e+38);
  if (lane < 2u) {
    let tail_bias = vec4<f32>(
      read_weight_f32x4(
        params.head_bias_offset,
        (lane + ${HEAD_FULL_VECTORS_PER_LANE * WORKGROUP_SIZE}u) * 4u
      )
    );
    tail_a = tail_bias;
    tail_b = tail_bias;
  }
${wgslFp32PairedHeadBlock()}
${wgslFp32HeadTailAndReduce("a")}
${wgslFp32HeadTailAndReduce("b")}`;
}

function fp16WeightReadersWgsl(): string {
  return `fn read_weight_f16(base_word: u32, element: u32) -> f32 {
  let pair = weight_pairs[base_word + element / 2u];
  return f32(select(pair.x, pair.y, (element & 1u) != 0u));
}

fn read_weight_vec4(base_word: u32, element: u32) -> vec4<f32> {
  let first = weight_pairs[base_word + element / 2u];
  let second = weight_pairs[base_word + element / 2u + 1u];
  return vec4<f32>(vec2<f32>(first), vec2<f32>(second));
}

fn read_weight_vec4_f16(base_word: u32, element: u32) -> vec4<f16> {
  let first = weight_pairs[base_word + element / 2u];
  let second = weight_pairs[base_word + element / 2u + 1u];
  return vec4<f16>(first, second);
}

fn read_weight_f32(base_word: u32, element: u32) -> f32 {
  return bitcast<f32>(weights[base_word + element]);
}

fn read_weight_f32x4(base_word: u32, element: u32) -> vec4<f32> {
  return vec4<f32>(
    read_weight_f32(base_word, element),
    read_weight_f32(base_word, element + 1u),
    read_weight_f32(base_word, element + 2u),
    read_weight_f32(base_word, element + 3u)
  );
}`;
}

function fp32WeightReadersWgsl(): string {
  return `fn read_weight_f32(base_word: u32, element: u32) -> f32 {
  return bitcast<f32>(weights[base_word + element]);
}

fn read_weight_vec4(base_word: u32, element: u32) -> vec4<f32> {
  return weight_vectors[(base_word + element) / 4u];
}

fn read_weight_f32x4(base_word: u32, element: u32) -> vec4<f32> {
  return read_weight_vec4(base_word, element);
}`;
}

function fp16ArenaReadersWgsl(): string {
  return `fn read_arena_f16(base_word: u32, element: u32) -> f32 {
  let pair = unpack2x16float(arena[base_word + element / 2u]);
  return select(pair.x, pair.y, (element & 1u) != 0u);
}

fn read_arena_vec4(base_word: u32, element: u32) -> vec4<f32> {
  let first = unpack2x16float(arena[base_word + element / 2u]);
  let second = unpack2x16float(arena[base_word + element / 2u + 1u]);
  return vec4<f32>(first, second);
}`;
}

function fp32ArenaReadersWgsl(): string {
  return `fn read_arena_f32(base_word: u32, element: u32) -> f32 {
  return bitcast<f32>(arena[base_word + element]);
}

fn read_arena_vec4(base_word: u32, element: u32) -> vec4<f32> {
  return vec4<f32>(
    read_arena_f32(base_word, element),
    read_arena_f32(base_word, element + 1u),
    read_arena_f32(base_word, element + 2u),
    read_arena_f32(base_word, element + 3u)
  );
}`;
}

export function tdtDecoderWgsl(): string {
  return tdtDecoderWgslForPrecision("fp16");
}

export function tdtDecoderFp32Wgsl(): string {
  return tdtDecoderWgslForPrecision("fp32");
}

export function tdtDecoderWgslForProfile(
  executionProfile: ParakeetExecutionProfile =
    PARAKEET_FP16_EXECUTION_PROFILE,
): string {
  if (executionProfile.kernelBackend === "portable") {
    return executionProfile.precision === "fp16"
      ? portableTdtDecoderWgsl("fp16")
      : portableTdtDecoderWgsl("fp32");
  }
  return executionProfile.precision === "fp16"
    ? tdtDecoderWgsl()
    : tdtDecoderFp32Wgsl();
}

function tdtDecoderWgslForPrecision(
  precision: ParakeetExecutionProfile["precision"],
): string {
  const scalarEnable =
    precision === "fp16" ? "enable f16;" : "";
  const vectorWeightBinding =
    precision === "fp16"
      ? "@group(0) @binding(3) var<storage, read> weight_pairs: array<vec2<f16>>;"
      : "@group(0) @binding(3) var<storage, read> weight_vectors: array<vec4<f32>>;";
  const weightReaders =
    precision === "fp16"
      ? fp16WeightReadersWgsl()
      : fp32WeightReadersWgsl();
  const arenaReaders =
    precision === "fp16"
      ? fp16ArenaReadersWgsl()
      : fp32ArenaReadersWgsl();
  const lstmAccumulatorDeclarations =
    precision === "fp16"
      ? wgslLstmAccumulatorDeclarations
      : wgslFp32LstmAccumulatorDeclarations;
  const lstmAccumulate =
    precision === "fp16"
      ? wgslLstmAccumulate
      : wgslFp32LstmAccumulate;
  const pairedHeadBody =
    precision === "fp16"
      ? wgslFp16PairedHeadBody()
      : wgslFp32PairedHeadBody();
  const readScalarWeight =
    precision === "fp16"
      ? "read_weight_f16"
      : "read_weight_f32";
  return /* wgsl */ `
enable subgroups;
${scalarEnable}

struct Params {
  batch_size: u32,
  frames: u32,
  max_tokens: u32,
  encoder_offset: u32,
  lengths_offset: u32,
  state_offset: u32,
  tokens_offset: u32,
  token_frames_offset: u32,
  token_durations_offset: u32,
  counts_offset: u32,
  status_offset: u32,
  embedding_offset: u32,
  lstm0_offset: u32,
  lstm_bias0_offset: u32,
  lstm1_offset: u32,
  lstm_bias1_offset: u32,
  decoder_projector_offset: u32,
  decoder_projector_bias_offset: u32,
  head_offset: u32,
  head_bias_offset: u32,
  predictor_scratch_offset: u32,
  decode_starts_offset: u32,
  decode_ends_offset: u32,
  flush_finals_offset: u32,
  _pad3: u32,
  _pad4: u32,
  _pad5: u32,
  _pad6: u32,
  _pad7: u32,
  _pad8: u32,
  _pad9: u32,
  _pad10: u32,
};

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> arena: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
${vectorWeightBinding}

var<workgroup> candidate_h0: array<f32, ${HIDDEN_SIZE}>;
var<workgroup> candidate_c0: array<f32, ${HIDDEN_SIZE}>;
var<workgroup> candidate_h1: array<f32, ${HIDDEN_SIZE}>;
var<workgroup> candidate_c1: array<f32, ${HIDDEN_SIZE}>;
var<workgroup> gate_values: array<f32, ${GATE_SIZE}>;
var<workgroup> joint_input_a: array<f32, ${HIDDEN_SIZE}>;
var<workgroup> joint_input_b: array<f32, ${HIDDEN_SIZE}>;
var<workgroup> token_values: array<f32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> token_ids: array<u32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> duration_values: array<f32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> duration_ids: array<u32, ${SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> decode_time_a: u32;
var<workgroup> decode_time_b: u32;
var<workgroup> output_count_a: u32;
var<workgroup> output_count_b: u32;
var<workgroup> decode_status_a: u32;
var<workgroup> decode_status_b: u32;
var<workgroup> best_token_a: u32;
var<workgroup> best_token_b: u32;
var<workgroup> best_duration_a: u32;
var<workgroup> best_duration_b: u32;
var<workgroup> encoded_length_a: u32;
var<workgroup> encoded_length_b: u32;
var<workgroup> symbols_at_time_a: u32;
var<workgroup> symbols_at_time_b: u32;
var<workgroup> emitted_nonblank_a: u32;
var<workgroup> emitted_nonblank_b: u32;
var<workgroup> flush_blanks_a: u32;
var<workgroup> flush_blanks_b: u32;
var<workgroup> flush_time_a: u32;
var<workgroup> flush_time_b: u32;

${weightReaders}

${arenaReaders}

fn state_index(sequence: u32, slot: u32, channel: u32) -> u32 {
  return params.state_offset +
    sequence * ${4 * HIDDEN_SIZE}u +
    slot * ${HIDDEN_SIZE}u +
    channel;
}

fn predictor_scratch_index(sequence: u32, slot: u32, channel: u32) -> u32 {
  return params.predictor_scratch_offset +
    sequence * ${PREDICTOR_SCRATCH_SLOTS * HIDDEN_SIZE}u +
    slot * ${HIDDEN_SIZE}u +
    channel;
}

fn read_state(sequence: u32, slot: u32, channel: u32) -> f32 {
  return bitcast<f32>(arena[state_index(sequence, slot, channel)]);
}

fn read_state_vec4(sequence: u32, slot: u32, channel: u32) -> vec4<f32> {
  return vec4<f32>(
    read_state(sequence, slot, channel),
    read_state(sequence, slot, channel + 1u),
    read_state(sequence, slot, channel + 2u),
    read_state(sequence, slot, channel + 3u)
  );
}

fn read_predictor_scratch_vec4(
  sequence: u32,
  slot: u32,
  channel: u32,
) -> vec4<f32> {
  return vec4<f32>(
    bitcast<f32>(arena[predictor_scratch_index(sequence, slot, channel)]),
    bitcast<f32>(arena[predictor_scratch_index(sequence, slot, channel + 1u)]),
    bitcast<f32>(arena[predictor_scratch_index(sequence, slot, channel + 2u)]),
    bitcast<f32>(arena[predictor_scratch_index(sequence, slot, channel + 3u)])
  );
}

fn sigmoid(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-clamp(value, -20.0, 20.0)));
}

fn sigmoid_vec4(value: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(1.0) /
    (vec4<f32>(1.0) + exp(-clamp(value, vec4<f32>(-20.0), vec4<f32>(20.0))));
}

fn stable_tanh(value: f32) -> f32 {
  // Metal's fast tanh can produce NaN for large finite cell states instead of
  // saturating. At +/-10 the correctly rounded f32 result is already +/-1.
  return tanh(clamp(value, -10.0, 10.0));
}

fn stable_tanh_vec4(value: vec4<f32>) -> vec4<f32> {
  return tanh(clamp(value, vec4<f32>(-10.0), vec4<f32>(10.0)));
}

fn read_gate_vec4(element: u32) -> vec4<f32> {
  return vec4<f32>(
    gate_values[element],
    gate_values[element + 1u],
    gate_values[element + 2u],
    gate_values[element + 3u]
  );
}

fn sanitize_score(value: f32) -> f32 {
  return select(value, -3.402823466e+38, value != value);
}

fn prepare_predictor(
  sequence: u32,
  token: u32,
  initial: bool,
  lane: u32,
) {
  {
${lstmAccumulatorDeclarations("layer0_gate", "params.lstm_bias0_offset")}
    // SOS has exact zero x/h. Its layer-0 matrix product is therefore zero.
    if (!initial) {
      for (var k = 0u; k < ${LSTM_INPUT_SIZE}u; k += 2u) {
        var input_even = 0.0;
        if (k < ${HIDDEN_SIZE}u) {
          input_even = ${readScalarWeight}(
            params.embedding_offset,
            token * ${HIDDEN_SIZE}u + k
          );
        } else {
          input_even = read_state(sequence, 0u, k - ${HIDDEN_SIZE}u);
        }
        let odd_k = k + 1u;
        var input_odd = 0.0;
        if (odd_k < ${HIDDEN_SIZE}u) {
          input_odd = ${readScalarWeight}(
            params.embedding_offset,
            token * ${HIDDEN_SIZE}u + odd_k
          );
        } else {
          input_odd = read_state(sequence, 0u, odd_k - ${HIDDEN_SIZE}u);
        }
${lstmAccumulate("layer0_gate", "params.lstm0_offset", "input_even", "k", "")}
${lstmAccumulate("layer0_gate", "params.lstm0_offset", "input_odd", "odd_k", "_odd")}
      }
    }
${wgslLstmStore("layer0_gate")}
  }
  workgroupBarrier();

  if (lane < ${HIDDEN_SIZE / 4}u) {
    let channel = lane * 4u;
    let input_gate = sigmoid_vec4(read_gate_vec4(channel));
    let forget_gate = sigmoid_vec4(read_gate_vec4(${HIDDEN_SIZE}u + channel));
    let cell_gate = stable_tanh_vec4(
      read_gate_vec4(${HIDDEN_SIZE * 2}u + channel)
    );
    let output_gate = sigmoid_vec4(
      read_gate_vec4(${HIDDEN_SIZE * 3}u + channel)
    );
    let cell = fma(
      forget_gate,
      read_state_vec4(sequence, 1u, channel),
      input_gate * cell_gate
    );
    let hidden = output_gate * stable_tanh_vec4(cell);
    let stable_cell = cell;
    let stable_hidden = hidden;
    candidate_c0[channel] = stable_cell.x;
    candidate_c0[channel + 1u] = stable_cell.y;
    candidate_c0[channel + 2u] = stable_cell.z;
    candidate_c0[channel + 3u] = stable_cell.w;
    candidate_h0[channel] = stable_hidden.x;
    candidate_h0[channel + 1u] = stable_hidden.y;
    candidate_h0[channel + 2u] = stable_hidden.z;
    candidate_h0[channel + 3u] = stable_hidden.w;
  }
  workgroupBarrier();

  {
${lstmAccumulatorDeclarations("layer1_gate", "params.lstm_bias1_offset")}
    // Initial recurrent h1 is also exact zero, so only candidate h0 contributes.
    let layer1_input_limit = select(${LSTM_INPUT_SIZE}u, ${HIDDEN_SIZE}u, initial);
    for (var k = 0u; k < layer1_input_limit; k += 2u) {
      var input_even = 0.0;
      if (k < ${HIDDEN_SIZE}u) {
        input_even = candidate_h0[k];
      } else {
        input_even = read_state(sequence, 2u, k - ${HIDDEN_SIZE}u);
      }
      let odd_k = k + 1u;
      var input_odd = 0.0;
      if (odd_k < ${HIDDEN_SIZE}u) {
        input_odd = candidate_h0[odd_k];
      } else {
        input_odd = read_state(sequence, 2u, odd_k - ${HIDDEN_SIZE}u);
      }
${lstmAccumulate("layer1_gate", "params.lstm1_offset", "input_even", "k", "")}
${lstmAccumulate("layer1_gate", "params.lstm1_offset", "input_odd", "odd_k", "_odd")}
    }
${wgslLstmStore("layer1_gate")}
  }
  workgroupBarrier();

  if (lane < ${HIDDEN_SIZE / 4}u) {
    let channel = lane * 4u;
    let input_gate = sigmoid_vec4(read_gate_vec4(channel));
    let forget_gate = sigmoid_vec4(read_gate_vec4(${HIDDEN_SIZE}u + channel));
    let cell_gate = stable_tanh_vec4(
      read_gate_vec4(${HIDDEN_SIZE * 2}u + channel)
    );
    let output_gate = sigmoid_vec4(
      read_gate_vec4(${HIDDEN_SIZE * 3}u + channel)
    );
    let cell = fma(
      forget_gate,
      read_state_vec4(sequence, 3u, channel),
      input_gate * cell_gate
    );
    let hidden = output_gate * stable_tanh_vec4(cell);
    let stable_cell = cell;
    let stable_hidden = hidden;
    candidate_c1[channel] = stable_cell.x;
    candidate_c1[channel + 1u] = stable_cell.y;
    candidate_c1[channel + 2u] = stable_cell.z;
    candidate_c1[channel + 3u] = stable_cell.w;
    candidate_h1[channel] = stable_hidden.x;
    candidate_h1[channel + 1u] = stable_hidden.y;
    candidate_h1[channel + 2u] = stable_hidden.z;
    candidate_h1[channel + 3u] = stable_hidden.w;
  }
  workgroupBarrier();

  if (lane < ${HIDDEN_SIZE / 4}u) {
    let output = lane * 4u;
    for (var component = 0u; component < 4u; component += 1u) {
      let channel = output + component;
      arena[predictor_scratch_index(sequence, 0u, channel)] =
        bitcast<u32>(candidate_h0[channel]);
      arena[predictor_scratch_index(sequence, 1u, channel)] =
        bitcast<u32>(candidate_c0[channel]);
      arena[predictor_scratch_index(sequence, 2u, channel)] =
        bitcast<u32>(candidate_h1[channel]);
      arena[predictor_scratch_index(sequence, 3u, channel)] =
        bitcast<u32>(candidate_c1[channel]);
    }
    var projection = read_weight_f32x4(
      params.decoder_projector_bias_offset,
      output
    );
    for (var k = 0u; k < ${HIDDEN_SIZE}u; k += 1u) {
      projection = fma(
        vec4<f32>(candidate_h1[k]),
        read_weight_vec4(
          params.decoder_projector_offset,
          k * ${HIDDEN_SIZE}u + output
        ),
        projection
      );
    }
    arena[predictor_scratch_index(sequence, 4u, output)] =
      bitcast<u32>(projection.x);
    arena[predictor_scratch_index(sequence, 4u, output + 1u)] =
      bitcast<u32>(projection.y);
    arena[predictor_scratch_index(sequence, 4u, output + 2u)] =
      bitcast<u32>(projection.z);
    arena[predictor_scratch_index(sequence, 4u, output + 3u)] =
      bitcast<u32>(projection.w);
  }
  storageBarrier();
  workgroupBarrier();
}

fn commit_predictor(sequence: u32, lane: u32) {
  for (var channel = lane; channel < ${HIDDEN_SIZE}u; channel += ${WORKGROUP_SIZE}u) {
    arena[state_index(sequence, 0u, channel)] =
      arena[predictor_scratch_index(sequence, 0u, channel)];
    arena[state_index(sequence, 1u, channel)] =
      arena[predictor_scratch_index(sequence, 1u, channel)];
    arena[state_index(sequence, 2u, channel)] =
      arena[predictor_scratch_index(sequence, 2u, channel)];
    arena[state_index(sequence, 3u, channel)] =
      arena[predictor_scratch_index(sequence, 3u, channel)];
  }
  storageBarrier();
  workgroupBarrier();
}

fn choose_better(
  left_value: f32,
  left_id: u32,
  right_value: f32,
  right_id: u32,
) -> bool {
  return right_value > left_value ||
    (right_value == left_value && right_id < left_id);
}

fn reduce_joint_result(
  pair_slot: u32,
  lane: u32,
  subgroup_lane: u32,
  subgroup_slot: u32,
  local_token_value: f32,
  local_token_id: u32,
  local_duration_value: f32,
  local_duration_id: u32,
) {
  let subgroup_token_value = subgroupMax(local_token_value);
  let subgroup_token_id = subgroupMin(
    select(
      0xffffffffu,
      local_token_id,
      local_token_value == subgroup_token_value
    )
  );
  let subgroup_duration_value = subgroupMax(local_duration_value);
  let subgroup_duration_id = subgroupMin(
    select(
      0xffffffffu,
      local_duration_id,
      local_duration_value == subgroup_duration_value
    )
  );
  if (subgroup_lane == 0u) {
    token_values[subgroup_slot] = subgroup_token_value;
    token_ids[subgroup_slot] = subgroup_token_id;
    duration_values[subgroup_slot] = subgroup_duration_value;
    duration_ids[subgroup_slot] = subgroup_duration_id;
  }
  workgroupBarrier();

  if (lane == 0u) {
    var token_value = -3.402823466e+38;
    var token_id = 0xffffffffu;
    var duration_value = -3.402823466e+38;
    var duration_id = 0xffffffffu;
    for (
      var subgroup = 0u;
      subgroup < ${SUBGROUPS_PER_WORKGROUP}u;
      subgroup += 1u
    ) {
      if (
        choose_better(
          token_value,
          token_id,
          token_values[subgroup],
          token_ids[subgroup]
        )
      ) {
        token_value = token_values[subgroup];
        token_id = token_ids[subgroup];
      }
      if (
        choose_better(
          duration_value,
          duration_id,
          duration_values[subgroup],
          duration_ids[subgroup]
        )
      ) {
        duration_value = duration_values[subgroup];
        duration_id = duration_ids[subgroup];
      }
    }
    if (pair_slot == 0u) {
      best_token_a = token_id;
      best_duration_a = duration_id;
    } else {
      best_token_b = token_id;
      best_duration_b = duration_id;
    }
  }
  workgroupBarrier();
}

fn run_joint_pair(
  sequence_a: u32,
  sequence_b: u32,
  active_a: bool,
  active_b: bool,
  lane: u32,
  subgroup_lane: u32,
  subgroup_slot: u32,
) {
  if (lane < ${HIDDEN_SIZE / 4}u) {
    let channel = lane * 4u;
    var joint_a = vec4<f32>(0.0);
    if (active_a) {
      let encoder_element =
        (sequence_a * params.frames + decode_time_a) * ${HIDDEN_SIZE}u +
        channel;
      let encoder_value = read_arena_vec4(
        params.encoder_offset,
        encoder_element
      );
      let decoder_value = read_predictor_scratch_vec4(
        sequence_a,
        4u,
        channel
      );
      joint_a = max(encoder_value + decoder_value, vec4<f32>(0.0));
    }
    joint_input_a[channel] = joint_a.x;
    joint_input_a[channel + 1u] = joint_a.y;
    joint_input_a[channel + 2u] = joint_a.z;
    joint_input_a[channel + 3u] = joint_a.w;

    var joint_b = vec4<f32>(0.0);
    if (active_b) {
      let encoder_element =
        (sequence_b * params.frames + decode_time_b) * ${HIDDEN_SIZE}u +
        channel;
      let encoder_value = read_arena_vec4(
        params.encoder_offset,
        encoder_element
      );
      let decoder_value = read_predictor_scratch_vec4(
        sequence_b,
        4u,
        channel
      );
      joint_b = max(encoder_value + decoder_value, vec4<f32>(0.0));
    }
    joint_input_b[channel] = joint_b.x;
    joint_input_b[channel + 1u] = joint_b.y;
    joint_input_b[channel + 2u] = joint_b.z;
    joint_input_b[channel + 3u] = joint_b.w;
  }
  workgroupBarrier();

${pairedHeadBody}
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let sequence_a = group.x * 2u;
  let sequence_b = sequence_a + 1u;
  if (sequence_a >= params.batch_size) {
    return;
  }
  let has_sequence_b = sequence_b < params.batch_size;
  for (var item = lane; item < ${4 * HIDDEN_SIZE}u; item += ${WORKGROUP_SIZE}u) {
    arena[params.state_offset + sequence_a * ${4 * HIDDEN_SIZE}u + item] = 0u;
    if (has_sequence_b) {
      arena[params.state_offset + sequence_b * ${4 * HIDDEN_SIZE}u + item] = 0u;
    }
  }
  if (lane == 0u) {
    output_count_a = 0u;
    output_count_b = 0u;
    decode_status_a = 0u;
    decode_status_b = 0u;
    symbols_at_time_a = 0u;
    symbols_at_time_b = 0u;
    emitted_nonblank_a = 0u;
    emitted_nonblank_b = 0u;
    encoded_length_a =
      min(
        min(arena[params.lengths_offset + sequence_a], params.frames),
        arena[params.decode_ends_offset + sequence_a]
      );
    decode_time_a =
      min(arena[params.decode_starts_offset + sequence_a], encoded_length_a);
    encoded_length_b = 0u;
    decode_time_b = 0u;
    if (has_sequence_b) {
      encoded_length_b =
        min(
          min(arena[params.lengths_offset + sequence_b], params.frames),
          arena[params.decode_ends_offset + sequence_b]
        );
      decode_time_b =
        min(arena[params.decode_starts_offset + sequence_b], encoded_length_b);
    }
  }
  storageBarrier();
  workgroupBarrier();

  // Chrome/Metal exposes fixed subgroup-32 execution on the target Apple GPU.
  let subgroup_slot = lane / ${SUBGROUP_SIZE}u;

  // Prime each two-layer LSTM sequentially through the one predictor workspace.
  if (
    workgroupUniformLoad(&decode_time_a) <
    workgroupUniformLoad(&encoded_length_a)
  ) {
    prepare_predictor(sequence_a, 0u, true, lane);
  }
  if (
    workgroupUniformLoad(&decode_time_b) <
    workgroupUniformLoad(&encoded_length_b)
  ) {
    prepare_predictor(sequence_b, 0u, true, lane);
  }

  loop {
    let active_a =
      workgroupUniformLoad(&decode_time_a) <
      workgroupUniformLoad(&encoded_length_a);
    let active_b =
      workgroupUniformLoad(&decode_time_b) <
      workgroupUniformLoad(&encoded_length_b);
    if (!active_a && !active_b) {
      break;
    }
    run_joint_pair(
      sequence_a,
      sequence_b,
      active_a,
      active_b,
      lane,
      subgroup_lane,
      subgroup_slot
    );

    if (lane == 0u) {
      emitted_nonblank_a = 0u;
      emitted_nonblank_b = 0u;
      if (active_a) {
        let emission_time = decode_time_a;
        if (best_token_a != ${BLANK_TOKEN}u) {
          emitted_nonblank_a = 1u;
          if (output_count_a < params.max_tokens) {
            let output = sequence_a * params.max_tokens + output_count_a;
            arena[params.tokens_offset + output] = best_token_a;
            arena[params.token_frames_offset + output] = emission_time;
            arena[params.token_durations_offset + output] = best_duration_a;
            output_count_a += 1u;
          } else {
            decode_status_a |= ${TDT_DECODER_OVERFLOW}u;
          }
        }
        symbols_at_time_a += 1u;
        if (best_duration_a != 0u) {
          decode_time_a += best_duration_a;
          symbols_at_time_a = 0u;
        } else if (
          best_token_a == ${BLANK_TOKEN}u ||
          symbols_at_time_a >= 10u
        ) {
          decode_time_a += 1u;
          symbols_at_time_a = 0u;
        }
      }
      if (active_b) {
        let emission_time = decode_time_b;
        if (best_token_b != ${BLANK_TOKEN}u) {
          emitted_nonblank_b = 1u;
          if (output_count_b < params.max_tokens) {
            let output = sequence_b * params.max_tokens + output_count_b;
            arena[params.tokens_offset + output] = best_token_b;
            arena[params.token_frames_offset + output] = emission_time;
            arena[params.token_durations_offset + output] = best_duration_b;
            output_count_b += 1u;
          } else {
            decode_status_b |= ${TDT_DECODER_OVERFLOW}u;
          }
        }
        symbols_at_time_b += 1u;
        if (best_duration_b != 0u) {
          decode_time_b += best_duration_b;
          symbols_at_time_b = 0u;
        } else if (
          best_token_b == ${BLANK_TOKEN}u ||
          symbols_at_time_b >= 10u
        ) {
          decode_time_b += 1u;
          symbols_at_time_b = 0u;
        }
      }
    }
    workgroupBarrier();

    if (workgroupUniformLoad(&emitted_nonblank_a) != 0u) {
      let next_token = workgroupUniformLoad(&best_token_a);
      commit_predictor(sequence_a, lane);
      prepare_predictor(sequence_a, next_token, false, lane);
    }
    if (workgroupUniformLoad(&emitted_nonblank_b) != 0u) {
      let next_token = workgroupUniformLoad(&best_token_b);
      commit_predictor(sequence_b, lane);
      prepare_predictor(sequence_b, next_token, false, lane);
    }
  }

  // FluidAudio's v2/v3 decoder uses a bounded final-chunk continuation after
  // normal frame exhaustion. This adaptation preserves the native WebGPU decoder's
  // predictor state, while matching its probe-frame cycle and stopping rules.
  // Most windows skip this uniform branch entirely.
  // Stage storage-backed flags through workgroup memory so all barrier-
  // containing control flow is visibly uniform to WGSL validation.
  if (lane == 0u) {
    emitted_nonblank_a = select(
      0u,
      1u,
      arena[params.flush_finals_offset + sequence_a] != 0u &&
        arena[params.decode_starts_offset + sequence_a] < encoded_length_a
    );
    emitted_nonblank_b = 0u;
    if (has_sequence_b) {
      emitted_nonblank_b = select(
        0u,
        1u,
        arena[params.flush_finals_offset + sequence_b] != 0u &&
          arena[params.decode_starts_offset + sequence_b] < encoded_length_b
      );
    }
  }
  workgroupBarrier();
  let flush_requested_a =
    workgroupUniformLoad(&emitted_nonblank_a) != 0u;
  let flush_requested_b =
    workgroupUniformLoad(&emitted_nonblank_b) != 0u;
  if (flush_requested_a || flush_requested_b) {
    if (lane == 0u) {
      symbols_at_time_a = 0u;
      symbols_at_time_b = 0u;
      flush_blanks_a = 0u;
      flush_blanks_b = 0u;
      flush_time_a = decode_time_a;
      flush_time_b = decode_time_b;
    }
    workgroupBarrier();

    loop {
      let flush_active_a =
        flush_requested_a &&
        workgroupUniformLoad(&symbols_at_time_a) < 10u &&
        workgroupUniformLoad(&flush_blanks_a) < 5u;
      let flush_active_b =
        flush_requested_b &&
        workgroupUniformLoad(&symbols_at_time_b) < 10u &&
        workgroupUniformLoad(&flush_blanks_b) < 5u;
      if (!flush_active_a && !flush_active_b) {
        break;
      }

      if (lane == 0u) {
        if (flush_active_a) {
          let variation = symbols_at_time_a % 3u;
          if (variation == 0u) {
            decode_time_a = min(
              flush_time_a,
              min(
                arena[params.lengths_offset + sequence_a],
                params.frames
              ) - 1u
            );
          } else if (variation == 1u) {
            decode_time_a = min(encoded_length_a - 1u, params.frames - 1u);
          } else {
            decode_time_a = min(
              max(encoded_length_a, 2u) - 2u,
              params.frames - 1u
            );
          }
        }
        if (flush_active_b) {
          let variation = symbols_at_time_b % 3u;
          if (variation == 0u) {
            decode_time_b = min(
              flush_time_b,
              min(
                arena[params.lengths_offset + sequence_b],
                params.frames
              ) - 1u
            );
          } else if (variation == 1u) {
            decode_time_b = min(encoded_length_b - 1u, params.frames - 1u);
          } else {
            decode_time_b = min(
              max(encoded_length_b, 2u) - 2u,
              params.frames - 1u
            );
          }
        }
      }
      workgroupBarrier();
      run_joint_pair(
        sequence_a,
        sequence_b,
        flush_active_a,
        flush_active_b,
        lane,
        subgroup_lane,
        subgroup_slot
      );

      if (lane == 0u) {
        emitted_nonblank_a = 0u;
        emitted_nonblank_b = 0u;
        if (flush_active_a) {
          if (best_token_a == ${BLANK_TOKEN}u) {
            flush_blanks_a += 1u;
          } else {
            flush_blanks_a = 0u;
            emitted_nonblank_a = 1u;
            if (output_count_a < params.max_tokens) {
              let output = sequence_a * params.max_tokens + output_count_a;
              arena[params.tokens_offset + output] = best_token_a;
              arena[params.token_frames_offset + output] =
                min(flush_time_a, encoded_length_a - 1u);
              arena[params.token_durations_offset + output] = best_duration_a;
              output_count_a += 1u;
            } else {
              decode_status_a |= ${TDT_DECODER_OVERFLOW}u;
            }
          }
          flush_time_a = min(
            flush_time_a + max(1u, best_duration_a),
            encoded_length_a
          );
          symbols_at_time_a += 1u;
        }
        if (flush_active_b) {
          if (best_token_b == ${BLANK_TOKEN}u) {
            flush_blanks_b += 1u;
          } else {
            flush_blanks_b = 0u;
            emitted_nonblank_b = 1u;
            if (output_count_b < params.max_tokens) {
              let output = sequence_b * params.max_tokens + output_count_b;
              arena[params.tokens_offset + output] = best_token_b;
              arena[params.token_frames_offset + output] =
                min(flush_time_b, encoded_length_b - 1u);
              arena[params.token_durations_offset + output] = best_duration_b;
              output_count_b += 1u;
            } else {
              decode_status_b |= ${TDT_DECODER_OVERFLOW}u;
            }
          }
          flush_time_b = min(
            flush_time_b + max(1u, best_duration_b),
            encoded_length_b
          );
          symbols_at_time_b += 1u;
        }
      }
      workgroupBarrier();

      if (workgroupUniformLoad(&emitted_nonblank_a) != 0u) {
        let next_token = workgroupUniformLoad(&best_token_a);
        commit_predictor(sequence_a, lane);
        prepare_predictor(sequence_a, next_token, false, lane);
      }
      if (workgroupUniformLoad(&emitted_nonblank_b) != 0u) {
        let next_token = workgroupUniformLoad(&best_token_b);
        commit_predictor(sequence_b, lane);
        prepare_predictor(sequence_b, next_token, false, lane);
      }
    }
  }

  if (lane == 0u) {
    arena[params.counts_offset + sequence_a] = output_count_a;
    arena[params.status_offset + sequence_a] = decode_status_a;
    if (has_sequence_b) {
      arena[params.counts_offset + sequence_b] = output_count_b;
      arena[params.status_offset + sequence_b] = decode_status_b;
    }
  }
}
`;
}

/**
 * Construct the standards-only decoder from the accepted precision shader.
 * The four 256-lane candidate arrays stay below the existing 32 KiB device
 * requirement. Eight lanes independently scan one 32-candidate segment, then
 * lane zero folds those winners in increasing order. Argmax comparisons do
 * not round, so this three-barrier reduction preserves both the maximum value
 * and lowest-ID tie rule without paying for five shared-memory tree levels.
 * Keeping this as a post-generation specialization leaves
 * both established subgroup shader strings byte-for-byte unchanged.
 */
function portableTdtDecoderWgsl(
  precision: ParakeetExecutionProfile["precision"],
): string {
  let result =
    precision === "fp16" ? tdtDecoderWgsl() : tdtDecoderFp32Wgsl();
  result = replacePortableRequired(result, "enable subgroups;\n", "");
  for (const declaration of [
    "token_values: array<f32",
    "token_ids: array<u32",
    "duration_values: array<f32",
    "duration_ids: array<u32",
  ]) {
    result = replacePortableRequired(
      result,
      `var<workgroup> ${declaration}, ${SUBGROUPS_PER_WORKGROUP}>;`,
      `var<workgroup> ${declaration}, ${WORKGROUP_SIZE}>;`,
    );
  }

  const reductionStart = result.indexOf("fn reduce_joint_result(");
  const jointStart = result.indexOf("fn run_joint_pair(", reductionStart);
  if (reductionStart < 0 || jointStart < 0) {
    throw new Error("Unable to construct portable TDT reduction");
  }
  result =
    result.slice(0, reductionStart) +
    portableTdtReductionWgsl() +
    "\n\n" +
    result.slice(jointStart);

  result = replacePortableRequired(
    result,
    `  lane: u32,
  subgroup_lane: u32,
  subgroup_slot: u32,
) {`,
    `  lane: u32,
) {`,
  );
  result = result.replaceAll(
    `    lane,
    subgroup_lane,
    subgroup_slot,
    local_token_value_`,
    `    lane,
    local_token_value_`,
  );
  result = replacePortableRequired(
    result,
    `  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,`,
    `  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,`,
  );
  result = replacePortableRequired(
    result,
    `  // Chrome/Metal exposes fixed subgroup-32 execution on the target Apple GPU.
  let subgroup_slot = lane / ${SUBGROUP_SIZE}u;

`,
    "",
  );
  result = result.replaceAll(
    `      lane,
      subgroup_lane,
      subgroup_slot
    );`,
    `      lane
    );`,
  );
  result = result.replaceAll(
    `        lane,
        subgroup_lane,
        subgroup_slot
      );`,
    `        lane
      );`,
  );
  const subgroupIndex = result.indexOf("subgroup");
  if (subgroupIndex >= 0) {
    throw new Error(
      "Portable TDT shader retained a subgroup dependency near: " +
        result.slice(Math.max(0, subgroupIndex - 40), subgroupIndex + 80),
    );
  }
  return result;
}

function portableTdtReductionWgsl(): string {
  return `fn reduce_joint_result(
  pair_slot: u32,
  lane: u32,
  local_token_value: f32,
  local_token_id: u32,
  local_duration_value: f32,
  local_duration_id: u32,
) {
  token_values[lane] = local_token_value;
  token_ids[lane] = local_token_id;
  duration_values[lane] = local_duration_value;
  duration_ids[lane] = local_duration_id;
  workgroupBarrier();

  if (lane < ${SUBGROUPS_PER_WORKGROUP}u) {
    let base = lane * ${SUBGROUP_SIZE}u;
    var token_value = token_values[base];
    var token_id = token_ids[base];
    var duration_value = duration_values[base];
    var duration_id = duration_ids[base];
    for (var member = 1u; member < ${SUBGROUP_SIZE}u; member += 1u) {
      let right = base + member;
      if (
        choose_better(
          token_value,
          token_id,
          token_values[right],
          token_ids[right]
        )
      ) {
        token_value = token_values[right];
        token_id = token_ids[right];
      }
      if (
        choose_better(
          duration_value,
          duration_id,
          duration_values[right],
          duration_ids[right]
        )
      ) {
        duration_value = duration_values[right];
        duration_id = duration_ids[right];
      }
    }
    token_values[base] = token_value;
    token_ids[base] = token_id;
    duration_values[base] = duration_value;
    duration_ids[base] = duration_id;
  }
  workgroupBarrier();

  if (lane == 0u) {
    var token_value = -3.402823466e+38;
    var token_id = 0xffffffffu;
    var duration_value = -3.402823466e+38;
    var duration_id = 0xffffffffu;
    for (
      var virtual_group = 0u;
      virtual_group < ${SUBGROUPS_PER_WORKGROUP}u;
      virtual_group += 1u
    ) {
      let winner = virtual_group * ${SUBGROUP_SIZE}u;
      if (
        choose_better(
          token_value,
          token_id,
          token_values[winner],
          token_ids[winner]
        )
      ) {
        token_value = token_values[winner];
        token_id = token_ids[winner];
      }
      if (
        choose_better(
          duration_value,
          duration_id,
          duration_values[winner],
          duration_ids[winner]
        )
      ) {
        duration_value = duration_values[winner];
        duration_id = duration_ids[winner];
      }
    }
    if (pair_slot == 0u) {
      best_token_a = token_id;
      best_duration_a = duration_id;
    } else {
      best_token_b = token_id;
      best_duration_b = duration_id;
    }
  }
  workgroupBarrier();
}`;
}

function replacePortableRequired(
  source: string,
  search: string,
  replacement: string,
): string {
  if (!source.includes(search)) {
    throw new Error(`Unable to construct portable TDT shader: ${search}`);
  }
  return source.replace(search, replacement);
}
