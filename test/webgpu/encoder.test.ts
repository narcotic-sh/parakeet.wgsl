import { describe, expect, it } from "vitest";

import {
  deriveEncoderValidLength,
  deriveEncoderValidLengths,
  deriveSubsamplingLengths,
  PARAKEET_ENCODER_BATCH_CAPACITY,
  PARAKEET_ENCODER_CHANNELS,
  PARAKEET_ENCODER_FRAMES,
  PARAKEET_ENCODER_LAYERS,
  PARAKEET_ENCODER_SUBMISSION_CHUNKS,
  PARAKEET_FEATURE_BINS,
  PARAKEET_FEATURE_FRAMES,
  PARAKEET_INTERMEDIATE_CHANNELS,
  PARAKEET_PROJECTED_CHANNELS,
  PARAKEET_SUBSAMPLING_MICROBATCH_SIZE,
  planEncoderB1Checkpoints,
  planEncoderLayerArena,
  planEncoderLayerStructure,
  planEncoderSubmissionChunks,
  planEncoderUniformParameterPools,
  planParakeetEncoderExecution,
  planParakeetEncoderArena,
  planSubsamplingDispatchStructure,
  subsamplingSliceNeedsFinalMask,
} from "../../src/webgpu/encoder";

describe("static Parakeet encoder dimensions", () => {
  it("plans one backend-neutral submission chunk per stable encoder boundary", () => {
    const chunks = planEncoderSubmissionChunks();

    expect(PARAKEET_ENCODER_SUBMISSION_CHUNKS).toBe(26);
    expect(chunks).toHaveLength(PARAKEET_ENCODER_SUBMISSION_CHUNKS);
    expect(chunks.map(({ label }) => label)).toEqual([
      "parakeet-encoder-subsampling",
      ...Array.from(
        { length: PARAKEET_ENCODER_LAYERS },
        (_, layer) =>
          `parakeet-encoder-layer-${layer.toString().padStart(2, "0")}`,
      ),
      "parakeet-encoder-projector",
    ]);
    expect(chunks[0]).toEqual({
      kind: "subsampling",
      label: "parakeet-encoder-subsampling",
      layerIndex: null,
    });
    expect(chunks.slice(1, -1).map(({ layerIndex }) => layerIndex)).toEqual(
      Array.from(
        { length: PARAKEET_ENCODER_LAYERS },
        (_, layer) => layer,
      ),
    );
    expect(chunks.at(-1)).toEqual({
      kind: "projector",
      label: "parakeet-encoder-projector",
      layerIndex: null,
    });
    expect(planEncoderSubmissionChunks()).toBe(chunks);
  });

  it("plans a bounded B1 diagnostic capture at every stable boundary", () => {
    const fp16 = planEncoderB1Checkpoints("fp16");
    const fp32 = planEncoderB1Checkpoints("fp32");

    expect(fp16.checkpoints).toHaveLength(PARAKEET_ENCODER_LAYERS + 2);
    expect(fp16.checkpoints.map(({ label }) => label)).toEqual([
      "subsampling",
      ...Array.from(
        { length: PARAKEET_ENCODER_LAYERS },
        (_, layer) => `layer-${layer.toString().padStart(2, "0")}`,
      ),
      "projector",
    ]);
    expect(fp16.checkpoints[0]).toMatchObject({
      kind: "subsampling",
      layerIndex: null,
      rows: PARAKEET_ENCODER_FRAMES,
      channels: PARAKEET_ENCODER_CHANNELS,
      elementOffset: 0,
      elementCount:
        PARAKEET_ENCODER_FRAMES * PARAKEET_ENCODER_CHANNELS,
      byteOffset: 0,
      byteLength:
        PARAKEET_ENCODER_FRAMES * PARAKEET_ENCODER_CHANNELS * 2,
    });
    expect(fp16.checkpoints[1]).toMatchObject({
      kind: "layer",
      layerIndex: 0,
    });
    expect(fp16.checkpoints.at(-1)).toMatchObject({
      kind: "projector",
      layerIndex: null,
      channels: PARAKEET_PROJECTED_CHANNELS,
      elementCount:
        PARAKEET_ENCODER_FRAMES * PARAKEET_PROJECTED_CHANNELS,
    });
    expect(fp16).toMatchObject({
      precision: "fp16",
      elementBytes: 2,
      elementCount: 4_933_120,
      byteLength: 9_866_240,
      widenedFloat32ByteLength: 19_732_480,
    });
    expect(fp32).toMatchObject({
      precision: "fp32",
      elementBytes: 4,
      elementCount: fp16.elementCount,
      byteLength: 19_732_480,
      widenedFloat32ByteLength: 19_732_480,
    });
    expect(
      fp32.checkpoints.map(({ elementOffset }) => elementOffset),
    ).toEqual(
      fp16.checkpoints.map(({ elementOffset }) => elementOffset),
    );
    for (let index = 0; index < fp16.checkpoints.length; index += 1) {
      expect(fp32.checkpoints[index]!.byteOffset).toBe(
        fp16.checkpoints[index]!.byteOffset * 2,
      );
      expect(fp32.checkpoints[index]!.byteLength).toBe(
        fp16.checkpoints[index]!.byteLength * 2,
      );
    }
  });

  it("uses six bounded non-GEMM parameter pools", () => {
    expect(planEncoderUniformParameterPools(256)).toEqual({
      bufferCount: 6,
      uniqueSlotCount: 123,
      byteLength: 31_488,
    });
    expect(
      planEncoderUniformParameterPools(512).byteLength,
    ).toBe(62_976);
  });

  it("derives all three same-padded stride-2 lengths", () => {
    expect(deriveSubsamplingLengths(1501)).toEqual([751, 376, 188]);
    expect(deriveSubsamplingLengths(1500)).toEqual([750, 375, 188]);
    expect(deriveSubsamplingLengths(1)).toEqual([1, 1, 1]);
    expect(deriveEncoderValidLength(1497)).toBe(188);
    expect([...deriveEncoderValidLengths([1501, 800, 1])]).toEqual([
      188, 100, 1,
    ]);
  });

  it("masks only lengths with padded rows after all three strides", () => {
    const rawLengths = [1, 799, 1496, 1497, 1500, 1501];
    const encoderLengths = deriveEncoderValidLengths(rawLengths);
    expect([...encoderLengths]).toEqual([1, 100, 187, 188, 188, 188]);
    expect(
      rawLengths.map((_, batchOffset) =>
        subsamplingSliceNeedsFinalMask(
          encoderLengths,
          batchOffset,
          1,
        ),
      ),
    ).toEqual([true, true, true, false, false, false]);
  });

  it("gates final masks only within active mixed batch slices", () => {
    const mixed = deriveEncoderValidLengths([1501, 1500, 1496, 1497, 1]);
    expect(subsamplingSliceNeedsFinalMask(mixed, 0, 2)).toBe(false);
    expect(subsamplingSliceNeedsFinalMask(mixed, 1, 3)).toBe(true);
    expect(subsamplingSliceNeedsFinalMask(mixed, 3, 1)).toBe(false);
    expect(subsamplingSliceNeedsFinalMask(mixed, 3, 2)).toBe(true);

    const structure = planSubsamplingDispatchStructure();
    expect(structure).toEqual({
      unconditionalPerMicrobatch: 5,
      conditionalFinalMasksPerMicrobatch: 1,
    });
  });

  it("plans one dynamic active-prefix path for every B1-B40 tail", () => {
    const executions = Array.from(
      { length: PARAKEET_ENCODER_BATCH_CAPACITY },
      (_, index) => planParakeetEncoderExecution(index + 1),
    );
    expect(executions[0]).toEqual({
      batchSize: 1,
      rows: PARAKEET_ENCODER_FRAMES,
      microbatchCount: 1,
    });
    expect(executions[38]).toEqual({
      batchSize: 39,
      rows: 39 * PARAKEET_ENCODER_FRAMES,
      microbatchCount: 13,
    });
    expect(executions[39]).toEqual({
      batchSize: 40,
      rows: 40 * PARAKEET_ENCODER_FRAMES,
      microbatchCount: 14,
    });
    expect(
      executions.map(({ rows }) => rows),
    ).toEqual(
      Array.from(
        { length: PARAKEET_ENCODER_BATCH_CAPACITY },
        (_, index) => (index + 1) * PARAKEET_ENCODER_FRAMES,
      ),
    );
    expect(
      executions.map(({ microbatchCount }) => microbatchCount),
    ).toEqual(
      Array.from(
        { length: PARAKEET_ENCODER_BATCH_CAPACITY },
        (_, index) =>
          Math.ceil(
            (index + 1) / PARAKEET_SUBSAMPLING_MICROBATCH_SIZE,
          ),
      ),
    );
    expect(() => planParakeetEncoderExecution(0)).toThrow(/active batch/);
    expect(() =>
      planParakeetEncoderExecution(PARAKEET_ENCODER_BATCH_CAPACITY + 1),
    ).toThrow(/active batch/);
  });

  it("rejects invalid feature lengths and batch cardinality", () => {
    expect(() => deriveEncoderValidLength(0)).toThrow(/positive/);
    expect(() => deriveEncoderValidLength(PARAKEET_FEATURE_FRAMES + 1)).toThrow(
      /must not exceed/,
    );
    expect(() => deriveEncoderValidLengths([1501], 2)).toThrow(/Expected 2/);
    expect(() => subsamplingSliceNeedsFinalMask([188], -1, 1)).toThrow(
      /offset/,
    );
    expect(() => subsamplingSliceNeedsFinalMask([188], 0, 0)).toThrow(
      /positive/,
    );
    expect(() => subsamplingSliceNeedsFinalMask([188], 1, 1)).toThrow(
      /exceeds/,
    );
    expect(() => subsamplingSliceNeedsFinalMask([0], 0, 1)).toThrow(
      /valid length/,
    );
    expect(() => subsamplingSliceNeedsFinalMask([189], 0, 1)).toThrow(
      /valid length/,
    );
    expect(() => subsamplingSliceNeedsFinalMask([1, 189], 0, 2)).toThrow(
      /valid length/,
    );
  });
});

describe("overlapped Parakeet encoder arena", () => {
  it("plans feature-major input and exact B3/B1/B2 command variants", () => {
    const plan = planParakeetEncoderArena();
    const perBatchEncoderBytes =
      PARAKEET_ENCODER_FRAMES * PARAKEET_ENCODER_CHANNELS * 2;

    expect(plan.rows).toBe(
      PARAKEET_ENCODER_BATCH_CAPACITY * PARAKEET_ENCODER_FRAMES,
    );
    expect(plan.subsamplingMicrobatchSize).toBe(
      PARAKEET_SUBSAMPLING_MICROBATCH_SIZE,
    );
    expect(plan.featureInput.byteLength).toBe(
      PARAKEET_ENCODER_BATCH_CAPACITY *
        PARAKEET_FEATURE_BINS *
        PARAKEET_FEATURE_FRAMES *
        4,
    );
    expect(plan.microbatches).toHaveLength(PARAKEET_ENCODER_BATCH_CAPACITY);
    expect(plan.microbatches.at(0)).toMatchObject({
      batchOffset: 0,
      batchSize: 1,
      output: {
        label: "encoder-slot-0-batch-0-size-1",
        byteOffset: plan.encoderSlots[0].byteOffset,
        byteLength: perBatchEncoderBytes,
      },
    });
    expect(plan.microbatches.at(1)).toMatchObject({
      batchOffset: 0,
      batchSize: 2,
      output: {
        label: "encoder-slot-0-batch-0-size-2",
        byteOffset: plan.encoderSlots[0].byteOffset,
        byteLength: 2 * perBatchEncoderBytes,
      },
    });
    expect(plan.microbatches.at(2)).toMatchObject({
      batchOffset: 0,
      batchSize: 3,
      output: {
        label: "encoder-slot-0-batch-0-size-3",
        byteOffset: plan.encoderSlots[0].byteOffset,
        byteLength: 3 * perBatchEncoderBytes,
      },
    });
    expect(plan.microbatches.at(-1)).toMatchObject({
      batchOffset: PARAKEET_ENCODER_BATCH_CAPACITY - 1,
      batchSize: 1,
      output: {
        label:
          `encoder-slot-0-batch-${PARAKEET_ENCODER_BATCH_CAPACITY - 1}-size-1`,
        byteOffset:
          plan.encoderSlots[0].byteOffset +
          (PARAKEET_ENCODER_BATCH_CAPACITY - 1) * perBatchEncoderBytes,
        byteLength: perBatchEncoderBytes,
      },
    });
    for (const [index, microbatch] of plan.microbatches.entries()) {
      const activeBatchEnd = index + 1;
      const expectedBatchSize =
        activeBatchEnd % PARAKEET_SUBSAMPLING_MICROBATCH_SIZE ||
        PARAKEET_SUBSAMPLING_MICROBATCH_SIZE;
      const expectedBatchOffset = activeBatchEnd - expectedBatchSize;
      expect(microbatch.batchOffset).toBe(expectedBatchOffset);
      expect(microbatch.batchSize).toBe(expectedBatchSize);
      expect(microbatch.output.byteOffset).toBe(
        plan.encoderSlots[0].byteOffset +
          expectedBatchOffset * perBatchEncoderBytes,
      );
      expect(microbatch.output.byteOffset % 256).toBe(0);
      expect(microbatch.output.byteLength).toBe(
        expectedBatchSize * perBatchEncoderBytes,
      );
      expect(overlaps(microbatch.output, plan.subsampling.projected188)).toBe(
        false,
      );
      expect(
        microbatch.output.byteOffset + microbatch.output.byteLength,
      ).toBe(
        plan.encoderSlots[0].byteOffset +
          activeBatchEnd * perBatchEncoderBytes,
      );
    }
    const finalOutput = plan.microbatches.at(-1)!.output;
    expect(finalOutput.byteOffset + finalOutput.byteLength).toBe(
      plan.encoderSlots[0].byteOffset +
        plan.encoderSlots[0].byteLength,
    );

    for (
      let activeBatch = 1;
      activeBatch <= PARAKEET_ENCODER_BATCH_CAPACITY;
      activeBatch += 1
    ) {
      const execution = planParakeetEncoderExecution(activeBatch);
      let coveredBatchEnd = 0;
      for (
        let microbatchIndex = 0;
        microbatchIndex < execution.microbatchCount;
        microbatchIndex += 1
      ) {
        const activeBatchEnd = Math.min(
          (microbatchIndex + 1) *
            PARAKEET_SUBSAMPLING_MICROBATCH_SIZE,
          activeBatch,
        );
        const command = plan.microbatches[activeBatchEnd - 1]!;
        expect(command.batchOffset).toBe(coveredBatchEnd);
        coveredBatchEnd += command.batchSize;
        expect(coveredBatchEnd).toBe(activeBatchEnd);
      }
      expect(coveredBatchEnd).toBe(activeBatch);
    }
  });

  it("uses two alternating 1024 slots and one 4096-channel temporary", () => {
    const plan = planParakeetEncoderArena();
    const slotBytes =
      plan.rows * PARAKEET_ENCODER_CHANNELS * Uint16Array.BYTES_PER_ELEMENT;

    expect(plan.encoderSlots).toHaveLength(2);
    expect(plan.encoderSlots.map((slot) => slot.byteLength)).toEqual([
      slotBytes,
      slotBytes,
    ]);
    expect(plan.bigTemporary.byteLength).toBe(
      plan.rows *
        PARAKEET_INTERMEDIATE_CHANNELS *
        Uint16Array.BYTES_PER_ELEMENT,
    );
    expect(plan.projectedOutput.byteLength).toBe(
      plan.rows * PARAKEET_PROJECTED_CHANNELS * Uint16Array.BYTES_PER_ELEMENT,
    );
    expect(plan.projectedOutput.byteOffset).toBe(
      plan.bigTemporary.byteOffset,
    );
  });

  it("uses the complete dense FFN temporary", () => {
    const plan = planParakeetEncoderArena();
    expect(plan.intermediateChannels).toBe(
      PARAKEET_INTERMEDIATE_CHANNELS,
    );
    expect(plan.feedForwardTemporary.byteOffset).toBe(
      plan.bigTemporary.byteOffset,
    );
    expect(plan.feedForwardTemporary.byteLength).toBe(
      plan.rows *
        PARAKEET_INTERMEDIATE_CHANNELS *
        Uint16Array.BYTES_PER_ELEMENT,
    );
    expect(plan.feedForwardTemporary).toEqual({
      ...plan.bigTemporary,
      label: "encoder-feed-forward-temporary",
    });
  });

  it("uses disjoint big-temporary tails for exact attention and convolution", () => {
    const plan = planParakeetEncoderArena();
    const slotBytes =
      plan.rows * PARAKEET_ENCODER_CHANNELS * Uint16Array.BYTES_PER_ELEMENT;
    const bigEnd =
      plan.bigTemporary.byteOffset + plan.bigTemporary.byteLength;

    expect(plan.attentionQkv).toEqual({
      label: "encoder-attention-qkv",
      byteOffset: plan.bigTemporary.byteOffset,
      byteLength: slotBytes * 3,
    });
    expect(plan.attentionContext).toEqual({
      label: "encoder-attention-context",
      byteOffset: plan.bigTemporary.byteOffset + slotBytes * 3,
      byteLength: slotBytes,
    });
    expect(
      plan.attentionQkv.byteOffset + plan.attentionQkv.byteLength,
    ).toBe(plan.attentionContext.byteOffset);
    expect(
      plan.attentionContext.byteOffset + plan.attentionContext.byteLength,
    ).toBe(bigEnd);

    expect(plan.convolutionPointwise).toEqual({
      label: "encoder-convolution-pointwise",
      byteOffset: plan.bigTemporary.byteOffset,
      byteLength: slotBytes * 2,
    });
    expect(plan.convolutionOutput).toEqual({
      label: "encoder-convolution-output",
      byteOffset: plan.bigTemporary.byteOffset + slotBytes * 2,
      byteLength: slotBytes,
    });
    expect(
      plan.convolutionPointwise.byteOffset +
        plan.convolutionPointwise.byteLength,
    ).toBe(plan.convolutionOutput.byteOffset);
  });

  it("overlaps mutually exclusive phases without overlapping live tensors", () => {
    const plan = planParakeetEncoderArena();
    const [slot0, slot1] = plan.encoderSlots;

    expect(slot1.byteOffset).toBe(plan.featureInput.byteOffset);
    expect(slot0.byteOffset + slot0.byteLength).toBeLessThanOrEqual(
      plan.featureLengths.byteOffset,
    );
    expect(plan.featureLengths.byteOffset + plan.featureLengths.byteLength)
      .toBeLessThanOrEqual(plan.encoderLengths.byteOffset);
    expect(plan.encoderLengths.byteOffset + plan.encoderLengths.byteLength)
      .toBeLessThanOrEqual(plan.featureInput.byteOffset);

    expect(slot1.byteOffset + slot1.byteLength).toBeLessThanOrEqual(
      plan.bigTemporary.byteOffset,
    );
    expect(plan.featureInput.byteOffset + plan.featureInput.byteLength)
      .toBeLessThanOrEqual(plan.subsampling.stage376.byteOffset);
    expect(
      plan.subsampling.stage376.byteOffset +
        plan.subsampling.stage376.byteLength,
    ).toBeLessThanOrEqual(plan.subsampling.projected376.byteOffset);
    expect("stage751" in plan.subsampling).toBe(false);
    expect(plan.byteLength % 256).toBe(0);
  });

  it("toggles back to slot zero after exactly 24 Conformer layers", () => {
    const plan = planParakeetEncoderArena();
    expect(PARAKEET_ENCODER_LAYERS).toBe(24);
    expect(plan.layerOutputSlots).toHaveLength(24);
    expect(plan.layerOutputSlots).toEqual(
      Array.from({ length: 12 }, () => [1, 0]).flat(),
    );
    expect(plan.layerOutputSlots.at(-1)).toBe(0);
  });

  it("plans every live two-slot layer role without overlapping bindings", () => {
    const plan = planParakeetEncoderArena();
    let inputSlot = 0;
    for (let layer = 0; layer < PARAKEET_ENCODER_LAYERS; layer += 1) {
      const flow = planEncoderLayerArena(plan, inputSlot);
      expect(overlaps(flow.input, flow.alternate)).toBe(false);
      expect(overlaps(flow.input, flow.bigTemporary)).toBe(false);
      expect(overlaps(flow.alternate, flow.bigTemporary)).toBe(false);
      expect(overlaps(flow.attentionQkv, flow.attentionContext)).toBe(false);
      expect(
        overlaps(flow.convolutionPointwise, flow.convolutionOutput),
      ).toBe(false);
      expect(flow.outputSlot).toBe((inputSlot + 1) % 2);
      inputSlot = flow.outputSlot;
    }
    expect(inputSlot).toBe(0);
    expect(() => planEncoderLayerArena(plan, -1)).toThrow(/out of range/);
    expect(() => planEncoderLayerArena(plan, 2)).toThrow(/out of range/);
  });

  it("locks the transient-expansion layer dispatch structure", () => {
    expect(planEncoderLayerStructure()).toEqual({
      logicalDispatches: 15,
      layerNormPasses: 5,
      feedForward1MatrixPasses: 2,
      totalComputePasses: 23,
    });
  });

  it("locks the B40 arena capacity and byte length", () => {
    const plan = planParakeetEncoderArena();
    expect(plan.batchSize).toBe(PARAKEET_ENCODER_BATCH_CAPACITY);
    expect(plan.precision).toBe("fp16");
    expect(plan.byteLength).toBe(92_406_272);
    expect(plan.buffers).toEqual([
      {
        label: "parakeet-encoder-arena",
        byteLength: 92_406_272,
      },
    ]);
  });

  it("keeps the FP32 B40 arena in three stock-limit buffers", () => {
    const plan = planParakeetEncoderArena("fp32");
    const slotBytes =
      plan.rows *
      PARAKEET_ENCODER_CHANNELS *
      Float32Array.BYTES_PER_ELEMENT;

    expect(plan.precision).toBe("fp32");
    expect(plan.batchSize).toBe(PARAKEET_ENCODER_BATCH_CAPACITY);
    expect(plan.byteLength).toBe(184_812_032);
    expect(plan.buffers).toEqual([
      {
        label: "parakeet-encoder-fp32-slot-zero",
        byteLength: 30_802_176,
      },
      {
        label: "parakeet-encoder-fp32-slot-one",
        byteLength: 30_802_176,
      },
      {
        label: "parakeet-encoder-fp32-temporary",
        byteLength: 123_207_680,
      },
    ]);
    expect(
      Math.max(...plan.buffers.map(({ byteLength }) => byteLength)),
    ).toBeLessThanOrEqual(128 * 1024 * 1024);
    expect(plan.encoderSlots.map((slot) => slot.byteLength)).toEqual([
      slotBytes,
      slotBytes,
    ]);
    expect(plan.encoderSlots[0].bufferIndex ?? 0).toBe(0);
    expect(plan.encoderSlots[1].bufferIndex).toBe(1);
    expect(plan.bigTemporary.bufferIndex).toBe(2);
    expect(plan.bigTemporary.byteLength).toBe(123_207_680);
    expect(plan.featureInput.bufferIndex).toBe(1);
    expect(plan.projectedOutput.bufferIndex).toBe(1);
    expect(plan.encoderLengths.bufferIndex).toBe(1);
    expect(plan.decoderScratch.bufferIndex).toBe(1);
    expect(plan.projectedOutput.byteOffset).toBe(0);
    expect(
      plan.projectedOutput.byteOffset + plan.projectedOutput.byteLength,
    ).toBeLessThanOrEqual(plan.decoderScratch.byteOffset);
    expect(
      plan.decoderScratch.byteOffset + plan.decoderScratch.byteLength,
    ).toBeLessThanOrEqual(plan.encoderSlots[1].byteLength);
    expect(plan.layerOutputSlots.at(-1)).toBe(0);
    expect(plan.microbatches).toHaveLength(40);
    expect(
      plan.microbatches.every(
        ({ output }) => (output.bufferIndex ?? 0) === 0,
      ),
    ).toBe(true);
  });

  it("keeps B3 subsampling memory-neutral across both precision paths", () => {
    const plan = planParakeetEncoderArena();
    const fp32 = planParakeetEncoderArena("fp32");
    expect(PARAKEET_SUBSAMPLING_MICROBATCH_SIZE).toBe(3);
    expect(plan.subsamplingMicrobatchSize).toBe(3);
    expect(fp32.subsamplingMicrobatchSize).toBe(3);
    expect(plan.microbatches).toHaveLength(PARAKEET_ENCODER_BATCH_CAPACITY);
    expect(plan.microbatches.map(({ batchSize }) => batchSize)).toEqual(
      Array.from(
        { length: PARAKEET_ENCODER_BATCH_CAPACITY },
        (_, index) =>
          (index + 1) % PARAKEET_SUBSAMPLING_MICROBATCH_SIZE ||
          PARAKEET_SUBSAMPLING_MICROBATCH_SIZE,
      ),
    );
    expect(plan.subsampling.stage376.byteLength).toBe(18_481_152);
    expect(plan.subsampling.projected376.byteLength).toBe(18_481_152);
    expect(fp32.subsampling.stage376.byteLength).toBe(36_962_304);
    expect(fp32.subsampling.projected376.byteLength).toBe(36_962_304);
    expect(plan.byteLength).toBe(92_406_272);
    expect(fp32.byteLength).toBe(184_812_032);
  });
});

function overlaps(
  left: { readonly byteOffset: number; readonly byteLength: number },
  right: { readonly byteOffset: number; readonly byteLength: number },
): boolean {
  return (
    left.byteOffset < right.byteOffset + right.byteLength &&
    right.byteOffset < left.byteOffset + left.byteLength
  );
}
