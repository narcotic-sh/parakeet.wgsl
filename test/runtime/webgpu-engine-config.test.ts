import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  COOPERATIVE_GPU_IDLE_MILLISECONDS,
  COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH,
  DECODER_CONTROL_STAGING_BYTES,
  ENCODER_GRAPH_COUNT,
  FEATURE_STAGING_BYTES,
  FEATURE_UPLOAD_MODE,
  INFERENCE_BATCH_SIZE,
  INFERENCE_COMMAND_BUFFERS_PER_GRAPH,
  INFERENCE_SLOT_COUNT,
  planDecoderArena,
  planDecoderReadback,
  PRODUCTION_GPU_TIMING_SOURCE,
  PRODUCTION_OPTIONAL_WEBGPU_FEATURES,
  PRODUCTION_GPU_SUBMISSION_MODE,
  selectModelManifestUrl,
  validateTokenizerModelContract,
  validateDecoderFrameRange,
} from "../../src/runtime/webgpu-engine";
import type { ArenaSlice } from "../../src/webgpu/arena";
import {
  PARAKEET_TRANSIENT_PALETTE_DISPATCHES_PER_GRAPH,
  PARAKEET_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH,
  PARAKEET_TRANSIENT_PALETTE_MATRIX_COUNT,
  PARAKEET_SUBSAMPLING_MICROBATCH_SIZE,
  planParakeetEncoderArena,
  type ParakeetEncoderGraph,
} from "../../src/webgpu/encoder";

const engineSource = readFileSync(
  new URL("../../src/runtime/webgpu-engine.ts", import.meta.url),
  "utf8",
);

describe("raw WebGPU production configuration", () => {
  it("reports the selected portable FP16 GEMM load encoding without changing subgroup diagnostics", () => {
    expect(engineSource).toContain(
      "primaryGraph.portableF16GemmLoadEncoding === null",
    );
    expect(engineSource).toContain(
      "portableF16GemmLoadEncoding:\n              primaryGraph.portableF16GemmLoadEncoding",
    );
  });

  it("selects exactly one capability-matched package URL", () => {
    const configuration = {
      fp16ModelUrl: "/model/files/manifest.json",
      fp32ModelUrl: "/model/files-fp32/manifest.json",
    };
    expect(
      selectModelManifestUrl(configuration, "fp16"),
    ).toBe("/model/files/manifest.json");
    expect(
      selectModelManifestUrl(configuration, "fp32"),
    ).toBe("/model/files-fp32/manifest.json");

    const capabilityOffset = engineSource.indexOf(
      "const webgpu = await requestWebGpuContext({",
    );
    const selectionOffset = engineSource.indexOf(
      "const modelUrl = selectModelManifestUrl(",
      capabilityOffset,
    );
    const loadOffset = engineSource.indexOf(
      "model = await loadSelectedModelPackage(",
      selectionOffset,
    );
    expect(capabilityOffset).toBeGreaterThanOrEqual(0);
    expect(selectionOffset).toBeGreaterThan(capabilityOffset);
    expect(loadOffset).toBeGreaterThan(selectionOffset);
    expect(engineSource.match(/ParakeetGpuPackage\.load\(/g)).toHaveLength(
      1,
    );
    expect(
      engineSource.slice(loadOffset, loadOffset + 300),
    ).toContain("webgpu.executionProfile.precision");
    expect(engineSource).toContain("expectedPrecision: precision");
    expect(engineSource).toContain("cache.registerAssets(selectedPackageAssets(");
  });

  it("requires tokenizer blank and vocabulary dimensions to match the package", () => {
    const tokenizer = {
      blankTokenId: 1024,
      vocabularySize: 1025,
    };
    const config = {
      blankTokenId: 1024,
      vocabularySize: 1025,
    };

    expect(() =>
      validateTokenizerModelContract(tokenizer, config)
    ).not.toThrow();
    expect(() =>
      validateTokenizerModelContract(
        { ...tokenizer, blankTokenId: 8192 },
        config,
      )
    ).toThrow("Tokenizer blank token does not match the model");
    expect(() =>
      validateTokenizerModelContract(
        { ...tokenizer, vocabularySize: 8193 },
        config,
      )
    ).toThrow("Tokenizer vocabulary size does not match the model");

    expect(engineSource).toContain(
      "validateTokenizerModelContract(tokenizer, model.manifest.config);",
    );
  });

  it("lets WGSL clamp a Fluid actual-frame end to the encoder-valid tail", () => {
    expect(() =>
      validateDecoderFrameRange(
        { index: 0, decodeStartFrame: 0, decodeEndFrame: 2 },
        1,
      ),
    ).not.toThrow();
    expect(() =>
      validateDecoderFrameRange(
        { index: 1, decodeStartFrame: 1, decodeEndFrame: 186 },
        187,
      ),
    ).not.toThrow();
    expect(() =>
      validateDecoderFrameRange(
        { index: 2, decodeStartFrame: 1, decodeEndFrame: 2 },
        1,
      ),
    ).toThrow(/invalid decoder range/);
  });

  it("keeps two logical B40 slots on one shared compute graph", () => {
    expect(INFERENCE_BATCH_SIZE).toBe(40);
    expect(PARAKEET_SUBSAMPLING_MICROBATCH_SIZE).toBe(3);
    expect(planParakeetEncoderArena().subsamplingMicrobatchSize).toBe(3);
    expect(FEATURE_UPLOAD_MODE).toBe("batch");
    expect(INFERENCE_SLOT_COUNT).toBe(2);
    expect(ENCODER_GRAPH_COUNT).toBe(1);
    expect(INFERENCE_COMMAND_BUFFERS_PER_GRAPH).toBe(27);
    expect(COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH).toBe(26);
    expect(COOPERATIVE_GPU_IDLE_MILLISECONDS).toBe(1);
    expect(PRODUCTION_GPU_SUBMISSION_MODE).toBe(
      "queue-drained-subsampling-layer-projector-decoder",
    );
    expect(FEATURE_STAGING_BYTES).toBe(30_740_480);
    expect(DECODER_CONTROL_STAGING_BYTES).toBe(480);
    expect(engineSource).toContain(
      "...encoderGraph.encodeSubmissionChunks(validFeatureLengths),",
    );
    const encodeOffset = engineSource.indexOf(
      "...encoderGraph.encodeSubmissionChunks(validFeatureLengths),",
    );
    const controlCopyOffset = engineSource.indexOf(
      "encodeDecoderControlCopies(",
      encodeOffset,
    );
    const decoderOffset = engineSource.indexOf(
      "decoderDispatch.encode(",
      controlCopyOffset,
    );
    expect(controlCopyOffset).toBeGreaterThan(encodeOffset);
    expect(decoderOffset).toBeGreaterThan(controlCopyOffset);
    expect(engineSource).not.toContain(
      "encoderGraph.arena.upload(\n        this.device,\n        decoderLayout.decode",
    );
    expect(engineSource).not.toContain("slot.encoderGraph");
    expect(engineSource).not.toContain("slot.decoderDispatch");
    expect(engineSource).toContain(
      "windows.length,\n      );",
    );
    expect(engineSource).not.toContain(
      "this.preferredBatchSize,\n      ).fill(1)",
    );
    expect(engineSource).not.toContain("clearUnusedFeatures(");
    expect(engineSource).toContain(
      "this.inferenceCompute.encoderGraph.uploadFeatures(",
    );
    expect(engineSource).toContain(
      "featureStaging: this.featureStagingForBatch(batchSize)",
    );
    expect(engineSource.match(/new Float32Array\(/g)).toHaveLength(1);
    expect(engineSource).toContain(
      "this.mapDecoderOutput(slot, readbackLayout)",
    );
    expect(engineSource).not.toContain(".getMappedRange(0, layout.readbackByteLength)\n        .slice(0)");
    expect(engineSource).toContain(
      "samples = this.fbank.beginPreparedBatch(batchSize, sampleCount);",
    );
    expect(engineSource).toContain(
      'fbankMode: "local-simd-wasm"',
    );
    expect(engineSource).not.toContain("startupFbank");
    expect(engineSource).not.toContain("StartupFbank");
    expect(engineSource).not.toContain("startup-transferred");
    expect(engineSource).not.toContain("new Worker(");
    expect(engineSource).toContain(
      "window.samples.length === window.windowSampleCount",
    );
  });

  it("keeps timestamp queries out of the production device and command buffers", () => {
    expect(PRODUCTION_OPTIONAL_WEBGPU_FEATURES).toEqual([]);
    expect(PRODUCTION_GPU_TIMING_SOURCE).toBe("submit-to-readback-wall");
  });

  it("holds exclusive graph ownership from feature upload through decoder readback", () => {
    const prepareStart = engineSource.indexOf(
      "prepareWindowBatch(",
    );
    const transcribeStart = engineSource.indexOf(
      "async transcribeWindows(",
      prepareStart,
    );
    const prepareSource = engineSource.slice(
      prepareStart,
      transcribeStart,
    );
    const gateAcquireOffset = prepareSource.indexOf(
      "await this.graphSubmissionGate.acquire()",
    );
    const ownershipStoreOffset = prepareSource.indexOf(
      "prepared.releaseGraphSubmission =",
      gateAcquireOffset,
    );
    const featureUploadOffset = prepareSource.indexOf(
      "this.inferenceCompute.encoderGraph.uploadFeatures(",
      ownershipStoreOffset,
    );
    expect(prepareStart).toBeGreaterThanOrEqual(0);
    expect(transcribeStart).toBeGreaterThan(prepareStart);
    expect(gateAcquireOffset).toBeGreaterThanOrEqual(0);
    expect(ownershipStoreOffset).toBeGreaterThan(gateAcquireOffset);
    expect(featureUploadOffset).toBeGreaterThan(ownershipStoreOffset);

    const transcribeEnd = engineSource.indexOf(
      "\n  decodeTokenIds(",
      transcribeStart,
    );
    const transcribeSource = engineSource.slice(
      transcribeStart,
      transcribeEnd,
    );
    const submitOffset = transcribeSource.indexOf(
      "await submitCommandBuffersCooperatively(",
    );
    expect(submitOffset).toBeGreaterThan(0);
    const readbackAwaitOffset = transcribeSource.indexOf(
      "const [validationError, words] = await Promise.all([",
      submitOffset,
    );
    const mapOffset = transcribeSource.indexOf(
      "this.mapDecoderOutput(slot, readbackLayout)",
      readbackAwaitOffset,
    );
    const finalProgressOffset = transcribeSource.indexOf(
      "onExecutionProgress?.({",
      mapOffset,
    );
    const finalAbortOffset = transcribeSource.lastIndexOf(
      "throwIfAborted(signal);",
      finalProgressOffset,
    );
    const successfulReleaseOffset = transcribeSource.indexOf(
      "releaseGraphSubmission();",
      finalProgressOffset,
    );
    const finallyOffset = transcribeSource.indexOf(
      "} finally {",
      successfulReleaseOffset,
    );
    const fallbackReleaseOffset = transcribeSource.indexOf(
      "releaseGraphSubmission?.();",
      finallyOffset,
    );
    expect(readbackAwaitOffset).toBeGreaterThan(submitOffset);
    expect(mapOffset).toBeGreaterThan(readbackAwaitOffset);
    expect(finalAbortOffset).toBeGreaterThan(mapOffset);
    expect(finalProgressOffset).toBeGreaterThan(mapOffset);
    expect(finalProgressOffset).toBeGreaterThan(finalAbortOffset);
    expect(successfulReleaseOffset).toBeGreaterThan(
      finalProgressOffset,
    );
    expect(finallyOffset).toBeGreaterThan(successfulReleaseOffset);
    expect(fallbackReleaseOffset).toBeGreaterThan(finallyOffset);
    expect(transcribeSource).toContain(
      "encodeDecoderReadbackCopies(\n        decoderCommands,\n        encoderGraph.arena.bufferFor(decoderLayout.tokens),\n        slot.decoderReadback,",
    );
    expect(transcribeSource).toContain(
      "commandBuffers.length !==\n        INFERENCE_COMMAND_BUFFERS_PER_GRAPH",
    );
    expect(engineSource).toContain(
      "setTimeout(resolve, COOPERATIVE_GPU_IDLE_MILLISECONDS)",
    );
  });

  it("requires only stock Chrome WebGPU features", () => {
    expect(engineSource).toContain(
      "report.requiredFeatures.some((feature) => !device.features.has(feature))",
    );
  });

  it("reports packed, retained, expanded, runtime, and loader memory", () => {
    for (const field of [
      "modelPackedDownloadBytes",
      "modelRetainedGpuBytes",
      "modelExpandedGpuBytes",
      "modelRuntimeGpuBytes",
      "modelPackageFormat",
      "modelManifestUrl",
      "modelManifestSha256",
      "modelManifestListedBytes",
      "modelLargestRuntimeShardBytes",
      "modelLoaderPeakGpuBytes",
      "modelRuntimeBufferCount",
      "inferenceSlotCount",
      "encoderGraphCount",
      "gpuSubmissionMode",
      "encoderCommandBuffersPerGraph",
      "commandBuffersPerGraph",
      "queueDrainFencesPerGraph",
      "cooperativeGpuTaskYieldsPerGraph",
      "cooperativeGpuIdleMilliseconds",
      "maxQueuedParakeetCommandBuffers",
      "graphSubmissionConcurrency",
      "conformerLayersPerCommandBuffer",
      "modelTransientWeightScratchBytesPerGraph",
      "modelTransientWeightScratchCount",
      "modelTransientWeightScratchTotalBytes",
      "modelTransientWeightParameterPoolBytesPerGraph",
      "modelTransientWeightParameterPoolCount",
      "modelTransientWeightParameterPoolBufferCountPerGraph",
      "modelTransientWeightParameterPoolUsedSlotsPerGraph",
      "modelTransientWeightParameterPoolTotalBytes",
      "modelTransientWeightMatrixCount",
      "modelTransientWeightExpansionDispatchesPerGraph",
      "modelTransientWeightExpandedBytesPerGraph",
      "modelSteadyGpuBytes",
      "activationArenaCount",
      "activationArenaBufferCount",
      "activationArenaBufferBytes",
      "activationArenaLargestBufferBytes",
      "encoderUniformParameterPoolBytesPerGraph",
      "encoderUniformParameterPoolCount",
      "encoderUniformParameterPoolBufferCountPerGraph",
      "encoderUniformParameterPoolUsedSlotsPerGraph",
      "encoderUniformParameterPoolTotalBytes",
      "decoderDispatchCount",
      "decoderReadbackBytesPerSlot",
      "decoderReadbackCount",
      "decoderReadbackTotalBytes",
    ]) {
      expect(engineSource).toContain(field);
    }
    expect(engineSource).toContain(
      'primaryGraph.plan.precision === "fp16"',
    );
    expect(engineSource).toContain(
      "FP32_TRANSIENT_PALETTE_WEIGHT_SCRATCH_BYTES",
    );
    expect(PARAKEET_TRANSIENT_PALETTE_MATRIX_COUNT).toBe(196);
    expect(
      PARAKEET_TRANSIENT_PALETTE_DISPATCHES_PER_GRAPH,
    ).toBe(196);
    expect(
      PARAKEET_TRANSIENT_PALETTE_EXPANDED_BYTES_PER_GRAPH,
    ).toBe(1_167_589_376);
  });

  it("compacts decoder readback for every active B1-B40 tail", () => {
    const capacity = INFERENCE_BATCH_SIZE;
    const maxTokens = 2_048;
    const tokenCapacityBytes = capacity * maxTokens * 4;
    const layout = {
      batchCapacity: capacity,
      maxTokens,
      decodeStarts: {
        label: "decode-starts",
        byteOffset: 19_744,
        byteLength: capacity * 4,
      },
      decodeEnds: {
        label: "decode-ends",
        byteOffset: 19_840,
        byteLength: capacity * 4,
      },
      flushFinals: {
        label: "flush-finals",
        byteOffset: 19_936,
        byteLength: capacity * 4,
      },
      state: { label: "state", byteOffset: 0, byteLength: capacity * 10 },
      predictorScratch: {
        label: "scratch",
        byteOffset: 10_000,
        byteLength: capacity * 10,
      },
      tokens: {
        label: "tokens",
        byteOffset: 20_000,
        byteLength: tokenCapacityBytes,
      },
      tokenFrames: {
        label: "frames",
        byteOffset: 20_000 + tokenCapacityBytes,
        byteLength: tokenCapacityBytes,
      },
      tokenDurations: {
        label: "durations",
        byteOffset: 20_000 + tokenCapacityBytes * 2,
        byteLength: tokenCapacityBytes,
      },
      metadata: {
        label: "metadata",
        byteOffset: 20_000 + tokenCapacityBytes * 3,
        byteLength: capacity * 2 * 4,
      },
    };
    const plans = Array.from(
      { length: capacity },
      (_, index) => planDecoderReadback(layout, index + 1),
    );
    for (const [index, plan] of plans.entries()) {
      const active = index + 1;
      const outputBytes = active * maxTokens * 4;
      expect(plan.activeBatchSize).toBe(active);
      expect(plan.readbackByteLength).toBe(outputBytes * 3 + active * 8);
      expect(plan.copies).toHaveLength(active === capacity ? 1 : 5);
      if (active === capacity) {
        expect(plan.copies[0]).toEqual({
          sourceByteOffset: layout.tokens.byteOffset,
          destinationByteOffset: 0,
          byteLength: plan.readbackByteLength,
        });
        continue;
      }
      expect(plan.copies[3]).toMatchObject({
        sourceByteOffset: layout.metadata.byteOffset,
        byteLength: active * 4,
      });
      expect(plan.copies[4]).toMatchObject({
        sourceByteOffset:
          layout.metadata.byteOffset + capacity * 4,
        destinationByteOffset:
          outputBytes * 3 + active * 4,
        byteLength: active * 4,
      });
    }
    expect(() => planDecoderReadback(layout, 0)).toThrow(/active batch/);
    expect(() => planDecoderReadback(layout, capacity + 1)).toThrow(
      /active batch/,
    );
  });

  it("places all FP32 decoder state beside the projected input", () => {
    const plan = planParakeetEncoderArena("fp32");
    const arena = {
      slice(
        label: string,
        byteOffset: number,
        byteLength: number,
        bufferIndex = 0,
      ): ArenaSlice {
        return bufferIndex === 0
          ? { label, byteOffset, byteLength }
          : { label, byteOffset, byteLength, bufferIndex };
      },
    };
    const encoder = {
      batchSize: plan.batchSize,
      plan,
      arena,
    } as unknown as ParakeetEncoderGraph;
    const decoder = planDecoderArena(encoder, 2_048);

    expect(plan.decoderScratch.bufferIndex).toBe(1);
    for (const slice of [
      decoder.decodeStarts,
      decoder.decodeEnds,
      decoder.flushFinals,
      decoder.state,
      decoder.predictorScratch,
      decoder.tokens,
      decoder.tokenFrames,
      decoder.tokenDurations,
      decoder.metadata,
    ]) {
      expect(slice.bufferIndex).toBe(1);
      expect(slice.byteOffset).toBeGreaterThanOrEqual(
        plan.decoderScratch.byteOffset,
      );
      expect(slice.byteOffset + slice.byteLength).toBeLessThanOrEqual(
        plan.decoderScratch.byteOffset + plan.decoderScratch.byteLength,
      );
    }
  });
});
