#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";

const repositoryRoot = resolve(import.meta.dirname, "..");
const coreDirectory = resolve(
  process.env.PARAKEET_FFMPEG_CORE_DIR ??
    resolve(repositoryRoot, "src/cpp/wasm"),
);
const coreScript = resolve(
  coreDirectory,
  "parakeet-wgsl-ffmpeg-core.js",
);
const coreWasm = resolve(
  coreDirectory,
  "parakeet-wgsl-ffmpeg-core.wasm",
);
const FIRST_B40_READY_SAMPLES = 39 * 206_080 + 238_080 + 1;
const FIRST_PULL_SAMPLES = FIRST_B40_READY_SAMPLES;
const STEADY_PULL_SAMPLES = 40 * 206_080;

// The generated core deliberately targets a browser worker. These two globals
// let Node exercise its deterministic MEMFS-facing ABI without changing the
// production worker build. Browser WORKERFS/OPFS integration has a separate
// acceptance test.
globalThis.self = globalThis;
globalThis.location ??= { href: import.meta.url };

const createCore = (await import(coreScript)).default;
if (typeof createCore !== "function") {
  throw new Error("The packaged FFmpeg core has no default module factory");
}
const wasmBytes = await readFile(coreWasm);
const wasmURL = `data:application/wasm;base64,${wasmBytes.toString("base64")}`;
const [referenceCore, incrementalCore] = await Promise.all([
  createCore({ wasmURL }),
  createCore({ wasmURL }),
]);
if (typeof incrementalCore.createIncrementalAudioDecoder !== "function") {
  throw new Error("The packaged core has no incremental decoder API");
}

const requestedPaths = process.argv.slice(2).map((value) => resolve(value));
const fixtures =
  requestedPaths.length === 0
    ? [
        {
          name: "synthetic-stereo-8khz.wav",
          bytes: stereoPcm16Wav8Khz(2_003),
          expectedEstimatedSampleCount: 4_006,
          // Deliberately split encoded PCM packets and the final filter drain.
          pullMaximums: [1, 3, 511, 17, 1_024],
        },
      ]
    : await Promise.all(
        requestedPaths.map(async (path) => ({
          name: basename(path),
          bytes: await readFile(path),
        })),
      );

const reports = [];
for (const [index, fixture] of fixtures.entries()) {
  reports.push(
    await verifyFixture(referenceCore, incrementalCore, fixture, index),
  );
}
process.stdout.write(`${JSON.stringify({ ok: true, fixtures: reports }, null, 2)}\n`);

async function verifyFixture(referenceCore, incrementalCore, fixture, index) {
  const suffix = safeExtension(fixture.name);
  const inputPath = `/fixture-${index}${suffix}`;
  const referencePath = `/reference-${index}.wav`;
  referenceCore.FS.writeFile(inputPath, fixture.bytes);
  incrementalCore.FS.writeFile(inputPath, fixture.bytes);

  const logTail = [];
  referenceCore.setLogger?.(({ type, message }) => {
    if (type !== "stderr") return;
    logTail.push(message);
    while (logTail.length > 20) logTail.shift();
  });

  try {
    const referenceStartedAt = performance.now();
    const exitCode = referenceCore.exec(
      "-hide_banner",
      "-loglevel",
      "warning",
      "-i",
      inputPath,
      "-map",
      "0:a:0",
      "-vn",
      "-sn",
      "-dn",
      "-ac",
      "1",
      "-ar",
      "16000",
      "-c:a",
      "pcm_s16le",
      "-map_metadata",
      "-1",
      "-fflags",
      "+bitexact",
      "-flags:a",
      "+bitexact",
      referencePath,
    );
    const referenceMs = performance.now() - referenceStartedAt;
    if (exitCode !== 0) {
      throw new Error(
        `One-shot reference failed for ${fixture.name} with ${exitCode}: ` +
          logTail.join("\n"),
      );
    }
    const referenceWave = referenceCore.FS.readFile(referencePath);
    const referencePcm = wavePcmPayload(referenceWave);
    const referenceSha256 = sha256(referencePcm);

    const beforeOpenEstimate =
      incrementalCore._parakeet_audio_decoder_estimated_sample_count();
    if (beforeOpenEstimate !== -1n) {
      throw new Error("Incremental decoder retained an estimate before open");
    }
    const decoder = incrementalCore.createIncrementalAudioDecoder(inputPath);
    const estimatedSampleCount = decoder.estimatedSampleCount;
    if (
      estimatedSampleCount !== null &&
      (!Number.isSafeInteger(estimatedSampleCount) ||
        estimatedSampleCount < 0)
    ) {
      throw new Error(`Invalid sample-count estimate for ${fixture.name}`);
    }
    if (
      fixture.expectedEstimatedSampleCount !== undefined &&
      estimatedSampleCount !== fixture.expectedEstimatedSampleCount
    ) {
      throw new Error(
        `${fixture.name} estimated ${estimatedSampleCount} samples; expected ` +
          fixture.expectedEstimatedSampleCount,
      );
    }
    const incrementalHash = createHash("sha256");
    let totalSamples = 0;
    let pulls = 0;
    let firstB40ReadyMs;
    let decoderActiveMs = 0;
    const incrementalStartedAt = performance.now();
    try {
      while (true) {
        const maximum =
          fixture.pullMaximums?.[pulls % fixture.pullMaximums.length] ??
          (pulls === 0 ? FIRST_PULL_SAMPLES : STEADY_PULL_SAMPLES);
        const result = decoder.decodeNext(maximum);
        pulls += 1;
        if (
          !(result.samples instanceof Int16Array) ||
          result.samples.length !== result.sampleCount ||
          result.sampleCount > maximum
        ) {
          throw new Error(`Invalid incremental pull for ${fixture.name}`);
        }
        incrementalHash.update(
          new Uint8Array(
            result.samples.buffer,
            result.samples.byteOffset,
            result.samples.byteLength,
          ),
        );
        totalSamples += result.sampleCount;
        decoderActiveMs = result.totalDecodeMs;
        if (
          firstB40ReadyMs === undefined &&
          totalSamples >= FIRST_B40_READY_SAMPLES
        ) {
          firstB40ReadyMs = performance.now() - incrementalStartedAt;
        }
        if (result.done) break;
      }
    } finally {
      decoder.close();
    }
    if (
      incrementalCore._parakeet_audio_decoder_estimated_sample_count() !== -1n
    ) {
      throw new Error("Incremental decoder retained an estimate after close");
    }
    const incrementalWallMs = performance.now() - incrementalStartedAt;
    const incrementalSha256 = incrementalHash.digest("hex");
    const outputByteLength = totalSamples * Int16Array.BYTES_PER_ELEMENT;

    if (outputByteLength !== referencePcm.byteLength) {
      throw new Error(
        `${fixture.name} produced ${outputByteLength} incremental bytes; ` +
          `the one-shot core produced ${referencePcm.byteLength}`,
      );
    }
    if (incrementalSha256 !== referenceSha256) {
      throw new Error(
        `${fixture.name} incremental SHA-256 ${incrementalSha256} did not ` +
          `match one-shot ${referenceSha256}`,
      );
    }

    return {
      name: fixture.name,
      inputByteLength: fixture.bytes.byteLength,
      outputByteLength,
      sampleCount: totalSamples,
      estimatedSampleCount,
      sha256: incrementalSha256,
      pulls,
      referenceMs,
      incrementalWallMs,
      decoderActiveMs,
      ...(firstB40ReadyMs === undefined ? {} : { firstB40ReadyMs }),
    };
  } finally {
    unlinkIfPresent(referenceCore.FS, referencePath);
    unlinkIfPresent(referenceCore.FS, inputPath);
    unlinkIfPresent(incrementalCore.FS, inputPath);
  }
}

function wavePcmPayload(bytes) {
  if (bytes.byteLength < 12) throw new Error("Reference WAV is truncated");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (fourCc(bytes, 0) !== "RIFF" || fourCc(bytes, 8) !== "WAVE") {
    throw new Error("Reference output is not RIFF/WAVE");
  }
  const riffEnd = Math.min(bytes.byteLength, view.getUint32(4, true) + 8);
  let offset = 12;
  while (offset + 8 <= riffEnd) {
    const id = fourCc(bytes, offset);
    const length = view.getUint32(offset + 4, true);
    const start = offset + 8;
    const end = start + length;
    if (end > riffEnd) throw new Error(`Reference WAV ${id} chunk is truncated`);
    if (id === "data") return bytes.subarray(start, end);
    offset = end + (length & 1);
  }
  throw new Error("Reference WAV has no data chunk");
}

function fourCc(bytes, offset) {
  return String.fromCharCode(
    bytes[offset],
    bytes[offset + 1],
    bytes[offset + 2],
    bytes[offset + 3],
  );
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function safeExtension(name) {
  const extension = extname(name).toLowerCase();
  return /^\.[a-z0-9]{1,10}$/.test(extension) ? extension : ".audio";
}

function unlinkIfPresent(fileSystem, path) {
  try {
    fileSystem.unlink(path);
  } catch (error) {
    if (!(error instanceof fileSystem.ErrnoError)) throw error;
  }
}

function stereoPcm16Wav8Khz(frameCount) {
  const channels = 2;
  const sampleRate = 8_000;
  const bytesPerSample = 2;
  const dataBytes = frameCount * channels * bytesPerSample;
  const bytes = new Uint8Array(44 + dataBytes);
  const view = new DataView(bytes.buffer);
  writeAscii(bytes, 0, "RIFF");
  view.setUint32(4, bytes.byteLength - 8, true);
  writeAscii(bytes, 8, "WAVE");
  writeAscii(bytes, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeAscii(bytes, 36, "data");
  view.setUint32(40, dataBytes, true);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const phase = (2 * Math.PI * 440 * frame) / sampleRate;
    view.setInt16(44 + frame * 4, Math.round(Math.sin(phase) * 18_000), true);
    view.setInt16(46 + frame * 4, Math.round(Math.cos(phase) * 9_000), true);
  }
  return bytes;
}

function writeAscii(target, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    target[offset + index] = value.charCodeAt(index);
  }
}
