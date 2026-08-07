#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFile as execFileCallback } from "node:child_process";
import {
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";

const execFile = promisify(execFileCallback);
const repositoryRoot = resolve(import.meta.dirname, "..");
const temporaryRoot = await mkdtemp(join(tmpdir(), "parakeet-wgsl-package-"));
const packDirectory = join(temporaryRoot, "pack");
const consumerDirectory = join(temporaryRoot, "consumer");

const expectedArtifacts = [
  {
    label: "filter-bank Wasm",
    byteLength: 23_214,
    sha256: "bd3970899334d2dec114465df6b8c5cee014bc21572badbda15d0b6647bec39d",
  },
  {
    label: "FFmpeg Wasm core",
    byteLength: 1_599_767,
    sha256: "8703a1a66936cf39a10ef94490e9e594dc602e4d094fa7b0325d59fee0488fa5",
  },
];

try {
  await Promise.all([
    mkdir(packDirectory, { recursive: true }),
    mkdir(consumerDirectory, { recursive: true }),
  ]);
  const { stdout: packOutput } = await execFile(
    "npm",
    [
      "pack",
      "--json",
      "--ignore-scripts",
      "--pack-destination",
      packDirectory,
    ],
    { cwd: repositoryRoot, maxBuffer: 10 * 1024 * 1024 },
  );
  const packReport = JSON.parse(packOutput);
  const packageReports = Array.isArray(packReport)
    ? packReport
    : Object.values(packReport);
  const filename = packageReports.at(-1)?.filename;
  assert(typeof filename === "string", "npm pack did not report a tarball");
  const tarball = join(packDirectory, filename);

  const { stdout: tarOutput } = await execFile("tar", ["-tzf", tarball], {
    maxBuffer: 10 * 1024 * 1024,
  });
  const entries = tarOutput.trim().split(/\r?\n/).filter(Boolean).sort();
  const unexpected = entries.filter((entry) => !isAllowedTarEntry(entry));
  assert(
    unexpected.length === 0,
    `Unexpected npm tarball entries:\n${unexpected.join("\n")}`,
  );
  for (const required of [
    "package/package.json",
    "package/dist/index.js",
    "package/dist/index.d.ts",
    "package/README.md",
    "package/DOCS.md",
    "package/LICENSE",
    "package/THIRD_PARTY_LICENSES",
  ]) {
    assert(entries.includes(required), `npm tarball is missing ${required}`);
  }

  await writeFile(
    join(consumerDirectory, "package.json"),
    `${JSON.stringify({ private: true, type: "module" }, null, 2)}\n`,
  );
  await execFile(
    "npm",
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--no-package-lock",
      "--no-save",
      tarball,
    ],
    { cwd: consumerDirectory, maxBuffer: 10 * 1024 * 1024 },
  );

  const consumerSource = join(consumerDirectory, "consumer.ts");
  await writeFile(
    consumerSource,
    `import {
  createTranscriber,
  requiresAudioDecoding,
  type AudioDecodingMetrics,
  type TranscriptionResult,
} from "parakeet.wgsl";

declare const audio: Blob;
const transcriber = createTranscriber();
const cachedWarmup: Promise<boolean> = transcriber.loadCachedModel();
const inspection: Promise<boolean> = requiresAudioDecoding(audio);
const transcription: Promise<TranscriptionResult> = transcriber.transcribe(audio, {
  sourceName: "recording.webm",
  onProgress(progress) {
    const decoding: AudioDecodingMetrics = progress.metrics.audioDecoding;
    void decoding.applied;
  },
});
void inspection;
void cachedWarmup;
void transcription;
`,
  );
  await execFile(
    process.execPath,
    [
      join(repositoryRoot, "node_modules", "typescript", "bin", "tsc"),
      "--noEmit",
      "--strict",
      "--target",
      "ES2024",
      "--module",
      "ESNext",
      "--moduleResolution",
      "Bundler",
      "--lib",
      "ES2024,DOM,DOM.Iterable",
      consumerSource,
    ],
    { cwd: consumerDirectory, maxBuffer: 10 * 1024 * 1024 },
  );

  const installedRoot = join(
    consumerDirectory,
    "node_modules",
    "parakeet.wgsl",
  );
  const installedPackage = JSON.parse(
    await readFile(join(installedRoot, "package.json"), "utf8"),
  );
  assert(
    installedPackage.name === "parakeet.wgsl",
    "Installed tarball has the wrong package name",
  );
  assert(
    Object.keys(installedPackage.dependencies ?? {}).length === 0,
    "Installed package acquired runtime dependencies",
  );

  const installedEntrySource = await readFile(
    join(installedRoot, "dist", "index.js"),
    "utf8",
  );
  assert(
    !/(["'])\1\s*\+\s*import\.meta\.url/.test(installedEntrySource),
    "Published runtime asset URLs are hidden from consuming bundlers",
  );
  const runtimeAssetReferences = [
    ...installedEntrySource.matchAll(
      /new URL\(["']\.\/assets\/([^"']+)["'],\s*import\.meta\.url\)/g,
    ),
  ].map((match) => match[1]);
  assert(
    runtimeAssetReferences.length === 4 &&
      new Set(runtimeAssetReferences).size === 4,
    `Expected four distinct relocatable runtime assets; found ${runtimeAssetReferences.length}`,
  );
  for (const asset of runtimeAssetReferences) {
    assert(
      entries.includes(`package/dist/assets/${asset}`),
      `Published entry references missing runtime asset ${asset}`,
    );
  }

  const publicModule = await import(
    `${pathToFileURL(join(installedRoot, "dist", "index.js")).href}?smoke=${Date.now()}`
  );
  for (const exportName of [
    "checkSupport",
    "createTranscriber",
    "deleteCachedModels",
    "getModelCacheInfo",
    "requiresAudioDecoding",
  ]) {
    assert(
      typeof publicModule[exportName] === "function",
      `Installed package is missing ${exportName}()`,
    );
  }

  const installedAssets = await filesRecursively(
    join(installedRoot, "dist", "assets"),
  );
  for (const expected of expectedArtifacts) {
    const matches = [];
    for (const path of installedAssets) {
      const bytes = await readFile(path);
      if (
        bytes.byteLength === expected.byteLength &&
        createHash("sha256").update(bytes).digest("hex") === expected.sha256
      ) {
        matches.push(basename(path));
      }
    }
    assert(
      matches.length === 1,
      `Expected one byte-exact ${expected.label} in dist/assets; found ${matches.length}`,
    );
  }

  await Promise.all([
    writeFile(
      join(consumerDirectory, "index.html"),
      '<script type="module" src="/main.js"></script>\n',
    ),
    writeFile(
      join(consumerDirectory, "main.js"),
      'import { createTranscriber } from "parakeet.wgsl";\n' +
        "globalThis.parakeet = createTranscriber;\n",
    ),
  ]);
  await execFile(
    process.execPath,
    [
      join(repositoryRoot, "node_modules", "vite", "bin", "vite.js"),
      "build",
      "--logLevel",
      "error",
    ],
    { cwd: consumerDirectory, maxBuffer: 10 * 1024 * 1024 },
  );
  const consumerAssets = await filesRecursively(
    join(consumerDirectory, "dist", "assets"),
  );
  for (const expectedName of [
    /^streaming-decoder\.worker-.*\.js$/,
    /^worker-entry-.*\.js$/,
    /^parakeet-fbank-.*\.wasm$/,
    /^parakeet-wgsl-ffmpeg-core-.*\.wasm$/,
  ]) {
    const matches = consumerAssets.filter((path) =>
      expectedName.test(basename(path)),
    );
    assert(
      matches.length === 1,
      `Default Vite consumer build emitted ${matches.length} files matching ${expectedName}`,
    );
  }
  const consumerJavaScript = (
    await Promise.all(
      consumerAssets
        .filter((path) => path.endsWith(".js"))
        .map((path) => readFile(path, "utf8")),
    )
  ).join("\n");
  assert(
    !/new Worker\([^)]*(?:data:|blob:)/.test(consumerJavaScript),
    "Default Vite consumer build inlined a package worker",
  );

  process.stdout.write(
    `${JSON.stringify(
      {
        ok: true,
        tarball: filename,
        entryCount: entries.length,
        assetCount: installedAssets.length,
        typescriptConsumerCompiled: true,
        viteConsumerBuilt: true,
        viteConsumerAssetCount: consumerAssets.length,
        verifiedArtifacts: expectedArtifacts.map(({ label, sha256 }) => ({
          label,
          sha256,
        })),
      },
      null,
      2,
    )}\n`,
  );
} finally {
  await rm(temporaryRoot, {
    recursive: true,
    force: true,
    maxRetries: 3,
    retryDelay: 100,
  });
}

function isAllowedTarEntry(entry) {
  return [
    /^package\/(?:LICENSE|README\.md|DOCS\.md|THIRD_PARTY_LICENSES|package\.json)$/,
    /^package\/dist\/(?:index\.js|index\.d\.ts|api\.d\.ts)$/,
    /^package\/dist\/runtime\/(?:transcriber|protocol|paced-transcript|paced-transcript-dom)\.d\.ts$/,
    /^package\/dist\/assets\/[^/]+$/,
  ].some((pattern) => pattern.test(entry));
}

async function filesRecursively(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await filesRecursively(path)));
    else if (entry.isFile()) files.push(path);
  }
  return files.sort();
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
