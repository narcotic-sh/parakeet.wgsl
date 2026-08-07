# Browser pipelined-audio acceptance

Run the real packaged FFmpeg core in an isolated stock Chrome profile:

```sh
pnpm test:browser:audio
```

The command first builds the package. The fixture then imports the emitted
`dist/index.js` public API and supplies a fake inference worker, so no WebGPU or
model fetch is involved. A deterministic stereo 8 kHz PCM16 WAV cannot take
the canonical-input bypass: the emitted incremental worker mounts it with
WORKERFS, decodes it through the pull API, closes an immutable raw PCM16 OPFS
segment, and transfers that segment's `File` snapshot over the dedicated data
port. The fake inference worker checks the exact output SHA-256 and contiguous
16 kHz mono sample timeline, plus the required
`prepare-stream` → `initialize` → `transcribe-stream` controller sequence.
Resolution of `transcribe()` also proves the job's OPFS scratch directory was
removed. Set `CHROME_PATH` when Chrome is not installed at the standard macOS
location.

After `pnpm build`, verify the npm boundary separately:

```sh
pnpm verify:package
```

That check creates a temporary tarball and consumer install, rejects files
outside the release allowlist, imports the installed public module, and proves
that the FFmpeg JavaScript and Wasm assets remain byte-identical in
`dist/assets`. Neither check publishes the package.
