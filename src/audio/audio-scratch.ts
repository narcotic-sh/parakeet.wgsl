export const AUDIO_DECODING_SCRATCH_ROOT =
  "parakeet-wgsl-audio-decoding";

const STALE_AFTER_MS = 24 * 60 * 60 * 1_000;

interface IterableFileSystemDirectoryHandle extends FileSystemDirectoryHandle {
  entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
}

/** Remove decode jobs abandoned by a crash or terminated browser worker. */
export async function cleanupStaleAudioDecodingScratchData(): Promise<void> {
  if (!hasOpfs()) return;
  try {
    const storageRoot = await navigator.storage.getDirectory();
    const scratchRoot = await storageRoot.getDirectoryHandle(
      AUDIO_DECODING_SCRATCH_ROOT,
    );
    const iterableRoot = scratchRoot as IterableFileSystemDirectoryHandle;
    const cutoff = Date.now() - STALE_AFTER_MS;
    for await (const [name, handle] of iterableRoot.entries()) {
      if (handle.kind !== "directory") continue;
      const timestamp = Number.parseInt(name.split("-", 1)[0] ?? "", 10);
      if (Number.isFinite(timestamp) && timestamp < cutoff) {
        await scratchRoot
          .removeEntry(name, { recursive: true })
          .catch(() => undefined);
      }
    }
  } catch (error) {
    if (isNotFoundError(error)) return;
    console.warn(
      "[parakeet.wgsl] Failed to remove stale audio-decoding scratch data",
      error,
    );
  }
}

export async function removeAudioDecodingScratchDirectory(
  name: string,
  retryMissing = false,
): Promise<void> {
  if (!hasOpfs()) return;
  const retryDelaysMs = [0, 20, 80] as const;
  let lastError: unknown;
  for (const [attempt, delayMs] of retryDelaysMs.entries()) {
    if (delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    try {
      const storageRoot = await navigator.storage.getDirectory();
      const scratchRoot = await storageRoot.getDirectoryHandle(
        AUDIO_DECODING_SCRATCH_ROOT,
      );
      await scratchRoot.removeEntry(name, { recursive: true });
      return;
    } catch (error) {
      lastError = error;
      const isLastAttempt = attempt === retryDelaysMs.length - 1;
      if (isNotFoundError(error) && (!retryMissing || isLastAttempt)) return;
      if (!isNotFoundError(error) && !isTransientOpfsError(error)) break;
    }
  }
  console.warn(
    "[parakeet.wgsl] Failed to remove audio-decoding scratch data",
    lastError,
  );
}

export function assertAudioDecodingOpfsAvailable(): void {
  if (!hasOpfs()) {
    throw new Error(
      "Streaming decode for non-canonical audio requires Origin Private File System support in this browser.",
    );
  }
}

function hasOpfs(): boolean {
  return (
    typeof navigator !== "undefined" &&
    navigator.storage !== undefined &&
    typeof navigator.storage.getDirectory === "function"
  );
}

function isNotFoundError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}

function isTransientOpfsError(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    (error.name === "InvalidStateError" ||
      error.name === "NoModificationAllowedError")
  );
}
