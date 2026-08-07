import { createRawWebGpuParakeetEngine } from "./webgpu-engine";
import {
  installParakeetWorker,
  type ParakeetWorkerScope,
} from "./worker";

installParakeetWorker(
  globalThis as unknown as ParakeetWorkerScope,
  createRawWebGpuParakeetEngine,
);
