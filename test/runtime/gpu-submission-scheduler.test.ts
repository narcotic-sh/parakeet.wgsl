import { describe, expect, it } from "vitest";

import {
  COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH,
  ExclusiveAsyncGate,
  INFERENCE_COMMAND_BUFFERS_PER_GRAPH,
  submitCommandBuffersCooperatively,
} from "../../src/runtime/webgpu-engine";

function makeCommandBuffers(count: number): GPUCommandBuffer[] {
  return Array.from(
    { length: count },
    (_, index) => ({ label: `command-${index}` }) as GPUCommandBuffer,
  );
}

async function flushMicrotasks(): Promise<void> {
  for (let turn = 0; turn < 6; turn += 1) {
    await Promise.resolve();
  }
}

describe("cooperative GPU submission scheduler", () => {
  it("submits 27 singleton buffers with 26 queue drains and idle yields", async () => {
    const commandBuffers = makeCommandBuffers(
      INFERENCE_COMMAND_BUFFERS_PER_GRAPH,
    );
    const submissions: GPUCommandBuffer[][] = [];
    const completionCallbacks: Array<
      readonly [completed: number, total: number]
    > = [];
    let drainCount = 0;
    let yieldCount = 0;
    const queue = {
      submit(buffers: Iterable<GPUCommandBuffer>): undefined {
        submissions.push([...buffers]);
        return undefined;
      },
      onSubmittedWorkDone(): Promise<undefined> {
        drainCount += 1;
        return Promise.resolve(undefined);
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;

    await submitCommandBuffersCooperatively(
      queue,
      commandBuffers,
      new AbortController().signal,
      () => {
        yieldCount += 1;
        return Promise.resolve();
      },
      (completed, total) => {
        completionCallbacks.push([completed, total]);
      },
    );

    expect(submissions).toHaveLength(
      INFERENCE_COMMAND_BUFFERS_PER_GRAPH,
    );
    expect(
      submissions.every((submission) => submission.length === 1),
    ).toBe(true);
    expect(submissions.flat()).toEqual(commandBuffers);
    expect(drainCount).toBe(
      COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH,
    );
    expect(yieldCount).toBe(
      COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH,
    );
    expect(completionCallbacks).toEqual(
      Array.from(
        { length: COOPERATIVE_GPU_QUEUE_DRAINS_PER_GRAPH },
        (_, index) =>
          [
            index + 1,
            INFERENCE_COMMAND_BUFFERS_PER_GRAPH,
          ] as const,
      ),
    );
  });

  it("waits for both the queue drain and idle interval before each next submit", async () => {
    const commandBuffers = makeCommandBuffers(3);
    const drainDeferreds: PromiseWithResolvers<undefined>[] = [];
    const yieldDeferreds: PromiseWithResolvers<void>[] = [];
    const events: string[] = [];
    let outstanding = 0;
    let maxOutstanding = 0;
    const queue = {
      submit(buffers: Iterable<GPUCommandBuffer>): undefined {
        const [buffer, ...extra] = [...buffers];
        if (buffer === undefined || extra.length !== 0) {
          throw new Error("Expected one command buffer per submission");
        }
        if (outstanding !== 0) {
          throw new Error("Submitted before the prior buffer drained");
        }
        const index = commandBuffers.indexOf(buffer);
        events.push(`submit:${index}`);
        outstanding += 1;
        maxOutstanding = Math.max(maxOutstanding, outstanding);
        return undefined;
      },
      onSubmittedWorkDone(): Promise<undefined> {
        const index = drainDeferreds.length;
        const deferred = Promise.withResolvers<undefined>();
        drainDeferreds.push(deferred);
        events.push(`fence:${index}`);
        return deferred.promise.then(() => {
          outstanding -= 1;
          events.push(`drained:${index}`);
          return undefined;
        });
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
    const yieldQueueIdle = (): Promise<void> => {
      const index = yieldDeferreds.length;
      const deferred = Promise.withResolvers<void>();
      yieldDeferreds.push(deferred);
      events.push(`yield:${index}`);
      return deferred.promise.then(() => {
        events.push(`yielded:${index}`);
      });
    };

    const completion = submitCommandBuffersCooperatively(
      queue,
      commandBuffers,
      new AbortController().signal,
      yieldQueueIdle,
      (completed, total) => {
        events.push(`completed:${completed}/${total}`);
      },
    );

    expect(events).toEqual(["submit:0", "fence:0"]);
    drainDeferreds[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(events).toEqual([
      "submit:0",
      "fence:0",
      "drained:0",
      "yield:0",
      "completed:1/3",
    ]);

    yieldDeferreds[0]!.resolve();
    await flushMicrotasks();
    expect(events).toEqual([
      "submit:0",
      "fence:0",
      "drained:0",
      "yield:0",
      "completed:1/3",
      "yielded:0",
      "submit:1",
      "fence:1",
    ]);

    drainDeferreds[1]!.resolve(undefined);
    await flushMicrotasks();
    expect(events.at(-1)).toBe("completed:2/3");
    expect(events).not.toContain("submit:2");

    yieldDeferreds[1]!.resolve();
    await completion;
    expect(events).toEqual([
      "submit:0",
      "fence:0",
      "drained:0",
      "yield:0",
      "completed:1/3",
      "yielded:0",
      "submit:1",
      "fence:1",
      "drained:1",
      "yield:1",
      "completed:2/3",
      "yielded:1",
      "submit:2",
    ]);
    expect(maxOutstanding).toBe(1);
    expect(outstanding).toBe(1);
    expect(drainDeferreds).toHaveLength(2);
    expect(yieldDeferreds).toHaveLength(2);
  });

  it("does not submit another buffer when aborted during the idle interval", async () => {
    const controller = new AbortController();
    const idle = Promise.withResolvers<void>();
    const submissions: GPUCommandBuffer[][] = [];
    let drainCount = 0;
    let yieldCount = 0;
    const completionCallbacks: number[] = [];
    const queue = {
      submit(buffers: Iterable<GPUCommandBuffer>): undefined {
        submissions.push([...buffers]);
        return undefined;
      },
      onSubmittedWorkDone(): Promise<undefined> {
        drainCount += 1;
        return Promise.resolve(undefined);
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
    const completion = submitCommandBuffersCooperatively(
      queue,
      makeCommandBuffers(2),
      controller.signal,
      () => {
        yieldCount += 1;
        return idle.promise;
      },
      (completed) => {
        completionCallbacks.push(completed);
      },
    );
    const rejection = expect(completion).rejects.toMatchObject({
      name: "AbortError",
    });

    await flushMicrotasks();
    expect(submissions).toHaveLength(1);
    expect(drainCount).toBe(1);
    expect(yieldCount).toBe(1);
    expect(completionCallbacks).toEqual([1]);

    controller.abort();
    idle.resolve();
    await rejection;
    expect(submissions).toHaveLength(1);
  });

  it("waits for the already-started idle interval when a completion callback throws", async () => {
    const failure = new Error("completion callback failed");
    const idle = Promise.withResolvers<void>();
    const submissions: GPUCommandBuffer[][] = [];
    let rejected = false;
    const queue = {
      submit(buffers: Iterable<GPUCommandBuffer>): undefined {
        submissions.push([...buffers]);
        return undefined;
      },
      onSubmittedWorkDone(): Promise<undefined> {
        return Promise.resolve(undefined);
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
    const completion = submitCommandBuffersCooperatively(
      queue,
      makeCommandBuffers(2),
      new AbortController().signal,
      () => idle.promise,
      () => {
        throw failure;
      },
    );
    void completion.catch(() => {
      rejected = true;
    });

    await flushMicrotasks();
    expect(submissions).toHaveLength(1);
    expect(rejected).toBe(false);

    idle.resolve();
    await expect(completion).rejects.toBe(failure);
    expect(rejected).toBe(true);
    expect(submissions).toHaveLength(1);
  });

  it("rejects an empty command-buffer list without touching the queue", async () => {
    const queue = {
      submit(): undefined {
        throw new Error("queue.submit must not run");
      },
      onSubmittedWorkDone(): Promise<undefined> {
        throw new Error("queue.onSubmittedWorkDone must not run");
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;

    await expect(
      submitCommandBuffersCooperatively(
        queue,
        [],
        new AbortController().signal,
      ),
    ).rejects.toThrow(/At least one GPU command buffer/);
  });

  it("stops immediately when a queue drain rejects", async () => {
    const failure = new Error("queue drain failed");
    const submissions: GPUCommandBuffer[][] = [];
    let yieldCount = 0;
    const queue = {
      submit(buffers: Iterable<GPUCommandBuffer>): undefined {
        submissions.push([...buffers]);
        return undefined;
      },
      onSubmittedWorkDone(): Promise<undefined> {
        return Promise.reject(failure);
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;

    await expect(
      submitCommandBuffersCooperatively(
        queue,
        makeCommandBuffers(2),
        new AbortController().signal,
        () => {
          yieldCount += 1;
          return Promise.resolve();
        },
      ),
    ).rejects.toBe(failure);
    expect(submissions).toHaveLength(1);
    expect(yieldCount).toBe(0);
  });

  it("stops immediately when the idle interval rejects", async () => {
    const failure = new Error("idle interval failed");
    const submissions: GPUCommandBuffer[][] = [];
    let drainCount = 0;
    const queue = {
      submit(buffers: Iterable<GPUCommandBuffer>): undefined {
        submissions.push([...buffers]);
        return undefined;
      },
      onSubmittedWorkDone(): Promise<undefined> {
        drainCount += 1;
        return Promise.resolve(undefined);
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;

    await expect(
      submitCommandBuffersCooperatively(
        queue,
        makeCommandBuffers(2),
        new AbortController().signal,
        () => Promise.reject(failure),
      ),
    ).rejects.toBe(failure);
    expect(submissions).toHaveLength(1);
    expect(drainCount).toBe(1);
  });
});

describe("exclusive async graph gate", () => {
  it("grants leases in FIFO order and makes every release idempotent", async () => {
    const gate = new ExclusiveAsyncGate();
    const acquisitionOrder: string[] = [];
    const firstRelease = await gate.acquire();
    acquisitionOrder.push("first");
    const secondLease = gate.acquire().then((release) => {
      acquisitionOrder.push("second");
      return release;
    });
    const thirdLease = gate.acquire().then((release) => {
      acquisitionOrder.push("third");
      return release;
    });

    await flushMicrotasks();
    expect(acquisitionOrder).toEqual(["first"]);

    firstRelease();
    const secondRelease = await secondLease;
    expect(acquisitionOrder).toEqual(["first", "second"]);

    firstRelease();
    await flushMicrotasks();
    expect(acquisitionOrder).toEqual(["first", "second"]);

    secondRelease();
    const thirdRelease = await thirdLease;
    expect(acquisitionOrder).toEqual(["first", "second", "third"]);

    secondRelease();
    thirdRelease();
    thirdRelease();
    const fourthRelease = await gate.acquire();
    acquisitionOrder.push("fourth");
    fourthRelease();
    expect(acquisitionOrder).toEqual([
      "first",
      "second",
      "third",
      "fourth",
    ]);
  });
});
