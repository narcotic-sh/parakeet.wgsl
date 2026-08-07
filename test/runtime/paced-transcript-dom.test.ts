import { beforeEach, describe, expect, it } from "vitest";

import {
  PacedTranscriptDomRenderer,
  pacedTranscriptWordAtOffset,
} from "../../src/runtime/paced-transcript-dom";
import type {
  PacedTranscriptUpdate,
  PacedTranscriptWord,
} from "../../src/runtime/paced-transcript";

describe("paced transcript DOM renderer", () => {
  beforeEach(() => {
    FakeIntersectionObserver.instances.length = 0;
  });

  it("keeps one stable text node while applying append and repair splices", () => {
    const { container } = fakeDom();
    const renderer = new PacedTranscriptDomRenderer(
      container as unknown as HTMLElement,
      { autoFollow: false },
    );
    const textNode = ownedTextNode(container);

    renderer.applyUpdate(
      update(1, {
        startOffset: 0,
        deleteCount: 0,
        insertText: "hello",
      }, {
        startIndex: 0,
        deleteCount: 0,
        insertWords: [word("hello", 0, 5, 0, 0.5)],
      }),
    );
    renderer.applyUpdate(
      update(2, {
        startOffset: 5,
        deleteCount: 0,
        insertText: " world",
      }, {
        startIndex: 1,
        deleteCount: 0,
        insertWords: [word("world", 6, 11, 0.5, 1)],
      }),
    );
    renderer.applyUpdate(
      update(3, {
        startOffset: 5,
        deleteCount: 6,
        insertText: " there",
      }, {
        startIndex: 1,
        deleteCount: 1,
        insertWords: [word("there", 6, 11, 0.6, 1.1)],
      }),
    );
    renderer.applyUpdate(
      update(
        4,
        { startOffset: 11, deleteCount: 0, insertText: "" },
        { startIndex: 2, deleteCount: 0, insertWords: [] },
        true,
      ),
    );

    expect(ownedTextNode(container)).toBe(textNode);
    expect(textNode.data).toBe("hello there");
    expect(renderer.text).toBe("hello there");
    expect(renderer.wordRanges.map(({ text }) => text)).toEqual([
      "hello",
      "there",
    ]);
    expect(renderer.wordAtTextOffset(4)?.text).toBe("hello");
    expect(renderer.wordAtTextOffset(5)).toBeUndefined();
    expect(renderer.wordAtTextOffset(6)?.startSeconds).toBe(0.6);
    expect(renderer.revision).toBe(4);
    expect(renderer.isFinal).toBe(true);
    expect(() =>
      renderer.applyUpdate(
        update(
          5,
          { startOffset: 11, deleteCount: 0, insertText: "!" },
          { startIndex: 2, deleteCount: 0, insertWords: [] },
        ),
      ),
    ).toThrow("after the paced transcript is final");
  });

  it("rejects gaps and malformed ranges without changing rendered state", () => {
    const { container } = fakeDom();
    const renderer = new PacedTranscriptDomRenderer(
      container as unknown as HTMLElement,
      { autoFollow: false },
    );
    renderer.applyUpdate(
      update(1, {
        startOffset: 0,
        deleteCount: 0,
        insertText: "one",
      }, {
        startIndex: 0,
        deleteCount: 0,
        insertWords: [word("one", 0, 3, 0, 1)],
      }),
    );

    expect(() =>
      renderer.applyUpdate(
        update(3, {
          startOffset: 3,
          deleteCount: 0,
          insertText: " two",
        }, {
          startIndex: 1,
          deleteCount: 0,
          insertWords: [word("two", 4, 7, 1, 2)],
        }),
      ),
    ).toThrow("must be contiguous");
    expect(() =>
      renderer.applyUpdate(
        update(2, {
          startOffset: 3,
          deleteCount: 0,
          insertText: " two",
        }, {
          startIndex: 1,
          deleteCount: 0,
          insertWords: [word("wrong", 4, 7, 1, 2)],
        }),
      ),
    ).toThrow("does not match inserted text");
    expect(renderer.text).toBe("one");
    expect(renderer.revision).toBe(1);

    const textNode = ownedTextNode(container);
    renderer.reset();
    expect(ownedTextNode(container)).toBe(textNode);
    expect(renderer.text).toBe("");
    expect(renderer.wordRanges).toEqual([]);
    expect(renderer.revision).toBe(0);
    expect(renderer.isFinal).toBe(false);
  });

  it("creates exact DOM ranges only for current word entries", () => {
    const { container } = fakeDom();
    const renderer = new PacedTranscriptDomRenderer(
      container as unknown as HTMLElement,
      { autoFollow: false },
    );
    renderer.applyUpdate(
      update(1, {
        startOffset: 0,
        deleteCount: 0,
        insertText: "hello world",
      }, {
        startIndex: 0,
        deleteCount: 0,
        insertWords: [
          word("hello", 0, 5, 0, 0.5),
          word("world", 6, 11, 0.5, 1),
        ],
      }),
    );

    const textNode = ownedTextNode(container);
    const hello = renderer.wordRanges[0]!;
    const helloRange = renderer.domRangeForWord(hello);
    expect(helloRange?.startContainer).toBe(textNode);
    expect(helloRange?.startOffset).toBe(0);
    expect(helloRange?.endContainer).toBe(textNode);
    expect(helloRange?.endOffset).toBe(5);
    expect(helloRange?.toString()).toBe("hello");

    expect(renderer.domRangeForWord({ ...hello })).toBeUndefined();

    renderer.applyUpdate(
      update(2, {
        startOffset: 0,
        deleteCount: 5,
        insertText: "there",
      }, {
        startIndex: 0,
        deleteCount: 1,
        insertWords: [word("there", 0, 5, 0.1, 0.6)],
      }),
    );
    expect(renderer.domRangeForWord(hello)).toBeUndefined();
    expect(
      renderer.domRangeForWord(renderer.wordRanges[0]!)?.toString(),
    ).toBe("there");

    const there = renderer.wordRanges[0]!;
    renderer.reset();
    expect(renderer.domRangeForWord(there)).toBeUndefined();

    renderer.dispose();
    expect(() => renderer.domRangeForWord(there)).toThrow(
      "renderer is disposed",
    );
  });

  it("activates the native bottom anchor once and rearms on reset", () => {
    const { container } = fakeDom();
    const renderer = new PacedTranscriptDomRenderer(
      container as unknown as HTMLElement,
    );
    const firstObserver = FakeIntersectionObserver.instances[0]!;
    expect(firstObserver.observed).toHaveLength(1);

    firstObserver.emit(101, 100);
    expect(container.scrollTop).toBe(2_147_483_647);
    expect(firstObserver.disconnected).toBe(true);

    renderer.reset();
    expect(container.scrollTop).toBe(0);
    const secondObserver = FakeIntersectionObserver.instances[1]!;
    expect(secondObserver.observed).toHaveLength(1);
    firstObserver.emit(101, 100);
    expect(container.scrollTop).toBe(0);
    expect(secondObserver.disconnected).toBe(false);

    renderer.jumpToLatest();
    expect(container.scrollTop).toBe(2_147_483_647);
    expect(secondObserver.disconnected).toBe(true);

    renderer.dispose();
    expect(() => renderer.reset()).toThrow("renderer is disposed");
    expect(container.textContent).toBe("");
  });

  it("disables native anchoring and observation when auto-follow is off", () => {
    const { container } = fakeDom();
    const renderer = new PacedTranscriptDomRenderer(
      container as unknown as HTMLElement,
      { autoFollow: false },
    );
    const anchor = container.childNodes[1];
    if (!(anchor instanceof FakeElement)) {
      throw new Error("Missing renderer anchor element");
    }

    expect(
      anchor.style.properties.get("overflow-anchor"),
    ).toBe("none");
    expect(FakeIntersectionObserver.instances).toHaveLength(0);
    expect(container.scrollTop).toBe(0);

    renderer.reset();
    expect(FakeIntersectionObserver.instances).toHaveLength(0);
    expect(container.scrollTop).toBe(0);
  });
});

describe("paced transcript word range lookup", () => {
  it("uses half-open UTF-16 character ranges", () => {
    const words = [
      word("one", 0, 3, 0, 1),
      word("two", 4, 7, 1, 2),
    ];
    expect(pacedTranscriptWordAtOffset(words, 0)?.text).toBe("one");
    expect(pacedTranscriptWordAtOffset(words, 2)?.text).toBe("one");
    expect(pacedTranscriptWordAtOffset(words, 3)).toBeUndefined();
    expect(pacedTranscriptWordAtOffset(words, 4)?.text).toBe("two");
    expect(pacedTranscriptWordAtOffset(words, 7)).toBeUndefined();
    expect(pacedTranscriptWordAtOffset(words, -1)).toBeUndefined();
    expect(pacedTranscriptWordAtOffset(words, 1.5)).toBeUndefined();
  });
});

function update(
  revision: number,
  textSplice: PacedTranscriptUpdate["textSplice"],
  wordSplice: PacedTranscriptUpdate["wordSplice"],
  isFinal = false,
): PacedTranscriptUpdate {
  return {
    revision,
    sourceRevision: 1,
    textSplice,
    wordSplice,
    audioDurationSeconds: 10,
    processedAudioSeconds: isFinal ? 10 : 5,
    isFinal,
  };
}

function word(
  text: string,
  startOffset: number,
  endOffset: number,
  startSeconds: number,
  endSeconds: number,
): PacedTranscriptWord {
  return {
    text,
    startOffset,
    endOffset,
    startSeconds,
    endSeconds,
  };
}

function fakeDom(): {
  readonly container: FakeElement;
  readonly document: FakeDocument;
} {
  const view = {
    IntersectionObserver: FakeIntersectionObserver,
  };
  const document = new FakeDocument(view);
  return {
    container: document.createElement("div"),
    document,
  };
}

function ownedTextNode(container: FakeElement): FakeText {
  const content = container.childNodes[0];
  if (!(content instanceof FakeElement)) {
    throw new Error("Missing renderer content element");
  }
  const text = content.childNodes[0];
  if (!(text instanceof FakeText)) {
    throw new Error("Missing renderer text node");
  }
  return text;
}

abstract class FakeNode {
  parentNode: FakeElement | undefined;

  abstract get textContent(): string;
}

class FakeText extends FakeNode {
  constructor(public data: string) {
    super();
  }

  get length(): number {
    return this.data.length;
  }

  get textContent(): string {
    return this.data;
  }

  appendData(value: string): void {
    this.data += value;
  }

  replaceData(offset: number, count: number, value: string): void {
    this.data =
      this.data.slice(0, offset) +
      value +
      this.data.slice(offset + count);
  }
}

class FakeStyle {
  readonly properties = new Map<string, string>();

  setProperty(name: string, value: string): void {
    this.properties.set(name, value);
  }
}

class FakeElement extends FakeNode {
  readonly childNodes: FakeNode[] = [];
  readonly style = new FakeStyle();
  readonly attributes = new Map<string, string>();
  scrollTop = 0;

  constructor(
    readonly ownerDocument: FakeDocument,
    readonly tagName: string,
  ) {
    super();
  }

  get textContent(): string {
    return this.childNodes.map((child) => child.textContent).join("");
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
  }

  replaceChildren(...nodes: FakeNode[]): void {
    for (const child of this.childNodes) {
      child.parentNode = undefined;
    }
    this.childNodes.length = 0;
    for (const node of nodes) {
      node.parentNode = this;
      this.childNodes.push(node);
    }
  }
}

class FakeDocument {
  readonly defaultView: unknown;

  constructor(view: unknown) {
    this.defaultView = view;
  }

  createElement(tagName: string): FakeElement {
    return new FakeElement(this, tagName);
  }

  createTextNode(data: string): FakeText {
    return new FakeText(data);
  }

  createRange(): FakeRange {
    return new FakeRange();
  }
}

class FakeRange {
  startContainer: FakeNode | undefined;
  startOffset = 0;
  endContainer: FakeNode | undefined;
  endOffset = 0;

  setStart(node: FakeNode, offset: number): void {
    this.startContainer = node;
    this.startOffset = offset;
  }

  setEnd(node: FakeNode, offset: number): void {
    this.endContainer = node;
    this.endOffset = offset;
  }

  getClientRects(): readonly DOMRect[] {
    return [];
  }

  toString(): string {
    if (
      !(this.startContainer instanceof FakeText) ||
      this.endContainer !== this.startContainer
    ) {
      return "";
    }
    return this.startContainer.data.slice(
      this.startOffset,
      this.endOffset,
    );
  }
}

class FakeIntersectionObserver {
  static readonly instances: FakeIntersectionObserver[] = [];

  readonly observed: unknown[] = [];
  disconnected = false;

  constructor(
    private readonly callback: IntersectionObserverCallback,
  ) {
    FakeIntersectionObserver.instances.push(this);
  }

  observe(target: unknown): void {
    this.observed.push(target);
  }

  disconnect(): void {
    this.disconnected = true;
  }

  emit(anchorBottom: number, rootBottom: number): void {
    this.callback(
      [
        {
          boundingClientRect: { bottom: anchorBottom },
          rootBounds: { bottom: rootBottom },
        } as IntersectionObserverEntry,
      ],
      this as unknown as IntersectionObserver,
    );
  }
}
