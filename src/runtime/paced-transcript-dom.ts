import type {
  PacedTranscriptTextSplice,
  PacedTranscriptUpdate,
  PacedTranscriptWord,
  PacedTranscriptWordSplice,
} from "./paced-transcript.js";

const WORD_HIT_RECTANGLE_TOLERANCE_PIXELS = 1;
const MAX_SCROLL_OFFSET = 2_147_483_647;

export interface PacedTranscriptDomWordHit {
  readonly word: PacedTranscriptWord;
  readonly clientRects: readonly DOMRect[];
}

export interface PacedTranscriptDomRendererOptions {
  /**
   * Follow the growing transcript until the user scrolls away. Native CSS
   * scroll anchoring handles all updates after one initial activation.
   */
  readonly autoFollow?: boolean;
}

/**
 * Incrementally renders one PacedTranscriptUpdate stream into one stable text
 * node. The supplied container is exclusively owned until dispose().
 */
export class PacedTranscriptDomRenderer {
  private readonly document: Document;
  private readonly view: Window;
  private readonly contentElement: HTMLSpanElement;
  private readonly anchorElement: HTMLSpanElement;
  private readonly textNode: Text;
  private readonly words: PacedTranscriptWord[] = [];
  private readonly autoFollow: boolean;

  private initialAutoFollowObserver: IntersectionObserver | undefined;
  private lastRevision = 0;
  private lastSourceRevision = 0;
  private final = false;
  private disposed = false;

  constructor(
    private readonly container: HTMLElement,
    options: PacedTranscriptDomRendererOptions = {},
  ) {
    this.document = container.ownerDocument;
    const view = this.document.defaultView;
    if (view === null) {
      throw new Error(
        "Paced transcript DOM rendering requires a live document",
      );
    }
    this.view = view;
    this.autoFollow = options.autoFollow ?? true;

    this.contentElement = this.document.createElement("span");
    this.contentElement.setAttribute(
      "data-parakeet-transcript-content",
      "",
    );
    this.contentElement.style.setProperty("overflow-anchor", "none");
    this.textNode = this.document.createTextNode("");
    this.contentElement.replaceChildren(this.textNode);

    this.anchorElement = this.document.createElement("span");
    this.anchorElement.setAttribute(
      "data-parakeet-transcript-anchor",
      "",
    );
    this.anchorElement.setAttribute("aria-hidden", "true");
    this.anchorElement.style.setProperty("display", "block");
    this.anchorElement.style.setProperty("inline-size", "1px");
    this.anchorElement.style.setProperty("block-size", "1px");
    this.anchorElement.style.setProperty(
      "overflow-anchor",
      this.autoFollow ? "auto" : "none",
    );
    this.anchorElement.style.setProperty("pointer-events", "none");

    this.container.replaceChildren(
      this.contentElement,
      this.anchorElement,
    );
    this.container.scrollTop = 0;
    this.armInitialAutoFollow();
  }

  get text(): string {
    return this.textNode.data;
  }

  /**
   * The live range array. Callers must treat it as immutable.
   */
  get wordRanges(): readonly PacedTranscriptWord[] {
    return this.words;
  }

  get revision(): number {
    return this.lastRevision;
  }

  get isFinal(): boolean {
    return this.final;
  }

  /**
   * Arrow-bound so it can be passed directly as PacedTranscriptDelivery's
   * onUpdate callback.
   */
  readonly applyUpdate = (update: PacedTranscriptUpdate): void => {
    this.assertUsable();
    this.validateUpdate(update);

    const { textSplice, wordSplice } = update;
    if (textSplice.deleteCount !== 0 || textSplice.insertText.length !== 0) {
      if (
        textSplice.startOffset === this.textNode.length &&
        textSplice.deleteCount === 0
      ) {
        this.textNode.appendData(textSplice.insertText);
      } else {
        this.textNode.replaceData(
          textSplice.startOffset,
          textSplice.deleteCount,
          textSplice.insertText,
        );
      }
    }
    applyWordSplice(this.words, wordSplice);
    this.lastRevision = update.revision;
    this.lastSourceRevision = update.sourceRevision;
    this.final = update.isFinal;
  };

  wordAtTextOffset(textOffset: number): PacedTranscriptWord | undefined {
    this.assertUsable();
    return pacedTranscriptWordAtOffset(this.words, textOffset);
  }

  /**
   * Create a DOM range for one current word entry. Stale or foreign word
   * objects return undefined, even when their values match a current entry.
   */
  domRangeForWord(word: PacedTranscriptWord): Range | undefined {
    this.assertUsable();
    if (
      pacedTranscriptWordAtOffset(this.words, word.startOffset) !== word
    ) {
      return undefined;
    }

    const range = this.document.createRange();
    range.setStart(this.textNode, word.startOffset);
    range.setEnd(this.textNode, word.endOffset);
    return range;
  }

  /**
   * Resolve a timestamped word at viewport coordinates. Geometry is read only
   * when a presentation layer explicitly asks for a hit test.
   */
  wordHitAtPoint(
    clientX: number,
    clientY: number,
  ): PacedTranscriptDomWordHit | undefined {
    this.assertUsable();
    if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) {
      return undefined;
    }
    const textOffset = this.caretOffsetAtPoint(clientX, clientY);
    if (textOffset === undefined) return undefined;

    const atOffset = pacedTranscriptWordAtOffset(this.words, textOffset);
    const beforeOffset =
      textOffset > 0
        ? pacedTranscriptWordAtOffset(this.words, textOffset - 1)
        : undefined;
    const candidates =
      atOffset === beforeOffset
        ? [atOffset]
        : [atOffset, beforeOffset];
    for (const word of candidates) {
      if (word === undefined) continue;
      const clientRects = this.clientRectsForWord(word);
      if (
        pointIntersectsRectangles(
          clientRects,
          clientX,
          clientY,
        )
      ) {
        return { word, clientRects };
      }
    }
    return undefined;
  }

  /**
   * Explicitly resume following after a user has scrolled away.
   */
  jumpToLatest(): void {
    this.assertUsable();
    this.disconnectInitialAutoFollow();
    this.container.scrollTop = MAX_SCROLL_OFFSET;
  }

  /**
   * Start a new paced revision domain while preserving the owned DOM nodes.
   */
  reset(): void {
    this.assertUsable();
    this.disconnectInitialAutoFollow();
    this.textNode.replaceData(0, this.textNode.length, "");
    this.words.length = 0;
    this.lastRevision = 0;
    this.lastSourceRevision = 0;
    this.final = false;
    this.container.scrollTop = 0;
    this.armInitialAutoFollow();
  }

  /**
   * Release observation state while leaving the last rendered text visible.
   */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.disconnectInitialAutoFollow();
    this.words.length = 0;
  }

  private validateUpdate(update: PacedTranscriptUpdate): void {
    if (
      !Number.isSafeInteger(update.revision) ||
      update.revision !== this.lastRevision + 1
    ) {
      throw new RangeError(
        "Paced transcript DOM updates must be contiguous",
      );
    }
    if (
      !Number.isSafeInteger(update.sourceRevision) ||
      update.sourceRevision < 1 ||
      update.sourceRevision < this.lastSourceRevision
    ) {
      throw new RangeError(
        "Paced transcript source revisions must be nondecreasing",
      );
    }
    if (this.final) {
      throw new Error(
        "Cannot apply an update after the paced transcript is final",
      );
    }

    validateTextSplice(this.textNode.length, update.textSplice);
    validateWordSplice(this.words.length, update.wordSplice);
    this.validateWordRanges(update.textSplice, update.wordSplice);
  }

  private validateWordRanges(
    textSplice: PacedTranscriptTextSplice,
    wordSplice: PacedTranscriptWordSplice,
  ): void {
    const nextTextLength =
      this.textNode.length -
      textSplice.deleteCount +
      textSplice.insertText.length;
    const insertedTextEnd =
      textSplice.startOffset + textSplice.insertText.length;
    const previousWord = this.words[wordSplice.startIndex - 1];
    const nextWord =
      this.words[wordSplice.startIndex + wordSplice.deleteCount];
    const insertedWords = wordSplice.insertWords;

    if (
      previousWord !== undefined &&
      previousWord.endOffset > textSplice.startOffset
    ) {
      throw new RangeError(
        "Paced transcript text splice overlaps a retained prior word",
      );
    }
    if (
      nextWord !== undefined &&
      textSplice.insertText.length !== textSplice.deleteCount
    ) {
      throw new RangeError(
        "A length-changing text splice must replace the affected word suffix",
      );
    }
    if (
      nextWord !== undefined &&
      insertedTextEnd > nextWord.startOffset
    ) {
      throw new RangeError(
        "Paced transcript text splice overlaps a retained following word",
      );
    }

    let previousEnd = previousWord?.endOffset ?? 0;
    for (const word of insertedWords) {
      validateWord(word, previousEnd, nextTextLength);
      if (
        word.startOffset < textSplice.startOffset ||
        word.endOffset > insertedTextEnd ||
        textSplice.insertText.slice(
          word.startOffset - textSplice.startOffset,
          word.endOffset - textSplice.startOffset,
        ) !== word.text
      ) {
        throw new RangeError(
          "Paced transcript word range does not match inserted text",
        );
      }
      previousEnd = word.endOffset;
    }
    if (
      nextWord !== undefined &&
      previousEnd > nextWord.startOffset
    ) {
      throw new RangeError(
        "Paced transcript word splice produces overlapping ranges",
      );
    }
  }

  private caretOffsetAtPoint(
    clientX: number,
    clientY: number,
  ): number | undefined {
    const caretPosition =
      this.document.caretPositionFromPoint(clientX, clientY);
    if (caretPosition?.offsetNode === this.textNode) {
      return caretPosition.offset;
    }
    return undefined;
  }

  private clientRectsForWord(
    word: PacedTranscriptWord,
  ): readonly DOMRect[] {
    const range = this.domRangeForWord(word);
    return range === undefined ? [] : Array.from(range.getClientRects());
  }

  private armInitialAutoFollow(): void {
    if (!this.autoFollow) return;
    const view = this.view as Window & {
      readonly IntersectionObserver: typeof IntersectionObserver;
    };
    const observer = new view.IntersectionObserver(
      (entries: readonly IntersectionObserverEntry[]) => {
        const entry = entries.at(-1);
        const rootBounds = entry?.rootBounds;
        if (
          this.disposed ||
          this.initialAutoFollowObserver !== observer ||
          entry === undefined ||
          rootBounds === null ||
          rootBounds === undefined ||
          entry.boundingClientRect.bottom <= rootBounds.bottom
        ) {
          return;
        }
        // The observer runs after layout, so this single write activates the
        // native bottom anchor without a synchronous scrollHeight read.
        this.container.scrollTop = MAX_SCROLL_OFFSET;
        this.disconnectInitialAutoFollow();
      },
      { root: this.container },
    );
    this.initialAutoFollowObserver = observer;
    observer.observe(this.anchorElement);
  }

  private disconnectInitialAutoFollow(): void {
    this.initialAutoFollowObserver?.disconnect();
    this.initialAutoFollowObserver = undefined;
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new Error("Paced transcript DOM renderer is disposed");
    }
  }
}

export function pacedTranscriptWordAtOffset(
  words: readonly PacedTranscriptWord[],
  textOffset: number,
): PacedTranscriptWord | undefined {
  if (!Number.isSafeInteger(textOffset) || textOffset < 0) {
    return undefined;
  }
  let low = 0;
  let high = words.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (words[middle]!.startOffset <= textOffset) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const candidate = words[low - 1];
  return candidate !== undefined &&
    textOffset >= candidate.startOffset &&
    textOffset < candidate.endOffset
    ? candidate
    : undefined;
}

function validateTextSplice(
  textLength: number,
  splice: PacedTranscriptTextSplice,
): void {
  if (
    !Number.isSafeInteger(splice.startOffset) ||
    !Number.isSafeInteger(splice.deleteCount) ||
    splice.startOffset < 0 ||
    splice.deleteCount < 0 ||
    splice.startOffset + splice.deleteCount > textLength
  ) {
    throw new RangeError("Invalid paced transcript DOM text splice");
  }
}

function validateWordSplice(
  wordCount: number,
  splice: PacedTranscriptWordSplice,
): void {
  if (
    !Number.isSafeInteger(splice.startIndex) ||
    !Number.isSafeInteger(splice.deleteCount) ||
    splice.startIndex < 0 ||
    splice.deleteCount < 0 ||
    splice.startIndex + splice.deleteCount > wordCount
  ) {
    throw new RangeError("Invalid paced transcript DOM word splice");
  }
}

function validateWord(
  word: PacedTranscriptWord,
  previousEnd: number,
  textLength: number,
): void {
  if (
    !Number.isSafeInteger(word.startOffset) ||
    !Number.isSafeInteger(word.endOffset) ||
    word.startOffset < previousEnd ||
    word.endOffset <= word.startOffset ||
    word.endOffset > textLength ||
    !Number.isFinite(word.startSeconds) ||
    !Number.isFinite(word.endSeconds) ||
    word.startSeconds < 0 ||
    word.endSeconds < word.startSeconds
  ) {
    throw new RangeError("Invalid paced transcript DOM word range");
  }
}

function applyWordSplice(
  words: PacedTranscriptWord[],
  splice: PacedTranscriptWordSplice,
): void {
  const oldLength = words.length;
  const insertCount = splice.insertWords.length;
  const removedEnd = splice.startIndex + splice.deleteCount;
  const tailCount = oldLength - removedEnd;
  const lengthDelta = insertCount - splice.deleteCount;

  if (lengthDelta > 0) {
    words.length = oldLength + lengthDelta;
    for (let index = tailCount - 1; index >= 0; index -= 1) {
      words[removedEnd + lengthDelta + index] =
        words[removedEnd + index]!;
    }
  } else if (lengthDelta < 0) {
    for (let index = 0; index < tailCount; index += 1) {
      words[splice.startIndex + insertCount + index] =
        words[removedEnd + index]!;
    }
    words.length = oldLength + lengthDelta;
  }
  for (let index = 0; index < insertCount; index += 1) {
    words[splice.startIndex + index] = splice.insertWords[index]!;
  }
}

function pointIntersectsRectangles(
  rectangles: readonly DOMRect[],
  clientX: number,
  clientY: number,
): boolean {
  for (const rectangle of rectangles) {
    if (
      clientX >=
        rectangle.left - WORD_HIT_RECTANGLE_TOLERANCE_PIXELS &&
      clientX <=
        rectangle.right + WORD_HIT_RECTANGLE_TOLERANCE_PIXELS &&
      clientY >=
        rectangle.top - WORD_HIT_RECTANGLE_TOLERANCE_PIXELS &&
      clientY <=
        rectangle.bottom + WORD_HIT_RECTANGLE_TOLERANCE_PIXELS
    ) {
      return true;
    }
  }
  return false;
}
