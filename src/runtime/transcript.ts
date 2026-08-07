import type {
  TranscriptionToken,
  TranscriptionWord,
} from "./protocol";
import type { TimedToken } from "./stitch";

export interface TranscriptDecoder {
  decodeTokenIds(tokenIds: readonly number[]): string;
  tokenPiece(tokenId: number): string | undefined;
}

export interface SerializedTimedTranscript {
  readonly text: string;
  readonly tokens: readonly TranscriptionToken[];
  readonly words: readonly TranscriptionWord[];
}

interface WordTokenRange {
  readonly startIndex: number;
  readonly endIndex: number;
  readonly startSample: number;
  readonly endSample: number;
}

/**
 * Serialize one timestamped token stream without discarding either the raw
 * token timing or word-level timing used by presentation layers.
 *
 * The pinned tokenizer uses a SentencePiece-style `▁` boundary marker.
 * Punctuation and continuation pieces remain attached to the preceding word.
 */
export function serializeTimedTranscript(
  timedTokens: readonly TimedToken[],
  sampleRate: number,
  decoder: TranscriptDecoder,
): SerializedTimedTranscript {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError("sampleRate must be a positive finite number");
  }

  const tokenIds = timedTokens.map((token) => token.tokenId);
  const text = decoder.decodeTokenIds(tokenIds);
  const tokens = timedTokens.map((token) => ({
    tokenId: token.tokenId,
    startSeconds: token.startSample / sampleRate,
    endSeconds: token.endSample / sampleRate,
  }));
  const wordRanges: WordTokenRange[] = [];
  let activeRange: WordTokenRange | undefined;
  for (let index = 0; index < timedTokens.length; index += 1) {
    const token = timedTokens[index]!;
    const piece = decoder.tokenPiece(timedTokens[index]!.tokenId);
    if (isSkippedSpecialPiece(piece)) continue;

    if (
      activeRange === undefined ||
      piece === undefined ||
      startsWord(piece)
    ) {
      if (activeRange !== undefined) wordRanges.push(activeRange);
      activeRange = {
        startIndex: index,
        endIndex: index + 1,
        startSample: token.startSample,
        endSample: token.endSample,
      };
    } else {
      activeRange = {
        ...activeRange,
        endIndex: index + 1,
        endSample: Math.max(activeRange.endSample, token.endSample),
      };
    }
  }
  if (activeRange !== undefined) wordRanges.push(activeRange);

  const trimmedText = text.trim();
  const surfaceWords =
    trimmedText.length === 0 ? [] : trimmedText.split(/\s+/u);
  const words =
    surfaceWords.length === wordRanges.length
      ? wordRanges.map((range, index) =>
          timedWord(surfaceWords[index]!, range, sampleRate),
        )
      : wordRanges.flatMap((range) => {
          const wordText = decoder
            .decodeTokenIds(
              timedTokens
                .slice(range.startIndex, range.endIndex)
                .map((token) => token.tokenId),
            )
            .trim();
          return wordText.length === 0
            ? []
            : [timedWord(wordText, range, sampleRate)];
        });

  return { text, tokens, words };
}

function timedWord(
  text: string,
  range: WordTokenRange,
  sampleRate: number,
): TranscriptionWord {
  return {
    text,
    startSeconds: range.startSample / sampleRate,
    endSeconds: range.endSample / sampleRate,
  };
}

function startsWord(piece: string): boolean {
  return piece.startsWith("▁") || piece.startsWith(" ");
}

function isSkippedSpecialPiece(piece: string | undefined): boolean {
  return piece === "<unk>" || piece === "<pad>" || piece === "<blank>";
}
