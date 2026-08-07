interface TokenizerAddedToken {
  readonly id: number;
  readonly content: string;
  readonly special: boolean;
}

interface TokenizerJson {
  readonly added_tokens: readonly unknown[];
  readonly decoder: unknown;
  readonly model: unknown;
}

interface BpeModelJson {
  readonly type: "BPE";
  readonly vocab: Record<string, number>;
}

const V2_MODEL_VOCABULARY_SIZE = 1024;
const V2_UNK_TOKEN_ID = 0;
const V2_BLANK_TOKEN_ID = 1024;

/**
 * Decode-only implementation of NVIDIA's official tokenizer.json.
 *
 * BPE merges are needed to encode text, not to decode model token IDs. Decoding
 * is exactly an ID-to-piece lookup followed by the fixed v2 Metaspace decoder.
 */
export class ParakeetTokenizer {
  readonly vocabularySize: number;
  readonly blankTokenId: number = V2_BLANK_TOKEN_ID;

  readonly #pieces: readonly string[];

  private constructor(pieces: readonly string[]) {
    this.#pieces = pieces;
    this.vocabularySize = pieces.length;
  }

  static fromJSON(value: unknown): ParakeetTokenizer {
    const root = requireRecord(value, "tokenizer");
    if (!Array.isArray(root.added_tokens)) {
      throw new TypeError("tokenizer.added_tokens must be an array");
    }
    const json: TokenizerJson = {
      added_tokens: root.added_tokens,
      decoder: root.decoder,
      model: root.model,
    };
    const model = parseModel(json.model);
    parseDecoder(json.decoder);

    const pieces: (string | undefined)[] = [];
    for (const [piece, idValue] of Object.entries(model.vocab)) {
      const id = requireTokenId(idValue, `vocab entry ${JSON.stringify(piece)}`);
      assignPiece(pieces, id, piece, "model vocabulary");
    }
    if (
      Object.keys(model.vocab).length !== V2_MODEL_VOCABULARY_SIZE ||
      pieces.length !== V2_MODEL_VOCABULARY_SIZE ||
      pieces.filter((piece) => piece !== undefined).length !==
        V2_MODEL_VOCABULARY_SIZE ||
      pieces[V2_UNK_TOKEN_ID] !== "<unk>"
    ) {
      throw new TypeError(
        "tokenizer.model.vocab must contain the fixed contiguous v2 vocabulary",
      );
    }

    const addedTokens = json.added_tokens.map(parseAddedToken);
    if (
      addedTokens.length !== 2 ||
      !hasFixedAddedToken(
        addedTokens,
        V2_UNK_TOKEN_ID,
        "<unk>",
      ) ||
      !hasFixedAddedToken(
        addedTokens,
        V2_BLANK_TOKEN_ID,
        "<blank>",
      )
    ) {
      throw new TypeError(
        "tokenizer.added_tokens must contain only fixed v2 <unk> and <blank>",
      );
    }
    assignPiece(pieces, V2_BLANK_TOKEN_ID, "<blank>", "added token");

    return new ParakeetTokenizer(
      Array.from({ length: pieces.length }, (_, id) => {
        const piece = pieces[id];
        if (piece === undefined) {
          throw new TypeError("tokenizer contains an unmapped token ID");
        }
        return piece;
      }),
    );
  }

  #pieceForId(id: number): string {
    requireTokenId(id, "token ID");
    const piece = this.#pieces[id];
    if (piece !== undefined) return piece;
    throw new RangeError(`Token ID ${id} is not present in tokenizer.json`);
  }

  /** Raw tokenizer piece used for token-aware long-form seam handling. */
  pieceForTokenId(id: number): string {
    return this.#pieceForId(id);
  }

  decode(tokenIds: Iterable<number>): string {
    const rawPieces: string[] = [];

    for (const id of tokenIds) {
      requireTokenId(id, "token ID");
      if (id === V2_UNK_TOKEN_ID || id === V2_BLANK_TOKEN_ID) continue;
      rawPieces.push(this.#pieceForId(id));
    }

    const text = rawPieces.join("").split("▁").join(" ");
    return text.startsWith(" ") ? text.slice(1) : text;
  }
}

function parseModel(value: unknown): BpeModelJson {
  const model = requireRecord(value, "tokenizer.model");
  if (model.type !== "BPE") {
    throw new TypeError(`Expected tokenizer.model.type to be "BPE"`);
  }
  const vocab = requireRecord(model.vocab, "tokenizer.model.vocab");

  const parsedVocab: Record<string, number> = {};
  for (const [piece, id] of Object.entries(vocab)) {
    parsedVocab[piece] = requireTokenId(id, `vocab entry ${JSON.stringify(piece)}`);
  }

  if (model.unk_token !== "<unk>") {
    throw new TypeError('tokenizer.model.unk_token must be "<unk>"');
  }
  if (model.byte_fallback !== false) {
    throw new TypeError("tokenizer.model.byte_fallback must be false");
  }

  return {
    type: "BPE",
    vocab: parsedVocab,
  };
}

function parseDecoder(value: unknown): void {
  const decoder = requireRecord(value, "tokenizer.decoder");
  if (decoder.type !== "Metaspace") {
    throw new TypeError(`Expected tokenizer.decoder.type to be "Metaspace"`);
  }
  if (decoder.replacement !== "▁") {
    throw new TypeError('tokenizer.decoder.replacement must be "▁"');
  }
  if (decoder.prepend_scheme !== "always") {
    throw new TypeError(
      'tokenizer.decoder.prepend_scheme must be "always"',
    );
  }
}

function parseAddedToken(value: unknown): TokenizerAddedToken {
  const token = requireRecord(value, "added token");
  if (typeof token.content !== "string") {
    throw new TypeError("added token content must be a string");
  }
  if (typeof token.special !== "boolean") {
    throw new TypeError("added token special must be a boolean");
  }
  return {
    id: requireTokenId(token.id, "added token ID"),
    content: token.content,
    special: token.special,
  };
}

function assignPiece(
  pieces: (string | undefined)[],
  id: number,
  piece: string,
  source: string,
): void {
  const existing = pieces[id];
  if (existing !== undefined && existing !== piece) {
    throw new TypeError(
      `Token ID ${id} maps to both ${JSON.stringify(existing)} and ${JSON.stringify(piece)} in ${source}`,
    );
  }
  pieces[id] = piece;
}

function hasFixedAddedToken(
  tokens: readonly TokenizerAddedToken[],
  id: number,
  content: string,
): boolean {
  return tokens.some(
    (token) =>
      token.id === id &&
      token.content === content &&
      token.special,
  );
}

function requireRecord(
  value: unknown,
  name: string,
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireTokenId(value: unknown, name: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0
  ) {
    throw new TypeError(`${name} must be a non-negative safe integer`);
  }
  return value;
}
