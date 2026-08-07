import { describe, expect, it } from "vitest";

import { ParakeetTokenizer } from "../../src/model/tokenizer";

function tokenizerJson(): {
  added_tokens: Array<{
    id: number;
    content: string;
    special: boolean;
  }>;
  decoder: {
    type: string;
    replacement: string;
    prepend_scheme: string;
  };
  model: {
    type: string;
    unk_token: string;
    byte_fallback: boolean;
    vocab: Record<string, number>;
  };
} {
  const vocab: Record<string, number> = {
    "<unk>": 0,
    "▁Hello": 1,
    ",": 2,
    "▁world": 3,
    "!": 4,
    "▁again": 5,
  };
  for (let id = 6; id < 1024; id += 1) {
    vocab[`piece-${id}`] = id;
  }
  return {
    added_tokens: [
      {
        id: 0,
        content: "<unk>",
        special: true,
      },
      {
        id: 1024,
        content: "<blank>",
        special: true,
      },
    ],
    decoder: {
      type: "Metaspace",
      replacement: "▁",
      prepend_scheme: "always",
    },
    model: {
      type: "BPE",
      unk_token: "<unk>",
      byte_fallback: false,
      vocab,
    },
  };
}

describe("ParakeetTokenizer", () => {
  it("decodes the fixed v2 BPE vocabulary with Metaspace semantics", () => {
    const tokenizer = ParakeetTokenizer.fromJSON(tokenizerJson());

    expect(tokenizer.decode([1, 2, 3, 4])).toBe("Hello, world!");
    expect(tokenizer.vocabularySize).toBe(1025);
    expect(tokenizer.blankTokenId).toBe(1024);
  });

  it("always skips the fixed unknown and blank tokens", () => {
    const tokenizer = ParakeetTokenizer.fromJSON(tokenizerJson());

    expect(tokenizer.decode([1, 0, 1024, 5])).toBe("Hello again");
  });

  it("exposes raw pieces for token-aware long-form merging", () => {
    const tokenizer = ParakeetTokenizer.fromJSON(tokenizerJson());

    expect(tokenizer.pieceForTokenId(1)).toBe("▁Hello");
    expect(tokenizer.pieceForTokenId(2)).toBe(",");
    expect(() => tokenizer.pieceForTokenId(1025)).toThrow(
      /not present/,
    );
  });

  it("rejects any non-v2 decoder or model behavior", () => {
    const invalidDecoder = tokenizerJson();
    invalidDecoder.decoder.type = "ByteLevel";
    expect(() => ParakeetTokenizer.fromJSON(invalidDecoder)).toThrow(
      "Metaspace",
    );

    const invalidReplacement = tokenizerJson();
    invalidReplacement.decoder.replacement = "_";
    expect(() => ParakeetTokenizer.fromJSON(invalidReplacement)).toThrow(
      "replacement",
    );

    const invalidPrepend = tokenizerJson();
    invalidPrepend.decoder.prepend_scheme = "never";
    expect(() => ParakeetTokenizer.fromJSON(invalidPrepend)).toThrow(
      "prepend_scheme",
    );

    const byteFallback = tokenizerJson();
    byteFallback.model.byte_fallback = true;
    expect(() => ParakeetTokenizer.fromJSON(byteFallback)).toThrow(
      "byte_fallback",
    );

    const wrongUnknown = tokenizerJson();
    wrongUnknown.model.unk_token = "other";
    expect(() => ParakeetTokenizer.fromJSON(wrongUnknown)).toThrow(
      "unk_token",
    );
  });

  it("rejects language tokens and malformed fixed token IDs", () => {
    const languageToken = tokenizerJson();
    languageToken.added_tokens.push({
      id: 1025,
      content: "<|en|>",
      special: true,
    });
    expect(() => ParakeetTokenizer.fromJSON(languageToken)).toThrow(
      "only fixed v2",
    );

    const wrongBlank = tokenizerJson();
    wrongBlank.added_tokens[1]!.id = 1023;
    expect(() => ParakeetTokenizer.fromJSON(wrongBlank)).toThrow(
      "only fixed v2",
    );

    const missingPiece = tokenizerJson();
    delete missingPiece.model.vocab["piece-1023"];
    expect(() => ParakeetTokenizer.fromJSON(missingPiece)).toThrow(
      "contiguous v2 vocabulary",
    );

    const wrongUnknownPiece = tokenizerJson();
    delete wrongUnknownPiece.model.vocab["<unk>"];
    wrongUnknownPiece.model.vocab.other = 0;
    expect(() => ParakeetTokenizer.fromJSON(wrongUnknownPiece)).toThrow(
      "contiguous v2 vocabulary",
    );
  });
});
