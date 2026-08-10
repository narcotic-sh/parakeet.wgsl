import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

const demoRoot = new URL("../demo/", import.meta.url);

describe("demo markup", () => {
  it("sizes the short-page layout against the usable mobile viewport", async () => {
    const css = await readFile(new URL("src/style.css", demoRoot), "utf8");
    const flexBodyRule = css.match(
      /body\s*\{\s*display:\s*flex;(?<declarations>[^}]*)\}/,
    );

    expect(flexBodyRule?.groups?.declarations).toContain(
      "min-height: 100dvh;",
    );
    expect(flexBodyRule?.groups?.declarations).not.toContain(
      "min-height: 100vh;",
    );
  });

  it("uses the project gear and portable favicon assets", async () => {
    const html = await readFile(new URL("index.html", demoRoot), "utf8");

    expect(html).not.toContain('rel="canonical"');
    expect(html).toContain('src="./src/assets/gear.png"');
    expect(html).toContain('href="./src/assets/favicon-32x32.png"');
    expect(html).toContain('href="./src/assets/favicon-16x16.png"');
    expect(html).toContain('href="./src/assets/apple-touch-icon.png"');
  });

  it("exposes desktop and mobile project links", async () => {
    const html = await readFile(new URL("index.html", demoRoot), "utf8");

    expect(html).toContain('class="desktop-project-links"');
    expect(html).toContain('class="mobile-project-links"');
    expect(html.match(/https:\/\/github\.com\/narcotic-sh\/parakeet\.wgsl/g))
      .toHaveLength(2);
    expect(html.match(/https:\/\/www\.npmjs\.com\/package\/parakeet\.wgsl/g))
      .toHaveLength(2);
  });

  it("places sample download feedback between the file input and actions", async () => {
    const html = await readFile(new URL("index.html", demoRoot), "utf8");
    const audioInput = /<input\s+id="audio"/.exec(html)?.index ?? -1;
    const sampleStatus = html.indexOf('id="sample-status"');
    const actions = html.indexOf('class="actions"');

    expect(audioInput).toBeGreaterThan(-1);
    expect(sampleStatus).toBeGreaterThan(audioInput);
    expect(actions).toBeGreaterThan(sampleStatus);
  });

  it("uses the final English-only transcription copy", async () => {
    const html = await readFile(new URL("index.html", demoRoot), "utf8");

    expect(html).toContain(
      "Fast, accurate transcription in the browser, powered by WebGPU",
    );
    expect(html).toMatch(
      /NVIDIA Parakeet TDT 0\.6B v2<\/a\s*>\s*\(English-only\) entirely on-device/,
    );
    expect(html).toContain("This demo runs");
    expect(html).not.toContain("parakeet.wgsl runs");
  });

  it("places an italic browser warning after the model-cache copy", async () => {
    const [html, css] = await Promise.all([
      readFile(new URL("index.html", demoRoot), "utf8"),
      readFile(new URL("src/style.css", demoRoot), "utf8"),
    ]);
    const cacheCopy = html.indexOf("The model is downloaded");
    const warning = html.indexOf('id="browser-performance-warning"');
    const mobileLinks = html.indexOf('class="mobile-project-links"');

    expect(warning).toBeGreaterThan(cacheCopy);
    expect(mobileLinks).toBeGreaterThan(warning);
    expect(html).toMatch(
      /id="browser-performance-warning"\s+class="browser-performance-warning"\s+hidden/,
    );
    expect(css).toMatch(
      /\.browser-performance-warning\s*{[^}]*font-style:\s*italic;/,
    );
  });

  it("adds a final-transcript copy control to the transcript header", async () => {
    const [html, css, main] = await Promise.all([
      readFile(new URL("index.html", demoRoot), "utf8"),
      readFile(new URL("src/style.css", demoRoot), "utf8"),
      readFile(new URL("src/main.ts", demoRoot), "utf8"),
    ]);
    const heading = html.indexOf('class="panel-heading transcript-heading"');
    const copyButton = html.indexOf('id="copy-transcript"');
    const transcript = html.indexOf('id="output"');

    expect(heading).toBeGreaterThan(-1);
    expect(copyButton).toBeGreaterThan(heading);
    expect(transcript).toBeGreaterThan(copyButton);
    expect(html).toMatch(
      /id="copy-transcript"[\s\S]*?type="button"[\s\S]*?aria-label="Copy transcript"[\s\S]*?hidden[\s\S]*?disabled/,
    );
    expect(html).toContain(
      'class="transcript-copy-icon transcript-copy-icon-copy"',
    );
    expect(html).toContain('stroke="currentColor"');
    expect(css).toMatch(
      /\.transcript-heading\s*{[^}]*display:\s*flex;[^}]*justify-content:\s*space-between;/,
    );
    expect(css).toMatch(
      /\.transcript-copy-button\[hidden\]\s*{[^}]*display:\s*none;/,
    );
    expect(main).toContain(
      "Hover for timestamps. Click a word to seek.",
    );
    expect(main).toContain(
      "navigator.clipboard.writeText(transcriptRenderer.text)",
    );
    expect(main).toContain("setTranscriptCopyAvailable(false)");
    expect(main).toContain(
      "setTranscriptCopyAvailable(result.text.length > 0)",
    );
  });

  it("links the footer to the canonical repository license file", async () => {
    const [html, viteConfig] = await Promise.all([
      readFile(new URL("index.html", demoRoot), "utf8"),
      readFile(new URL("vite.config.ts", demoRoot), "utf8"),
    ]);

    expect(html).toContain('class="site-footer"');
    expect(html).toContain(
      "<!-- parakeet:deployment-footer-attribution -->",
    );
    expect(html).not.toContain("https://hamzaq.com");
    expect(html).not.toContain("Hamza Q.");
    expect(html).toMatch(/>Narcotic Software<\/a\s*>\s*&copy; 2026/);
    expect(html).toContain('href="./THIRD_PARTY_LICENSES"');
    expect(viteConfig).toContain(
      'new URL("../THIRD_PARTY_LICENSES", import.meta.url)',
    );
    expect(viteConfig).toContain(
      'new URL("./dist/THIRD_PARTY_LICENSES", import.meta.url)',
    );
    expect(viteConfig).toContain('fileName: THIRD_PARTY_LICENSES_NAME');
    expect(viteConfig).toContain(
      "server.middlewares.use(serveLicense(THIRD_PARTY_LICENSES_OUTPUT_PATH))",
    );
  });
});
