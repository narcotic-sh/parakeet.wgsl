import { posix as path } from "node:path";

import { defineConfig, type Plugin } from "vite";

const EXPECTED_RUNTIME_ASSET_URLS = 4;

/**
 * Vite 8 deliberately blinds its own Rolldown pass by emitting
 * `"" + import.meta.url` as the base of a built asset URL. That output is
 * correct for an application, but an npm package must remain analyzable by a
 * second consumer build. Restore the web-standard literal base after this
 * build is complete and fail closed if Vite's output shape ever changes.
 */
function publishRelocatableRuntimeAssets(): Plugin {
  return {
    name: "parakeet:publish-relocatable-runtime-assets",
    enforce: "post",
    generateBundle(_options, bundle) {
      const entry = bundle["index.js"];
      if (entry?.type !== "chunk") {
        throw new Error("Parakeet package build did not emit dist/index.js");
      }
      let replacements = 0;
      entry.code = entry.code.replace(
        /new URL\(("\.\/assets\/[^"]+"|'\.\/assets\/[^']+'),\s*(["'])\2\s*\+\s*import\.meta\.url\)/g,
        (_match, assetExpression: string) => {
          replacements += 1;
          return `new URL(${assetExpression},import.meta.url)`;
        },
      );
      if (replacements !== EXPECTED_RUNTIME_ASSET_URLS) {
        throw new Error(
          `Expected ${EXPECTED_RUNTIME_ASSET_URLS} relocatable runtime asset URLs; found ${replacements}`,
        );
      }
    },
  };
}

/**
 * This deliberately uses Vite's no-HTML application build instead of
 * `build.lib`. Vite always inlines assets in library mode, while Parakeet's
 * module worker and Wasm frontend need to remain separate, cacheable files.
 */
export default defineConfig({
  base: "./",
  plugins: [publishRelocatableRuntimeAssets()],
  experimental: {
    // Keep package-owned URL and Worker references in the standard literal
    // `new URL("./assets/...", import.meta.url)` form in the published entry.
    // A consuming bundler can then discover and relocate every asset again.
    renderBuiltUrl(filename, { hostId }) {
      const relative = path.relative(path.dirname(hostId), filename);
      return relative.startsWith(".") ? relative : `./${relative}`;
    },
  },
  build: {
    target: "esnext",
    outDir: "dist",
    emptyOutDir: true,
    assetsInlineLimit: 0,
    minify: "esbuild",
    sourcemap: false,
    rollupOptions: {
      input: "src/index.ts",
      preserveEntrySignatures: "strict",
      output: {
        format: "es",
        entryFileNames: "index.js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]"
      }
    }
  },
  worker: {
    format: "es",
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]"
      }
    }
  }
});
