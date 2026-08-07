import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { defineConfig, type Connect, type Plugin } from "vite";

const THIRD_PARTY_LICENSES_NAME = "THIRD_PARTY_LICENSES";
const THIRD_PARTY_LICENSES_PATH = fileURLToPath(
  new URL("../THIRD_PARTY_LICENSES", import.meta.url),
);
const THIRD_PARTY_LICENSES_OUTPUT_PATH = fileURLToPath(
  new URL("./dist/THIRD_PARTY_LICENSES", import.meta.url),
);

function exposeThirdPartyLicenses(): Plugin {
  const serveLicense = (
    path: string,
  ): Connect.NextHandleFunction => async (request, response, next) => {
    const pathname = new URL(
      request.url ?? "/",
      "http://localhost",
    ).pathname;
    if (pathname !== `/${THIRD_PARTY_LICENSES_NAME}`) {
      next();
      return;
    }
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.statusCode = 405;
      response.setHeader("Allow", "GET, HEAD");
      response.end();
      return;
    }

    try {
      const source = await readFile(path);
      response.statusCode = 200;
      response.setHeader("Content-Type", "text/plain; charset=utf-8");
      response.setHeader("Content-Length", String(source.byteLength));
      response.setHeader("Cache-Control", "no-cache");
      response.end(request.method === "HEAD" ? undefined : source);
    } catch (error) {
      next(error);
    }
  };

  return {
    name: "parakeet:demo-third-party-licenses",
    buildStart() {
      this.addWatchFile(THIRD_PARTY_LICENSES_PATH);
    },
    configureServer(server) {
      server.middlewares.use(serveLicense(THIRD_PARTY_LICENSES_PATH));
    },
    configurePreviewServer(server) {
      server.middlewares.use(serveLicense(THIRD_PARTY_LICENSES_OUTPUT_PATH));
    },
    async generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: THIRD_PARTY_LICENSES_NAME,
        source: await readFile(THIRD_PARTY_LICENSES_PATH),
      });
    },
  };
}

export default defineConfig({
  plugins: [exposeThirdPartyLicenses()],
  build: {
    target: "esnext",
  },
  server: {
    port: 5173,
    strictPort: true,
  },
  preview: {
    port: 4173,
    strictPort: true,
  },
});
