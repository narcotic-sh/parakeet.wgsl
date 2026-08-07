#!/usr/bin/env node

import { execFile as execFileCallback, spawn } from "node:child_process";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

import { build, createServer, preview } from "vite";

const execFile = promisify(execFileCallback);
const repositoryRoot = resolve(import.meta.dirname, "..");
const chrome =
  process.env.CHROME_PATH ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const temporaryRoot = await realpath(
  await mkdtemp(join(tmpdir(), "parakeet-wgsl-packed-browser-")),
);
const consumerDirectory = join(temporaryRoot, "consumer");
const packDirectory = join(temporaryRoot, "pack");
const profileDirectory = await mkdtemp(
  join(tmpdir(), "parakeet-wgsl-audio-decoding-"),
);

async function main() {
  let browser;
  let cdp;
  let server;
  try {
    await preparePackedConsumer();

    browser = spawn(
      chrome,
      [
        `--user-data-dir=${profileDirectory}`,
        "--headless=new",
        "--remote-debugging-address=127.0.0.1",
        "--remote-debugging-port=0",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "about:blank",
      ],
      { detached: true, stdio: ["ignore", "ignore", "pipe"] },
    );
    const stderr = [];
    browser.stderr?.setEncoding("utf8");
    browser.stderr?.on("data", (chunk) => {
      stderr.push(String(chunk));
      while (stderr.length > 30) stderr.shift();
    });

    const webSocketUrl = await waitForDevTools(
      profileDirectory,
      browser,
      stderr,
    );
    cdp = await CdpConnection.connect(webSocketUrl);
    const version = await cdp.send("Browser.getVersion");

    server = await createServer({
      configFile: false,
      root: consumerDirectory,
      logLevel: "error",
      server: { host: "127.0.0.1", port: 0, strictPort: false },
    });
    await server.listen();
    const developmentOrigin = serverOrigin(server);
    const development = await runAcceptancePage(
      cdp,
      `${developmentOrigin}/`,
      "Vite development",
    );
    const developmentRuntimeAssets =
      await verifyDevelopmentRuntimeAssets(developmentOrigin);
    await closeViteServer(server);
    server = undefined;

    await build({
      configFile: false,
      root: consumerDirectory,
      logLevel: "error",
    });
    server = await preview({
      configFile: false,
      root: consumerDirectory,
      logLevel: "error",
      preview: { host: "127.0.0.1", port: 0, strictPort: false },
    });
    const production = await runAcceptancePage(
      cdp,
      `${serverOrigin(server)}/`,
      "Vite production",
    );
    process.stdout.write(
      `${JSON.stringify(
        {
          ok: true,
          packedConsumer: {
            development: {
              ...development,
              runtimeAssets: developmentRuntimeAssets,
            },
            production,
          },
          chrome: {
            product: version.product,
            userAgent: version.userAgent,
            jsVersion: version.jsVersion,
          },
        },
        null,
        2,
      )}\n`,
    );
  } finally {
    cdp?.close();
    await terminateProcessGroup(browser);
    await closeViteServer(server);
    await Promise.all(
      [profileDirectory, temporaryRoot].map((directory) =>
        rm(directory, {
          recursive: true,
          force: true,
          maxRetries: 3,
          retryDelay: 100,
        }),
      ),
    );
  }
}

async function preparePackedConsumer() {
  await Promise.all([
    mkdir(consumerDirectory, { recursive: true }),
    mkdir(packDirectory, { recursive: true }),
  ]);
  const { stdout } = await execFile(
    "npm",
    [
      "pack",
      "--json",
      "--ignore-scripts",
      "--pack-destination",
      packDirectory,
    ],
    { cwd: repositoryRoot, maxBuffer: 10 * 1024 * 1024 },
  );
  const report = JSON.parse(stdout);
  const packages = Array.isArray(report) ? report : Object.values(report);
  const filename = packages.at(-1)?.filename;
  if (typeof filename !== "string") {
    throw new Error("npm pack did not report a tarball");
  }
  await writeFile(
    join(consumerDirectory, "package.json"),
    `${JSON.stringify({ private: true, type: "module" }, null, 2)}\n`,
  );
  await execFile(
    "npm",
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--no-package-lock",
      "--no-save",
      join(packDirectory, filename),
    ],
    { cwd: consumerDirectory, maxBuffer: 10 * 1024 * 1024 },
  );
  await Promise.all([
    copyFile(
      join(repositoryRoot, "test", "browser", "audio-decoding.html"),
      join(consumerDirectory, "index.html"),
    ),
    copyFile(
      join(repositoryRoot, "test", "browser", "audio-decoding.ts"),
      join(consumerDirectory, "audio-decoding.ts"),
    ),
  ]);
}

async function runAcceptancePage(connection, url, label) {
  const { targetId } = await connection.send("Target.createTarget", { url });
  try {
    const { sessionId } = await connection.send(
      "Target.attachToTarget",
      { targetId, flatten: true },
    );
    await Promise.all([
      connection.send("Page.enable", {}, sessionId),
      connection.send("Runtime.enable", {}, sessionId),
    ]);
    const report = await waitForReport(connection, sessionId);
    if (!report.ok) {
      throw new Error(`${label}: ${report.error ?? "audio decoding failed"}`);
    }
    return report;
  } finally {
    await connection.send("Target.closeTarget", { targetId }).catch(() => {});
  }
}

async function verifyDevelopmentRuntimeAssets(origin) {
  const entryResponse = await fetch(`${origin}/audio-decoding.ts`);
  if (!entryResponse.ok) {
    throw new Error(
      `Could not inspect Vite development entry (HTTP ${entryResponse.status})`,
    );
  }
  const entrySource = await entryResponse.text();
  const packageImport = [
    ...entrySource.matchAll(/(?:from|import)\s*["']([^"']+)["']/g),
  ]
    .map((match) => match[1])
    .find((specifier) => specifier.includes("parakeet"));
  if (packageImport === undefined) {
    throw new Error("Vite development entry did not import parakeet.wgsl");
  }
  const packageUrl = new URL(packageImport, `${origin}/`);
  const packageResponse = await fetch(packageUrl);
  if (!packageResponse.ok) {
    throw new Error(
      `Could not inspect optimized parakeet.wgsl (HTTP ${packageResponse.status})`,
    );
  }
  const packageSource = await packageResponse.text();
  const assetSpecifiers = [
    ...packageSource.matchAll(/["'`]([^"'`]*\/dist\/assets\/[^"'`]+)["'`]/g),
  ].map((match) => match[1]);
  const uniqueSpecifiers = [...new Set(assetSpecifiers)];
  if (uniqueSpecifiers.length !== 4) {
    throw new Error(
      `Expected four optimized development asset URLs; found ${uniqueSpecifiers.length}`,
    );
  }
  const loaded = [];
  for (const specifier of uniqueSpecifiers) {
    const assetUrl = new URL(specifier, packageUrl);
    const response = await fetch(assetUrl);
    if (!response.ok) {
      throw new Error(
        `Vite development asset ${assetUrl.pathname} returned HTTP ${response.status}`,
      );
    }
    loaded.push(assetUrl.pathname.slice(assetUrl.pathname.lastIndexOf("/") + 1));
    await response.body?.cancel();
  }
  return loaded.sort();
}

function serverOrigin(server) {
  const address = server?.httpServer?.address();
  if (
    address === null ||
    address === undefined ||
    typeof address === "string"
  ) {
    throw new Error("Vite did not expose a TCP test address");
  }
  return `http://127.0.0.1:${address.port}`;
}

async function closeViteServer(server) {
  if (server === undefined) return;
  if (typeof server.close === "function") {
    await server.close();
    return;
  }
  const httpServer = server.httpServer;
  if (httpServer === undefined || !httpServer.listening) return;
  await new Promise((resolvePromise, rejectPromise) => {
    httpServer.close((error) => {
      if (error === undefined) resolvePromise();
      else rejectPromise(error);
    });
  });
}

class CdpConnection {
  #socket;
  #nextId = 1;
  #pending = new Map();

  static async connect(webSocketUrl) {
    if (typeof WebSocket !== "function") {
      throw new Error("The browser runner requires Node.js with global WebSocket");
    }
    const socket = new WebSocket(webSocketUrl);
    await new Promise((resolvePromise, rejectPromise) => {
      const timer = setTimeout(
        () => rejectPromise(new Error("CDP connection timed out")),
        10_000,
      );
      socket.addEventListener(
        "open",
        () => {
          clearTimeout(timer);
          resolvePromise();
        },
        { once: true },
      );
      socket.addEventListener(
        "error",
        (event) => {
          clearTimeout(timer);
          rejectPromise(new Error(event.message ?? "CDP connection failed"));
        },
        { once: true },
      );
    });
    return new CdpConnection(socket);
  }

  constructor(socket) {
    this.#socket = socket;
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data));
      if (message.id === undefined) return;
      const pending = this.#pending.get(message.id);
      if (pending === undefined) return;
      this.#pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error === undefined) pending.resolve(message.result ?? {});
      else pending.reject(new Error(`${pending.method}: ${message.error.message}`));
    });
  }

  send(method, params = {}, sessionId, timeoutMs = 30_000) {
    const id = this.#nextId++;
    return new Promise((resolvePromise, rejectPromise) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id);
        rejectPromise(new Error(`${method} timed out`));
      }, timeoutMs);
      this.#pending.set(id, {
        method,
        resolve: resolvePromise,
        reject: rejectPromise,
        timer,
      });
      this.#socket.send(
        JSON.stringify({
          id,
          method,
          params,
          ...(sessionId === undefined ? {} : { sessionId }),
        }),
      );
    });
  }

  close() {
    this.#socket.close();
  }
}

async function waitForDevTools(profile, child, stderr) {
  const activePort = join(profile, "DevToolsActivePort");
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(
        `Chrome exited before CDP was ready:\n${stderr.join("")}`,
      );
    }
    try {
      const [portSource, browserPath] = (await readFile(activePort, "utf8"))
        .trim()
        .split(/\r?\n/);
      const port = Number(portSource);
      if (Number.isSafeInteger(port) && browserPath) {
        const response = await fetch(`http://127.0.0.1:${port}/json/version`);
        if (response.ok) {
          const version = await response.json();
          return (
            version.webSocketDebuggerUrl ??
            `ws://127.0.0.1:${port}${browserPath}`
          );
        }
      }
    } catch {
      // Chrome has not written the active port yet.
    }
    await delay(100);
  }
  throw new Error(`Timed out waiting for Chrome:\n${stderr.join("")}`);
}

async function waitForReport(connection, sessionId) {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    try {
      const response = await connection.send(
        "Runtime.evaluate",
        {
          expression: `(() => {
            const text = document.querySelector("#result")?.textContent ?? "";
            let report;
            try { report = JSON.parse(text); } catch {}
            return report?.ok === true || report?.ok === false ? report : null;
          })()`,
          awaitPromise: true,
          returnByValue: true,
        },
        sessionId,
      );
      if (response.exceptionDetails !== undefined) {
        throw new Error(
          response.exceptionDetails.exception?.description ??
            response.exceptionDetails.text,
        );
      }
      if (response.result?.value !== null) return response.result.value;
    } catch {
      // Page.navigate may briefly replace the JavaScript execution context.
    }
    await delay(100);
  }
  throw new Error("Timed out waiting for browser audio decoding");
}

async function terminateProcessGroup(child) {
  if (child === undefined) return;
  const signal = (name) => {
    try {
      process.kill(-child.pid, name);
    } catch (error) {
      if (error.code !== "ESRCH") throw error;
    }
  };
  signal("SIGTERM");
  await Promise.race([
    new Promise((resolvePromise) => child.once("exit", resolvePromise)),
    delay(5_000),
  ]);
  try {
    process.kill(-child.pid, 0);
    signal("SIGKILL");
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
}

function delay(milliseconds) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));
}

await main();
