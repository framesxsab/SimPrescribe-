import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:net";
import { closeSync, existsSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { randomBytes, randomUUID } from "node:crypto";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const tempRoot = join(root, "tmp");
mkdirSync(tempRoot, { recursive: true });
const tempDir = mkdtempSync(join(tempRoot, "a11y-checkpoint-"));
const dbPath = join(tempDir, "accessibility.sqlite");
const storageDir = join(tempDir, "prescriptions");
const logPath = join(tempDir, "uvicorn.log");
const sessionSecret = randomBytes(32).toString("hex");
const python = process.env.SIMPLISCRIBE_PYTHON || (
  process.platform === "win32"
    ? join(root, ".venv", "Scripts", "python.exe")
    : existsSync(join(root, ".venv", "bin", "python"))
      ? join(root, ".venv", "bin", "python")
      : "python3"
);
const databasePath = dbPath.replace(/\\/g, "/");
const databaseUrl = process.platform === "win32"
  ? "sqlite:///" + databasePath
  : "sqlite:///" + databasePath;
const port = await freePort();
const baseUrl = "http://127.0.0.1:" + port;
const email = "a11y-" + randomUUID() + "@example.test";
const password = "A11y-" + randomBytes(24).toString("hex") + "!";

const env = {
  ...process.env,
  APP_ENV: "development",
  AUTH_REQUIRED: "true",
  SESSION_SECRET: sessionSecret,
  DATABASE_URL: databaseUrl,
  PRESCRIPTION_STORAGE_DIR: storageDir,
  PYTHONPATH: [root, process.env.PYTHONPATH || ""].filter(Boolean).join(process.platform === "win32" ? ";" : ":"),
  ALTERNATIVES_ENABLED: "false",
  A11Y_EMAIL: email,
  A11Y_PASSWORD: password,
  SIMPLISCRIBE_BASE_URL: baseUrl,
};

let server = null;
let logDescriptor = null;
let exitCode = 1;

try {
  const seeded = spawnSync(python, [join(root, "tests", "a11y_seed.py")], {
    cwd: root,
    env,
    encoding: "utf8",
    maxBuffer: 4 * 1024 * 1024,
  });
  if (seeded.error || seeded.status !== 0) {
    throw new Error("Could not seed accessibility fixture: " + (seeded.stderr || seeded.error || seeded.status));
  }
  const seedLines = seeded.stdout.trim().split(/\r?\n/);
  const fixture = JSON.parse(seedLines[seedLines.length - 1]);
  Object.assign(env, fixture);

  logDescriptor = openSync(logPath, "w");
  server = spawn(python, [
    "-m",
    "uvicorn",
    "tests.a11y_app:app",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
    "--no-access-log",
  ], {
    cwd: root,
    env,
    stdio: ["ignore", logDescriptor, logDescriptor],
  });

  await waitForHealth(server, baseUrl, logPath);
  console.log("Accessibility fixture ready: isolated SQLite DB, synthetic patient, four analysis states, owned order.");

  const cliPath = join(root, "node_modules", "playwright", "cli.js");
  if (!existsSync(cliPath)) {
    throw new Error("Playwright is missing; install the repository lockfile dependencies first.");
  }
  const run = spawnSync(process.execPath, [
    cliPath,
    "test",
    "tests/a11y.spec.mjs",
    ...process.argv.slice(2),
  ], {
    cwd: root,
    env,
    stdio: "inherit",
  });
  if (run.error) throw run.error;
  exitCode = run.status ?? 1;
} catch (error) {
  console.error(error instanceof Error ? error.stack : String(error));
  exitCode = 1;
} finally {
  if (exitCode !== 0 && existsSync(logPath)) {
    console.error("Accessibility server log:\n" + readFileSync(logPath, "utf8"));
  }
  if (server && server.exitCode === null) {
    server.kill();
    await Promise.race([
      new Promise((resolveExit) => server.once("exit", resolveExit)),
      delay(3000),
    ]);
    if (server.exitCode === null && process.platform === "win32") {
      spawnSync("taskkill.exe", ["/PID", String(server.pid), "/T", "/F"], { stdio: "ignore" });
    }
  }
  if (logDescriptor !== null) closeSync(logDescriptor);
  for (let attempt = 0; attempt < 16; attempt += 1) {
    try {
      rmSync(tempDir, { recursive: true, force: true });
      break;
    } catch (error) {
      if (attempt === 15) {
        console.error("Could not remove temporary accessibility fixture: " + String(error));
        exitCode = 1;
      }
      await delay(250);
    }
  }
  console.log("Accessibility fixture cleanup complete.");
}
process.exitCode = exitCode;

async function freePort() {
  const serverSocket = createServer();
  await new Promise((resolveListen, reject) => {
    serverSocket.once("error", reject);
    serverSocket.listen(0, "127.0.0.1", resolveListen);
  });
  const address = serverSocket.address();
  const available = address.port;
  await new Promise((resolveClose) => serverSocket.close(resolveClose));
  return available;
}

async function waitForHealth(child, url, serverLog) {
  for (let attempt = 0; attempt < 240; attempt += 1) {
    if (child.exitCode !== null) break;
    try {
      const response = await fetch(url + "/api/health", { signal: AbortSignal.timeout(1000) });
      if (response.ok) return;
    } catch {
      // The app is still starting.
    }
    await delay(250);
  }
  const log = existsSync(serverLog) ? readFileSync(serverLog, "utf8") : "";
  throw new Error("Accessibility app did not become ready within 60 seconds. Server log:\n" + log);
}

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}
