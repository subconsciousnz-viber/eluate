// Builds desktop/dist-engine/eluate-engine/ (PyInstaller onedir) from the repo's .venv.
// Cross-platform replacement for build-engine.ps1: same spec, same paths, same flags.
// PyInstaller builds for the host, so this produces an arm64 engine on Apple Silicon, an
// x64 one on Windows, and a native one on Linux.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scripts = dirname(fileURLToPath(import.meta.url));
const desktop = dirname(scripts);
const root = dirname(desktop);

const py =
  process.platform === "win32"
    ? join(root, ".venv", "Scripts", "python.exe")
    : join(root, ".venv", "bin", "python");

if (!existsSync(py)) {
  console.error(
    `No virtualenv python at ${py}\n` +
      `Create the repo .venv on Python 3.12 and 'pip install -e ".[dev]"' first.`,
  );
  process.exit(1);
}

function run(args, what) {
  const r = spawnSync(py, args, { stdio: "inherit" });
  if (r.error) {
    console.error(`${what} could not start: ${r.error.message}`);
    process.exit(1);
  }
  if (r.status !== 0) {
    console.error(`\n${what} failed (${r.status})`);
    process.exit(r.status ?? 1);
  }
}

run(["-m", "pip", "install", "-q", "pyinstaller>=6.10"], "pip install pyinstaller");
run(
  [
    "-m",
    "PyInstaller",
    join(desktop, "engine", "eluate-engine.spec"),
    "--distpath",
    join(desktop, "dist-engine"),
    "--workpath",
    join(desktop, "build-engine"),
    "--noconfirm",
    "--log-level",
    "WARN",
  ],
  "PyInstaller",
);
