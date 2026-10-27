// Copies ../ui into dist/ for Tauri, leaving out the browser-preview data and notes.
import { cpSync, readdirSync, rmSync, utimesSync } from "node:fs";
import { dirname, join, basename } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "..", "ui");
const out = join(here, "..", "dist");
const skip = new Set(["data.js", "README.md"]);

rmSync(out, { recursive: true, force: true });
cpSync(src, out, { recursive: true, filter: (p) => !skip.has(basename(p)) });
// the copies get the time of staging, not the source's: Windows' copy keeps the source's
// mtime, and the shell's build embeds dist/ only when a file is newer than its last build,
// so an edit made before that build would ship the page it replaced
const now = new Date();
for (const f of readdirSync(out, { recursive: true, withFileTypes: true })) {
  if (f.isFile()) utimesSync(join(f.parentPath ?? f.path, f.name), now, now);
}
console.log(`staged ${src} -> ${out}`);
