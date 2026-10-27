# Builds desktop/dist-engine/eluate-engine/ (PyInstaller onedir) from the repo's .venv.
$ErrorActionPreference = "Stop"
$desktop = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $desktop
$py = Join-Path $root ".venv\Scripts\python.exe"

& $py -m pip install -q "pyinstaller>=6.10"
& $py -m PyInstaller (Join-Path $desktop "engine\eluate-engine.spec") `
    --distpath (Join-Path $desktop "dist-engine") `
    --workpath (Join-Path $desktop "build-engine") `
    --noconfirm --log-level WARN
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }
