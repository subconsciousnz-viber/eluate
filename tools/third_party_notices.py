"""Write THIRD_PARTY_NOTICES.txt at the repo root: the licences of everything the installer ships.

Run with the venv the engine bundle is built from (standard library only):

    .venv/Scripts/python tools/third_party_notices.py

This script is the one entry point: the file is reproducible from it. With --rust-only it
regenerates Part 4 alone (and the Rust lines of the UNKNOWN list) inside the existing file,
which needs only cargo and any Python 3.9+, not the engine venv. CI runs that and fails on
`git diff --exit-code THIRD_PARTY_NOTICES.txt` (.github/workflows/tests.yml, job rust).

Sections:
  1. Python distributions: every runtime package pinned in requirements/lock-win-x64-py312.txt,
     plus any distribution whose top-level package appears in a built engine bundle
     (desktop/dist-engine/eluate-engine/_internal) but not in the lock (PyInstaller hooks
     pull in a few). Name, version, licence (License-Expression, License, classifiers) and the
     text of every LICENSE*/LICENCE*/COPYING*/NOTICE* file the distribution installed.
  2. Native libraries inside those wheels (libsndfile, OpenBLAS, LLVM, the MSVC runtime) and
     which licence text above covers each, plus the licence texts of the codecs statically
     linked into soundfile's libsndfile DLL (kept in tools/licenses/, fetched from upstream).
  3. CPython (sys.base_prefix/LICENSE.txt, which also carries the licences of the libraries the
     Windows build of CPython bundles: OpenSSL, libffi, bzip2, xz, expat, zlib and others).
  4. Rust crates in the desktop shell's release graph (normal dependencies of the `eluate`
     crate for x86_64-pc-windows-msvc, from `cargo metadata`): name, version, licence field,
     and the text of every LICENSE*/LICENCE*/COPYING*/NOTICE*/COPYRIGHT* file the published
     crate carries (read from cargo's registry checkout, `manifest_path`), each distinct text
     once with the crates that carry it. A crate that publishes no licence file is covered by
     a text kept in tools/licenses/rust/ (RUST_LICENCE_FILES), else listed as UNKNOWN.

Anything whose licence cannot be determined is listed under "UNKNOWN" rather than dropped.

TODO (release checklist, audit 2026-11-10 desktop S2 #3/#4; desktop/README.md "Release
checklist" has the same list). This script describes the WINDOWS installer only:
  - Per platform: a lock per platform (requirements/lock-<os>-<arch>-py312.txt) and a notices
    file generated from it, with RUST_TARGET set per target (`--filter-platform`), for the
    macOS and Linux packages. Not done: those builds cannot run on the Windows machine this
    is generated on.
  - AppImage: credit the system libraries linuxdeploy bundles (GStreamer, WebKitGTK and
    their dependencies, LGPL-2.1+ among others), e.g. from the AppDir's usr/lib and dpkg's
    copyright files.
  - Tag the exact commit each platform's packages were built from, so the AGPL
    Corresponding Source (build scripts included) is at the tag the release names.
  - LGPL source offer: attach the libsndfile 1.2.2, mpg123 1.32.9 and LAME 3.100 source
    tarballs (and the vcpkg port versions used) to the GitHub release, and say in Part 2
    where they are, rather than relying on upstream URLs only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "requirements" / "lock-win-x64-py312.txt"
BUNDLE = ROOT / "desktop" / "dist-engine" / "eluate-engine" / "_internal"
TAURI = ROOT / "desktop" / "src-tauri"
OUT = ROOT / "THIRD_PARTY_NOTICES.txt"
LICENCES = Path(__file__).resolve().parent / "licenses"
RUST_TARGET = "x86_64-pc-windows-msvc"

RULE = "=" * 78
THIN = "-" * 78
LICENCE_NAME = re.compile(r"(LICEN[CS]E|COPYING|NOTICE)", re.IGNORECASE)
CODE_SUFFIXES = {".py", ".pyc", ".pyi", ".pyd", ".dll", ".so", ".h", ".c"}
RUST_LICENCE_NAME = re.compile(r"(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT)", re.IGNORECASE)
RUST_CODE_SUFFIXES = {".rs", ".toml", ".json", ".yml", ".yaml", ".sh", ".py", ".c", ".h"}
# Crates that publish no licence file: the text that covers each, in tools/licenses/rust/
# (each file names its source).
RUST_LICENCE_FILES = {
    "alloc-stdlib": "alloc-stdlib-LICENSE.txt",
    "selectors": "selectors-MPL-2.0.txt",
    "webview2-com": "webview2-rs-LICENSE.txt",
    "webview2-com-macros": "webview2-rs-LICENSE.txt",
    "webview2-com-sys": "webview2-rs-LICENSE.txt",
}
PART4_TITLE = "PART 4. RUST CRATES IN THE DESKTOP SHELL (desktop/src-tauri)"
RUST_UNKNOWN = "Rust crate"


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_names() -> list[str]:
    names = []
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(re.split(r"[=<>!~;\[ ]", line, maxsplit=1)[0])
    return names


def bundle_extras(locked: set[str]) -> dict[str, str]:
    """Distributions present in a built engine bundle but not in the runtime lock."""
    if not BUNDLE.is_dir():
        return {}
    top_to_dist = md.packages_distributions()
    found: dict[str, str] = {}
    for entry in BUNDLE.iterdir():
        if entry.name.endswith(".dist-info"):
            dist = entry.name[: -len(".dist-info")].rsplit("-", 1)[0]
            found.setdefault(norm(dist), f"{entry.name} in the built engine bundle")
        elif entry.is_dir():
            for dist in top_to_dist.get(entry.name, []):
                found.setdefault(norm(dist), f"package {entry.name}/ in the built engine bundle")
    own = {"eluate"}
    return {d: why for d, why in sorted(found.items()) if d not in locked and d not in own}


def licence_of(dist: md.Distribution) -> str:
    meta = dist.metadata
    parts = []
    expr = (meta.get("License-Expression") or "").strip()
    if expr:
        parts.append(expr)
    field = (meta.get("License") or "").strip()
    if field and field.upper() != "UNKNOWN":
        # Some projects paste the whole licence text into License; keep the first line.
        first = field.splitlines()[0].strip()
        if len(field.splitlines()) > 1 or len(first) > 80:
            first = "licence text in the License field, reproduced in the files below"
        if first not in parts:
            parts.append(first)
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    for c in classifiers:
        if c not in parts:
            parts.append(c)
    return "; ".join(parts) if parts else "UNKNOWN"


def licence_files(dist: md.Distribution) -> list[tuple[str, str]]:
    out = []
    for f in dist.files or []:
        p = Path(str(f))
        if not LICENCE_NAME.search(p.name) or p.suffix.lower() in CODE_SUFFIXES:
            continue
        try:
            text = Path(dist.locate_file(f)).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text.strip():
            out.append((str(f).replace("\\", "/"), text.strip()))
    return sorted(out)


def python_section(unknown: list[str]) -> list[str]:
    locked = lock_names()
    extras = bundle_extras({norm(n) for n in locked})
    lines = [RULE, "PART 1. PYTHON PACKAGES IN THE ENGINE (desktop/dist-engine)", RULE, ""]
    lines.append("Runtime packages pinned in requirements/lock-win-x64-py312.txt"
                 + (", plus packages found in the built engine bundle that the lock does not"
                    " name (marked below)." if extras else "."))
    lines.append("")
    entries = [(n, None) for n in locked] + [(n, why) for n, why in extras.items()]
    index, bodies = [], []
    for name, why in sorted(entries, key=lambda e: norm(e[0])):
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            unknown.append(f"Python package {name}: not installed in this environment")
            index.append(f"  {name:<22} (not installed; licence UNKNOWN)")
            continue
        lic = licence_of(dist)
        files = licence_files(dist)
        real = dist.metadata["Name"]
        if lic == "UNKNOWN":
            unknown.append(f"Python package {real} {dist.version}: no licence metadata")
        if not files:
            unknown.append(f"Python package {real} {dist.version}: no licence file installed")
        tag = f"  [{why}]" if why else ""
        index.append(f"  {real:<22} {dist.version:<12} {lic}{tag}")
        body = [THIN, f"{real} {dist.version}", f"Licence: {lic}"]
        if why:
            body.append(f"Included because: {why}")
        home = dist.metadata.get("Home-page") or next(
            (u.split(",", 1)[1].strip() for u in dist.metadata.get_all("Project-URL") or []
             if u.split(",", 1)[0].strip().lower() in {"source", "source code", "homepage",
                                                        "repository", "home"}), "")
        if home:
            body.append(f"Project: {home}")
        body.append(THIN)
        if files:
            for path, text in files:
                body += ["", f"--- {path} ---", "", text]
        else:
            body += ["", "(no licence file in the installed distribution)"]
        bodies += body + [""]
    return lines + ["Index:"] + index + [""] + bodies


# soundfile 0.14.0's _soundfile_data/libsndfile_x64.dll is bastibe/libsndfile-binaries commit
# 1ac5f94 (2024-12-30, same SHA-256), built by vcpkg (libsndfile 1.2.2, default features
# external-libs + mpeg) with every dependency static. Versions from the DLL's strings (libVorbis
# 1.3.7, FLAC 1.4.3, LAME 3.100, opus v1.5.2 buildtree), the vcpkg ports at that date (libogg
# 1.3.5) and the DLL's mpg123 buildtree hash, which matches vcpkg's mpg123 1.32.9 port-version 1.
# A different DLL means the list must be re-checked, so the hash is pinned.
SNDFILE_DLL_SHA256 = "22518c16f9d13eda5ae5adf999c4740d7d16cc2d6b178b36dd4ac2f759a38559"
SNDFILE_CODECS = [  # (name, version, licence, files in tools/licenses)
    ("libogg", "1.3.5", "BSD-3-Clause", ["libogg-1.3.5.txt"]),
    ("libvorbis", "1.3.7", "BSD-3-Clause", ["libvorbis-1.3.7.txt"]),
    ("libFLAC", "1.4.3", "BSD-3-Clause (COPYING.Xiph; FLAC's GPL/LGPL/FDL texts cover the"
     " command-line tools and docs, which are not shipped)", ["flac-1.4.3.txt"]),
    ("libopus", "1.5.2", "BSD-3-Clause, with royalty-free patent licences", ["opus-1.5.2.txt"]),
    ("libmpg123", "1.32.9", "LGPL-2.1 (COPYING also carries the GPL-2 text for the mpg123"
     " program, which is not shipped)", ["mpg123-1.32.9.txt"]),
    ("libmp3lame (LAME)", "3.100", "LGPL-2.0-or-later (Library GPL version 2)",
     ["lame-3.100.txt", "lame-3.100-LICENSE.txt"]),
]


def sndfile_codecs(dll: Path, unknown: list[str]) -> list[str]:
    """The codec libraries linked into libsndfile, with their licence texts."""
    digest = hashlib.sha256(dll.read_bytes()).hexdigest()
    if digest != SNDFILE_DLL_SHA256:
        unknown.append(f"libsndfile's statically linked codecs: {dll.name} has changed"
                       f" (SHA-256 {digest[:16]}...), re-check the codec list and versions")
    lines = [THIN, f"Codec libraries statically linked into {dll.name}", THIN, "",
             "libsndfile 1.2.2 itself: LGPL-2.1-or-later, text in soundfile's"
             " _soundfile_data/COPYING in Part 1. The codecs below are linked into the same DLL"
             " (built with vcpkg by https://github.com/bastibe/libsndfile-binaries; the DLL can"
             " be replaced). Sources: https://github.com/xiph/ogg, https://github.com/xiph/vorbis,"
             " https://github.com/xiph/flac, https://github.com/xiph/opus,"
             " https://www.mpg123.de, https://lame.sourceforge.io.", ""]
    lines += [f"  {n:<20} {v:<8} {lic}" for n, v, lic, _ in SNDFILE_CODECS] + [""]
    for name, version, _, files in SNDFILE_CODECS:
        for f in files:
            path = LICENCES / f
            if not path.is_file():
                unknown.append(f"{name} {version} (in libsndfile): tools/licenses/{f} missing")
                continue
            lines += [f"--- {name} {version}: tools/licenses/{f} ---", "",
                      path.read_text(encoding="utf-8").strip(), ""]
    return lines


def native_section(unknown: list[str]) -> list[str]:
    site = Path(md.distribution("numpy").locate_file(""))
    lines = [RULE, "PART 2. NATIVE LIBRARIES INSIDE THOSE PACKAGES", RULE, ""]
    known = [
        ("_soundfile_data", "libsndfile*.dll",
         "libsndfile (LGPL-2.1-or-later), shipped by soundfile, with the codec libraries"
         " linked into it (libogg, libvorbis, FLAC, Opus: BSD-3-Clause; mpg123, LAME: LGPL; build"
         " recipe at https://github.com/bastibe/libsndfile-binaries). Licence text: soundfile's"
         " _soundfile_data/COPYING in Part 1 (LGPL-2.1); the codecs' texts are at the end of"
         " this part. It is a separate DLL that can be replaced; source:"
         " https://github.com/libsndfile/libsndfile"),
        ("numpy.libs", "*openblas*.dll",
         "OpenBLAS (BSD-3-Clause) with LAPACK (BSD-3-Clause) and the GCC/gfortran runtime"
         " (GPL-3.0 with the GCC Runtime Library Exception), shipped by numpy. Licence texts:"
         " numpy's LICENSE.txt in Part 1"),
        ("scipy.libs", "*openblas*.dll",
         "OpenBLAS, as above, shipped by scipy. Licence texts: scipy's LICENSE.txt in Part 1"),
        ("llvmlite/binding", "llvmlite.dll",
         "LLVM (Apache-2.0 WITH LLVM-exception), statically linked into llvmlite.dll. Licence"
         " texts: llvmlite's LICENSE.thirdparty in Part 1"),
        ("soxr", "*.pyd",
         "libsoxr (LGPL-2.1-or-later) and PFFFT (BSD-like), compiled into soxr. Licence texts:"
         " soxr's licences in Part 1; source: https://github.com/dofuuz/python-soxr"),
    ]
    codecs: list[str] = []
    for rel, pattern, what in known:
        hits = sorted((site / rel).glob(pattern)) if (site / rel).is_dir() else []
        if hits:
            names = ", ".join(f"{rel}/{h.name}" for h in hits)
            lines += [f"* {names}", f"  {what}", ""]
            if rel == "_soundfile_data":
                for h in hits:
                    codecs += sndfile_codecs(h, unknown)
        else:
            unknown.append(f"native library {rel}/{pattern}: not found to check")
    vc = sorted({p.name.lower() for d in ("numpy.libs", "scipy.libs", "llvmlite.libs")
                 for p in (site / d).glob("msvcp140*.dll")} if site.is_dir() else set())
    lines += ["* Microsoft Visual C++ runtime (msvcp140, vcruntime140, vcomp140 and similar,"
              " shipped inside the wheels above and by PyInstaller): Microsoft Visual C++"
              " Redistributable, distributed under Microsoft's redistribution terms for those"
              " files." + (f" Found: {', '.join(vc)}." if vc else ""), ""]
    return lines + codecs


def cpython_section(unknown: list[str]) -> list[str]:
    lines = [RULE, f"PART 3. CPYTHON {sys.version.split()[0]} (the engine's interpreter)", RULE, ""]
    lic = Path(sys.base_prefix) / "LICENSE.txt"
    if not lic.is_file():
        lic = Path(sys.base_prefix) / "LICENSE"
    if lic.is_file():
        lines += [f"From {lic.name} of the CPython installation. It includes the licences of the"
                  " libraries CPython's Windows build ships (OpenSSL, libffi, bzip2, xz,"
                  " expat, zlib and others).", "",
                  lic.read_text(encoding="utf-8", errors="replace").strip(), ""]
    else:
        unknown.append("CPython: LICENSE.txt not found under sys.base_prefix")
        lines += ["CPython licence file not found in this installation: see"
                  " https://docs.python.org/3/license.html", ""]
    return lines


def crate_licence_files(pkg: dict) -> list[tuple[str, str]]:
    """(label, text) of every licence file a published crate carries: files at the crate
    root, and files under a root folder named like one (LICENSES/)."""
    root = Path(pkg["manifest_path"]).parent
    found: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if not RUST_LICENCE_NAME.search(entry.name):
            continue
        if entry.is_file() and entry.suffix.lower() not in RUST_CODE_SUFFIXES:
            found.append(entry)
        elif entry.is_dir():
            found += sorted((f for f in entry.rglob("*") if f.is_file()),
                            key=lambda f: f.relative_to(root).as_posix())
    out = []
    for f in found:
        text = f.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").strip()
        if text:
            out.append((f.relative_to(root).as_posix(), text))
    return out


def rust_section(unknown: list[str]) -> list[str]:
    lines = [RULE, PART4_TITLE, RULE, ""]
    try:
        raw = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--locked",
             "--filter-platform", RUST_TARGET],
            cwd=TAURI, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        unknown.append(f"{RUST_UNKNOWN}s: cargo metadata failed ({e})")
        return lines + ["cargo metadata failed; Rust crates not listed.", ""]
    meta = json.loads(raw)
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen, stack = set(), [root]
    while stack:  # normal (shipped) dependencies only: no build or dev dependencies
        node = nodes[stack.pop()]
        for dep in node["deps"]:
            if any(k["kind"] is None for k in dep["dep_kinds"]) and dep["pkg"] not in seen:
                seen.add(dep["pkg"])
                stack.append(dep["pkg"])
    crates = sorted((pkgs[i] for i in seen), key=lambda p: (p["name"], p["version"]))
    mpl = [p["name"] for p in crates if "MPL" in (p.get("license") or "")]
    lines += [
        f"{len(crates)} crates: the normal dependencies of the `eluate` crate for {RUST_TARGET},"
        " from `cargo metadata` (build and dev dependencies are not shipped and are left out),"
        " statically linked into eluate.exe, unmodified. Index: name, version, licence (the"
        " crate's licence field) and the licence texts below that the crate's own files carry."
        " A dual-licensed crate (\"MIT OR Apache-2.0\") is used under either; all its texts are"
        " reproduced."
        + (f" {len(mpl)} crates are under MPL-2.0 (file-level copyleft: {', '.join(mpl)});"
           " their source is the published crate on https://crates.io at the version listed."
           if mpl else ""),
        "The installer also downloads Microsoft's WebView2 bootstrapper at install time (not"
        " bundled), under Microsoft's terms.", "",
    ]
    # Each distinct text once (whitespace ignored when comparing), numbered in order of first
    # use, with the crates (and file names) that carry it.
    texts: dict[str, tuple[str, str, list[str]]] = {}
    index = []
    for p in crates:
        name, version = p["name"], p["version"]
        lic = p.get("license") or ""
        if not lic and p.get("license_file"):
            lic = f"see {p['license_file']}"
        if not lic:
            lic = "UNKNOWN"
            unknown.append(f"{RUST_UNKNOWN} {name} {version}: no license field")
        files = crate_licence_files(p)
        if not files and name in RUST_LICENCE_FILES:
            f = LICENCES / "rust" / RUST_LICENCE_FILES[name]
            if f.is_file():
                files = [(f"tools/licenses/rust/{f.name}, as the crate publishes no licence file",
                          f.read_text(encoding="utf-8").replace("\r\n", "\n").strip())]
        if not files:
            unknown.append(f"{RUST_UNKNOWN} {name} {version}: no licence file in the published"
                           " crate (add one to tools/licenses/rust/ and RUST_LICENCE_FILES)")
        ids = []
        for label, text in files:
            key = " ".join(text.split())
            if key not in texts:
                texts[key] = (f"T{len(texts) + 1}", text, [])
            tid, _, users = texts[key]
            users.append(f"{name} {version} ({label})")
            if tid not in ids:
                ids.append(tid)
        refs = f"  [{', '.join(ids)}]" if ids else "  [no licence text]"
        index.append(f"  {name:<36} {version:<14} {lic}{refs}")
    lines += ["Index:"] + index + ["", f"Licence texts ({len(texts)} distinct):", ""]
    for tid, text, users in texts.values():
        users_line = textwrap.fill(f"{tid}, carried by: " + "; ".join(users), width=78,
                                   subsequent_indent="  ", break_on_hyphens=False)
        lines += [THIN, users_line, THIN, "", text, ""]
    return lines


UNKNOWN_PREFIX = "  UNKNOWN: "
PART1_START = f"{RULE}\nPART 1. "


def head_lines(unknown: list[str]) -> list[str]:
    head = [
        "THIRD-PARTY NOTICES",
        "",
        "Eluate is licensed under AGPL-3.0-or-later (LICENSE). The Windows installer also"
        " ships the third-party software listed below, each under its own licence.",
        "Generated by tools/third_party_notices.py; do not edit by hand.",
        "",
        "Licences that could not be determined (listed, not dropped):",
    ]
    return head + ([f"{UNKNOWN_PREFIX}{u}" for u in unknown] or ["  none"])


def rust_only() -> str | None:
    """The existing file with Part 4 (and the Rust UNKNOWN lines) regenerated, or None if
    the file does not have the expected layout."""
    old = OUT.read_text(encoding="utf-8").replace("\r\n", "\n")
    body_at, part4_at = old.find(PART1_START), old.find(f"{RULE}\n{PART4_TITLE}\n")
    if body_at < 0 or part4_at < body_at:
        return None
    kept = [line[len(UNKNOWN_PREFIX):] for line in old[:body_at].splitlines()
            if line.startswith(UNKNOWN_PREFIX)
            and not line.startswith(f"{UNKNOWN_PREFIX}{RUST_UNKNOWN}")]
    rust_unknown: list[str] = []
    part4 = rust_section(rust_unknown)
    text = ("\n".join(head_lines(kept + rust_unknown)) + "\n\n" + old[body_at:part4_at]
            + "\n".join(part4))
    return text.rstrip() + "\n"


def main(argv: list[str]) -> int:
    if argv not in ([], ["--rust-only"]):
        print("usage: third_party_notices.py [--rust-only]", file=sys.stderr)
        return 2
    if argv:
        text = rust_only()
        if text is None:
            print(f"{OUT.name}: Part 1 or Part 4 not found; run without --rust-only",
                  file=sys.stderr)
            return 1
        unknown = [line[len(UNKNOWN_PREFIX):] for line in text.splitlines()
                   if line.startswith(UNKNOWN_PREFIX)]
    else:
        unknown = []
        body = (python_section(unknown) + native_section(unknown) + cpython_section(unknown)
                + rust_section(unknown))
        text = "\n".join(head_lines(unknown) + [""] + body).rstrip() + "\n"
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(text.encode('utf-8')) // 1024} KiB,"
          f" {len(unknown)} unknown")
    for u in unknown:
        print(f"  UNKNOWN: {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
