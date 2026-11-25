"""Emits one closure of functions as one C file, and compiles it.

Lifting a function on its own was the wrong unit. Allowing `call` took the
accepted set from 121 to 4,713 in this image — because a call between two
lifted functions needs nothing: both take the same register file, so

    f_<target>(r, stk);

*is* the call. No marshalling, no ABI layer, no arguments to agree on.

What a closure still needs is a decision for the targets it does not have:
a call to an import, to a function with AVX in it, or to something indirect.
Those become `missing_<rva>`, which records the address and returns, so the
build links and the gap is visible at run time instead of silently wrong.

    python tools/closure.py --root 0xa35d40 --out out/closure.c
    python tools/closure.py --root 0xa35d40 --out out/closure.c --compile
"""

import argparse
import os
import struct
import subprocess
import sys

import pefile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import branch  # noqa: E402
import lift  # noqa: E402


def table(pe):
    """rva -> bytes, from `.pdata`; the entry point and call targets fill gaps."""
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    raw = pe.get_data(entry.VirtualAddress, entry.Size)
    out = {}
    for at in range(0, len(raw) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", raw, at)
        if begin and end > begin:
            out[begin] = pe.get_data(begin, end - begin)
    return out


def build(root, code_of, max_functions=4000, max_bytes=2048):
    """Breadth-first over call edges, lifting what it reaches."""
    lifted = {}
    missing = set()
    queue = [root]
    while queue and len(lifted) < max_functions:
        rva = queue.pop(0)
        if rva in lifted or rva in missing:
            continue
        code = code_of(rva)
        if code is None or len(code) > max_bytes:
            missing.add(rva)
            continue
        try:
            lines = branch.lift_function(code, rva)
        except (branch.Refused, lift.Refused):
            missing.add(rva)
            continue
        lifted[rva] = lines
        for callee in sorted(branch.CALLS.get(rva, ())):
            if callee not in lifted and callee not in missing:
                queue.append(callee)
    return lifted, missing


def emit(lifted, missing, path):
    out = [
        "#include <stdint.h>",
        "static long missing_hits;",
        "static void miss(uint64_t *r, uint8_t *stk, uint64_t at) { (void)r; (void)stk; missing_hits++; }",
    ]
    for rva in sorted(missing):
        out.append(f"static void f_{rva:#x}(uint64_t *r, uint8_t *stk) {{ miss(r, stk, {rva:#x}); }}")
    for rva in sorted(lifted):
        out.append(f"static void f_{rva:#x}(uint64_t *r, uint8_t *stk);")
    for rva in sorted(lifted):
        out.append(f"static void f_{rva:#x}(uint64_t *r, uint8_t *stk) {{")
        out.extend("    " + line for line in lifted[rva])
        out.append("}")
    out.append("long closure_missing_hits(void) { return missing_hits; }")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")


def compile_c(path, work):
    cl = lift.find_cl()
    if not cl:
        return False, "no cl.exe"
    env = dict(os.environ, INCLUDE=";".join(lift.include_dirs()), LIB=";".join(lift.lib_dirs()))
    r = subprocess.run([cl, "/nologo", "/O1", "/c", f"/Fo{work}\\", path],
                       capture_output=True, text=True, encoding="mbcs", errors="replace", env=env)
    if r.returncode != 0:
        return False, ((r.stdout or "") + (r.stderr or ""))[-2500:]
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
    ap.add_argument("--root", type=lambda v: int(v, 0), required=True)
    ap.add_argument("--out", default="out/closure.c")
    ap.add_argument("--max-functions", type=int, default=4000)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    pe = pefile.PE(args.exe, fast_load=True)
    funcs = table(pe)

    def code_of(rva):
        return funcs.get(rva)

    lifted, missing = build(args.root, code_of, args.max_functions)
    print(f"closure from {args.root:#x}: {len(lifted):,} lifted, {len(missing):,} missing")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    emit(lifted, missing, args.out)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out} ({size/2**20:.1f} MB)")
    if args.compile:
        ok, why = compile_c(args.out, os.path.dirname(args.out) or ".")
        print("compiles" if ok else "compile failed:\n" + why)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
