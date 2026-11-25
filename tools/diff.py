"""Differential test against the hardware, not against ourselves.

`lift.py` compares the generated C with an interpreter written from the same
understanding, which catches typos and nothing else. This runs the *original
bytes* — same ISA, same ABI, so the original is the oracle — and compares the
C against it. A function that passes here is the same function, for the inputs
tried.

    python tools/diff.py --limit 400
"""

import argparse
import os
import random
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import branch  # noqa: E402
import capstone  # noqa: E402
import lift  # noqa: E402

STACK = 256
SP0 = STACK // 2
CHUNK = 6  # cases per oracle run: a fault costs one chunk, not the run


def balanced(code, rva):
    """True when every `ret` leaves `rsp` where it found it.

    A function whose exit frame differs from its entry one is a fragment of
    another function (a cold block, a funclet): its `ret` reads a return address
    the parent pushed, so calling it on its own pops whatever the stack holds
    and jumps into it. Statically this is just a running `rsp` delta: a
    straight-line body has one at every point.
    """
    delta = 0
    for insn in lift.MD.disasm(code, rva):
        m, ops = insn.mnemonic, insn.operands
        if m == "push":
            delta -= 8
        elif m == "pop":
            delta += 8
        elif m in ("add", "sub") and ops and lift.MD.reg_name(ops[0].reg) == "rsp" if ops and ops[0].type == 1 else False:
            if ops[1].type != 1:  # not an immediate
                return False
            delta += ops[1].imm if m == "add" else -ops[1].imm
        elif m == "lea" and ops[0].type == 1 and lift.MD.reg_name(ops[0].reg) == "rsp":
            if ops[1].type != 3 or ops[1].mem.index or ops[1].mem.base == 0:
                return False
            if lift.MD.reg_name(ops[1].mem.base) != "rsp":
                return False
            delta += ops[1].mem.disp
        elif m in ("leave", "mov") and any(o.type == 1 and lift.MD.reg_name(o.reg) == "rsp" for o in ops):
            return False
        elif m == "ret":
            if delta != 0:
                return False
    return True


def safe(code, rva):
    """True when every memory operand is one this harness can give it.

    The oracle gives the function a real 256-byte stack and lets it run, so an
    access through an arbitrary register would fault. Bases are therefore
    limited to `rsp`/`rbp` (the frame the function builds itself) and offsets to
    one screen of that stack. `[rip + d]` is fine: it is a fixed offset into the
    image, and both sides have the image mapped.
    """
    for insn in lift.MD.disasm(code, rva):
        for op in insn.operands:
            if op.type != capstone.x86.X86_OP_MEM:
                continue
            base = lift.MD.reg_name(op.mem.base) if op.mem.base else ""
            if base == "rip":
                if not 0 <= insn.address + insn.size + op.mem.disp < lift.IMAGE_SIZE:
                    return False
                continue
            if base not in ("rsp", "rbp"):
                return False
            if not -SP0 <= op.mem.disp < STACK - SP0 - 8:
                return False
    return True


def candidates_branch(exe, limit, max_bytes):
    """Functions lifted with the branch pass, in `.pdata` order."""
    import pefile

    pe = pefile.PE(exe, fast_load=True)
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    table = pe.get_data(entry.VirtualAddress, entry.Size)
    picked = []
    for at in range(0, len(table) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", table, at)
        if begin == 0 or not 0 < end - begin <= max_bytes:
            continue
        code = pe.get_data(begin, end - begin)
        try:
            lines = branch.lift_function(code, begin)
        except (branch.Refused, lift.Refused):
            continue
        if lines:
            picked.append((begin, code, lines))
        if len(picked) >= limit:
            break
    return pe, picked


def emit(picked, path, pe=None):
    """The lifted functions plus a driver: reads cases, writes results."""
    out = ["#include <stdint.h>", "#include <stdio.h>", "#include <string.h>"]
    out += lift.image_loader(pe)
    for i, (_, _, lines) in enumerate(picked):
        out.append(f"void f{i}(uint64_t *r, uint8_t *stk) {{")
        out.extend("    " + line for line in lines)
        out.append("}")
    out += [
        "typedef void (*fn)(uint64_t *, uint8_t *);",
        "static const fn table[] = {" + ", ".join(f"f{i}" for i in range(len(picked))) + "};",
        "int main(int argc, char **argv) {",
        "    FILE *in = fopen(argv[1], \"rb\"), *out = fopen(argv[2], \"wb\");",
        "    if (!in || !out) return 1;",
        "    if (argc < 4 || !map_image(argv[3])) return 4;",
        "    size_t n = 0, count = sizeof table / sizeof table[0];",
        "    uint32_t head[2];",
        "    while (fread(head, sizeof head, 1, in) == 1) {",
        "        uint64_t r[16]; uint8_t stk[256];",
        "        if (fread(r, sizeof r, 1, in) != 1) return 2;",
        "        if (fread(stk, sizeof stk, 1, in) != 1) return 3;",
        "        table[n % count](r, stk);",
        "        n++;",
        "        fwrite(r, sizeof r, 1, out);",
        "        fwrite(stk, sizeof stk, 1, out);",
        "    }",
        "    return 0;",
        "}",
    ]
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-bytes", type=int, default=220)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--oracle", default=r"D:\zjou-build\release\oracle.exe")
    ap.add_argument("--branches", action="store_true", help="use the branch pass")
    ap.add_argument("--keep", help="write the generated C here, for reading")
    args = ap.parse_args()

    if args.branches:
        pe, picked = candidates_branch(args.exe, args.limit, args.max_bytes)
        kept = [(rva, code, lines) for rva, code, lines in picked if safe(code, rva) and balanced(code, rva)]
    else:
        pe, picked = lift.candidates(args.exe, args.limit, args.max_bytes)
        kept = [(rva, code, lines) for rva, code, lines in picked if safe(code, rva) and balanced(code, rva)]
    print(f"lifted {len(picked)} functions, {len(kept)} runnable here")

    work = tempfile.mkdtemp(prefix="diff-")
    csrc = os.path.join(work, "lifted.c")
    emit(kept, csrc, pe)
    if args.keep:
        os.makedirs(os.path.dirname(os.path.abspath(args.keep)), exist_ok=True)
        with open(csrc, "rb") as src, open(args.keep, "wb") as dst:
            dst.write(src.read())

    cl = lift.find_cl()
    env = dict(os.environ, INCLUDE=";".join(lift.include_dirs()), LIB=";".join(lift.lib_dirs()))
    exe = os.path.join(work, "lifted.exe")
    build = subprocess.run([cl, "/nologo", "/O1", "/std:c11", f"/Fo{work}\\", csrc, "/link", f"/OUT:{exe}"],
                           capture_output=True, text=True, encoding="mbcs", errors="replace", env=env)
    if build.returncode != 0:
        print("compile failed:", ((build.stdout or "") + (build.stderr or ""))[-1500:])
        return 1

    rng = random.Random(0x0DDBA11)
    # Each round is a fresh register/stack state; the four argument registers
    # get random values, the rest point into the stack so an unlucky access
    # stays inside the buffer.
    cases = []
    for _ in range(args.rounds):
        for rva, code, _ in kept:
            regs = [SP0] * 16
            for slot in (1, 2, 8, 9):
                regs[slot] = rng.getrandbits(64)
            stk = bytes(rng.getrandbits(8) for _ in range(STACK))
            cases.append((rva, len(code), regs, stk))

    def write_cases(items, path):
        with open(path, "wb") as f:
            for rva, length, regs, stk in items:
                f.write(struct.pack("<II", rva, length))
                f.write(struct.pack("<16Q", *regs))
                f.write(stk)

    # The hardware's answers, a chunk at a time: a function that faults takes
    # its chunk down, not the run.
    expected = []
    faulted = set()
    for start in range(0, len(cases), CHUNK):
        chunk = cases[start:start + CHUNK]
        cin, cout = os.path.join(work, "c.bin"), os.path.join(work, "o.bin")
        write_cases(chunk, cin)
        run = subprocess.run([args.oracle, args.exe, cin, cout], capture_output=True)
        if run.returncode != 0:
            print(f"  chunk {start // CHUNK}: oracle faulted ({run.returncode})")
            for rva, *_ in chunk:
                faulted.add(rva)
            expected.extend([None] * len(chunk))
            continue
        data = open(cout, "rb").read()
        for i in range(len(chunk)):
            base = i * (128 + STACK)
            expected.append((struct.unpack_from("<16Q", data, base), data[base + 128: base + 128 + STACK]))

    # Ours.
    cin, cout = os.path.join(work, "all.bin"), os.path.join(work, "ours.bin")
    write_cases(cases, cin)
    run = subprocess.run([exe, cin, cout, args.exe], capture_output=True)
    if run.returncode != 0:
        print(f"lifted driver failed: {run.returncode}")
        return 1
    ours = open(cout, "rb").read()

    bad, checked, skipped = 0, 0, 0
    shown = 0
    for i, (rva, length, regs, stk) in enumerate(cases):
        want = expected[i]
        if want is None or rva in faulted:
            skipped += 1
            continue
        base = i * (128 + STACK)
        got_regs = struct.unpack_from("<16Q", ours, base)
        got_stk = ours[base + 128: base + 128 + STACK]
        checked += 1
        # The `call` that starts the run pushes a return address at `rsp-8`;
        # the lifted C has no such push, so that one slot is not compared.
        cut = slice(SP0 - 8, SP0)
        same_stack = (got_stk[:cut.start] == want[1][:cut.start] and got_stk[cut.stop:] == want[1][cut.stop:])
        if list(got_regs) != list(want[0]) or not same_stack:
            bad += 1
            if shown < 5:
                shown += 1
                diff = [f"{lift.REGS[k]}: hw {want[0][k]:#x} vs c {got_regs[k]:#x}"
                        for k in range(16) if want[0][k] != got_regs[k]]
                if not diff:
                    diff = ["stack differs"]
                print(f"  {rva:#x}: " + ", ".join(diff))
    print(f"against hardware: {checked} case(s) compared, {checked - bad} agree, {bad} differ, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
