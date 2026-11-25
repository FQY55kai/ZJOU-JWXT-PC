"""Differential test against the hardware, not against ourselves.

`lift.py` compares the generated C with an interpreter written from the same
understanding, which catches typos and nothing else. This runs the *original
bytes* — same ISA, same ABI, so the original is the oracle — and compares the C
against it. A function that passes here is the same function, for the inputs
tried.

Two things make it cheap, and both of them are the reason a port like this can
be checked at all:

**Guest addresses are host addresses.** The image is mapped at its preferred
base on both sides, the arena is mapped at a fixed address on both sides, and
every register but `rsp` points into the arena on entry. So `mov [rcx], rax` is
a store where rcx points — the same store on both sides — and the comparison is
a page of memory, not a scratch buffer every address had to be folded into.

**A call to code that is not lifted yet still runs the original.** 99.5% of the
functions this lifter accepts contain a `call`, so a harness that can only enter
leaves can only see 0.5% of them. Those call sites go through `guest_bridge`,
which is the same stub the oracle itself uses, so the function under test is
lifted C and everything below it is the game's own code — on both sides. One
slot of memory differs there (the pushed return address: a host address in the
bridge, an image address in a guest `call`) and the generated C reports exactly
which slot that was, so the comparison can exclude it instead of guessing.

    python tools/diff.py --limit 400 --branches
"""

import argparse
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import branch  # noqa: E402
import capstone  # noqa: E402
import lift  # noqa: E402
import pefile  # noqa: E402

ARENA = lift.ARENA
ARENA_SIZE = lift.ARENA_SIZE
RSP0 = lift.RSP0

SLOTS = 32  # bridged call sites a function may report; more is clamped
RECORD_IN = 8 + 16 * 8 + ARENA_SIZE
RECORD_OUT = 16 * 8 + ARENA_SIZE + 4 + SLOTS * 8 + 4
CHUNK = 64  # cases per oracle run: a fault is bisected, not paid for


balanced = lift.balanced
reachable = lift.reachable


def pick(exe, limit, max_bytes, branches, skip=0):
    """Functions the lifter accepts, in `.pdata` order, from entry `skip` on."""
    pe = pefile.PE(exe, fast_load=True)
    branch.set_image(pe)
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    table = pe.get_data(entry.VirtualAddress, entry.Size)
    picked = []
    for at in range(12 * skip, len(table) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", table, at)
        if begin == 0 or not 0 < end - begin <= max_bytes:
            continue
        code = pe.get_data(begin, end - begin)
        try:
            lines = branch.lift_function(code, begin) if branches else lift.lift_function(code, begin)
        except (branch.Refused, lift.Refused):
            continue
        if lines:
            picked.append((begin, code, lines))
        if len(picked) >= limit:
            break
    return pe, picked


def externals(kept):
    """Call targets that are not themselves lifted here: the bridge's job."""
    inside = {rva for rva, _, _ in kept}
    out = set()
    for rva, _, _ in kept:
        out.update(t for t in branch.CALLS.get(rva, ()) if t not in inside)
    return out


def emit(kept, path, pe):
    """The lifted functions, a bridge for everything else, and a driver."""
    outer = externals(kept)
    names = sorted({rva for rva, _, _ in kept} | outer)
    lines = ["#include <stdint.h>", "#include <stdio.h>", "#include <string.h>"]
    lines += lift.image_loader(pe)
    lines += [
        "#include <windows.h>",
        f"#define ARENA_ADDR 0x{ARENA:x}ull",
        f"#define ARENA_SIZE 0x{ARENA_SIZE:x}u",
        f"#define SLOTS {SLOTS}",
        "",
        "// The one slot a bridged call leaves different: the return address it",
        "// pushed, a host address here and an image address in a guest `call`.",
        "extern void guest_bridge(uint64_t *r, const void *target);",
        "static uint64_t slots[SLOTS];",
        "static uint32_t nslots;",
        "static void note(uint64_t at) { if (nslots < SLOTS) slots[nslots++] = at; }",
        "",
    ]
    for rva in names:
        lines.append(f"void f_{rva:#x}(uint64_t *r);")
        lines.append(f"void j_{rva:#x}(uint64_t *r);")
    lines.append("")
    for rva in sorted(outer):
        # The lift pushed a return address and lowered `rsp` before calling;
        # undo that here, because the bridge's own `call` is going to push one
        # and lower `rsp` itself. It pushes a host address where the original
        # pushed an image address, so that slot is reported and skipped.
        lines += [
            f"void f_{rva:#x}(uint64_t *r) {{",
            "    r[4] += 8;",
            "    note(r[4] - 8);",
            f"    guest_bridge(r, img + {rva:#x});",
            "    r[4] -= 8;",
            "}",
            f"void j_{rva:#x}(uint64_t *r) {{",
            "    note(r[4] - 8);",
            f"    guest_bridge(r, img + {rva:#x});",
            "}",
        ]
    for rva, _, body in kept:
        lines.append(f"void f_{rva:#x}(uint64_t *r) {{")
        lines.extend("    " + line for line in body)
        lines.append("}")
    # A tail jump into a lifted function is just a call to it: `rsp` already
    # holds the caller's return address, which is what the callee's `ret` needs.
    for rva, _, _ in kept:
        if rva not in outer:
            lines += [f"void j_{rva:#x}(uint64_t *r) {{ f_{rva:#x}(r); }}"]
    lines += [
        "static uint8_t *arena;",
        "static LONG CALLBACK on_fault(EXCEPTION_POINTERS *ep) {",
        "    CONTEXT *c = ep->ContextRecord;",
        "    fprintf(stderr, \"fault code=%lx at=%p rip=%p rsp=%p rax=%p rcx=%p rdx=%p\",",
        "            (unsigned long)ep->ExceptionRecord->ExceptionCode,",
        "            ep->ExceptionRecord->ExceptionAddress,",
        "            (void *)c->Rip, (void *)c->Rsp, (void *)c->Rax, (void *)c->Rcx, (void *)c->Rdx);",
        "    fputc(10, stderr);",
        "    return EXCEPTION_CONTINUE_SEARCH;",
        "}",
        "int main(int argc, char **argv) {",
        "    FILE *in = fopen(argv[1], \"rb\"), *out = fopen(argv[2], \"wb\");",
        "    if (!in || !out) return 1;",
        "    AddVectoredExceptionHandler(1, on_fault);",
        "    arena = (uint8_t *)VirtualAlloc((void *)(uintptr_t)ARENA_ADDR, ARENA_SIZE,",
        "                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);",
        "    if (!arena) return 5;",
        "    if (argc < 4 || !map_image(argv[3])) return 4;",
        "    uint32_t head[2];",
        "    while (fread(head, sizeof head, 1, in) == 1) {",
        "        uint64_t r[16];",
        "        if (fread(r, sizeof r, 1, in) != 1) return 2;",
        "        if (fread(arena, ARENA_SIZE, 1, in) != 1) return 3;",
        "        nslots = 0;",
        "        uint32_t ran = 0;",
        "        __try {",
        "        switch (head[0]) {",
    ]
    lines += [f"        case {rva:#x}: f_{rva:#x}(r); break;" for rva, _, _ in kept]
    lines += [
        "        default: break;",
        "        }",
        "        ran = 1;",
        "        } __except (EXCEPTION_EXECUTE_HANDLER) { ran = 0; }",
        "        fwrite(r, sizeof r, 1, out);",
        "        fwrite(arena, ARENA_SIZE, 1, out);",
        "        fwrite(&nslots, sizeof nslots, 1, out);",
        "        fwrite(slots, sizeof slots, 1, out);",
        "        fwrite(&ran, sizeof ran, 1, out);",
        "    }",
        "    return 0;",
        "}",
    ]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def isolate(launch, read, items, work, tag, limit):
    """Run a batch, and on a fault or a hang, bisect until the case is alone.

    A case that dereferences a register the record did not prepare takes the
    whole process down with it, and nothing in the process can say which one it
    was — the guest switches stacks, so the handler cannot even unwind to the
    loop. Asking again with fewer cases can: log2(batch) runs, one identified
    case, and the rest of the batch still checked.
    """
    if not items:
        return []
    cin = os.path.join(work, tag + ".in")
    cout = os.path.join(work, tag + ".out")
    with open(cin, "wb") as f:
        for rva, length, regs, arena in items:
            f.write(struct.pack("<II", rva, length))
            f.write(struct.pack("<16Q", *regs))
            f.write(arena)
    try:
        failed = subprocess.run(launch(cin, cout), capture_output=True, cwd=work,
                                timeout=limit).returncode != 0
    except subprocess.TimeoutExpired:
        failed = True
    if not failed:
        return read(cout, len(items))
    if len(items) == 1:
        return [None]
    half = len(items) // 2
    return isolate(launch, read, items[:half], work, tag + "a", limit) +         isolate(launch, read, items[half:], work, tag + "b", limit)


def read_hardware(path, count):
    """Registers then arena, per case: what the original bytes did."""
    with open(path, "rb") as f:
        data = f.read()
    got = []
    for i in range(count):
        base = i * (128 + ARENA_SIZE)
        got.append((struct.unpack_from("<16Q", data, base), data[base + 128:base + 128 + ARENA_SIZE]))
    return got


def read_port(path, count):
    """The same, plus the slots a bridged call had to excuse."""
    with open(path, "rb") as f:
        data = f.read()
    got = []
    for i in range(count):
        base = i * RECORD_OUT
        count_slots = struct.unpack_from("<I", data, base + 128 + ARENA_SIZE)[0]
        got.append({
            "regs": struct.unpack_from("<16Q", data, base),
            "arena": data[base + 128:base + 128 + ARENA_SIZE],
            "slots": struct.unpack_from(f"<{SLOTS}Q", data, base + 128 + ARENA_SIZE + 4),
            "nslots": count_slots,
            "ran": struct.unpack_from("<I", data, base + 128 + ARENA_SIZE + 4 + SLOTS * 8)[0],
        })
    return got


def self_referential(rng):
    """An arena whose every word is a pointer into itself.

    Random bytes are useless as an argument: the first `mov rcx, [rsi+0x18]`
    loads a wild pointer and the second instruction faults, on both sides, and
    the case says nothing. A word that is *also* a small integer gives the code
    somewhere to go — a chain of dereferences stays inside the arena instead of
    falling off it — and the low 32 bits, which is what a size or an index
    usually reads, stay small. Both sides see the same bytes, so a value that is
    nonsense is at least the same nonsense.
    """
    words = [ARENA + rng.randrange(0, 0x1000, 8) for _ in range(ARENA_SIZE // 8)]
    return struct.pack(f"<{len(words)}Q", *words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-bytes", type=int, default=220)
    ap.add_argument("--only", type=lambda v: int(v, 0), default=None,
                    help="one function, for looking at a fault")
    ap.add_argument("--skip", type=int, default=0, help="start this many .pdata entries in")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120,
                    help="seconds a case batch may run before it counts as hung")
    ap.add_argument("--oracle", default=r"D:\zjou-build\release\oracle.exe")
    ap.add_argument("--bridge", default=r"D:\zjou-build\release\oracle_guest.dll",
                    help="the DLL that exports guest_bridge")
    ap.add_argument("--branches", action="store_true", help="use the branch pass")
    ap.add_argument("--keep", help="write the generated C here, for reading")
    args = ap.parse_args()

    pe, picked = pick(args.exe, args.limit, args.max_bytes, args.branches, args.skip)
    kept = [(rva, code, lines) for rva, code, lines in picked
            if reachable(code, rva) and balanced(code, rva)]
    if args.only is not None:
        kept = [item for item in kept if item[0] == args.only]
        args.rounds = 1
    bridged = len(externals(kept))
    print(f"lifted {len(picked)} functions, {len(kept)} runnable here, {bridged} bridged callee(s)")

    work = tempfile.mkdtemp(prefix="diff-")
    csrc = os.path.join(work, "lifted.c")
    emit(kept, csrc, pe)
    if args.keep:
        os.makedirs(os.path.dirname(os.path.abspath(args.keep)), exist_ok=True)
        shutil.copyfile(csrc, args.keep)

    cl = lift.find_cl()
    env = dict(os.environ, INCLUDE=";".join(lift.include_dirs()), LIB=";".join(lift.lib_dirs()))
    exe = os.path.join(work, "lifted.exe")
    lib = os.path.splitext(args.bridge)[0] + ".dll.lib"
    # `/BASE` keeps the test program off the image's preferred base (0x1_4000_0000
    # is also the default for an x64 exe), without which the mapping of the guest
    # image fails with ERROR_INVALID_ADDRESS.
    build = subprocess.run([cl, "/nologo", "/O1", "/std:c11", f"/Fo{work}\\", csrc, "/link",
                            f"/OUT:{exe}", lib, "/BASE:0x10000000", "/DYNAMICBASE:NO"],
                           capture_output=True, text=True, encoding="mbcs", errors="replace", env=env)
    if build.returncode != 0:
        print("compile failed:", ((build.stdout or "") + (build.stderr or ""))[-1500:])
        return 1
    shutil.copyfile(args.bridge, os.path.join(work, os.path.basename(args.bridge)))

    # Each round is a fresh register/arena state. Every register but `rsp`
    # points into the arena, which is what a real caller hands a function.
    #
    # A function that faults does so for every input — it is dereferencing a
    # global the title never initialised, not disagreeing about this one — so it
    # is dropped after the first round instead of paying for the bisection again.
    rng = random.Random(0x0DDBA11)
    by_rva = {rva: code for rva, code, _ in kept}
    live = [rva for rva, _, _ in kept]
    faulted_both, faulted_port, records = set(), set(), []
    top = RSP0 - ARENA

    for rnd in range(args.rounds):
        cases = []
        for rva in live:
            regs = [ARENA + 0x1000 + rng.randrange(0, 0x800, 8) for _ in range(16)]
            regs[4] = RSP0
            cases.append((rva, len(by_rva[rva]), regs, self_referential(rng)))
        if not cases:
            break
        expected, port = [], []
        for start in range(0, len(cases), CHUNK):
            tag = f"r{rnd}c{start // CHUNK}"
            slice_ = cases[start:start + CHUNK]
            expected.extend(isolate(lambda i, o: [args.oracle, args.exe, i, o], read_hardware,
                                    slice_, work, "o" + tag, args.timeout))
            port.extend(isolate(lambda i, o: [exe, i, o, args.exe], read_port,
                                slice_, work, "p" + tag, args.timeout))
        for i, case in enumerate(cases):
            records.append((case[0], expected[i], port[i]))
            if expected[i] is None:
                faulted_both.add(case[0])
            elif port[i] is None or not port[i]["ran"]:
                faulted_port.add(case[0])
        live = [rva for rva in live if rva not in faulted_both and rva not in faulted_port]

    bad, checked, shown = 0, 0, 0
    for rva, want, mine in records:
        if want is None or mine is None or not mine["ran"]:
            continue
        # The `call` that starts the run pushed a return address at `rsp - 8`,
        # which is `RSP0` once the guest is running — the lifted C never pushes
        # it. Every bridged call leaves the same kind of hole one frame further
        # down, and the driver reported those.
        skip = {top} | {(s - ARENA) for s in mine["slots"][:mine["nslots"]]
                        if ARENA <= s < ARENA + ARENA_SIZE - 8}
        got_regs, got_arena = mine["regs"], mine["arena"]
        checked += 1
        same_memory = all(got_arena[off:off + 8] == want[1][off:off + 8]
                          for off in range(0, ARENA_SIZE, 8) if off not in skip)
        if list(got_regs) != list(want[0]) or not same_memory:
            bad += 1
            if shown < 6:
                shown += 1
                clash = [f"{lift.REGS[k]}: hw {want[0][k]:#x} vs c {got_regs[k]:#x}"
                         for k in range(16) if want[0][k] != got_regs[k]]
                if not clash:
                    where = [f"+{off:#x} hw={want[1][off:off + 8][::-1].hex()} c={got_arena[off:off + 8][::-1].hex()}"
                             for off in range(0, ARENA_SIZE, 8)
                             if off not in skip and got_arena[off:off + 8] != want[1][off:off + 8]][:4]
                    clash = ["arena at " + ", ".join(where) if where else "arena differs"]
                print(f"  {rva:#x}: " + ", ".join(clash))
    print(f"functions: {len(kept)} runnable, {len(kept) - len(faulted_both) - len(faulted_port)} "
          f"compared, {len(faulted_both)} fault in both, {len(faulted_port)} fault only in the port")
    print(f"against hardware: {checked} case(s) compared, {checked - bad} agree, {bad} differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
