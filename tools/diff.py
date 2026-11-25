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
    python tools/diff.py --snippets --rounds 6     # one function per instruction
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
VEC_BYTES = 16 * 32  # sixteen 256-bit vector registers
RECORD_IN = 8 + 16 * 8 + ARENA_SIZE + VEC_BYTES
RECORD_OUT = 16 * 8 + ARENA_SIZE + VEC_BYTES + 4 + SLOTS * 8 + 4
CHUNK = 64  # cases per oracle run: a fault is bisected, not paid for


balanced = lift.balanced
reachable = lift.reachable


def teb_dependent(pe, rva, lifted, depth=3):
    """True when the function, or anything it calls, reads the TEB.

    `mov rax, gs:[0x58]` is the TLS array pointer, and the TEB is *per process*.
    The oracle and the port are two processes, so that pointer names two
    different objects and everything reachable from it — TLS blocks, and the
    allocation state a title keeps in them — is unrelated between the sides.
    Such a function is translated correctly and still cannot agree here, so it
    is counted apart instead of as a disagreement: the harness's boundary, not
    the translation's.

    The walk follows `branch.CALLS`, which the CFG pass recorded for every
    function it lifted, and reads the bytes of the ones it did not.
    """
    seen = set()
    todo = [(rva, 0)]
    while todo:
        at, level = todo.pop()
        if at in seen or level > depth:
            continue
        seen.add(at)
        code = lifted.get(at)
        if code is None:
            code = pe.get_data(at, 0x200)
        for insn in lift.MD.disasm(code, at):
            if any(op.type == capstone.x86.X86_OP_MEM and op.mem.segment
                   for op in insn.operands):
                return True
        for target in branch.CALLS.get(at, ()):
            todo.append((target, level + 1))
    return False


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


# --- instruction snippets ---------------------------------------------------
#
# A whole function is the right unit to prove *a function* right, and the wrong
# unit to prove *an instruction* right: the functions that use `vpsignb` are
# mostly functions that read a global the file leaves zero, so they fault on
# both sides and the instruction is never compared. Sixty per cent of what this
# round added would go unverified that way.
#
# So each instruction gets its own one-instruction function. The bytes are the
# image's own, copied out of a real function, which means no assembler and no
# hand-encoded operands: whatever the compiler chose is what gets tested. They
# are written into a run of padding inside `.text` — in a *copy* of the image,
# so both sides map the same patched file — followed by a `ret` and, when the
# instruction only speaks through the flags, by four `setcc`s that turn them
# into bits.
#
# The trailer writes r10b..r13b, which no snippet's operands name, and
# `flags_for` checks that by disassembling the bytes it just built.
FLAG_TRAILER = {
    "cf": bytes.fromhex("410f92c2"),  # setb  r10b
    "zf": bytes.fromhex("410f94c3"),  # sete  r11b
    "sf": bytes.fromhex("410f98c4"),  # sets  r12b
    "of": bytes.fromhex("410f90c5"),  # seto  r13b
}

# (mnemonic, which flags the result is read through). Empty flags means the
# instruction's result lands in a register or memory and is compared directly.
#
# Only instructions this image actually contains: a name nobody uses costs a
# full rescan of `.pdata` and proves nothing. `ptest`, `vpsignw`, `blsmsk` and
# the `lock`-prefixed atomics are all absent from this title, so they are not
# listed even though the lifter accepts them.
SNIPPETS = [
    # 向量族：这一轮新升举的，加上本来就是多操作数、值得单独试的
    ("vpsignb", ""),
    ("vinsertps", ""), ("vmovshdup", ""), ("vmovsldup", ""), ("vmovddup", ""),
    ("vblendps", ""), ("vpblendd", ""), ("vpblendw", ""),
    ("vcmpeqps", ""), ("vcmpltps", ""), ("vcmplepd", ""),
    ("vpcmpgtb", ""), ("vpcmpgtw", ""),
    ("vshufps", ""), ("vshufpd", ""), ("vunpcklps", ""), ("vpunpckhwd", ""),
    ("vhaddps", ""), ("vhaddpd", ""),
    ("vpmovmskb", ""), ("vmovmskps", ""), ("vmovmskpd", ""),
    ("vpsrad", ""), ("vpslld", ""), ("vpsrlq", ""), ("vpsllw", ""),
    ("vpsubb", ""), ("vpaddw", ""),
    ("vsqrtps", ""), ("vsqrtss", ""), ("vsqrtsd", ""),
    ("vcvttss2si", ""), ("vcvttsd2si", ""),
    # 只通过标志说话的：结果不落在任何寄存器，四个 setcc 就是全部可观测输出
    ("vtestps", ("cf", "zf")), ("vtestpd", ("cf", "zf")),
    # 整数侧：BMI1 位提取与计数、旋转、以及比较本身（czso 会走到这一轮新加的
    # js/jns 与 jo/jno 两条路径）
    ("tzcnt", ""), ("lzcnt", ""), ("blsr", ""), ("blsi", ""),
    ("rol", ""), ("ror", ""),
    ("cmp", ("cf", "zf", "sf", "of")),
]


def snippet_code(flags):
    """The trailer of a snippet: "setcc*; ret" — the instruction bytes come first."""
    trailer = b"".join(FLAG_TRAILER[f] for f in flags)  # one setcc per flag
    return trailer + b"\xc3"


def padding(pe, need):
    """The largest run of one repeated padding byte in `.text` no function owns.

    A snippet has to live somewhere the original code never enters and that no
    unwinder claims, or writing it there changes what another test sees. Gaps
    between `.pdata` ranges are exactly that.
    """
    text = next((s for s in pe.sections if s.Name.rstrip(b"\0") == b".text"), None)
    if text is None:
        raise SystemExit("no .text")
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    raw = pe.get_data(entry.VirtualAddress, entry.Size)
    spans = []
    for at in range(0, len(raw) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", raw, at)
        if begin and end > begin:
            spans.append((begin, end))
    spans.sort()
    best, best_len = None, 0
    at = text.VirtualAddress
    limit = text.VirtualAddress + text.Misc_VirtualSize
    for begin, end in spans + [(limit, limit)]:
        if begin - at > best_len and at >= text.VirtualAddress:
            best, best_len = at, begin - at
        at = max(at, end)
    if best is None or best_len < need:
        raise SystemExit(f"no padding run of {need} bytes in .text (largest {best_len})")
    return best


def snippets(exe, work):
    """One function per instruction, written into a patched copy of the image.

    Returns (path to the patched copy, the listing, the names). The listing is
    the same shape `pick` returns, so everything downstream — `emit`, the
    driver, the comparison — runs unchanged: the only difference is how the
    functions got there.
    """
    pe = pefile.PE(exe, fast_load=True)
    branch.set_image(pe)
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    table = pe.get_data(entry.VirtualAddress, entry.Size)

    # One real instance of each instruction, from anywhere in the image. A
    # rip-relative memory operand is skipped: the displacement belongs to the
    # function the bytes came from, and the snippet lives somewhere else, so it
    # would address an unrelated place — sometimes mapped, sometimes not, and
    # never the same place it meant. What the compiler chose is what gets
    # tested: no assembler, and no hand-picked operands.
    wanted = {name for name, _ in SNIPPETS}
    found = {}
    for at in range(0, len(table) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", table, at)
        if not begin or end <= begin or end - begin > 4096:
            continue
        for insn in lift.MD.disasm(pe.get_data(begin, end - begin), begin):
            if insn.mnemonic not in wanted or insn.mnemonic in found:
                continue
            if any(op.type == capstone.x86.X86_OP_MEM
                   and op.mem.base and lift.MD.reg_name(op.mem.base) == "rip"
                   for op in insn.operands):
                continue
            # The first instance is not always a testable one: `vcmpeqps
            # xmm0, xmm0, [rsi + 0x3320]` reads past the arena, so the harness
            # cannot give it a place to read. A later instance of the same
            # instruction usually has a shorter displacement, so keep looking
            # instead of settling for an instance that would fault on both
            # sides and prove nothing.
            if not reachable(bytes(insn.bytes), 0):
                continue
            found[insn.mnemonic] = bytes(insn.bytes)
        if len(found) == len(wanted):
            break

    # Lift each one before choosing where it goes: `branch` can still refuse,
    # and a refusal here is information — it says which instruction the round
    # has *not* covered, which is the opposite of what a passing count says.
    pieces, refused = [], []
    for name, flags in SNIPPETS:
        body = found.get(name)
        if body is None:
            refused.append((name, "not in the image"))
            continue
        code = body + snippet_code(flags)
        # `branch.lift_function` gets exactly what the guest will execute, so the
        # bytes are the specification on both sides. rva is only ever used to
        # resolve rip-relative targets, and a snippet has none.
        try:
            lines = branch.lift_function(code, 0)
        except (branch.Refused, lift.Refused) as why:
            refused.append((name, str(why)))
            continue
        if not lines:
            refused.append((name, "lifted to nothing"))
            continue
        pieces.append((name, code, lines))

    # Sixteen-byte starts, packed: the run has to fit in a gap between two
    # `.pdata` ranges, and gaps in a stripped `.text` are hundreds of bytes, not
    # thousands. A ret is a full stop, so alignment is for readability only.
    offsets, at = [], 0
    for _, code, _ in pieces:
        at = (at + 15) & ~15
        offsets.append(at)
        at += len(code)

    slots = padding(pe, at)
    image = bytearray(open(exe, "rb").read())
    rows = []
    for (name, code, lines), off in zip(pieces, offsets):
        rva = slots + off
        rows.append((rva, name, code, lines))

    for rva, _, code, _ in rows:
        for sec in pe.sections:
            if sec.VirtualAddress <= rva and rva + len(code) <= \
                    sec.VirtualAddress + sec.SizeOfRawData:
                fo = rva - sec.VirtualAddress + sec.PointerToRawData
                image[fo:fo + len(code)] = code
                break
        else:
            raise SystemExit(f"{rva:#x} ({len(code)} bytes) is not in raw data")

    path = os.path.join(work, "snippets.exe")
    open(path, "wb").write(image)
    for name, why in refused:
        print(f"  {name:14s} skipped: {why}")
    print(f"  {len(rows)} snippets in {at} bytes at {slots:#x}, "
          f"image patched to {os.path.basename(path)}")
    return path, [(rva, code, lines) for rva, _, code, lines in rows], \
        {rva: name for rva, name, _, _ in rows}



def emit(kept, path, pe):
    """The lifted functions, a bridge for everything else, and a driver."""
    outer = externals(kept)
    names = sorted({rva for rva, _, _ in kept} | outer)
    lines = ["#include <stdint.h>", "#include <stdio.h>", "#include <string.h>",
             "#include <math.h>", "#include <intrin.h>", "#include <setjmp.h>"]
    lines += lift.HELPERS
    lines += lift.image_loader(pe)
    lines += [
        "#include <windows.h>",
        f"#define ARENA_ADDR 0x{ARENA:x}ull",
        f"#define ARENA_SIZE 0x{ARENA_SIZE:x}u",
        f"#define SLOTS {SLOTS}",
        "",
        "// The vector register file: sixteen 256-bit registers, four lanes each.",
        "// A global because they *are* global: a nested lifted call must see the",
        "// same registers its caller does. Reset from the case record, so both",
        "// sides of the comparison start from the same bytes.",
        "static uint64_t v[16][4];",
        "",
        "// The one slot a bridged call leaves different: the return address it",
        "// pushed, a host address here and an image address in a guest `call`.",
        "extern void guest_bridge(uint64_t *r, uint64_t *v, const void *target);",
        "static void call_indirect(uint64_t *r, uint64_t target);",
        "static void jump_indirect(uint64_t *r, uint64_t target);",
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
            f"    guest_bridge(r, v, img + {rva:#x});",
            "    r[4] -= 8;",
            "}",
            f"void j_{rva:#x}(uint64_t *r) {{",
            "    note(r[4] - 8);",
            f"    guest_bridge(r, v, img + {rva:#x});",
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
    # The two hooks the branch pass emits where the call or jump target is a
    # value. `call rax` cannot be resolved at translate time, but it can be
    # resolved against the only thing that knows what the address means: the
    # function table. One table, two entry points, because the stack protocol
    # differs — a call returns here, a tail jump does not.
    lines += [
        "",
        "// An indirect call names its target in a register, so the target is a",
        "// value. This is the table that gives it a meaning: a lifted body where",
        "// there is one, and the original bytes where there is not — which is not",
        "// a fallback in the sense of being worse, it is the code the console ran.",
        "static void route(uint64_t *r, uint64_t target, int tail) {",
        "    switch (target - (uint64_t)(uintptr_t)img) {",
    ]
    lines += [f"    case {rva:#x}: if (tail) j_{rva:#x}(r); else f_{rva:#x}(r); return;"
              for rva in names]
    lines += [
        "    default: break;",
        "    }",
        "    // Nothing lifted at that address. For a call the lift has already",
        "    // pushed an image address at `r[4] - 8`; the bridge's own `call`",
        "    // pushes a host address into that same slot, so it is reported and",
        "    // the comparison skips it. For a tail jump `r[4]` already holds the",
        "    // caller's return address and there is nothing to push.",
        "    if (tail) {",
        "        note(r[4] - 8);",
        "        guest_bridge(r, v, (const void *)(uintptr_t)target);",
        "    } else {",
        "        r[4] += 8;",
        "        note(r[4] - 8);",
        "        guest_bridge(r, v, (const void *)(uintptr_t)target);",
        "        r[4] -= 8;",
        "    }",
        "}",
        "static void call_indirect(uint64_t *r, uint64_t target) { route(r, target, 0); }",
        "static void jump_indirect(uint64_t *r, uint64_t target) { route(r, target, 1); }",
    ]
    lines += [
        "static uint8_t *arena;",
        "",
        "// One case's fault is that case's business, including one inside the",
        "// bridge: `__try` cannot help there, because the guest runs on the arena",
        "// and the unwinder walks a stack that is not this thread's. A vectored",
        "// handler plus `longjmp` does not care whose stack it is on — it restores",
        "// rsp and rip from the buffer `setjmp` filled on the host's.",
        "static jmp_buf jb;",
        "static int armed;",
        "static LONG CALLBACK on_fault(EXCEPTION_POINTERS *ep) {",
        "    if (!armed) return EXCEPTION_CONTINUE_SEARCH;",
        "    armed = 0;",
        "    longjmp(jb, 1);",
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
        "        if (fread(v, sizeof v, 1, in) != 1) return 6;",
        "        nslots = 0;",
        "        uint32_t ran = 1;",
        "        armed = 1;",
        "        if (setjmp(jb) == 0) {",
        "        switch (head[0]) {",
    ]
    lines += [f"        case {rva:#x}: f_{rva:#x}(r); break;" for rva, _, _ in kept]
    lines += [
        "        default: break;",
        "        }",
        "        armed = 0;",
        "        } else { ran = 0; }",
        "        fwrite(r, sizeof r, 1, out);",
        "        fwrite(arena, ARENA_SIZE, 1, out);",
        "        fwrite(v, sizeof v, 1, out);",
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
        for rva, length, regs, arena, vec in items:
            f.write(struct.pack("<II", rva, length))
            f.write(struct.pack("<16Q", *regs))
            f.write(arena)
            f.write(struct.pack("<64Q", *vec))
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
    """Registers, arena, vector registers: what the original bytes did."""
    with open(path, "rb") as f:
        data = f.read()
    got = []
    stride = 128 + ARENA_SIZE + VEC_BYTES + 4
    for i in range(count):
        base = i * stride
        if struct.unpack_from("<I", data, base + stride - 4)[0] == 0:
            got.append(None)  # the guest faulted on this one
            continue
        got.append((struct.unpack_from("<16Q", data, base),
                    data[base + 128:base + 128 + ARENA_SIZE],
                    data[base + 128 + ARENA_SIZE:base + stride - 4]))
    return got


def read_port(path, count):
    """The same, plus the slots a bridged call had to excuse."""
    with open(path, "rb") as f:
        data = f.read()
    got = []
    tail = 128 + ARENA_SIZE + VEC_BYTES
    for i in range(count):
        base = i * RECORD_OUT
        got.append({
            "regs": struct.unpack_from("<16Q", data, base),
            "arena": data[base + 128:base + 128 + ARENA_SIZE],
            "vec": data[base + 128 + ARENA_SIZE:base + tail],
            "slots": struct.unpack_from(f"<{SLOTS}Q", data, base + tail + 4),
            "nslots": struct.unpack_from("<I", data, base + tail)[0],
            "ran": struct.unpack_from("<I", data, base + tail + 4 + SLOTS * 8)[0],
        })
    return got


def vector_state(rng):
    """Entry state for xmm0-xmm15: four kinds of register, one per index mod 4.

    Not random bytes. A random 64-bit lane is a NaN or a denormal far more often
    than not, and denormals are where hardware and a compiled expression are
    most likely to differ for reasons that have nothing to do with the
    translation. Every lane here is a normal float in [1, 2), or its negation,
    or zero.

    The variety is the point: `vtestps` looks only at sign bits, `vpsignb` reads
    the sign of each byte of its third operand, and `vrsqrtps`-shaped code cares
    whether a lane is zero. Sixteen registers of +1.0 would make every one of
    those flags constant and the comparison vacuous — it would agree for the
    wrong reason.
    """
    words = []
    for reg in range(16):
        style = reg % 4                      # +, -, zero, mixed
        for _ in range(4):
            low = 0x3F800000 | rng.getrandbits(23)
            high = 0x3F800000 | rng.getrandbits(23)
            if style == 1:
                low, high = low | 0x80000000, high | 0x80000000
            elif style == 2:
                low = high = 0
            elif style == 3:
                low |= rng.getrandbits(1) << 31
                high |= rng.getrandbits(1) << 31
            words.append(low | (high << 32))
    return words


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
    ap.add_argument("--timeout", type=int, default=5,
                    help="seconds a case batch may run before it counts as hung")
    ap.add_argument("--oracle", default=r"D:\zjou-build\release\oracle.exe")
    ap.add_argument("--bridge", default=r"D:\zjou-build\release\oracle_guest.dll",
                    help="the DLL that exports guest_bridge")
    ap.add_argument("--branches", action="store_true", help="use the branch pass")
    ap.add_argument("--snippets", action="store_true",
                    help="one function per instruction, bytes taken from the image")
    ap.add_argument("--keep", help="write the generated C here, for reading")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="diff-")
    names = {}
    if args.snippets:
        # Every instruction gets the function it appears in somewhere in the
        # image, cut down to itself. `--limit` does not apply: the unit here is
        # the instruction, not the function.
        patched, picked, names = snippets(args.exe, work)
        args.exe = patched
    else:
        _, picked = pick(args.exe, args.limit, args.max_bytes, args.branches, args.skip)
    kept = [(rva, code, lines) for rva, code, lines in picked
            if reachable(code, rva) and balanced(code, rva)]
    if names:
        gone = [names[rva] for rva, _, _ in picked if rva not in {k[0] for k in kept}]
        if gone:
            print(f"  not runnable as a snippet: {', '.join(gone)}")
    if args.only is not None:
        kept = [item for item in kept if item[0] == args.only]
        args.rounds = 1
    # Both sides map `args.exe`, so a snippet run patches a copy and both sides
    # see the same file: the bytes are the specification, and there is no way
    # for the oracle to test something other than what the port executes.
    pe = pefile.PE(args.exe, fast_load=True)
    branch.set_image(pe)
    bridged = len(externals(kept))
    print(f"lifted {len(picked)} functions, {len(kept)} runnable here, {bridged} bridged callee(s)")
    lifted = {rva: code for rva, code, _ in kept}
    teb = {rva for rva in lifted if teb_dependent(pe, rva, lifted)}
    if teb:
        print(f"  {len(teb)} of them read the TEB, here or through a callee: "
              f"compared apart, since the two processes have different TEBs")

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
            cases.append((rva, len(by_rva[rva]), regs, self_referential(rng), vector_state(rng)))
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

    bad, checked, shown, apart = 0, 0, 0, 0
    per_name = {}
    for rva, want, mine in records:
        if want is None or mine is None or not mine["ran"]:
            continue
        # The `call` that starts the run pushed a return address at `rsp - 8`,
        # which is `RSP0` once the guest is running — the lifted C never pushes
        # it. Every bridged call leaves the same kind of hole one frame further
        # down, and the driver reported those.
        skip = {top} | {(s - ARENA) for s in mine["slots"][:mine["nslots"]]
                        if ARENA <= s < ARENA + ARENA_SIZE - 8}
        got_regs, got_arena, got_vec = mine["regs"], mine["arena"], mine["vec"]
        if rva in teb:
            apart += 1
            continue
        checked += 1
        same_memory = all(got_arena[off:off + 8] == want[1][off:off + 8]
                          for off in range(0, ARENA_SIZE, 8) if off not in skip)
        agree = list(got_regs) == list(want[0]) and same_memory and got_vec == want[2]
        if names:
            name = names[rva]
            ok, n = per_name.get(name, (0, 0))
            per_name[name] = (ok + (1 if agree else 0), n + 1)
        if not agree:
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
                if got_vec != want[2]:
                    at = [f"x{off // 32}[{(off % 32) // 8}]" for off in range(0, VEC_BYTES, 8)
                          if got_vec[off:off + 8] != want[2][off:off + 8]][:4]
                    clash.append("vector at " + ", ".join(at))
                print(f"  {rva:#x}: " + ", ".join(clash))
    print(f"functions: {len(kept)} runnable, {len(kept) - len(faulted_both) - len(faulted_port)} "
          f"compared, {len(faulted_both)} fault in both, {len(faulted_port)} fault only in the port")
    if names and faulted_port:
        print("  faulting only in the port: "
              + ", ".join(sorted(names[r] for r in faulted_port)))
    print(f"against hardware: {checked} case(s) compared, {checked - bad} agree, {bad} differ")
    if per_name:
        broken = sorted(n for n, (ok, tot) in per_name.items() if ok != tot)
        clean = sorted(n for n, (ok, tot) in per_name.items() if ok == tot)
        print(f"  {len(per_name)} instruction(s) covered, {len(clean)} of them agree on every case")
        if broken:
            print("  disagreeing: " + ", ".join(
                f"{n} ({per_name[n][0]}/{per_name[n][1]})" for n in broken))
    if apart:
        print(f"  {apart} further case(s) not compared: the TEB is per process, "
              f"so the two sides cannot see the same TLS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
