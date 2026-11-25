"""Lift a small, safe subset of a real x86-64 function to C, then check it.

The point of this file is not coverage: it is to prove the pipeline on the
title's own code before anyone writes the general case. It walks `.pdata`'s
function table, keeps the functions a first lifter can hand to C (no calls, no
branches, no flags read, memory only through `rsp`), emits them as
`void f(uint64_t *r, uint8_t *stk)` where `r[0..15]` are rax..r15, compiles that
with the host's own MSVC, runs it on random inputs, and compares against an
interpreter written here, independently of the emitter.

    python tools/recomp/lift.py --limit 400 --keep out/lifted.c

The original bytes are not run yet. They will be, and cheaply: the guest is
x86-64 and so is the host, so a five-instruction stub can call the original in
this same process — that is the harness the general recompiler needs, and it is
what turns "two of our own implementations agree" into "the lifted code matches
the hardware".
"""

import argparse
import os
import random
import struct
import subprocess
import sys
import tempfile

import collections

import capstone
import pefile

REGS = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
IDX = {name: i for i, name in enumerate(REGS)}
STACK = 0x100  # bytes of scratch stack each call gets

# The image's preferred base and extent. A RIP-relative operand is an offset
# from the instruction, so it becomes `img + rva`: a real host pointer once the
# image is mapped, which it has to be — most functions read a global in their
# first three instructions.
IMAGE_BASE = 0x140000000
IMAGE_SIZE = 0xB06D000

MD = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
MD.detail = True


class Refused(Exception):
    pass


# Sub-registers. Writing `esi` clears the top half of `rsi`, writing `al` keeps
# the other 56 bits, writing `ah` keeps 56 bits and replaces bits 8..15. Each of
# those is two lines of C here and a silent wrong answer if approximated with a
# cast, so the width and the shift travel with every register operand.
SUBS = {}


def _sub(names, index, bits, shift=0):
    for name in names:
        SUBS[name] = (index, bits, shift)


for _i, _q, _d, _w, _b, _bh in ((0, "rax", "eax", "ax", "al", "ah"),
                                (1, "rcx", "ecx", "cx", "cl", "ch"),
                                (2, "rdx", "edx", "dx", "dl", "dh"),
                                (3, "rbx", "ebx", "bx", "bl", "bh"),
                                (4, "rsp", "esp", "sp", "spl", None),
                                (5, "rbp", "ebp", "bp", "bpl", None),
                                (6, "rsi", "esi", "si", "sil", None),
                                (7, "rdi", "edi", "di", "dil", None)):
    _sub([_q], _i, 64)
    _sub([_d], _i, 32)
    _sub([_w], _i, 16)
    _sub([_b], _i, 8)
    if _bh:
        _sub([_bh], _i, 8, 8)
for _i in range(8, 16):
    _sub(["r%d" % _i], _i, 64)
    _sub(["r%dd" % _i], _i, 32)
    _sub(["r%dw" % _i], _i, 16)
    _sub(["r%db" % _i], _i, 8)

ONES = {8: "0xFFull", 16: "0xFFFFull", 32: "0xFFFFFFFFull", 64: "~0ull"}


def reg(op):
    """The parent register of an operand: `esi` -> the index of `rsi`."""
    if op.type != capstone.x86.X86_OP_REG:
        raise Refused("not a register")
    name = MD.reg_name(op.reg)
    if name not in SUBS:
        raise Refused(f"register {name}")
    return SUBS[name][0]


def is_imm(op):
    return op.type == capstone.x86.X86_OP_IMM


def width(op):
    """The width of an operand in bits."""
    if op.type == capstone.x86.X86_OP_REG:
        return SUBS[MD.reg_name(op.reg)][1]
    if op.type == capstone.x86.X86_OP_MEM:
        return op.size * 8
    raise Refused("width of an immediate")


def stack_slot(op, bits=64, insn=None):
    """The guest stack: `stk`, addressed by offset, at the accessed width.

    The width matters on the way in as well as out: `mov dword ptr [rbp-4], eax`
    leaves the four bytes above it alone, and a `uint64_t` store there would
    quietly flatten the neighbouring local.

    `[rip + d]` is not the stack: it is a fixed place in the image, known at
    lift time, so it becomes an absolute offset into `img`.
    """
    if op.type != capstone.x86.X86_OP_MEM:
        raise Refused("not memory")
    m = op.mem
    if m.segment != 0:
        raise Refused("segment")
    terms = []
    if m.base:
        base = MD.reg_name(m.base)
        if base == "rip":
            return f"*(uint{bits}_t *)(img + {rip_target(insn, m.disp):#x})"
        if base not in SUBS:
            raise Refused(f"base {base}")
        terms.append(f"r[{SUBS[base][0]}]")
    if m.index:
        index = MD.reg_name(m.index)
        if index not in SUBS:
            raise Refused(f"index {index}")
        terms.append(f"r[{SUBS[index][0]}] * {m.scale or 1}")
    if m.disp:
        terms.append(str(m.disp))
    inner = " + ".join(terms) if terms else "0"
    return f"*(uint{bits}_t *)(stk + (({inner}) & 255))"


def rip_target(insn, disp):
    """The image offset a `[rip + disp]` operand names."""
    if insn is None:
        raise Refused("rip without the instruction")
    return insn.address + insn.size + disp


def immediate(value):
    """An immediate, as a guest address when it points into the image.

    Absolute addresses are the other half of the same problem as `[rip + d]`:
    `mov rax, 0x1409a1b20` is a pointer to a global, and if the image is not
    loaded at its preferred base it has to move with it.
    """
    if IMAGE_BASE <= value < IMAGE_BASE + IMAGE_SIZE:
        return f"(uint64_t)(img + {value - IMAGE_BASE:#x})"
    return f"UINT64_C({value:#x})"


def src(op, insn=None):
    """An operand read by an operation, at its own width."""
    if is_imm(op):
        return immediate(op.imm & 0xFFFFFFFFFFFFFFFF)
    if op.type == capstone.x86.X86_OP_REG:
        index, bits, shift = SUBS[MD.reg_name(op.reg)]
        if bits == 64:
            return f"r[{index}]"
        inner = f"r[{index}]" if shift == 0 else f"(r[{index}] >> {shift})"
        return f"((uint{bits}_t){inner})"
    return stack_slot(op, op.size * 8, insn)


def wr(op, expr, insn=None):
    """A statement storing `expr` where `op` points, at `op`'s width."""
    if op.type == capstone.x86.X86_OP_REG:
        index, bits, shift = SUBS[MD.reg_name(op.reg)]
        if bits == 64:
            return f"r[{index}] = {expr};"
        if shift:
            mask = ONES[bits]
            return (f"r[{index}] = (r[{index}] & ~({mask} << {shift}))"
                    f" | (((uint64_t)({expr}) & {mask}) << {shift});")
        if bits == 32:
            # A 32-bit write zeroes the top half on x86-64; this is not a cast.
            return f"r[{index}] = (uint64_t)(uint32_t)({expr});"
        return f"r[{index}] = (r[{index}] & ~{ONES[bits]}) | ((uint64_t)({expr}) & {ONES[bits]});"
    return f"{stack_slot(op, op.size * 8, insn)} = {expr};"


def lift_one(insn):
    """C statements for one instruction, or Refused."""
    m = insn.mnemonic
    ops = insn.operands
    if m in ("nop", "endbr64"):
        return []
    if m in ("int3", "ud2"):
        raise Refused(m)
    if m == "ret":
        return ["return;"]
    if m == "push":
        return ["r[4] -= 8;", f"*(uint64_t *)(stk + ((r[4]) & 255)) = {src(ops[0], insn)};"]
    if m == "pop":
        return [wr(ops[0], "*(uint64_t *)(stk + ((r[4]) & 255))", insn), "r[4] += 8;"]
    if m in ("mov", "movabs"):
        # `movabs` is capstone's name for the 10-byte `mov r64, imm64`, which is
        # how an absolute address reaches a register.
        if len(ops) != 2:
            raise Refused("mov arity")
        return [wr(ops[0], src(ops[1], insn), insn)]
    if m in ("movzx", "movsx", "movsxd"):
        if len(ops) != 2:
            raise Refused("movzx arity")
        bits = ops[1].size * 8
        inner = src(ops[1], insn)
        if m == "movzx":
            return [wr(ops[0], f"(uint{bits}_t)({inner})", insn)]
        return [wr(ops[0], f"(int64_t)(int{bits}_t)({inner})", insn)]
    if m == "lea":
        if len(ops) != 2 or ops[1].type != capstone.x86.X86_OP_MEM:
            raise Refused("lea form")
        mem = ops[1].mem
        if mem.segment != 0:
            raise Refused("lea segment")
        if mem.base and MD.reg_name(mem.base) == "rip":
            return [wr(ops[0], f"(uint64_t)(img + {rip_target(insn, mem.disp):#x})", insn)]
        parts = []
        if mem.base:
            base = MD.reg_name(mem.base)
            if base not in SUBS:
                raise Refused(f"lea base {base}")
            parts.append(f"r[{SUBS[base][0]}]")
        if mem.index:
            index = MD.reg_name(mem.index)
            if index not in SUBS:
                raise Refused(f"lea index {index}")
            parts.append(f"r[{SUBS[index][0]}] * {mem.scale or 1}")
        if mem.disp or not parts:
            parts.append(f"UINT64_C({mem.disp & 0xFFFFFFFFFFFFFFFF})")
        return [wr(ops[0], " + ".join(parts), insn)]
    if m in ("add", "sub", "and", "or", "xor"):
        if len(ops) != 2:
            raise Refused("alu arity")
        op = {"add": "+", "sub": "-", "and": "&", "or": "|", "xor": "^"}[m]
        return [wr(ops[0], f"{src(ops[0], insn)} {op} {src(ops[1], insn)}", insn)]
    if m == "imul":
        if len(ops) == 1:
            raise Refused("one-operand imul")
        if len(ops) == 2:
            return [wr(ops[0], f"{src(ops[0], insn)} * {src(ops[1], insn)}", insn)]
        if len(ops) == 3:
            return [wr(ops[0], f"{src(ops[1], insn)} * {src(ops[2], insn)}", insn)]
        raise Refused("imul arity")
    if m in ("shl", "shr", "sar"):
        if len(ops) != 2:
            raise Refused("shift arity")
        bits = width(ops[0])
        # The count is masked to the width of the destination: `shl eax, 32` is
        # a no-op, not a 32-bit shift.
        count = f"({src(ops[1], insn)} & {bits - 1})"
        value = src(ops[0], insn)
        if m == "shl":
            expr = f"{value} << {count}"
        elif m == "shr":
            expr = f"{value} >> {count}"
        else:
            expr = f"(int{bits}_t){value} >> {count}"
        return [wr(ops[0], expr, insn)]
    if m in ("inc", "dec", "neg", "not"):
        one = {"inc": "+ 1", "dec": "- 1"}.get(m)
        if one:
            return [wr(ops[0], f"{src(ops[0], insn)} {one}", insn)]
        if m == "neg":
            return [wr(ops[0], f"-(int{width(ops[0])}_t){src(ops[0], insn)}", insn)]
        return [wr(ops[0], f"~{src(ops[0], insn)}", insn)]
    if m in ("cmp", "test"):
        # Nothing in this subset reads the flags, so nothing to keep. The branch
        # pass in `branch.py` is where they start to matter.
        return []
    raise Refused(m)


def image_loader(pe=None):
    """C that maps the image at its preferred base, so `img + rva` resolves.

    A RIP-relative operand and an absolute address in an immediate both name a
    place in the image, and the lifted code reaches it through `img`. That means
    the image has to actually be there: same base, same bytes, data sections
    included, writes allowed (the title writes its own globals).

    Copying sections by hand rather than `MapViewOfFile` is deliberate — a file
    mapping is laid out as the file is, `SEC_IMAGE` is not, and the loader's
    layout is what the RVAs mean.
    """
    lines = [
        "#include <windows.h>",
        "",
        f"#define IMG_BASE 0x{IMAGE_BASE:x}ull",
        f"#define IMG_SIZE 0x{IMAGE_SIZE:x}ull",
        "",
        "static uint8_t *img;",
        "",
    ]
    if pe is None:
        lines.append("static int map_image(const char *path) { (void)path; return 0; }")
        return lines
    rows = [f"{{ {s.VirtualAddress:#x}, {s.PointerToRawData:#x}, {s.SizeOfRawData:#x} }}"
            for s in pe.sections if s.SizeOfRawData]
    lines += [
        "static const struct { uint32_t rva, off, size; } IMG_SECTIONS[] = {",
        "    " + ",\n    ".join(rows),
        "};",
        "",
        "static int map_image(const char *path) {",
        "    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);",
        "    if (f == INVALID_HANDLE_VALUE) return 0;",
        "    img = (uint8_t *)VirtualAlloc((void *)(uintptr_t)IMG_BASE, IMG_SIZE,",
        "                                  MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);",
        "    if (!img) { CloseHandle(f); return 0; }",
        "    for (unsigned i = 0; i < sizeof IMG_SECTIONS / sizeof IMG_SECTIONS[0]; i++) {",
        "        LARGE_INTEGER at; DWORD done = 0;",
        "        at.QuadPart = IMG_SECTIONS[i].off;",
        "        if (!SetFilePointerEx(f, at, NULL, FILE_BEGIN)) continue;",
        "        ReadFile(f, img + IMG_SECTIONS[i].rva, IMG_SECTIONS[i].size, &done, NULL);",
        "    }",
        "    CloseHandle(f);",
        "    return 1;",
        "}",
        "",
    ]
    return lines


def lift_function(code, base_rva):
    """C body for a function's bytes, or Refused."""
    insns = list(MD.disasm(code, base_rva))
    # `.pdata` ranges carry the padding that follows a function (`int3`, `nop`),
    # and a tail call ends the range with `jmp` instead of `ret`.
    while insns and insns[-1].mnemonic in ("int3", "nop", "ud2"):
        insns.pop()
    if not insns:
        raise Refused("empty")
    if insns[-1].mnemonic == "jmp":
        raise Refused("ends in a tail call")
    if insns[-1].mnemonic != "ret":
        raise Refused("does not end in ret")
    if any(i.mnemonic.startswith("j") or i.mnemonic in ("call", "loop") for i in insns):
        raise Refused("has control flow")
    lines = []
    for insn in insns:
        lines.extend(lift_one(insn))
    return lines


# --- the interpreter, written from the instruction semantics, not the emitter


def load_image(pe):
    """The image as the loader would lay it out, for the interpreter."""
    data = bytearray(IMAGE_SIZE)
    for s in pe.sections:
        if s.SizeOfRawData:
            start = s.VirtualAddress
            data[start:start + s.SizeOfRawData] = pe.get_data(start, s.SizeOfRawData)
    return data


def interpret(code, base_rva, regs, stk, image=None):
    # Eight bytes of slack, so a read at the very top of the stack does not
    # raise. Writing it back through `stk` at the end keeps the caller's view of
    # the stack exactly as long as it was.
    mem = bytearray(bytes(stk) + bytes(8))
    for insn in MD.disasm(code, base_rva):
        m = insn.mnemonic
        ops = insn.operands

        def absolute(op):
            """A `[rip + d]` operand: an offset into the image, not the stack."""
            return insn.address + insn.size + op.mem.disp

        def address(op):
            m_ = op.mem
            at = m_.disp
            if m_.base:
                at += regs[SUBS[MD.reg_name(m_.base)][0]]
            if m_.index:
                at += regs[SUBS[MD.reg_name(m_.index)][0]] * (m_.scale or 1)
            return at & 255

        def get(op):
            if op.type == capstone.x86.X86_OP_IMM:
                return op.imm & 0xFFFFFFFFFFFFFFFF
            if op.type == capstone.x86.X86_OP_REG:
                index, bits, shift = SUBS[MD.reg_name(op.reg)]
                return (regs[index] >> shift) & ((1 << bits) - 1 if bits < 64 else 0xFFFFFFFFFFFFFFFF)
            fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[op.size]
            if op.mem.base and MD.reg_name(op.mem.base) == "rip":
                if image is None:
                    raise Refused("rip without an image")
                return struct.unpack_from(fmt, image, absolute(op))[0]
            return struct.unpack_from(fmt, mem, address(op))[0]

        def put(op, value):
            value &= 0xFFFFFFFFFFFFFFFF
            if op.type == capstone.x86.X86_OP_REG:
                index, bits, shift = SUBS[MD.reg_name(op.reg)]
                mask = (1 << bits) - 1 if bits < 64 else 0xFFFFFFFFFFFFFFFF
                regs[index] = (regs[index] & ~(mask << shift)) | ((value & mask) << shift)
                return
            fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[op.size]
            value &= (1 << (op.size * 8)) - 1
            if op.mem.base and MD.reg_name(op.mem.base) == "rip":
                if image is None:
                    raise Refused("rip without an image")
                struct.pack_into(fmt, image, absolute(op), value)
            else:
                struct.pack_into(fmt, mem, address(op), value)

        if m in ("nop", "endbr64", "cmp", "test"):
            continue
        if m == "ret":
            break
        if m == "push":
            regs[4] -= 8
            struct.pack_into("<Q", mem, regs[4], get(ops[0]))
            continue
        if m == "pop":
            put(ops[0], struct.unpack_from("<Q", mem, regs[4])[0])
            regs[4] += 8
            continue
        if m == "mov":
            put(ops[0], get(ops[1]))
        elif m in ("movzx", "movsx", "movsxd"):
            size = ops[1].size * 8
            v = get(ops[1]) & ((1 << size) - 1)
            if m != "movzx":
                if v >> (size - 1):
                    v -= 1 << size
            put(ops[0], v)
        elif m == "lea":
            m_ = ops[1].mem
            if m_.base and MD.reg_name(m_.base) == "rip":
                # The emitted C computes `img + rva`, a host pointer whose value
                # is the preferred base plus the offset; model it the same way.
                put(ops[0], IMAGE_BASE + absolute(ops[1]))
            else:
                value = m_.disp
                if m_.base:
                    value += regs[SUBS[MD.reg_name(m_.base)][0]]
                if m_.index:
                    value += regs[SUBS[MD.reg_name(m_.index)][0]] * (m_.scale or 1)
                put(ops[0], value)
        elif m in ("add", "sub", "and", "or", "xor"):
            a, b = get(ops[0]), get(ops[1])
            put(ops[0], {"add": a + b, "sub": a - b, "and": a & b, "or": a | b, "xor": a ^ b}[m])
        elif m == "imul":
            a, b = get(ops[0]), get(ops[-1])
            put(ops[0], a * b)
        elif m in ("shl", "shr", "sar"):
            bits = width(ops[0])
            count = get(ops[1]) & (bits - 1)
            a = get(ops[0])
            if m == "shl":
                put(ops[0], a << count)
            elif m == "shr":
                put(ops[0], a >> count)
            else:
                signed = a - (1 << bits) if a >> (bits - 1) else a
                put(ops[0], signed >> count)
        elif m in ("inc", "dec", "neg", "not"):
            a = get(ops[0])
            put(ops[0], {"inc": a + 1, "dec": a - 1, "neg": -a, "not": ~a}[m])
        else:
            raise Refused(m)
    stk[:] = mem[:len(stk)]


def scan(path, sample):
    """Why functions are refused, over a sample: the map of what is left to write."""
    pe = pefile.PE(path, fast_load=True)
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    table = pe.get_data(entry.VirtualAddress, entry.Size)
    reasons = collections.Counter()
    ok = 0
    seen = 0
    for at in range(0, len(table) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", table, at)
        if begin == 0 or end <= begin:
            continue
        seen += 1
        if seen > sample:
            break
        try:
            lift_function(pe.get_data(begin, end - begin), begin)
            ok += 1
        except Refused as why:
            reasons[str(why).split("(")[0].strip()] += 1
    print(f"sampled {seen:,} functions: {ok:,} lift ({ok/seen:.1%}), refused {seen-ok:,}")
    for why, n in reasons.most_common(12):
        print(f"  {n:6,}  {why}")


def candidates(path, limit, max_bytes):
    pe = pefile.PE(path, fast_load=True)
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    table = pe.get_data(entry.VirtualAddress, entry.Size)
    picked = []
    for at in range(0, len(table) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", table, at)
        if begin == 0 or not 0 < end - begin <= max_bytes:
            continue
        code = pe.get_data(begin, end - begin)
        try:
            lines = lift_function(code, begin)
        except Refused:
            continue
        if lines:
            picked.append((begin, code, lines))
        if len(picked) >= limit:
            break
    return pe, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-bytes", type=int, default=160)
    ap.add_argument("--keep", help="write the generated C here")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--scan", type=int, default=0, help="tally why functions are refused, over this many")
    args = ap.parse_args()

    if args.scan:
        scan(args.exe, args.scan)
        return
    pe, picked = candidates(args.exe, args.limit, args.max_bytes)
    print(f"liftable subset: {len(picked)} functions (<= {args.max_bytes} bytes, no calls, no branches)")

    proto = "void f%d(uint64_t *r, uint8_t *stk)"
    body = []
    for i, (rva, code, lines) in enumerate(picked):
        body.append(f"{proto % i} {{")
        for line in lines:
            body.append(f"    {line}")
        body.append("}")
    driver = "\n".join([
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <string.h>",
        "typedef void (*fn)(uint64_t *, uint8_t *);",
        "static const fn table[] = {" + ", ".join(f"f{i}" for i in range(len(picked))) + "};",
        "int main(int argc, char **argv) {",
        "    FILE *in = fopen(argv[1], \"rb\"), *out = fopen(argv[2], \"wb\");",
        "    if (!in || !out) return 1;",
        "    if (argc < 4 || !map_image(argv[3])) return 4;",
        "    for (size_t n = 0; n < sizeof table / sizeof table[0]; n++) {",
        "        uint64_t r[16]; uint8_t stk[256];",
        "        if (fread(r, sizeof r, 1, in) != 1) return 2;",
        "        if (fread(stk, sizeof stk, 1, in) != 1) return 3;",
        "        table[n](r, stk);",
        "        fwrite(r, sizeof r, 1, out);",
        "        fwrite(stk, sizeof stk, 1, out);",
        "    }",
        "    return 0;",
        "}",
        "",
    ])

    work = tempfile.mkdtemp(prefix="lift-")
    csrc = os.path.join(work, "lifted.c")
    head = "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
    whole = head + "\n".join(image_loader(pe)) + "\n" + "\n".join(body) + "\n" + driver
    with open(csrc, "w", encoding="utf-8") as f:
        f.write(whole)
    if args.keep:
        with open(args.keep, "w", encoding="utf-8") as f:
            f.write(whole)
        print(f"generated C: {args.keep} ({len(body)} lines)")
    else:
        print(f"generated C: {csrc} ({len(body)} lines)")

    # Inputs, then the same inputs through the interpreter.
    rng = random.Random(0x5EED)
    # Only the four argument registers are defined on entry; the rest are
    # pointed into the scratch stack, so a function that reads `[rbp-x]`
    # before setting rbp stays inside it instead of faulting.
    inputs = []
    for _ in picked:
        regs = [STACK // 2] * 16
        for slot in (1, 2, 8, 9):  # rcx, rdx, r8, r9
            regs[slot] = rng.getrandbits(64)
        inputs.append(regs)
    stacks = [bytes(rng.getrandbits(8) for _ in range(256)) for _ in picked]
    with open(os.path.join(work, "in.bin"), "wb") as f:
        for regs, stk in zip(inputs, stacks):
            f.write(struct.pack("<16Q", *regs))
            f.write(stk)

    exe = os.path.join(work, "lifted.exe")
    cl = find_cl()
    if not cl:
        print("no cl.exe: emitted C only")
        return
    env = dict(os.environ)
    env["INCLUDE"] = ";".join(include_dirs())
    env["LIB"] = ";".join(lib_dirs())
    build = subprocess.run([cl, "/nologo", "/O1", "/std:c11", f"/Fo{work}\\", csrc, "/link", f"/OUT:{exe}"],
                           capture_output=True, text=True, encoding="mbcs", errors="replace", env=env)
    if build.returncode != 0:
        print("compile failed:")
        print(((build.stdout or "") + (build.stderr or ""))[-2500:])
        return
    run = subprocess.run([exe, os.path.join(work, "in.bin"), os.path.join(work, "out.bin"), args.exe],
                         capture_output=True)
    if run.returncode != 0:
        print(f"run failed: {run.returncode}")
        return

    data = open(os.path.join(work, "out.bin"), "rb").read()
    image = load_image(pe) if any("[rip" in line or "img +" in line for _, _, lines in picked for line in lines) else None
    bad = 0
    for i, (rva, code, _) in enumerate(picked):
        regs = list(inputs[i])
        stk = bytearray(stacks[i])
        try:
            interpret(code, rva, regs, stk, image)
        except Refused as why:
            print(f"  {rva:#x}: interpreter refused {why}")
            continue
        off = i * (16 * 8 + 256)
        got = struct.unpack_from("<16Q", data, off)
        got_stk = data[off + 128: off + 128 + 256]
        if list(got) != regs or bytes(got_stk) != bytes(stk):
            bad += 1
            if bad <= 3:
                diff = [f"{REGS[k]}: {regs[k]:#x} != {got[k]:#x}" for k in range(16) if regs[k] != got[k]]
                print(f"  {rva:#x}: {', '.join(diff) or 'stack differs'}")
    print(f"checked {len(picked)} functions x {args.rounds} calls: {len(picked) - bad} agree, {bad} differ")


def find_cl():
    import glob
    for pattern in (r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
                    r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[-1]
    return None


def lib_dirs():
    import glob
    dirs = []
    cl = find_cl()
    if cl:
        msvc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cl))))
        dirs.append(os.path.join(msvc, "lib", "x64"))
    for kit in sorted(glob.glob("C:/Program Files (x86)/Windows Kits/10/Lib/*")):
        for part in ("ucrt", "um"):
            path = os.path.join(kit, part, "x64")
            if os.path.isdir(path):
                dirs.append(path)
    return dirs


def include_dirs():
    """The headers cl.exe needs when it is run without vcvars."""
    import glob
    dirs = []
    for cl in [find_cl()]:
        if not cl:
            continue
        msvc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cl))))
        dirs.append(os.path.join(msvc, "include"))
    kits = sorted(glob.glob("C:/Program Files (x86)/Windows Kits/10/Include/*"))
    for kit in kits:
        for part in ("ucrt", "um", "shared", "winrt"):
            path = os.path.join(kit, part)
            if os.path.isdir(path):
                dirs.append(path)
    return dirs


if __name__ == "__main__":
    main()
