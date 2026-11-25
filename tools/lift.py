"""Lift a small, safe subset of a real x86-64 function to C, then check it.

The point of this file is not coverage: it is to prove the pipeline on the
title's own code before anyone writes the general case. It walks `.pdata`'s
function table, keeps the functions a first lifter can hand to C (no calls, no
branches, no flags read), emits them as
`void f(uint64_t *r)` where `r[0..15]` are rax..r15, compiles that
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

# A literal backslash-n, for the C strings emitted below: this file generates C,
# and writing the escape directly would put a line break in a string literal.
NL = chr(92) + "n"

# The arena: one fixed window, allocated by both sides of the differential
# harness at the same address, where every register but `rsp` points on entry.
# Guest pointers are host pointers, so this is not a buffer to index — it is
# 16 KiB of address space that happens to have something mapped in it. 128 GiB
# is below the Win64 user limit and far from the image, so a wild displacement
# faults instead of quietly landing in the other region.
ARENA = 0x2000000000
ARENA_SIZE = 0x4000
RSP0 = ARENA + 0x2000  # entry rsp, 8 KiB in, so a frame has room either way
MASK64 = 0xFFFFFFFFFFFFFFFF

# How far a displacement off a register may reach and still name a place the
# harness has mapped. Registers are placed at ARENA+0x1000..0x1800 and `rsp` at
# RSP0, so this bound keeps every access inside the arena and makes an
# out-of-bounds one a bug rather than a near miss.
MAX_DISP = 0x1000

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


def mem(op, bits=64, insn=None):
    """A memory operand as a real host address.

    Guest pointers are host pointers: the image is mapped at its preferred base
    and the heap is the host's, so `mov [rcx], rax` is a store where rcx points,
    with nothing in between. An earlier version addressed a 256-byte scratch
    buffer by taking the low byte of the address, which is only right when the
    address happens to be near that buffer — and rules out every function that
    writes through an argument, which is most of them.

    `[rip + d]` is not a pointer to compute: it is a fixed place in the image,
    known at lift time, so it becomes `img + rva`.
    """
    if op.type != capstone.x86.X86_OP_MEM:
        raise Refused("not memory")
    m = op.mem
    if m.segment != 0:
        raise Refused("segment")
    if m.base and MD.reg_name(m.base) == "rip":
        return f"*(uint{bits}_t *)(img + {rip_target(insn, m.disp):#x})"
    terms = []
    if m.base:
        base = MD.reg_name(m.base)
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
    return f"*(uint{bits}_t *)(uintptr_t)({inner})"


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
    return mem(op, op.size * 8, insn)


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
    return f"{mem(op, op.size * 8, insn)} = {expr};"


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
        return ["r[4] -= 8;", f"*(uint64_t *)(uintptr_t)(r[4]) = {src(ops[0], insn)};"]
    if m == "pop":
        return [wr(ops[0], "*(uint64_t *)(uintptr_t)(r[4])", insn), "r[4] += 8;"]
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


def imports(pe):
    """(IAT rva, DLL, name or None, ordinal) for every imported function."""
    import pefile

    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
    base = pe.OPTIONAL_HEADER.ImageBase
    out = []
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        dll = entry.dll.decode("ascii", "replace")
        for imp in entry.imports:
            name = imp.name.decode("ascii", "replace") if imp.name else None
            out.append((imp.address - base, dll, name, imp.ordinal or 0))
    return out


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
        "static void fix_imports(void);",
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
        f'    if (f == INVALID_HANDLE_VALUE) {{ fprintf(stderr, "open %s: %lu{NL}", path, GetLastError()); return 0; }}',
        "    // Execute as well as write: a bridged call runs the original bytes",
        "    // straight out of this mapping, and the title writes its own globals.",
        "    img = (uint8_t *)VirtualAlloc((void *)(uintptr_t)IMG_BASE, IMG_SIZE,",
        "                                  MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);",
        f'    if (!img) {{ fprintf(stderr, "map: %lu{NL}", GetLastError()); CloseHandle(f); return 0; }}',
        "    for (unsigned i = 0; i < sizeof IMG_SECTIONS / sizeof IMG_SECTIONS[0]; i++) {",
        "        LARGE_INTEGER at; DWORD done = 0;",
        "        at.QuadPart = IMG_SECTIONS[i].off;",
        "        if (!SetFilePointerEx(f, at, NULL, FILE_BEGIN)) continue;",
        "        ReadFile(f, img + IMG_SECTIONS[i].rva, IMG_SECTIONS[i].size, &done, NULL);",
        "    }",
        "    CloseHandle(f);",
        "    fix_imports();",
        "    return 1;",
        "}",
        "",
    ]
    rows = []
    for entry in imports(pe):
        iat, dll, name, ordinal = entry
        label = "NULL" if name is None else '"%s"' % name.replace('"', "")
        # Field order matters and matches the struct below: at, ordinal, name, dll.
        rows.append('{ %d, %d, %s, "%s" }'
                    % (iat, ordinal if name is None else 0, label, dll))
    if rows:
        lines += [
            "// The import table, so that a `jmp [__imp_Foo]` lands on Foo and not",
            "// on the hint/name pair the linker left in the slot. A recompiled",
            "// binary needs this for the same reason a loader does.",
            "static const struct { uint32_t at, ordinal; const char *name, *dll; } IMG_IMPORTS[] = {",
            "    " + ",\n    ".join(rows),
            "};",
            "",
            "static void fix_imports(void) {",
            "    HMODULE last = NULL; const char *dll = NULL;",
            "    for (unsigned i = 0; i < sizeof IMG_IMPORTS / sizeof IMG_IMPORTS[0]; i++) {",
            "        if (!dll || strcmp(dll, IMG_IMPORTS[i].dll)) {",
            "            dll = IMG_IMPORTS[i].dll;",
            "            last = LoadLibraryA(dll);",
            "        }",
            "        if (!last) continue;",
            "        FARPROC fn = IMG_IMPORTS[i].name",
            "            ? GetProcAddress(last, IMG_IMPORTS[i].name)",
            "            : GetProcAddress(last, (LPCSTR)(uintptr_t)IMG_IMPORTS[i].ordinal);",
            "        *(uint64_t *)(img + IMG_IMPORTS[i].at) = (uint64_t)(uintptr_t)fn;",
            "    }",
            "}",
            "",
        ]
    return lines


def balanced(code, rva):
    """True when every `ret` leaves `rsp` where it found it.

    A function whose exit frame differs from its entry one is a fragment of
    another function (a cold block, a funclet): its `ret` reads a return address
    the parent pushed, so entering it from a harness pops whatever the stack
    holds and jumps into it.

    Statically this is a running `rsp` delta, and `rbp` has to be carried along
    because the ordinary MSVC epilogue is `mov rsp, rbp; pop rbp; ret`. When
    either register is set to something this cannot follow — `and rsp, -16` —
    the delta becomes unknown and the function is not entered, which is the safe
    direction.
    """
    reg = capstone.x86.X86_OP_REG

    def immediate(ops):
        if len(ops) != 2 or ops[1].type != capstone.x86.X86_OP_IMM:
            return None
        return ops[1].imm

    rsp, rbp = 0, None  # offsets from the entry `rsp`
    for insn in MD.disasm(code, rva):
        m, ops = insn.mnemonic, insn.operands
        first = MD.reg_name(ops[0].reg) if ops and ops[0].type == reg else ""
        if m == "push":
            rsp -= 8
        elif m == "pop" and first == "rsp":
            return False  # `pop rsp` is a jump in disguise
        elif m == "pop":
            rsp += 8  # `pop rbp` reloads the frame pointer; rbp is then unknown
        elif m in ("add", "sub") and first == "rsp":
            n = immediate(ops)
            if n is None:
                return False
            rsp = rsp + n if m == "add" else rsp - n
        elif m == "lea" and first == "rsp":
            if len(ops) != 2 or ops[1].type != capstone.x86.X86_OP_MEM:
                return False
            if ops[1].mem.index or not ops[1].mem.base:
                return False
            if MD.reg_name(ops[1].mem.base) != "rsp":
                return False
            rsp += ops[1].mem.disp
        elif m == "leave":
            if rbp is None:
                return False
            rsp = rbp + 8
        elif m == "mov" and len(ops) == 2 and ops[1].type == reg:
            source = MD.reg_name(ops[1].reg)
            if first == "rsp" and source == "rbp":
                if rbp is None:
                    return False
                rsp = rbp
            elif first == "rbp" and source == "rsp":
                rbp = rsp
            elif first in ("rsp", "rbp"):
                return False
        elif m == "mov" and first in ("rsp", "rbp"):
            return False
        elif m == "ret":
            if rsp != 0:
                return False
    return True


def reachable(code, rva):
    """True when every memory operand names a place the harness has mapped.

    No `gs:`/`fs:`, no scaled index off a pointer (an address built from two
    arbitrary registers is not a place a test can predict), no displacement
    past the arena. `[rip+d]` is a fixed offset into the image and both sides
    have the image.
    """
    for insn in MD.disasm(code, rva):
        for op in insn.operands:
            if op.type != capstone.x86.X86_OP_MEM:
                continue
            if op.mem.segment:
                return False
            base = MD.reg_name(op.mem.base) if op.mem.base else ""
            if base == "rip":
                if not 0 <= insn.address + insn.size + op.mem.disp < IMAGE_SIZE:
                    return False
                continue
            if op.mem.index:
                return False
            if not base:
                # An absolute operand: correct only while the image sits at the
                # base it asks for, which it does on both sides.
                at = op.mem.disp & 0xFFFFFFFFFFFFFFFF
                if not IMAGE_BASE <= at < IMAGE_BASE + IMAGE_SIZE:
                    return False
                continue
            if not -MAX_DISP <= op.mem.disp < MAX_DISP:
                return False
    return True


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


class Space:
    """The guest's memory: an address space with two regions mapped.

    Guest addresses are host addresses on the other side of this file too, so
    the model has to be a map and not a buffer. The image sits at its preferred
    base, which is what makes `lea rax, [rip+d]` hand out a pointer that is
    still valid when it comes back as `[rax+8]`; the arena is where the harness
    points every other register.
    """

    def __init__(self, image=None):
        self.image = image
        self.arena = bytearray(ARENA_SIZE)

    def _at(self, addr, size):
        if addr + size > MASK64 + 1:
            raise Refused("address wraps")
        if IMAGE_BASE <= addr and addr + size <= IMAGE_BASE + IMAGE_SIZE:
            if self.image is None:
                raise Refused("image not mapped")
            return self.image, addr - IMAGE_BASE
        if ARENA <= addr and addr + size <= ARENA + ARENA_SIZE:
            return self.arena, addr - ARENA
        raise Refused(f"outside the mapped regions ({addr:#x})")

    def read(self, addr, size):
        buf, off = self._at(addr, size)
        return bytes(buf[off:off + size])

    def write(self, addr, size, value):
        buf, off = self._at(addr, size)
        buf[off:off + size] = (value & ((1 << (size * 8)) - 1)).to_bytes(size, "little")


def interpret(code, base_rva, regs, space=None):
    """Run a function's bytes in a model of its address space.

    Written from the instruction semantics rather than from the emitter, so that
    a disagreement between the two is a bug in one of them and not a shared
    misunderstanding. Control flow is not modelled here — `diff.py` is the
    checker that runs the hardware — so the input is a straight-line body.
    """
    if space is None:
        space = Space()
    for insn in MD.disasm(code, base_rva):
        m = insn.mnemonic
        ops = insn.operands

        def address(op):
            """The 64-bit address a memory operand names, without reaching it."""
            mo = op.mem
            at = mo.disp
            if mo.base:
                name = MD.reg_name(mo.base)
                if name == "rip":
                    return (IMAGE_BASE + insn.address + insn.size + mo.disp) & MASK64
                at += regs[SUBS[name][0]]
            if mo.index:
                at += regs[SUBS[MD.reg_name(mo.index)][0]] * (mo.scale or 1)
            return at & MASK64

        def get(op):
            if op.type == capstone.x86.X86_OP_IMM:
                return op.imm & MASK64
            if op.type == capstone.x86.X86_OP_REG:
                index, bits, shift = SUBS[MD.reg_name(op.reg)]
                if bits == 64:
                    return regs[index]
                return (regs[index] >> shift) & ((1 << bits) - 1)
            fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[op.size]
            return struct.unpack(fmt, space.read(address(op), op.size))[0]

        def put(op, value):
            value &= MASK64
            if op.type == capstone.x86.X86_OP_REG:
                index, bits, shift = SUBS[MD.reg_name(op.reg)]
                if bits == 32:
                    # A 32-bit write zeroes the top half on x86-64.
                    regs[index] = value & 0xFFFFFFFF
                else:
                    mask = (1 << bits) - 1
                    regs[index] = ((regs[index] & ~(mask << shift))
                                   | ((value & mask) << shift)) & MASK64
                return
            space.write(address(op), op.size, value)

        if m in ("nop", "endbr64", "cmp", "test"):
            continue
        if m == "ret":
            break
        if m == "push":
            regs[4] = (regs[4] - 8) & MASK64
            space.write(regs[4], 8, get(ops[0]))
            continue
        if m == "pop":
            put(ops[0], struct.unpack("<Q", space.read(regs[4], 8))[0])
            regs[4] = (regs[4] + 8) & MASK64
            continue
        if m in ("mov", "movabs"):
            put(ops[0], get(ops[1]))
        elif m in ("movzx", "movsx", "movsxd"):
            size = ops[1].size * 8
            v = get(ops[1]) & ((1 << size) - 1)
            if m != "movzx" and v >> (size - 1):
                v -= 1 << size
            put(ops[0], v)
        elif m == "lea":
            put(ops[0], address(ops[1]))
        elif m in ("add", "sub", "and", "or", "xor"):
            a, b = get(ops[0]), get(ops[1])
            put(ops[0], {"add": a + b, "sub": a - b, "and": a & b,
                         "or": a | b, "xor": a ^ b}[m])
        elif m == "imul":
            put(ops[0], get(ops[0]) * get(ops[-1]))
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
    return space


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
    picked = [(rva, code, lines) for rva, code, lines in picked
              if reachable(code, rva) and balanced(code, rva)]
    print(f"of those, {len(picked)} stay inside the arena")

    proto = "void f%d(uint64_t *r)"
    body = []
    for i, (rva, code, lines) in enumerate(picked):
        body.append(f"{proto % i} {{")
        for line in lines:
            body.append(f"    {line}")
        body.append("}")
    driver = "\n".join([
        f"#define ARENA_ADDR 0x{ARENA:x}ull",
        f"#define ARENA_SIZE 0x{ARENA_SIZE:x}u",
        "static uint8_t *arena;",
        "int main(int argc, char **argv) {",
        "    FILE *in = fopen(argv[1], \"rb\"), *out = fopen(argv[2], \"wb\");",
        "    if (!in || !out) return 1;",
        "    if (argc < 4 || !map_image(argv[3])) return 4;",
        "    arena = (uint8_t *)VirtualAlloc((void *)(uintptr_t)ARENA_ADDR, ARENA_SIZE,",
        "                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);",
        "    if (!arena) return 5;",
        "    uint32_t head[2];",
        "    while (fread(head, sizeof head, 1, in) == 1) {",
        "        uint64_t r[16];",
        "        if (fread(r, sizeof r, 1, in) != 1) return 2;",
        "        if (fread(arena, ARENA_SIZE, 1, in) != 1) return 3;",
        "        switch (head[0]) {",
    ] + [f"        case {rva:#x}: f{i}(r); break;" for i, (rva, _, _) in enumerate(picked)] + [
        "        default: break;",
        "        }",
        "        fwrite(r, sizeof r, 1, out);",
        "        fwrite(arena, ARENA_SIZE, 1, out);",
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

    # Inputs, then the same inputs through the interpreter. Every register but
    # `rsp` points into the arena, which is what a real caller hands a function:
    # guest pointers are host pointers, so a function that stores through its
    # first argument works instead of faulting.
    rng = random.Random(0x5EED)
    cases = []
    for rva, code, _ in picked:
        for _ in range(args.rounds):
            regs = [ARENA + rng.randrange(0x100, 0x1000, 8) for _ in range(16)]
            regs[4] = RSP0
            space = bytes(rng.getrandbits(8) for _ in range(ARENA_SIZE))
            cases.append((rva, len(code), regs, space))
    with open(os.path.join(work, "in.bin"), "wb") as f:
        for rva, length, regs, space in cases:
            f.write(struct.pack("<II", rva, length))
            f.write(struct.pack("<16Q", *regs))
            f.write(space)

    exe = os.path.join(work, "lifted.exe")
    cl = find_cl()
    if not cl:
        print("no cl.exe: emitted C only")
        return
    env = dict(os.environ)
    env["INCLUDE"] = ";".join(include_dirs())
    env["LIB"] = ";".join(lib_dirs())
    build = subprocess.run([cl, "/nologo", "/O1", "/std:c11", f"/Fo{work}\\", csrc, "/link", f"/OUT:{exe}",
                            "/BASE:0x10000000", "/DYNAMICBASE:NO"],
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
    image = load_image(pe)
    bad, checked, refused = 0, 0, 0
    for i, (rva, length, regs, space_bytes) in enumerate(cases):
        space = Space(image)
        space.arena[:] = space_bytes
        want = list(regs)
        try:
            interpret(pe.get_data(rva, length), rva, want, space)
        except Refused as why:
            refused += 1
            if refused <= 3:
                print(f"  {rva:#x}: interpreter refused {why}")
            continue
        off = i * (128 + ARENA_SIZE)
        got = list(struct.unpack_from("<16Q", data, off))
        got_arena = data[off + 128: off + 128 + ARENA_SIZE]
        checked += 1
        if got != want or got_arena != bytes(space.arena):
            bad += 1
            if bad <= 3:
                diff = [f"{REGS[k]}: {want[k]:#x} != {got[k]:#x}" for k in range(16) if want[k] != got[k]]
                print(f"  {rva:#x}: {', '.join(diff) or 'arena differs'}")
    print(f"emitter vs interpreter: {checked} case(s) over {len(picked)} functions: "
          f"{checked - bad} agree, {bad} differ, {refused} refused")


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
