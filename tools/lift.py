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


def addr(op, insn=None):
    """The 64-bit address a memory operand names, as a C expression.

    Split out of `mem` because a 128-bit operand is not a `uint128_t` in C: it
    is sixteen bytes at an address, moved with `memcpy`. The address is the
    only thing the two uses have in common.
    """
    if op.type != capstone.x86.X86_OP_MEM:
        raise Refused("not memory")
    m = op.mem
    terms = []
    if m.segment != 0:
        # Every segment override in this image is the same instruction:
        # `mov rax, gs:[0x58]`, the TLS array pointer in the TEB. Guest and host
        # are one process, so the TEB the original bytes see is the TEB this C
        # sees — `gs:[d]` is `*(T*)(teb + d)` with no translation in between.
        # `(uint64_t)__readgsqword(0x30)` is the TEB's own Self field, and on
        # x64 the GS base *is* the TEB address, so it is the base and not a
        # pointer to it. A standalone build has to establish that base itself;
        # that is loader work, and it is the same class of assumption as mapping
        # the image at its preferred address.
        seg = MD.reg_name(m.segment)
        if seg != "gs":
            raise Refused(f"segment {seg}")
        if m.base and MD.reg_name(m.base) == "rip":
            raise Refused("rip relative with a segment")
        terms.append("(uint64_t)__readgsqword(0x30)")
    if m.base and MD.reg_name(m.base) == "rip":
        # `[rip + d]` is not a pointer to compute: it is a fixed place in the
        # image, known at lift time, so it becomes `img + rva`.
        return f"(uint64_t)(img + {rip_target(insn, m.disp):#x})"
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
    return "(" + (" + ".join(terms) if terms else "0") + ")"


def mem(op, bits=64, insn=None):
    """A memory operand as a real host address.

    Guest pointers are host pointers: the image is mapped at its preferred base
    and the heap is the host's, so `mov [rcx], rax` is a store where rcx points,
    with nothing in between. An earlier version addressed a 256-byte scratch
    buffer by taking the low byte of the address, which is only right when the
    address happens to be near that buffer — and rules out every function that
    writes through an argument, which is most of them.
    """
    return f"*(uint{bits}_t *)(uintptr_t){addr(op, insn)}"


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


# --- the vector register file ------------------------------------------------
#
# Sixteen 256-bit registers, held as four 64-bit lanes each and moved with
# `memcpy` at the operand's own width, because that is what the instruction is:
# a byte copy, not a number. Nothing here is an `<immintrin.h>` type on purpose —
# the state has to be a byte image that both sides of the differential harness
# can be handed, and a `__m256` in a global is that image with a type on it that
# the C standard says may not be aliased.
#
# A VEX-encoded 128-bit form zeroes bits 255:128 of its destination, and that is
# modelled rather than ignored: the same register can be read as 256 bits later
# in the same function.

VEC = {}
for _i in range(16):
    VEC[f"xmm{_i}"] = (_i, 16)
    VEC[f"ymm{_i}"] = (_i, 32)
del _i

# `movntps`/`movntpd`/`movntdq` 是**非临时**存储：只改一个缓存提示（"别把它留在
# 缓存里"），数据落到哪、落在哪几个字节，与 `movaps` 逐字节相同。本移植没有缓存
# 模型，也就没有可区别的东西——与预取同理，不能因为它少见就拒绝整条函数。
VEC_MOVE = {"movaps", "movups", "movapd", "movupd", "movdqa", "movdqu", "lddqu",
            "movntps", "movntpd", "movntdq", "movntdqa"}

# Bitwise, three operands under VEX and two under SSE. `-` stands for "and not",
# which is `~a & b` and is not a subtraction.
VEC_LOGIC = {
    "xorps": "^", "xorpd": "^", "pxor": "^",
    "andps": "&", "andpd": "&", "pand": "&",
    "orps": "|", "orpd": "|", "por": "|",
    "andnps": "-", "andnpd": "-", "pandn": "-",
}

# Packed and scalar float arithmetic, in the width of the lane. `max`/`min` are
# ternaries and not `fmaxf`/`fminf`: the SSE instruction returns its *second*
# operand when either side is NaN, and the C functions return the other one.
# 硬件近似指令：答案就是那条指令的答案，不是它的数学定义。`1/sqrtf(x)` 与
# `1/x` 在最后几位上必然不同——rsqrt/rcp 是查表的近似，只在少数输入上与正确舍入
# 重合。要位相等只有一条路：让处理器自己算。MSVC 的 `_mm_rsqrt_ps`/`_mm_rcp_ps`
# **就是**这两条指令（SSE 基线，不需要 /arch），所以这里不是"重写语义"，是执行它。
#
# 256 位形式用两次 128 位代替：每个通道的近似只依赖它自己那一位，两个半边的结果
# 拼起来与一条 256 位指令逐位相同。片段差分（`diff.py --snippets`）拿硬件对过这一点。
VEC_APPROX = {"rsqrtps": ("rsqrt", True), "rcpps": ("rcp", True),
              "rsqrtss": ("rsqrt", False), "rcpss": ("rcp", False)}

# 融合乘加：`a*b+c` 在 C 里舍入两次，指令只舍入一次，所以没有任何 C 表达式能复现它。
# 唯一忠实的写法是那一条 intrinsic——它就是那条指令。三个数字说的是"第一个因子、
# 第二个因子、被加数"分别是哪一个操作数：132 是 DEST*SRC2+SRC1，213 是
# SRC1*DEST+SRC2，231 是 SRC1*SRC2+DEST（Intel SDM, VFMADD132PS 等）。
# 标量形式的 `127:32` 一律来自 SRC1，与 VEX 标量算术同一条规则。
FMA = {(kind + order + suffix): (kind, order, suffix)
       for kind in ("fmadd", "fmsub", "fnmadd", "fnmsub")
       for order in ("132", "213", "231")
       for suffix in ("ps", "pd", "ss", "sd")}

VEC_FLOAT = {
    "addps": ("+", True), "subps": ("-", True), "mulps": ("*", True),
    "divps": ("/", True), "maxps": ("MAX", True), "minps": ("MIN", True),
    "addpd": ("+", True), "subpd": ("-", True), "mulpd": ("*", True),
    "divpd": ("/", True), "maxpd": ("MAX", True), "minpd": ("MIN", True),
    "addss": ("+", False), "subss": ("-", False), "mulss": ("*", False),
    "divss": ("/", False), "maxss": ("MAX", False), "minss": ("MIN", False),
    "addsd": ("+", False), "subsd": ("-", False), "mulsd": ("*", False),
    "divsd": ("/", False), "maxsd": ("MAX", False), "minsd": ("MIN", False),
}

# Broadcast: one element in, every lane out. The number is the element size.
VEC_BROADCAST = {
    "broadcastss": 4, "broadcastsd": 8, "pbroadcastb": 1, "pbroadcastw": 2,
    "pbroadcastd": 4, "pbroadcastq": 8,
}

# Blends: a bitwise select per chunk, and the chunk is the whole story — the
# immediate's bit `n` picks src2's chunk `n` and drops src1's. Nothing here is a
# float, which is why `blendps` is expressible without a float model: 32 bits in,
# the same 32 bits out, from one side or the other.
VEC_BLEND = {"blendps": 4, "blendpd": 8, "pblendw": 2, "pblendd": 4}

# Packed compares. Each produces a mask of all-ones or all-zeros per chunk, so
# the emitted statement per chunk is a select between two constants — no
# comparison result reaches a register as a bool.
#
# The signedness is the instruction's, not C's: `pcmpgtb` is signed, and there
# is no unsigned form of the packed byte compare at all (the idiom for it is an
# xor with 0x80 first, which the title does itself).
VEC_CMP_INT = {
    "pcmpeqb": (1, "=="), "pcmpeqw": (2, "=="),
    "pcmpeqd": (4, "=="), "pcmpeqq": (8, "=="),
    "pcmpgtb": (1, ">"), "pcmpgtw": (2, ">"),
    "pcmpgtd": (4, ">"), "pcmpgtq": (8, ">"),
}

# The immediate predicate of `cmpps`/`cmppd`/`cmpss`/`cmpsd`, as the C relation
# that decides "all ones". Every one of the eight is written so that a NaN on
# either side lands on the side the instruction puts it: `eq` and `lt` and `le`
# are *ordered* and false for NaN, which C's `==`, `<`, `<=` already are; `neq`,
# `nlt` and `nle` are *unordered* and true for NaN, which the negation of those
# already is; `unord` and `ord` are the two predicates that name NaN directly.
VEC_CMP_FLOAT = {
    0: "{a} == {b}", 1: "{a} < {b}", 2: "{a} <= {b}",
    3: "!({a} == {a}) || !({b} == {b})", 4: "!({a} == {b})",
    5: "!({a} < {b})", 6: "!({a} <= {b})",
    7: "({a} == {a}) && ({b} == {b})",
    # AVX widened the immediate to five bits: the extra sixteen encodings are the
    # same eight relations with a different quiet/signaling pairing, plus `ge`,
    # `gt`, `false` and `true`. Quiet and signaling differ only in which
    # exceptions they raise; the mask they produce is the same, and this port
    # models no exceptions at all, so each one folds onto the relation it means.
    8: "{a} == {b}",             # eq_uq
    9: "!({a} >= {b})",          # nge_us
    10: "!({a} > {b})",          # ngt_us
    11: "0",                     # false_oq
    12: "!({a} == {b})",         # neq_oq
    13: "{a} >= {b}",            # ge_os
    14: "{a} > {b}",             # gt_os
    15: "1",                     # true_uq
    16: "{a} == {b}",            # eq_os
    17: "{a} < {b}",             # lt_oq
    18: "{a} <= {b}",            # le_oq
    19: "!({a} == {a}) || !({b} == {b})",   # unord_s
    20: "!({a} == {b})",         # neq_us
    21: "!({a} < {b})",          # nlt_uq
    22: "!({a} <= {b})",         # nle_uq
    23: "({a} == {a}) && ({b} == {b})",     # ord_s
    24: "{a} == {b}",            # eq_us
    25: "!({a} >= {b})",         # nge_uq
    26: "!({a} > {b})",          # ngt_uq
    27: "0",                     # false_os
    28: "!({a} == {b})",         # neq_os
    29: "{a} >= {b}",            # ge_oq
    30: "{a} > {b}",             # gt_oq
    31: "1",                     # true_us
}

# Capstone names `cmpps` by the predicate it was given — `vcmpeqps`,
# `vcmplt_oqps` — and then does not report the immediate, so the predicate has to
# be read back out of the mnemonic. The names are the encodings.
CMP_PREDICATE = {
    "eq": 0, "lt": 1, "le": 2, "unord": 3,
    "neq": 4, "nlt": 5, "nle": 6, "ord": 7,
    "eq_uq": 8, "nge_us": 9, "ngt_us": 10, "false_oq": 11,
    "neq_oq": 12, "ge_os": 13, "gt_os": 14, "true_uq": 15,
    "eq_os": 16, "lt_oq": 17, "le_oq": 18, "unord_s": 19,
    "neq_us": 20, "nlt_uq": 21, "nle_uq": 22, "ord_s": 23,
    "eq_us": 24, "nge_uq": 25, "ngt_uq": 26, "false_os": 27,
    "neq_os": 28, "ge_oq": 29, "gt_oq": 30, "true_us": 31,
}

# Shuffles and interleaves are all one shape: a list of which chunk of which
# source ends up at each output chunk. So they are all one emitter — build the
# list, read every chunk, then write. `unpck*` is the interleave of the low or
# high half of each 128-bit lane, which is why the second number is the half.
VEC_UNPACK = {
    "unpcklps": (4, 0), "unpckhps": (4, 1),
    "unpcklpd": (8, 0), "unpckhpd": (8, 1),
    "punpcklbw": (1, 0), "punpckhbw": (1, 1),
    "punpcklwd": (2, 0), "punpckhwd": (2, 1),
    "punpckldq": (4, 0), "punpckhdq": (4, 1),
    "punpcklqdq": (8, 0), "punpckhqdq": (8, 1),
}

# Packed integer add and subtract: chunk-local, wrapping, and signedness-free
# because two's complement does not care.
VEC_INT_ARITH = {}
for _op, _sym in (("add", "+"), ("sub", "-")):
    for _sz, _ch in (("b", 1), ("w", 2), ("d", 4), ("q", 8)):
        VEC_INT_ARITH[f"p{_op}{_sz}"] = (_ch, _sym)
del _op, _sym, _sz, _ch

# Horizontal add and subtract. Per 128-bit lane the pairs collapse inside one
# source and then inside the other: `haddps` gives a0+a1, a2+a3, b0+b1, b2+b3,
# and `haddpd` alternates, a0+a1, b0+b1 — which is the same rule with two lanes
# instead of four. The upper lane of a 256-bit register repeats it rather than
# mixing.
#
# The integer forms are the same rule with a different element size, and that is
# the whole of it: `phaddd` is `haddps` on dwords and `phaddw` on words, which
# is why one table can hold all of them. `phsub*` negates the second half of
# each pair. The three saturating names are the only ones that cannot be a plain
# C `+`: they clamp to the element's range instead of wrapping, so their lines
# are emitted separately below.
#
# `haddss`/`haddsd` stay at 0: the scalar forms add the low element *plus* the
# second operand's low element, which is not a pair inside one register.
VEC_HADD = {
    "haddps": (4, "float", "+"), "haddpd": (8, "double", "+"),
    "haddss": 0, "haddsd": 0,
    "phaddw": (2, "int16_t", "+"), "phaddd": (4, "int32_t", "+"),
    "phaddsw": (2, "int16_t", "SAT"), "phsubw": (2, "int16_t", "-"),
    "phsubd": (4, "int32_t", "-"), "phsubsw": (2, "int16_t", "-SAT"),
}

# Packed shifts. The second number is the chunk, and the third says whether the
# shift fills with the sign bit — which is why `psrad` is not `psrld` with a
# different width. The count is the low 64 bits of a second operand (or an
# immediate), and a count past the chunk's width saturates: all sign bits for an
# arithmetic shift, all zeros for a logical one.
VEC_SHIFT = {
    "psllw": (2, "<<"), "pslld": (4, "<<"), "psllq": (8, "<<"),
    "psrlw": (2, ">>"), "psrld": (4, ">>"), "psrlq": (8, ">>"),
    "psraw": (2, "SAR"), "psrad": (4, "SAR"),
}


# Every vector instruction this lifter translates, by stem. `branch.py` needs
# this to know that a vector instruction between a `cmp` and its branch did not
# eat the flags — no SSE or AVX instruction here writes EFLAGS, and the two that
# do (`ucomiss`, `ptest`) are not in this set. Derived rather than retyped, so a
# family added above is flag-neutral below without a second edit.
# 按符号取反。第三个源的元素是符号：负则把第一个源的那个元素取反，零则清零，
# 正则原样保留。数字是元素宽度。
VEC_SIGN = {"psignb": 1, "psignw": 2, "psignd": 4}

# 复制一半元素：`movshdup` 抄每个 128 位通道里的奇数元素，`movsldup` 抄偶数，
# `movddup` 把低 64 位复制到高 64 位。全是 gather，用 `weave` 就够。
VEC_DUP = {"movshdup": 1, "movsldup": 0}

# 平方根。SSE 的 `sqrt` 是**正确舍入**的，IEEE 754 要求 `sqrt` 也正确舍入，
# 所以 C 的 `sqrtf`/`sqrt` 与它是同一条指令——这一点与 `rsqrtps` 正相反。
VEC_SQRT = {"sqrtps": (4, True), "sqrtpd": (8, True),
            "sqrtss": (4, False), "sqrtsd": (8, False)}

# 取整。`imm[1:0]` 选方向，`imm[3]` 只抑制精度异常（本移植不建模异常，所以恒置
# 上，结果不变）。四个方向与 `_mm_round_*` 的四个常量一一对应，因此还是那条指令。
VEC_ROUND = {"roundps": (4, True), "roundpd": (8, True),
             "roundss": (4, False), "roundsd": (8, False)}
ROUND_MODE = ("_MM_FROUND_TO_NEAREST_INT", "_MM_FROUND_TO_NEG_INF",
              "_MM_FROUND_TO_POS_INF", "_MM_FROUND_TO_ZERO")

# AES-NI 的轮函数：名字对应 MSVC 里同名的那条 intrinsic，也就是这条指令本身。
# 第二个数字是操作数个数——`aesimc` 只有一源，其余四个是"状态 + 轮密钥"。
# 这个映像里用到的只有 `aesimc`（13 条），家族里其余几条一并接上：同一张表、
# 同一段发射，多一条不多一行代码。
VEC_AES = {"aesimc": ("_mm_aesimc_si128", 1),
           "aesenc": ("_mm_aesenc_si128", 2),
           "aesdec": ("_mm_aesdec_si128", 2),
           "aesenclast": ("_mm_aesenclast_si128", 2),
           "aesdeclast": ("_mm_aesdeclast_si128", 2)}

# 窄到宽的元素扩展：每个输出元素是源里一个更窄的元素，符号扩展或零扩展。
# 名字里的两个字母就是宽度——b/w/d/q 是 1/2/4/8 字节。
_WIDTH_LETTER = {1: "b", 2: "w", 4: "d", 8: "q"}
VEC_EXTEND = {}
for _s, _d in ((1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (4, 8)):
    for _k, _sign in (("s", True), ("z", False)):
        VEC_EXTEND[f"pmov{_k}x{_WIDTH_LETTER[_s]}{_WIDTH_LETTER[_d]}"] = (_s, _d, _sign)
del _s, _d, _k, _sign

# 饱和打包：两个源的元素各占输出的一半（第一个源在低半），元素被夹到目标宽度里。
# 四个形态就是"有符号还是有符号"与"窄多少"的两种组合。
VEC_PACK = {"packsswb": (2, 1, True), "packssdw": (4, 2, True),
            "packuswb": (2, 1, False), "packusdw": (4, 2, False)}

# 掩码访存：每个元素**自己的最高位**决定这一格走不走（SDM, VPMASKMOVD：`IF
# SRC1[i+31] THEN ... FI`），所以一条指令里既读又跳过，不能用"读一整块、写一整块"
# 的写法。数字是元素宽度。载入形式没选中的格子清零，存储形式没选中的格子**内存
# 保持原样**——这两个不对称，是这条指令被单独实现的原因。
VEC_MASKMOVE = {"pmaskmovd": 4, "pmaskmovq": 8, "maskmovps": 4, "maskmovpd": 8}

# 交替加减：偶数元素加、奇数元素减（`addsubps` 按 32 位、`addsubpd` 按 64 位）。
# 一条指令里两件事，所以不进 `VEC_FLOAT`。
VEC_ADDSUB = {"addsubps": 4, "addsubpd": 8}

# 逐元素乘法。宽的那半去哪一位、结果是有符号还是无符号，都由名字决定：
# `pmullw` 取低 16 位，`pmulhw` 取高 16 位（有符号），`pmulhuw` 取高 16 位（无符号），
# `pmulld` 取低 32 位。`pmuludq`/`pmaddwd` 的形状不同，单独处理。
VEC_MUL = {"pmullw": (2, "low"), "pmulhw": (2, "high_s"),
           "pmulhuw": (2, "high_u"), "pmulld": (4, "low")}

# 逐元素取绝对值：有符号进，同宽无符号出。
VEC_ABS = {"pabsb": 1, "pabsw": 2, "pabsd": 4}

# 按索引表逐个取元素（`vpermps` 是浮点那一份，`vpermd` 是整数那一份，规则相同）：
# 第 i 个输出取第 (索引[i] & 元素数-1) 个输入。与立即数置换的区别是索引是**数据**。
VEC_GATHER = {"permps": 4, "permd": 4}

# 每个元素自己的移位量（AVX2 的 `vpsrlvd` 一族）。移位数不是立即数，是**每个格子
# 一个**，所以一条指令里可以有四个不同的移位量。SDM 的规则与标量移位不同：位数
# 超过元素宽度不是"按位与"，而是结果直接饱和——逻辑移位得 0，算术移位得符号位。
VEC_VSHIFT = {"psrlvd": (4, ">>"), "psravd": (4, "SAR"), "psllvd": (4, "<<"),
              "psrlvq": (8, ">>"), "psllvq": (8, "<<")}

# 逐元素取小/取大。符号是**指令**的：`pminsd` 按有符号比，`pminud` 按无符号比，
# 两边的机器码只差一个前缀，结果却在 0x80000000 那一格分道扬镳。
VEC_MINMAX = {}
for _k in ("min", "max"):
    for _w, _n in ((1, "b"), (2, "w"), (4, "d")):
        for _s in ("s", "u"):
            VEC_MINMAX[f"p{_k}{_s}{_n}"] = (_w, _s == "s", _k)
del _k, _w, _n, _s



# 手写分支里的那几族：`lift_one` 里有些形态是逐条写的，不查表。它们同样是向量
# 指令、同样不碰 EFLAGS，但不在任何一张表里，于是 `branch` 那侧把它们当成"改了
# 标志"的指令——`shl`/`mov` 之外还会出现 `vcvtdq2ps` 这种根本不可能动标志的名字
# 把前面那条比较作废。这张表就是补上那个缺口：一张"这个升举器认得哪些 mnemonic"
# 的清单，和下面几张表并列，而不是让 `branch` 再抄一遍。
VEC_ADHOC = {
    "pshufd", "pshufb", "pshuflw", "pshufhw", "permilps",
    "pextrb", "pextrw", "pextrd", "pextrq",
    "pinsrb", "pinsrw", "pinsrd", "pinsrq",
    "cvtss2sd", "cvtsd2ss", "cvtdq2ps", "cvttps2dq", "cvttpd2dq",
    "cvtps2dq", "cvtpd2dq",
    "cvtps2ph", "cvtph2ps", "cvtsi2ss", "cvtsi2sd",
    "cvttss2si", "cvttsd2si",
    "inserti128", "insertf128", "extracti128", "extractf128",
    "hsubps", "hsubpd", "movlhps", "movhlps", "extractps",
    "movlps", "movhps", "movlpd", "movhpd", "movddup",
    "pslldq", "psrldq", "perm2f128", "perm2i128", "palignr",
    "aesimc", "aesenc", "aesdec", "aesenclast", "aesdeclast",
}

VEC_STEMS = (set(VEC_MOVE) | set(VEC_LOGIC) | set(VEC_FLOAT) | set(VEC_BROADCAST)
             | set(VEC_BLEND) | set(VEC_CMP_INT) | set(VEC_UNPACK)
             | set(VEC_APPROX) | set(FMA)
             | set(VEC_INT_ARITH) | set(VEC_HADD) | set(VEC_SHIFT)
             | set(VEC_SIGN) | set(VEC_DUP) | set(VEC_SQRT)
             | set(VEC_ROUND) | set(VEC_EXTEND) | set(VEC_PACK)
             | set(VEC_MASKMOVE)
             | set(VEC_VSHIFT) | set(VEC_MINMAX) | set(VEC_GATHER)
             | set(VEC_ADDSUB) | set(VEC_MUL) | set(VEC_ABS)
             | VEC_ADHOC
             | {"cmpps", "cmppd", "cmpss", "cmpsd", "shufps", "shufpd",
                "shufss", "shufsd", "blendvps", "blendvpd", "pblendvb",
                "pmovmskb", "movmskps", "movmskpd", "cvtsi2ss", "cvtsi2sd",
                "insertps", "cvttss2si", "cvttsd2si", "movddup",
                "movss", "movsd", "movd", "movq", "vzeroupper", "vzeroall"})



# C the emitted code leans on, kept in one place because two programs compile
# that code: this module's own check and the differential harness. A helper
# defined in only one of them is a link error in the other, found late.
HELPERS = [
    "#include <windows.h>",
    "// `#DE` 是这个移植唯一需要自己发的故障，而且必须用**调用**发，不能指望把一次",
    "// 除法留在原地：`_udiv128` 是纯 intrinsic，`if (q > 0xffff) q = _udiv128(d, 0, d,",
    "// &rem)` 会被 /O1 if-convert 成无条件执行——商没越界的那一支也跟着崩。这是",
    "// 本仓库第一次踩到**用一次会崩的指令当控制流**的坑：编译器有把纯运算搬出分支的",
    "// 自由，而故障正是它搬走之后才发生的。`RaiseException` 是一次真有副作用的调用，",
    "// 折不动；异常码 0xC0000094 就是硬件 `#DE` 在 Windows 上的那一个，收到它的处理器",
    "// 分不出区别。（换到 POSIX 上这条该写成 raise(SIGFPE)。）",
    "static __declspec(noreturn) void fault_divide(void) "
    "{ RaiseException(0xC0000094u, 0, 0, 0); }",
    "// `popcnt` 用逐位归并算，而不是 intrinsic：`__popcnt` 在没开 /arch:SSE4.2 时",
    "// 会变成一次运行库调用，链接不上；而这条式子对任何宽度都是精确的。",
    "static uint64_t popcount64(uint64_t x) {",
    "    x = x - ((x >> 1) & UINT64_C(0x5555555555555555));",
    "    x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));",
    "    x = (x + (x >> 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);",
    "    return (x * UINT64_C(0x0101010101010101)) >> 56;",
    "}",
    "// `tzcnt` and `lzcnt` are defined at zero — the count is the operand's",
    "// width — which `bsf`/`bsr` are not, so the intrinsic's failure return is",
    "// part of the answer rather than a case to refuse.",
    "static uint64_t count_trailing(uint64_t x, unsigned bits) {",
    "    unsigned long i;",
    "    if (x == 0) return bits;",
    "    _BitScanForward64(&i, x);",
    "    return i;",
    "}",
    "static uint64_t count_leading(uint64_t x, unsigned bits) {",
    "    unsigned long i;",
    "    if (x == 0) return bits;",
    "    _BitScanReverse64(&i, x);",
    "    return bits - 1 - i;",
    "}",
]


def weave(index, width, chunk, picks, insn, vex):
    """Copy chunk `picks[n]` of its source (`op`, chunk number) to chunk `n`.

    Every read is hoisted into a local before any write. The destination is
    almost always one of the sources — `vblendps xmm0, xmm0, xmm1, 8` — and a
    per-chunk copy would overwrite a chunk a later one still has to read.
    """
    bits = chunk * 8
    lines = ["{"]
    for n, (op, k) in enumerate(picks):
        lines.append(f"    uint{bits}_t w{n} = "
                     f"{element(op, k, f'uint{bits}_t', chunk, insn)};")
    for n in range(len(picks)):
        lines.append(f"    memcpy((char *){lane(index)} + {chunk * n}, &w{n}, {chunk});")
    if width == 16 and vex:
        lines += ["    " + z for z in upper_zeroed(index)]
    lines.append("}")
    return lines



def vec(op):
    """(index, bytes) for a vector register operand."""
    if op.type != capstone.x86.X86_OP_REG:
        raise Refused("not a vector register")
    name = MD.reg_name(op.reg)
    if name in VEC:
        return VEC[name]
    raise Refused(name)


def is_vec(op):
    return op.type == capstone.x86.X86_OP_REG and MD.reg_name(op.reg) in VEC


def lane(index):
    return f"&v[{index}][0]"


def src_vec(op, insn):
    """A `__m128i` value for a vector operand, wherever it lives.

    整条指令的 intrinsic 形态（AES 轮、按 MXCSR 舍入的转换）要的是 128 位值，
    而不是逐元素读法。`_mm_loadu_si128` 恰好是"读 16 字节、不要求对齐"——与这些
    指令对内存操作数的要求一样。
    """
    if op.type == capstone.x86.X86_OP_MEM:
        return f"_mm_loadu_si128((const __m128i *)(uintptr_t){addr(op, insn)})"
    return f"_mm_loadu_si128((const __m128i *){lane(vec(op)[0])})"


def lane_at(op, n, insn):
    """Lane `n` of a 64-bit-lane view of an operand, as a C expression."""
    if op.type == capstone.x86.X86_OP_MEM:
        return f"(*(const uint64_t *)(uintptr_t)({addr(op, insn)} + {8 * n}))"
    index, _ = vec(op)
    return f"v[{index}][{n}]"


def lane_ptr(op, n, insn):
    """A pointer to 64-bit lane `n` of an operand — for byte-wise readers.

    The plural of `lane_at`: that one is the lane's *value*, this one is where
    it lives, which is what a loop over the bytes of a register file entry
    needs. Casting the value to a pointer instead would make the register's
    contents the address, and the first read would fault.
    """
    if op.type == capstone.x86.X86_OP_MEM:
        return f"(const uint8_t *)(uintptr_t)({addr(op, insn)} + {8 * n})"
    index, _ = vec(op)
    return f"(const uint8_t *)&v[{index}][{n}]"


def upper_zeroed(index):
    """The bits a VEX-encoded 128-bit form clears."""
    return [f"v[{index}][2] = 0;", f"v[{index}][3] = 0;"]


def element(op, n, ctype, step, insn):
    """The `n`th element of an operand, typed, as a C expression."""
    if op.type == capstone.x86.X86_OP_MEM:
        return f"*(const {ctype} *)(uintptr_t)({addr(op, insn)} + {step * n})"
    return f"*(const {ctype} *)((const char *){lane(vec(op)[0])} + {step * n})"


# The scalar float comparisons. They write only the flags — no register, no
# memory — so the state they change lives in `branch.py`, which is where the
# flags are tracked. Listed here so `lift_one` does not refuse them.
FLOAT_COMPARE = {"ucomiss": 4, "ucomisd": 8, "comiss": 4, "comisd": 8,
                 "vucomiss": 4, "vucomisd": 8, "vcomiss": 4, "vcomisd": 8}


def float_operand(op, size, insn):
    """The scalar float an operand names, as a C expression."""
    ctype = "float" if size == 4 else "double"
    if op.type == capstone.x86.X86_OP_MEM:
        return f"(*(const {ctype} *)(uintptr_t){addr(op, insn)})"
    return f"(*(const {ctype} *){lane(vec(op)[0])})"


# Instructions that name their operand implicitly: the accumulator and rdx for
# the sign-extension group. They have no `ops`, so the guard above has to let
# them through to their own handler.
ZERO_OPERAND = {"cdqe", "cdq", "cqo", "cbw", "cwde", "cwd", "leave", "lahf", "sahf"}


def cmp_form(stem):
    """(predicate, packed, ctype) for a compare named by its predicate, or None.

    `cmpps xmm0, xmm1, 3` carries the predicate as an immediate and capstone
    reports it that way; `vcmpeqps` is the same instruction with the predicate
    folded into the name and the immediate no longer reported at all. Both have
    to reach the same place, so the name is read back apart.
    """
    if not stem.startswith("cmp"):
        return None
    for tail in ("ps", "pd", "ss", "sd"):
        if stem.endswith(tail) and stem[3:-2] in CMP_PREDICATE:
            ctype = "double" if tail in ("pd", "sd") else "float"
            return CMP_PREDICATE[stem[3:-2]], tail in ("ps", "pd"), ctype
    return None


def vec_line(insn):
    """C statements for a vector instruction, or None if it is not one."""
    m = insn.mnemonic
    ops = insn.operands
    if m in ("vzeroupper", "vzeroall"):
        # Bits 255:128 of every register, cleared. Readable later in the same
        # function, so it is emitted rather than skipped.
        return [f"v[{i}][2] = v[{i}][3] = 0;" for i in range(16)]
    if not ops:
        return None
    vex = m.startswith("v")
    stem = m[1:] if vex else m

    if stem in VEC_MOVE and len(ops) == 2:
        # A byte copy at the register's width. `movaps` and `movups` differ only
        # in whether the address must be aligned, and nothing here is modelled
        # as an address that could fail to be, so both are `memcpy`.
        dst, source = ops[0], ops[1]
        if is_vec(dst):
            index, size = vec(dst)
            if is_vec(source):
                lines = [f"memcpy({lane(index)}, {lane(vec(source)[0])}, {size});"]
            else:
                lines = [f"memcpy({lane(index)}, (const void *)(uintptr_t)"
                         f"{addr(source, insn)}, {size});"]
            if size == 16 and vex:
                lines += upper_zeroed(index)
            return lines
        if is_vec(source):
            index, size = vec(source)
            return [f"memcpy((void *)(uintptr_t){addr(dst, insn)}, {lane(index)}, {size});"]
        raise Refused(f"{m} form")

    if stem in ("movss", "movsd"):
        size = 4 if stem == "movss" else 8
        keep = ONES[size * 8]
        if not is_vec(ops[0]):
            if len(ops) != 2 or not is_vec(ops[1]) or ops[0].type != capstone.x86.X86_OP_MEM:
                raise Refused(f"{m} store")
            index, _ = vec(ops[1])
            return [f"*(uint{size * 8}_t *)(uintptr_t){addr(ops[0], insn)}"
                    f" = (uint{size * 8}_t)v[{index}][0];"]
        index, width = vec(ops[0])
        if len(ops) == 3:
            # VEX three-operand: the low field from src2, bits 127:32 from src1.
            lines = [f"v[{index}][0] = ({lane_at(ops[1], 0, insn)} & ~{keep})"
                     f" | ({lane_at(ops[2], 0, insn)} & {keep});",
                     f"v[{index}][1] = 0;"]
            if width == 16:
                lines += upper_zeroed(index)
            return lines
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        if ops[1].type == capstone.x86.X86_OP_MEM:
            # A scalar load zeroes everything above the bytes it read.
            lines = [f"v[{index}][0] = (*(const uint{size * 8}_t *)"
                     f"(uintptr_t){addr(ops[1], insn)});", f"v[{index}][1] = 0;"]
            if width == 16:
                lines += upper_zeroed(index)
            return lines
        # A register source: under VEX only the low field survives, but the
        # legacy encoding of the same mnemonic copies all 128 bits. The two look
        # identical here, so the legacy one is refused rather than guessed at —
        # `movaps` is what a compiler emits for a full copy anyway.
        if not vex:
            raise Refused(f"legacy {m} register")
        other, _ = vec(ops[1])
        lines = [f"v[{index}][0] = (v[{other}][0] & {keep});", f"v[{index}][1] = 0;"]
        if width == 16:
            lines += upper_zeroed(index)
        return lines

    if stem in ("movd", "movq") and len(ops) == 2:
        if is_vec(ops[0]) and is_vec(ops[1]):
            return [f"memcpy({lane(vec(ops[0])[0])}, {lane(vec(ops[1])[0])}, 16);"]
        bits = 32 if stem == "movd" else 64
        if is_vec(ops[0]):
            index, width = vec(ops[0])
            if ops[1].type == capstone.x86.X86_OP_MEM:
                value = f"(*(const uint{bits}_t *)(uintptr_t){addr(ops[1], insn)})"
            else:
                value = src(ops[1], insn)
            lines = [f"v[{index}][0] = {value};", f"v[{index}][1] = 0;"]
            if width == 16:
                lines += upper_zeroed(index)
            return lines
        if is_vec(ops[1]):
            index, _ = vec(ops[1])
            return [wr(ops[0], f"v[{index}][0]", insn)]
        raise Refused(f"{m} form")

    # 立即数每两位选一个双字；256 位形式在各自 128 位通道内重排。`vpermilps`
    # 的立即数形式与 `pshufd` 是同一件事（变量形式是另一个操作数，不走这里）。
    if stem in ("pshufd", "permilps") and len(ops) == 3 and \
            ops[2].type == capstone.x86.X86_OP_IMM:
        if len(ops) != 3 or ops[2].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, source, sel = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        lines = ["{"]
        pt = (f"(const uint8_t *)(uintptr_t){addr(source, insn)}"
              if source.type == capstone.x86.X86_OP_MEM
              else f"(const uint8_t *){lane(vec(source)[0])}")
        lines.append(f"    const uint8_t *ps = {pt};")
        for u in range(max(1, width // 16)):
            for n in range(4):
                pick = (sel.imm >> (2 * n)) & 3
                lines.append(f"    uint32_t s{u}{n}; memcpy(&s{u}{n}, ps + "
                             f"{u * 16 + 4 * pick}, 4);")
        for u in range(max(1, width // 16)):
            for n in range(4):
                lines.append(f"    memcpy((char *){lane(index)} + "
                             f"{u * 16 + 4 * n}, &s{u}{n}, 4);")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem == "permilps" and len(ops) == 3 and \
            (is_vec(ops[2]) or ops[2].type == capstone.x86.X86_OP_MEM):
        # 变量形式：第三操作数**逐元素**给控制字，每格的低两位选源里同一通道的
        # 哪一格。它和立即数形式是两条指令——立即数一条控制字管全部，这条每格
        # 各管自己，所以不能共用那段代码。
        dst, a, b = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        pt = (f"(const uint8_t *)(uintptr_t){addr(a, insn)}"
              if a.type == capstone.x86.X86_OP_MEM
              else f"(const uint8_t *){lane(vec(a)[0])}")
        lines = ["{", f"    const uint8_t *pa = {pt};"]
        k = 0
        for u in range(width // 16):
            for n in range(4):
                ctrl = element(b, u * 4 + n, "uint32_t", 4, insn)
                lines.append(f"    uint32_t c{k} = ({ctrl}) & 3;")
                lines.append(f"    uint32_t s{k}; memcpy(&s{k}, pa + {u * 16} + 4 * c{k}, 4);")
                k += 1
        for n in range(k):
            lines.append(f"    memcpy((char *){lane(index)} + {4 * n}, &s{n}, 4);")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem == "palignr":
        # 把两个源按 `src1:src2` 拼成 32 字节再右移 imm 个**字节**，取低 16 字节。
        # 256 位形式在**每个 128 位通道内部**各拼各的，不是把两个通道连起来。
        # imm 超过 31 时结果是全零——不是把计数取模。
        if len(ops) != 4 or ops[3].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, a, b = ops[0], ops[1], ops[2]
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        n = ops[3].imm & 0xFF
        lanes = width // 16

        def ptr(op):
            if op.type == capstone.x86.X86_OP_MEM:
                return f"(const void *)(uintptr_t){addr(op, insn)}"
            return f"(const void *){lane(vec(op)[0])}"

        lines = ["{", f"    char t[{32 * lanes}];"]
        for u in range(lanes):
            lines.append(f"    memcpy(t + {32 * u}, {ptr(b)}, 16);")
            lines.append(f"    memcpy(t + {32 * u + 16}, {ptr(a)}, 16);")
        for u in range(lanes):
            if n < 32:
                lines.append(f"    memcpy((char *){lane(index)} + {16 * u}, "
                             f"t + {32 * u + n}, 16);")
            else:
                lines.append(f"    memset((char *){lane(index)} + {16 * u}, 0, 16);")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem == "pextrb":
        # 抽一个字节出来。目标是通用寄存器时零扩展到 32 位，`wr` 负责这件事。
        if len(ops) != 3 or ops[2].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, source, sel = ops
        index, _ = vec(source)
        at = f"((const uint8_t *){lane(index)})[{sel.imm}]"
        return [wr(dst, at, insn)]

    if stem == "pextrw":
        # 同理，两字节一个元素；抽出来的 16 位写进 32 位寄存器时是零扩展。
        if len(ops) != 3 or ops[2].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, source, sel = ops
        index, _ = vec(source)
        at = (f"(*(const uint16_t *)((const char *){lane(index)} + {2 * sel.imm}))")
        return [wr(dst, at, insn)]

    if stem in ("pinsrb", "pinsrw", "pinsrd", "pinsrq"):
        # 把一个元素塞进第 imm 个位置，其余原样保留（VEX 的第二个源就是"其余"）。
        width_of = {"pinsrb": 1, "pinsrw": 2, "pinsrd": 4, "pinsrq": 8}[stem]
        if len(ops) != 4 or ops[3].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, base, value, sel = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        lines = ["{"]
        if vec(base)[0] != index:
            lines.append(f"    memcpy({lane(index)}, {lane(vec(base)[0])}, {width}u);")
        got = (f"(*(const uint{width_of * 8}_t *)(uintptr_t){addr(value, insn)})"
               if value.type == capstone.x86.X86_OP_MEM
               else f"(uint{width_of * 8}_t){src(value, insn)}")
        lines.append(f"    {{ uint{width_of * 8}_t w = {got}; "
                     f"memcpy((char *){lane(index)} + {width_of * sel.imm}, &w, "
                     f"{width_of}); }}")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("pextrd", "pextrq"):
        # 四字节/八字节一个元素；写进 64 位寄存器时窄的那个零扩展。
        if len(ops) != 3 or ops[2].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, source, sel = ops
        step = 8 if stem == "pextrq" else 4
        index, _ = vec(source)
        at = (f"(*(const uint{step * 8}_t *)((const char *){lane(index)} + "
              f"{step * sel.imm}))")
        return [wr(dst, at, insn)]

    if stem in ("cvtss2sd", "cvtsd2ss"):
        # 单元素精度转换。低一个元素是换算的结果，`127:32` 按 VEX 的规矩取自
        # src1（两操作数的 legacy 形式则留在目标自己身上）。
        if len(ops) == 3:
            dst, a, source = ops
        elif len(ops) == 2:
            dst, a, source = ops[0], None, ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        if source.type == capstone.x86.X86_OP_MEM:
            sp = f"(const void *)(uintptr_t){addr(source, insn)}"
        else:
            sp = f"(const void *){lane(vec(source)[0])}"
        if stem == "cvtss2sd":
            step, src_t, dst_t = 8, "float", "double"
        else:
            step, src_t, dst_t = 4, "double", "float"
        lines = ["{",
                 f"    {dst_t} d = ({dst_t})(*(const {src_t} *){sp});"]
        if a is not None and vec(a)[0] != index:
            lines.append(f"    memcpy({lane(index)}, {lane(vec(a)[0])}, {width}u);")
        lines.append(f"    memcpy({lane(index)}, &d, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem == "pshufb":
        # 逐字节查表：掩码字节的最高位为 1 就清零，否则取通道内第 (掩码 & 0x0F)
        # 个字节。源是第二个操作数，掩码是第三个；256 位形式在各自 128 位通道内查。
        if len(ops) == 3:
            dst, source, mask = ops
        elif len(ops) == 2:
            dst, source, mask = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        lines = ["{"]

        def ptr(op, name):
            at = (f"(const uint8_t *)(uintptr_t){addr(op, insn)}"
                  if op.type == capstone.x86.X86_OP_MEM
                  else f"(const uint8_t *){lane(vec(op)[0])}")
            lines.append(f"    const uint8_t *{name} = {at};")

        ptr(source, "ps")
        ptr(mask, "pm")
        # 先算出全部 32 个字节再写：目标和源常常是同一个寄存器。
        for u in range(max(1, width // 16)):
            for k in range(16):
                lines.append(f"    uint8_t b{u}_{k};")
        for u in range(max(1, width // 16)):
            for k in range(16):
                mk = f"pm[{u * 16 + k}]"
                lines.append(f"    b{u}_{k} = ({mk} & 0x80) ? 0 : "
                             f"ps[{u * 16} + ({mk} & 0x0F)];")
        for u in range(max(1, width // 16)):
            for k in range(16):
                lines.append(f"    memcpy((char *){lane(index)} + {u * 16 + k}, "
                             f"&b{u}_{k}, 1);")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("inserti128", "insertf128"):
        # 把一个 128 位通道塞进第 imm 位，另一个通道保留（VEX 的第二个源就是它）。
        if len(ops) != 4 or ops[3].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, base, value, sel = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        chunk = sel.imm & 1
        src_at = (f"(const void *)(uintptr_t){addr(value, insn)}"
                  if value.type == capstone.x86.X86_OP_MEM
                  else f"(const void *){lane(vec(value)[0])}")
        # 插入的值可能就住在目标寄存器里（`vinserti128 ymm7, ymm8, xmm7, 1`），
        # 所以先把那 16 字节搬到栈上，再去覆盖目标。
        lines = ["{", f"    uint8_t t[16]; memcpy(t, {src_at}, 16);"]
        if vec(base)[0] != index:
            lines.append(f"    memcpy({lane(index)}, {lane(vec(base)[0])}, {width}u);")
        lines.append(f"    memcpy((char *){lane(index)} + {16 * chunk}, t, 16);")
        lines.append("}")
        return lines

    if stem in ("pshuflw", "pshufhw"):
        # 只重排低半或高半的四个字，另一半原样保留；256 位形式各自 128 位通道内做。
        if len(ops) != 3 or ops[2].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, source, sel = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        base = 0 if stem == "pshuflw" else 8
        lines = ["{"]
        pt = (f"(const uint8_t *)(uintptr_t){addr(source, insn)}"
              if source.type == capstone.x86.X86_OP_MEM
              else f"(const uint8_t *){lane(vec(source)[0])}")
        lines.append(f"    const uint8_t *ps = {pt};")
        # 先把四个字全读出来再写：源与目标常常是同一个寄存器。
        units = max(1, width // 16)
        for u in range(units):
            for n in range(4):
                pick = (sel.imm >> (2 * n)) & 3
                lines.append(f"    uint16_t w{u}_{n}; memcpy(&w{u}_{n}, ps + "
                             f"{u * 16 + base + 2 * pick}, 2);")
        lines.append(f"    memcpy((char *){lane(index)}, ps, {width}u);")
        for u in range(units):
            for n in range(4):
                lines.append(f"    memcpy((char *){lane(index)} + "
                             f"{u * 16 + base + 2 * n}, &w{u}_{n}, 2);")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("hsubps", "hsubpd"):
        # 水平相减：每个 128 位通道里，低半的元素减相邻的高半元素。
        wide = 8 if stem.endswith("pd") else 4
        ctype = "double" if wide == 8 else "float"
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        per = 16 // wide                 # 每个 128 位通道的元素个数
        lines = ["{"]
        for u in range(max(1, width // 16)):
            for k in range(per // 2):
                x = element(a, u * per + 2 * k, ctype, wide, insn)
                y = element(a, u * per + 2 * k + 1, ctype, wide, insn)
                lines.append(f"    {ctype} d{u}_{k} = ({x}) - ({y});")
            for k in range(per // 2):
                x = element(b, u * per + 2 * k, ctype, wide, insn)
                y = element(b, u * per + 2 * k + 1, ctype, wide, insn)
                lines.append(f"    {ctype} e{u}_{k} = ({x}) - ({y});")
        for u in range(max(1, width // 16)):
            for k in range(per // 2):
                lines.append(f"    memcpy((char *){lane(index)} + "
                             f"{u * 16 + wide * k}, &d{u}_{k}, {wide});")
            for k in range(per // 2):
                lines.append(f"    memcpy((char *){lane(index)} + "
                             f"{u * 16 + wide * (per // 2 + k)}, &e{u}_{k}, {wide});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("extracti128", "extractf128"):
        # 抽出 128 位的通道；目标只能是 xmm 或内存。
        if len(ops) != 3 or ops[2].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} arity")
        dst, source, sel = ops
        index, _ = vec(source)
        chunk = sel.imm & 1
        src_at = f"((const char *){lane(index)} + {16 * chunk})"
        if is_vec(dst):
            d, dw = vec(dst)
            lines = [f"memcpy({lane(d)}, {src_at}, 16);"]
            if dw == 16 and vex:
                lines += upper_zeroed(d)
            return lines
        return [f"memcpy((void *)(uintptr_t){addr(dst, insn)}, {src_at}, 16);"]

    if stem in ("cvtps2ph", "cvtph2ps"):
        if len(ops) not in (2, 3):
            raise Refused(f"{m} arity")
        dst, source = ops[0], ops[1]
        # `vcvtps2ph m64, xmm, imm` 是这条指令最常见的形态（把一整个向量压成
        # 半精度存出去），目标不在寄存器里。内存目标只是换一个落点：写到
        # `pd + 2*n` 而不是寄存器里，没有"其余清零"这回事，因为内存只收到
        # 写进去的那几个字节。
        if is_vec(dst):
            index, width = vec(dst)
            base = f"(char *){lane(index)}"
        elif stem == "cvtps2ph" and dst.type == capstone.x86.X86_OP_MEM:
            index, width = None, 0
            base = f"(char *)(uintptr_t)({addr(dst, insn)})"
        else:
            raise Refused(f"{m} to memory")
        lines = ["{"]
        pt = (f"(const void *)(uintptr_t){addr(source, insn)}"
              if source.type == capstone.x86.X86_OP_MEM
              else f"(const void *){lane(vec(source)[0])}")
        if stem == "cvtps2ph":
            sel = ops[2] if len(ops) == 3 else None
            if sel is None or sel.type != capstone.x86.X86_OP_IMM:
                raise Refused(f"{m} without a rounding mode")
            if sel.imm & 4:
                # bit 2 让指令去读 MXCSR。这个移植不建模它，但可以不算：
                # `_MM_FROUND_CUR_DIRECTION` 编译出来就是 imm8 的 bit 2，
                # 也就是同一条指令、同一份 MXCSR，舍入模式逐位相同。
                mode = "_MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC"
            else:
                mode = f"{ROUND_MODE[sel.imm & 3]} | _MM_FROUND_NO_EXC"
            sw = vec(source)[1] if is_vec(source) else ops[1].size
            lines.append(f"    const char *ps = (const char *){pt};")
            for off in range(0, sw, 16):
                lines.append(f"    {{ __m128i h = _mm_cvtps_ph(_mm_loadu_ps("
                             f"(const float *)(ps + {off})), {mode}); "
                             f"memcpy({base} + {off // 2}, &h, 8); }}")
            # 只写一半的字节数，剩下的（xmm 目标的高 8 字节）指令是清零的。
            written = (sw // 16) * 8
            if width > written:
                lines.append(f"    memset({base} + {written}, 0, "
                             f"{width - written});")
        else:
            lines.append(f"    const char *ps = (const char *){pt};")
            for off in range(0, width, 16):
                lines.append(f"    _mm_storeu_ps((float *)({base} + {off}), "
                             f"_mm_cvtph_ps(_mm_loadu_si128((const __m128i *)"
                             f"(ps + {off // 2}))));")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_PACK:
        wide, narrow, signed = VEC_PACK[stem]
        if len(ops) == 3:
            dst, s1, s2 = ops
        elif len(ops) == 2:
            dst, s1, s2 = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        # 打包在 128 位的通道内部进行：256 位形式是两个通道各自打包，不是把两半
        # 拼起来再打。所以按 16 字节一段处理，每段 8 字节来自第一个源。
        unit = 16
        half = unit // 2
        per_source = half // narrow
        lo = -(1 << (narrow * 8 - 1)) if signed else 0
        hi = (1 << (narrow * 8 - 1)) - 1 if signed else (1 << (narrow * 8)) - 1
        stype = f"{'int' if signed else 'uint'}{narrow * 8}_t"
        lines = ["{"]
        for side, op in enumerate((s1, s2)):
            pt = (f"(const uint8_t *)(uintptr_t){addr(op, insn)}"
                  if op.type == capstone.x86.X86_OP_MEM
                  else f"(const uint8_t *){lane(vec(op)[0])}")
            lines.append(f"    const uint8_t *p{side} = {pt};")
        for u in range(max(1, width // unit)):
            for side in range(2):
                for n in range(per_source):
                    t = f"t{u}{side}{n}"
                    # 源一律按有符号读：`packuswb` 的源是 int16，负值饱和成 0，
                    # 按无符号读会把它当成 65535 而夹到 255。
                    lines.append(
                        f"    int64_t {t} = (int64_t)(*(const int{wide * 8}_t *)"
                        f"(p{side} + {u * unit + wide * n}));")
                    lines.append(f"    {t} = {t} < {lo} ? {lo} : "
                                 f"({t} > {hi} ? {hi} : {t});")
        # 全部读完再写：目标常常就是其中一个源（`vpackssdw ymm3, ymm4, ymm3` 是
        # 编译器最常发的形状），边读边写会让后读的元素拿到已经被盖掉的值。
        for u in range(max(1, width // unit)):
            for side in range(2):
                for n in range(per_source):
                    lines.append(
                        f"    {{ {stype} w{u}{side}{n} = ({stype})t{u}{side}{n}; "
                        f"memcpy((char *){lane(index)} + "
                        f"{u * unit + side * half + narrow * n}, "
                        f"&w{u}{side}{n}, {narrow}); }}")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_EXTEND:
        narrow, wide, signed = VEC_EXTEND[stem]
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        dst, source = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        lanes = width // wide
        if source.type == capstone.x86.X86_OP_MEM:
            ps = f"(const uint8_t *)(uintptr_t){addr(source, insn)}"
        else:
            ps = f"(const uint8_t *){lane(vec(source)[0])}"
        # 逐元素搬：源在内存时每个元素只读 `narrow` 字节，整寄存器地去读会越过它。
        lines = ["{", f"    const uint8_t *ps = {ps};"]
        for n in range(lanes):
            src_at = f"(*(const {'int' if signed else 'uint'}{narrow * 8}_t *)" \
                     f"(ps + {narrow * n}))"
            lines.append(f"    {'int' if signed else 'uint'}{wide * 8}_t w{n} = "
                         f"({'int' if signed else 'uint'}{wide * 8}_t){src_at};")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {wide * n}, &w{n}, {wide});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem == "movlhps" and len(ops) in (2, 3):
        # 低半来自第一个源，高半来自第二个源。legacy 的两操作数形式没有第二个源：
        # 低半留在目标自己身上，高半接源的低半。
        dst = ops[0]
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        lo = ops[1] if len(ops) == 3 else dst
        hi = ops[2] if len(ops) == 3 else ops[1]
        lines = ["{",
                 f"    uint64_t lo, hi;",
                 f"    memcpy(&lo, {lane_ptr(lo, 0, insn)}, 8);",
                 f"    memcpy(&hi, {lane_ptr(hi, 0, insn)}, 8);",
                 f"    memcpy({lane(index)}, &lo, 8);",
                 f"    memcpy((char *){lane(index)} + 8, &hi, 8);"]
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem == "extractps":
        if len(ops) != 3:
            raise Refused(f"{m} arity")
        dst, source, sel = ops
        if sel.type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} without a lane")
        index, _ = vec(source)
        lane_at_imm = f"(*(const uint32_t *)((const char *){lane(index)} + {4 * sel.imm}))"
        return [wr(dst, lane_at_imm, insn)]

    if stem in VEC_LOGIC:
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        index, width = vec(dst)
        op = VEC_LOGIC[stem]
        lines = []
        for n in range(width // 8):
            x, y = lane_at(a, n, insn), lane_at(b, n, insn)
            expr = f"(~{x}) & {y}" if op == "-" else f"({x}) {op} ({y})"
            lines.append(f"v[{index}][{n}] = {expr};")
        # Each lane reads before it writes, and the lanes it has not reached are
        # untouched, so a destination that is also a source stays correct.
        if width == 16 and vex:
            lines += upper_zeroed(index)
        return lines

    if stem in VEC_APPROX:
        which, packed = VEC_APPROX[stem]
        if len(ops) == 3:
            dst, a, source = ops
        elif len(ops) == 2:
            dst, a, source = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        if source.type == capstone.x86.X86_OP_MEM:
            srco = f"(const char *)(uintptr_t){addr(source, insn)}"
        else:
            srco = f"(const char *){lane(vec(source)[0])}"
        call = "_mm_rsqrt_ps" if which == "rsqrt" else "_mm_rcp_ps"
        scall = "_mm_rsqrt_ss" if which == "rsqrt" else "_mm_rcp_ss"
        lines = ["{"]
        if packed:
            lines.append(f"    const char *s = {srco};")
            lines.append(f"    for (unsigned off = 0; off < {width}u; off += 16u) {{")
            lines.append(f"        __m128 x = _mm_loadu_ps((const float *)(s + off));")
            lines.append(f"        _mm_storeu_ps((float *)((char *){lane(index)} + off), {call}(x));")
            lines.append("    }")
        else:
            # 标量形式只换低 32 位：`127:32` 来自另一个源（VEX 的三操作数形式）
            # 或留在目标自己身上（legacy 的两操作数形式）——与 `sqrtss` 同一条
            # 规则，所以这里照它的写法：整寄存器先搬过来，再盖掉低 4 字节。
            if a is not None and (a.type == capstone.x86.X86_OP_MEM
                                  or vec(a)[0] != index):
                if a.type == capstone.x86.X86_OP_MEM:
                    lines.append(f"    memcpy({lane(index)}, (const void *)(uintptr_t)"
                                 f"{addr(a, insn)}, {width}u);")
                else:
                    lines.append(f"    memcpy({lane(index)}, {lane(vec(a)[0])}, {width}u);")
            lines.append(f"    float d = _mm_cvtss_f32({scall}("
                         f"_mm_loadu_ps((const float *)({srco}))));")
            lines.append(f"    memcpy({lane(index)}, &d, 4);")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in FMA:
        kind, order, suffix = FMA[stem]
        if len(ops) != 3:
            raise Refused(f"{m} arity")
        dst, s1, s2 = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        ctype = "double" if suffix in ("pd", "sd") else "float"
        packed = suffix in ("ps", "pd")
        # 132 → DEST*SRC2+SRC1, 213 → SRC1*DEST+SRC2, 231 → SRC1*SRC2+DEST.
        first, second, addend = {"132": (dst, s2, s1),
                                 "213": (s1, dst, s2),
                                 "231": (s1, s2, dst)}[order]
        call = {"fmadd": "_mm_fmadd_", "fmsub": "_mm_fmsub_",
                "fnmadd": "_mm_fnmadd_", "fnmsub": "_mm_fnmsub_"}[kind] + suffix

        def at(op):
            if op.type == capstone.x86.X86_OP_MEM:
                return f"(const {ctype} *)(uintptr_t){addr(op, insn)}"
            return f"(const {ctype} *)((const char *){lane(vec(op)[0])})"

        lines = ["{"]
        vtype = "__m128" if ctype == "float" else "__m128d"
        if packed:
            load = "_mm_loadu_ps" if ctype == "float" else "_mm_loadu_pd"
            store = "_mm_storeu_ps" if ctype == "float" else "_mm_storeu_pd"
            lines.append(f"    const {ctype} *pa = {at(first)};")
            lines.append(f"    const {ctype} *pb = {at(second)};")
            lines.append(f"    const {ctype} *pc = {at(addend)};")
            # 256 位用两次 128 位：融合发生在每一个通道内部，两个半边的结果拼起来
            # 与一条 256 位指令逐位相同。三个源都在同一个偏移上先读完再写回，
            # 所以目标与某个源重合（编译器最常发的形态）也是对的。
            lines.append(f"    for (unsigned off = 0; off < {width}u; off += 16u) {{")
            lines.append(f"        {vtype} x = {load}((const {ctype} *)((const char *)pa + off));")
            lines.append(f"        {vtype} y = {load}((const {ctype} *)((const char *)pb + off));")
            lines.append(f"        {vtype} z = {load}((const {ctype} *)((const char *)pc + off));")
            lines.append(f"        {store}(({ctype} *)((char *){lane(index)} + off), "
                         f"{call}(x, y, z));")
            lines.append("    }")
        else:
            load = "_mm_load_ss" if ctype == "float" else "_mm_load_sd"
            poke = "_mm_cvtss_f32" if ctype == "float" else "_mm_cvtsd_f64"
            step = 4 if ctype == "float" else 8
            # 只读低一个元素：标量形式的 r/m 操作数在内存里只有 4/8 字节，
            # 整寄存器宽度地去读会越过它所在的页。
            lines.append(f"    {vtype} x = {load}({at(first)});")
            lines.append(f"    {vtype} y = {load}({at(second)});")
            lines.append(f"    {vtype} z = {load}({at(addend)});")
            lines.append(f"    {ctype} d = {poke}({call}(x, y, z));")
            # 标量形式只盖低 4/8 字节，`127:32` 留着目标自己的原值。这一条是硬件
            # 定的，不是从文档推的：`vfmadd132ss xmm3, xmm1, xmm1`、`vfmadd231ss
            # xmm1, xmm0, xmm0`、`vfmadd213ss xmm2, xmm3, xmm5` 三个用例里，结果
            # 的 `127:32` 都等于目标原值而与 SRC1 无关（片段差分把它们逐个指了出来）。
            lines.append(f"    memcpy({lane(index)}, &d, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_FLOAT:
        op, packed = VEC_FLOAT[stem]
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        index, width = vec(dst)
        ctype = "double" if stem.endswith(("sd", "pd")) else "float"
        step = 8 if ctype == "double" else 4
        lanes = width // step if packed else 1
        lines = ["{"]
        for n in range(lanes):
            x, y = element(a, n, ctype, step, insn), element(b, n, ctype, step, insn)
            if op == "MAX":
                expr = f"({x} > {y} ? {x} : {y})"
            elif op == "MIN":
                expr = f"({x} < {y} ? {x} : {y})"
            else:
                expr = f"({x}) {op} ({y})"
            lines.append(f"    {ctype} d{n} = {expr};")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &d{n}, {step});")
        if not packed:
            # A scalar form keeps the low lane and clears the rest of the
            # register, upper half included.
            lines.append(f"    memset((char *){lane(index)} + {step}, 0, {width - step});")
        elif width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_MASKMOVE:
        step = VEC_MASKMOVE[stem]
        if len(ops) != 3:
            raise Refused(f"{m} arity")
        if ops[0].type == capstone.x86.X86_OP_MEM:
            # 存储：掩码选中的格子才写回内存，其余原样不动。
            index, width = vec(ops[2])
            mask = vec(ops[1])[0]
            lines = ["{"]
            lines.append(f"const uint8_t *mk = (const uint8_t *){lane(mask)};")
            lines.append(f"const uint8_t *rd = (const uint8_t *){lane(index)};")
            lines.append(f"uint8_t *wr = (uint8_t *)(uintptr_t){addr(ops[0], insn)};")
            for n in range(width // step):
                off = step * n
                lines.append(f"if (mk[{off + step - 1}] & 0x80) "
                             f"memcpy(wr + {off}, rd + {off}, {step});")
            lines.append("}")
            return lines
        index, width = vec(ops[0])
        mask = vec(ops[1])[0]
        if ops[2].type == capstone.x86.X86_OP_MEM:
            at = f"(const uint8_t *)(uintptr_t){addr(ops[2], insn)}"
        else:
            at = f"(const uint8_t *){lane(vec(ops[2])[0])}"
        lines = ["{"]
        lines.append(f"const uint8_t *mk = (const uint8_t *){lane(mask)};")
        lines.append(f"const uint8_t *rd = {at};")
        for n in range(width // step):
            off = step * n
            lines.append(f"if (mk[{off + step - 1}] & 0x80) "
                         f"memcpy((char *){lane(index)} + {off}, rd + {off}, {step});")
            lines.append(f"else memset((char *){lane(index)} + {off}, 0, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_VSHIFT:
        step, op = VEC_VSHIFT[stem]
        if len(ops) != 3:
            raise Refused(f"{m} arity")
        dst, a, b = ops
        index, width = vec(dst)
        bits = step * 8
        ut, st = f"uint{bits}_t", f"int{bits}_t"
        lines = ["{"]
        for n in range(width // step):
            x = element(a, n, ut, step, insn)
            c = element(b, n, ut, step, insn)
            if op == "<<":
                expr = f"(({c}) > {bits - 1} ? 0 : ({ut})(({x}) << ({c})))"
            elif op == ">>":
                expr = f"(({c}) > {bits - 1} ? 0 : ({ut})(({x}) >> ({c})))"
            else:
                # 算术右移过界不是 0，是符号位的复制——先升到 64 位再移，
                # 32 位的符号在 int64_t 里还是那个符号。
                expr = (f"(({c}) > {bits - 1} ? ({ut})((int64_t)({st})({x}) >> {bits - 1})"
                        f" : ({ut})((int64_t)({st})({x}) >> ({c})))")
            lines.append(f"    {ut} d{n} = {expr};")
        for n in range(width // step):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &d{n}, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_MINMAX:
        step, signed, which = VEC_MINMAX[stem]
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        index, width = vec(dst)
        t = f"{'int' if signed else 'uint'}{step * 8}_t"
        rel = "<" if which == "min" else ">"
        lines = ["{"]
        for n in range(width // step):
            x = element(a, n, t, step, insn)
            y = element(b, n, t, step, insn)
            lines.append(f"    {t} m{n} = ({x} {rel} {y} ? {x} : {y});")
        for n in range(width // step):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &m{n}, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_ADDSUB:
        step = VEC_ADDSUB[stem]
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        index, width = vec(dst)
        t = "double" if step == 8 else "float"
        lines = ["{"]
        for n in range(width // step):
            # 偶数元素**减**、奇数元素加：SDM 的 `VADDSUBPS` 从 `DEST[31:0] :=
            # SRC1 - SRC2` 起头，名字里的 "addsub" 是"加了又减"，不是顺序。
            op = "-" if n % 2 == 0 else "+"
            x = element(a, n, t, step, insn)
            y = element(b, n, t, step, insn)
            lines.append(f"    {t} d{n} = ({x}) {op} ({y});")
        for n in range(width // step):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &d{n}, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_MUL:
        step, where = VEC_MUL[stem]
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        index, width = vec(dst)
        ut, st = f"uint{step * 8}_t", f"int{step * 8}_t"
        wide_u, wide_s = f"uint{step * 16}_t", f"int{step * 16}_t"
        lines = ["{"]
        for n in range(width // step):
            x = element(a, n, ut, step, insn)
            y = element(b, n, ut, step, insn)
            if where == "low":
                lines.append(f"    {ut} d{n} = ({ut})(({wide_u})({x}) * ({wide_u})({y}));")
            elif where == "high_u":
                lines.append(f"    {ut} d{n} = ({ut})((({wide_u})({x}) * ({wide_u})({y})) >> {step * 8});")
            else:
                lines.append(f"    {ut} d{n} = ({ut})((({wide_s})({st})({x}) * ({wide_s})({st})({y}))"
                             f" >> {step * 8});")
        for n in range(width // step):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &d{n}, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("pmuludq", "pmuldq", "pmaddwd"):
        # 三条"32 位元素两两乘、结果 64 位"的指令：前两条每 64 位通道只乘低 32 位
        # （无符号/有符号），`pmaddwd` 把一对乘积相加后落成 32 位。
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        index, width = vec(dst)
        lines = ["{"]
        for n in range(width // 8):
            x0 = element(a, 2 * n, "uint32_t", 4, insn)
            y0 = element(b, 2 * n, "uint32_t", 4, insn)
            if stem == "pmaddwd":
                x1 = element(a, 2 * n + 1, "uint32_t", 4, insn)
                y1 = element(b, 2 * n + 1, "uint32_t", 4, insn)
                lines.append(f"    int64_t s{n} = (int64_t)(int32_t)({x0}) * (int64_t)(int32_t)({y0})"
                             f" + (int64_t)(int32_t)({x1}) * (int64_t)(int32_t)({y1});")
                lines.append(f"    uint64_t d{n} = (uint32_t)s{n};")
            elif stem == "pmuludq":
                lines.append(f"    uint64_t d{n} = (uint64_t)({x0}) * (uint64_t)({y0});")
            else:
                lines.append(f"    uint64_t d{n} = (uint64_t)((int64_t)(int32_t)({x0})"
                             f" * (int64_t)(int32_t)({y0}));")
        for n in range(width // 8):
            lines.append(f"    v[{index}][{n}] = d{n};")
        lines.append("}")
        return lines

    if stem in VEC_ABS:
        step = VEC_ABS[stem]
        if len(ops) == 3:
            dst, a = ops[0], ops[2]
        elif len(ops) == 2:
            dst, a = ops
        else:
            raise Refused(f"{m} arity")
        index, width = vec(dst)
        st, ut = f"int{step * 8}_t", f"uint{step * 8}_t"
        lines = ["{"]
        for n in range(width // step):
            x = element(a, n, st, step, insn)
            lines.append(f"    {ut} d{n} = ({ut})(({x}) < 0 ? -({x}) : ({x}));")
        for n in range(width // step):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &d{n}, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_GATHER:
        step = VEC_GATHER[stem]
        if len(ops) != 3:
            raise Refused(f"{m} arity")
        dst, a, b = ops
        index, width = vec(dst)
        lanes = width // step
        t = f"uint{step * 8}_t"
        at = (f"(const uint8_t *)(uintptr_t){addr(a, insn)}" if a.type == capstone.x86.X86_OP_MEM
              else f"(const uint8_t *){lane(vec(a)[0])}")
        ct = (f"(const uint8_t *)(uintptr_t){addr(b, insn)}" if b.type == capstone.x86.X86_OP_MEM
              else f"(const uint8_t *){lane(vec(b)[0])}")
        lines = ["{",
                 # 先把整份源抄进临时变量：目标与源常常是同一个寄存器，边读边写会把
                 # 后面还要用的元素先覆盖掉。
                 f"    {t} sv[{lanes}], ci[{lanes}];",
                 f"    memcpy(sv, {at}, {width}u);",
                 f"    memcpy(ci, {ct}, {width}u);"]
        for n in range(lanes):
            lines.append(f"    {t} g{n} = sv[ci[{n}] & {lanes - 1}];")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &g{n}, {step});")
        lines.append("}")
        return lines

    if stem == "psadbw":
        # 每 8 个字节一组，组内逐字节求差的绝对值再求和——结果是一个 16 位的数，
        # 零扩展成 64 位占满这一组。`psadbw` 因此是"两行像素差多远"那一条。
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        index, width = vec(dst)
        pa = (f"(const uint8_t *)(uintptr_t){addr(a, insn)}" if a.type == capstone.x86.X86_OP_MEM
              else f"(const uint8_t *){lane(vec(a)[0])}")
        pb = (f"(const uint8_t *)(uintptr_t){addr(b, insn)}" if b.type == capstone.x86.X86_OP_MEM
              else f"(const uint8_t *){lane(vec(b)[0])}")
        lines = ["{",
                 f"    const uint8_t *pa = {pa}, *pb = {pb};"]
        for n in range(width // 8):
            lines.append(f"    uint64_t s{n} = 0;")
            lines.append(f"    for (unsigned k = 0; k < 8u; k++) {{")
            lines.append(f"        int d = (int)pa[{8 * n} + k] - (int)pb[{8 * n} + k];")
            lines.append(f"        s{n} += (uint64_t)(d < 0 ? -d : d);")
            lines.append("    }")
        for n in range(width // 8):
            lines.append(f"    v[{index}][{n}] = s{n};")
        lines.append("}")
        return lines

    if stem in ("permq", "permpd", "permilpd") and len(ops) == 3 \
            and ops[2].type == capstone.x86.X86_OP_IMM:
        # 立即数置换：`permq` 在整个 256 位里挑（四位、两位一个），`permilpd` 在
        # 各自 128 位通道里挑（两位、一位一个）——控制位按通道重复，这是它与
        # `permq` 唯一的区别。
        dst, source, sel = ops
        index, width = vec(dst)
        if source.type == capstone.x86.X86_OP_MEM:
            at = f"(const uint64_t *)(uintptr_t){addr(source, insn)}"
        else:
            at = f"(const uint64_t *){lane(vec(source)[0])}"
        lines = ["{", f"    const uint64_t *ps = {at};"]
        lanes = width // 8
        for n in range(lanes):
            if stem in ("permq", "permpd"):
                q = (sel.imm >> (2 * n)) & 3
            else:
                q = (n & ~1) + ((sel.imm >> (n & 1)) & 1)
            lines.append(f"    uint64_t q{n} = ps[{q}];")
        for n in range(lanes):
            lines.append(f"    v[{index}][{n}] = q{n};")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("cvtps2pd", "cvtpd2ps", "cvtdq2pd"):
        # 精度换算的打包形式：`cvtpd2ps` 把双精度压成单精度，目标低半装满、其余
        # 清零（256 位源压进 128 位目标时，上半也一起清）；`cvtps2pd` 反过来变宽，
        # 目标写满。
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        dst, source = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        sbytes = source.size if source.type == capstone.x86.X86_OP_MEM \
            else vec(source)[1]
        lines = ["{"]
        if stem == "cvtps2pd":
            for n in range(width // 8):
                x = element(source, n, "float", 4, insn)
                lines.append(f"    double d{n} = (double)({x});")
            for n in range(width // 8):
                lines.append(f"    memcpy((char *){lane(index)} + {8 * n}, &d{n}, 8);")
        elif stem == "cvtdq2pd":
            # 整数到双精度：32 位整数在 double 里是精确的，所以直接转换就是那条
            # 指令，不经过任何舍入假设。
            for n in range(width // 8):
                x = element(source, n, "int32_t", 4, insn)
                lines.append(f"    double d{n} = (double)({x});")
            for n in range(width // 8):
                lines.append(f"    memcpy((char *){lane(index)} + {8 * n}, &d{n}, 8);")
        else:
            for n in range(sbytes // 8):
                x = element(source, n, "double", 8, insn)
                lines.append(f"    float f{n} = (float)({x});")
            for n in range(sbytes // 8):
                lines.append(f"    memcpy((char *){lane(index)} + {4 * n}, &f{n}, 4);")
            if width > 4 * (sbytes // 8):
                lines.append(f"    memset((char *){lane(index)} + {4 * (sbytes // 8)}, 0, "
                             f"{width - 4 * (sbytes // 8)});")
        # VEX 的 128 位形式把上半清零，legacy 的留着不动——三条打包换算各差这一条
        # 规矩，片段差分把它们一起指了出来（x[2]/x[3] 三处）。
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("broadcastf128", "broadcasti128"):
        # 把 128 位原封不动抄进两个通道（不是把某个元素撒开，所以不进
        # `VEC_BROADCAST`）。源必须是内存，抄之前先读到临时变量里。
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        dst, source = ops
        index, width = vec(dst)
        at = f"(const uint8_t *)(uintptr_t){addr(source, insn)}"
        lines = ["{", f"    const uint8_t *ps = {at};",
                 "    uint64_t a0, a1;",
                 "    memcpy(&a0, ps, 8); memcpy(&a1, ps + 8, 8);"]
        for n in range(width // 16):
            lines.append(f"    v[{index}][{2 * n}] = a0; v[{index}][{2 * n + 1}] = a1;")
        lines.append("}")
        return lines

    if stem in VEC_BROADCAST:
        size = VEC_BROADCAST[stem]
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        index, width = vec(ops[0])
        source = ops[1]
        if source.type == capstone.x86.X86_OP_MEM:
            at = f"(const void *)(uintptr_t){addr(source, insn)}"
        else:
            at = f"(const void *){lane(vec(source)[0])}"
        lines = ["{"]
        if size == 1:
            lines.append("    uint64_t z = (uint64_t)(*(const uint8_t *)"
                         + at + ") * UINT64_C(0x0101010101010101);")
        elif size == 2:
            lines.append("    uint64_t z = (uint64_t)(*(const uint16_t *)"
                         + at + ") * UINT64_C(0x0001000100010001);")
        elif size == 4:
            lines.append(f"    uint32_t e; memcpy(&e, {at}, 4);")
            lines.append("    uint64_t z = ((uint64_t)e << 32) | e;")
        else:
            lines.append(f"    uint64_t z; memcpy(&z, {at}, 8);")
        for n in range(width // 8):
            lines.append(f"    v[{index}][{n}] = z;")
        lines.append("}")
        return lines

    if stem in VEC_BLEND:
        chunk = VEC_BLEND[stem]
        if len(ops) == 4:
            dst, a, b, sel = ops
        elif len(ops) == 3:
            dst, a, b, sel = ops[0], ops[0], ops[1], ops[2]
        else:
            raise Refused(f"{m} arity")
        if sel.type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} without an immediate mask")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        lanes = width // chunk
        picks = [(b if (sel.imm >> n) & 1 else a, n) for n in range(lanes)]
        return weave(index, width, chunk, picks, insn, vex)

    if stem in VEC_INT_ARITH:
        chunk, sym = VEC_INT_ARITH[stem]
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        bits = chunk * 8
        lanes = width // chunk
        unsigned = f"uint{bits}_t"
        lines = ["{"]
        for n in range(lanes):
            x = element(a, n, unsigned, chunk, insn)
            y = element(b, n, unsigned, chunk, insn)
            # The result is truncated back to the chunk width by the store, so
            # the wraparound is the instruction's, not the type's.
            lines.append(f"    uint{bits}_t w{n} = ({x}) {sym} ({y});")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {chunk * n}, &w{n}, {chunk});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_UNPACK:
        chunk, high = VEC_UNPACK[stem]
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        per_lane = 16 // chunk
        picks = []
        for base in range(0, width // chunk, per_lane):
            half = per_lane // 2
            for k in range(half):
                at = base + k + (half if high else 0)
                picks.append((a, at))
                picks.append((b, at))
        return weave(index, width, chunk, picks, insn, vex)

    if stem in ("shufps", "shufpd", "shufss", "shufsd"):
        if stem in ("shufss", "shufsd"):
            raise Refused(m)
        chunk = 4 if stem == "shufps" else 8
        sel = ops[-1]
        if sel.type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} without an immediate")
        if len(ops) == 4:
            dst, a, b = ops[0], ops[1], ops[2]
        elif len(ops) == 3:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        per_lane = 16 // chunk
        picks = []
        for base in range(0, width // chunk, per_lane):
            for n in range(per_lane):
                # The low half of the lane comes from src1, the high half from
                # src2, and the immediate names one source chunk per output.
                shift = n * (2 if chunk == 4 else 1)
                which = (sel.imm >> shift) & (3 if chunk == 4 else 1)
                picks.append((a if n < per_lane // 2 else b, base + which))
        return weave(index, width, chunk, picks, insn, vex)

    if stem in ("blendvps", "blendvpd", "pblendvb"):
        # A mask in a register: bit 31 (or 63) of each mask chunk selects. No
        # immediate, so the choice is a runtime one — but still a bitwise one,
        # which is what makes it expressible without a float model.
        chunk = 8 if stem == "blendvpd" else 4
        if len(ops) == 4:
            dst, a, b, mask = ops
        elif len(ops) == 3:
            dst, a, b, mask = ops[0], ops[0], ops[1], ops[2]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst) or not is_vec(mask):
            raise Refused(f"{m} form")
        index, width = vec(dst)
        lines = ["{"]
        for n in range(width // chunk):
            x = element(a, n, f"uint{chunk * 8}_t", chunk, insn)
            y = element(b, n, f"uint{chunk * 8}_t", chunk, insn)
            sel = element(mask, n, f"uint{chunk * 8}_t", chunk, insn)
            top = f"(UINT{chunk * 8}_C(1) << {chunk * 8 - 1})"
            lines.append(f"    uint{chunk * 8}_t w{n} = (({sel}) & {top}) ? ({y}) : ({x});")
        for n in range(width // chunk):
            lines.append(f"    memcpy((char *){lane(index)} + {chunk * n}, &w{n}, {chunk});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_HADD and VEC_HADD[stem]:
        step, ctype, op = VEC_HADD[stem]
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        per_lane = 16 // step
        half = per_lane // 2
        limits = None
        if op.endswith("SAT"):
            limits = {2: ("INT16_MIN", "INT16_MAX"),
                      4: ("INT32_MIN", "INT32_MAX")}.get(step)
            if limits is None:
                raise Refused(f"{m} saturate width")
        lines = ["{"]
        k = 0
        for base in range(0, width // step, per_lane):
            for source in (a, b):
                for g in range(half):
                    x = element(source, base + 2 * g, ctype, step, insn)
                    y = element(source, base + 2 * g + 1, ctype, step, insn)
                    if limits:
                        # 饱和：先按更宽的类型算，再夹回元素的取值范围。
                        wide = "int32_t" if step == 2 else "int64_t"
                        sign = "-" if op.startswith("-") else "+"
                        lines.append(f"    {wide} s{k} = ({wide})({x}) {sign} ({wide})({y});")
                        lines.append(f"    {ctype} d{k} = s{k} > {limits[1]} ? {limits[1]}"
                                     f" : (s{k} < {limits[0]} ? {limits[0]} : ({ctype})s{k});")
                    else:
                        lines.append(f"    {ctype} d{k} = ({ctype})(({x}) {op} ({y}));")
                    k += 1
        for n in range(k):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &d{n}, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("pmovmskb", "movmskps", "movmskpd"):
        if len(ops) != 2 or is_vec(ops[0]) or not is_vec(ops[1]):
            raise Refused(f"{m} form")
        index, width = vec(ops[1])
        if stem == "pmovmskb":
            count, chunk = width, 1
        elif stem == "movmskps":
            count, chunk = width // 4, 4
        else:
            count, chunk = width // 8, 8
        # The sign bit of each element, one per element, into a GPR — a bit
        # image of the register rather than a number out of it.
        lines = ["{", "    uint32_t k = 0;",
                 f"    for (int n = 0; n < {count}; n++)",
                 f"        k |= (uint32_t)(((const uint8_t *){lane(index)})[n * {chunk}"
                 f" + {chunk - 1}] >> 7) << n;",
                 f"    {wr(ops[0], '((uint32_t)k)', insn)}",
                 "}"]
        return lines

    if stem in ("cvtsi2ss", "cvtsi2sd"):
        ctype = "float" if stem == "cvtsi2ss" else "double"
        step = 4 if stem == "cvtsi2ss" else 8
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        if b.type == capstone.x86.X86_OP_MEM:
            bits = b.size * 8
            value = f"(*(const int{bits}_t *)(uintptr_t){addr(b, insn)})"
        elif b.type == capstone.x86.X86_OP_REG:
            bi, bits, shift = SUBS[MD.reg_name(b.reg)]
            raw = f"r[{bi}]" if shift == 0 else f"(r[{bi}] >> {shift})"
            # Signed, which is what the instruction converts; `src` would hand
            # back the unsigned view of the same bits.
            value = f"((int{bits}_t){raw})"
        else:
            raise Refused(f"{m} source")
        lines = ["{", f"    {ctype} d = ({ctype})({value});"]
        if len(ops) == 3 and vec(a)[0] != index:
            lines.append(f"    memcpy((char *){lane(index)} + {step}, "
                         f"(const char *){lane(vec(a)[0])} + {step}, {width - step});")
        lines.append(f"    memcpy((char *){lane(index)}, &d, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("pslldq", "psrldq"):
        # 按**字节**在整个 128 位里移，不是按元素——这是它与 `psllq`/`psrlq` 的
        # 全部区别：后者按 64 位元素各自移，元素之间不通。也没有 256 位形式。
        # 计数超过 15 就把整个寄存器清零，而不是取模。
        if len(ops) == 3:
            dst, source, count = ops
        elif len(ops) == 2:
            dst, source, count = ops[0], None, ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst) or count.type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} form")
        index, width = vec(dst)
        if width != 16:
            raise Refused(f"{m} width")
        n = count.imm & 0xFF
        lines = []
        if source is not None:
            # VEX 形式的目的寄存器不是源：先把源那一格搬过来再移位，否则移的是
            # 目的寄存器里上一条指令留下的东西。
            ptr = (f"(const void *)(uintptr_t)({addr(source, insn)})"
                   if source.type == capstone.x86.X86_OP_MEM
                   else f"(const void *){lane(vec(source)[0])}")
            lines.append(f"memcpy({lane(index)}, {ptr}, 16);")
        # `lane()` 是 `uint64_t *`：加 n 是加 8n 个字节，所以必须先在字节上算。
        p = f"(char *){lane(index)}"
        if n == 0:
            pass
        elif n > 15:
            lines.append(f"memset({p}, 0, 16);")
        elif stem == "pslldq":
            lines.append(f"memmove({p} + {n}, {p}, {16 - n});")
            lines.append(f"memset({p}, 0, {n});")
        else:
            lines.append(f"memmove({p}, {p} + {n}, {16 - n});")
            lines.append(f"memset({p} + {16 - n}, 0, {n});")
        if vex:
            lines += upper_zeroed(index)
        return lines

    if stem in ("perm2f128", "perm2i128"):
        # 两个源按 128 位拼成四格（a0 a1 b0 b1），低两位挑结果的低半、bit4-5 挑
        # 高半；bit3 与 bit7 各自把那半边清零。四格里没有"来自自己"这一项——
        # 目的寄存器是纯输出，所以可以先把结果拼在临时数组里再整块写回。
        # `vperm2i128` 与 `vperm2f128` 是同一条规则的两种类型：编码分开只是为了
        # 让整数代码不碰浮点单元，选择位的含义一模一样。
        if len(ops) != 4 or not is_vec(ops[0]) or \
                ops[3].type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} form")
        index, width = vec(ops[0])
        if width != 32:
            raise Refused(f"{m} width")
        mode = ops[3].imm & 0xFF

        def half(op, n):
            if op.type == capstone.x86.X86_OP_MEM:
                return f"(const void *)(uintptr_t)({addr(op, insn)} + {16 * n})"
            return f"(const void *)((const char *){lane(vec(op)[0])} + {16 * n})"

        lines = ["{", "    char t[32];"]
        for out_half, pick, zero in ((0, mode & 3, mode & 8),
                                     (1, (mode >> 4) & 3, mode & 0x80)):
            if zero:
                lines.append(f"    memset(t + {16 * out_half}, 0, 16);")
            else:
                op = ops[1] if pick < 2 else ops[2]
                lines.append(f"    memcpy(t + {16 * out_half}, {half(op, pick & 1)}, 16);")
        lines.append(f"    memcpy({lane(index)}, t, 32);")
        lines.append("}")
        return lines

    if stem in VEC_SHIFT:
        chunk, how = VEC_SHIFT[stem]
        if len(ops) == 3:
            dst, a, count = ops
        elif len(ops) == 2:
            dst, a, count = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        bits = chunk * 8
        if count.type == capstone.x86.X86_OP_IMM:
            shift = f"UINT64_C({count.imm & 0xFF})"
        elif count.type == capstone.x86.X86_OP_MEM:
            shift = f"(*(const uint64_t *)(uintptr_t){addr(count, insn)})"
        else:
            ci, _ = vec(count)
            shift = f"v[{ci}][0]"
        unsigned = f"uint{bits}_t"
        lines = ["{", f"    uint64_t s = {shift};"]
        for n in range(width // chunk):
            x = element(a, n, unsigned, chunk, insn)
            if how == "<<":
                expr = f"(s > {bits - 1} ? 0 : (uint{bits}_t)(({x}) << s))"
            elif how == ">>":
                expr = f"(s > {bits - 1} ? 0 : (uint{bits}_t)(({x}) >> s))"
            else:
                # An arithmetic shift fills with the sign bit, and a count past
                # the width leaves only sign bits — so it saturates at the width
                # instead of being masked, which is what the instruction does.
                expr = (f"(s > {bits - 1} ? ((int{bits}_t)({x}) >> {bits - 1})"
                        f" : ((int{bits}_t)({x}) >> s))")
            lines.append(f"    uint{bits}_t w{n} = {expr};")
        for n in range(width // chunk):
            lines.append(f"    memcpy((char *){lane(index)} + {chunk * n}, &w{n}, {chunk});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("movlps", "movhps", "movlpd", "movhpd"):
        # Half a register in or out of memory. `movhps` is the same operation at
        # offset 8, and the register form of each is `movlhps`/`movhlps`, which
        # is a half-width copy inside the file and is not this instruction.
        at = 8 if stem in ("movhps", "movhpd") else 0
        size = 8
        if len(ops) == 3:
            dst, a, source = ops
        elif len(ops) == 2:
            dst, a, source = ops[0], None, ops[1]
        else:
            raise Refused(f"{m} arity")
        if is_vec(dst):
            index, width = vec(dst)
            if source.type != capstone.x86.X86_OP_MEM:
                raise Refused(f"{m} register form")
            lines = ["{"]
            if len(ops) == 3 and a is not None and vec(a)[0] != index:
                lines.append(f"    memcpy({lane(index)}, {lane(vec(a)[0])}, {width});")
            elif vex and len(ops) == 2:
                # The VEX form names only one half, and the other half is
                # zeroed rather than kept — `vmovlps xmm1, m64` is not
                # `movlps xmm1, m64`, which leaves 127:64 alone.
                lines.append(f"    memset((char *){lane(index)} + {8 - at}, 0, {width - 8});")
            lines.append(f"    memcpy((char *){lane(index)} + {at}, "
                         f"(const void *)(uintptr_t){addr(source, insn)}, {size});")
            if width == 16 and vex:
                lines += ["    " + z for z in upper_zeroed(index)]
            lines.append("}")
            return lines
        if source is not None and is_vec(source) and dst.type == capstone.x86.X86_OP_MEM:
            index, _ = vec(source)
            return [f"memcpy((void *)(uintptr_t){addr(dst, insn)}, "
                    f"(const char *){lane(index)} + {at}, {size});"]
        raise Refused(f"{m} form")

    if stem == "cvttpd2dq":
        # 双精度截断成 32 位整数，只写目标的一半：两个 double 进去、两个字出来，
        # 剩下的字节清零。它不需要 MXCSR——截断没有舍入模式可选，这正是它与
        # `vcvtpd2dq` 的全部区别（后者按 MXCSR 舍入，所以仍然拒绝）。
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        dst, source = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        if width != 16:
            raise Refused(f"{m} width")
        sw = vec(source)[1] if is_vec(source) else ops[1].size
        lanes = sw // 8
        if lanes not in (2, 4):
            raise Refused(f"{m} source width")
        lines = ["{"]
        for n in range(lanes):
            # 越界与 NaN 都落到 INT32_MIN（"integer indefinite"），不是 C 强转的
            # 未定义行为。边界写成 double：写成 float 时 -2147483649.0 会舍成
            # -2^31，把合法值排除掉。
            lines.append(f"    double f{n} = {element(source, n, 'double', 8, insn)};")
            lines.append(f"    int32_t w{n} = (f{n} > -2147483649.0 && f{n} < 2147483648.0)"
                         f" ? (int32_t)f{n} : INT32_MIN;")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {4 * n}, &w{n}, 4);")
        rest = width - 4 * lanes
        if rest:
            lines.append(f"    memset((char *){lane(index)} + {4 * lanes}, 0, {rest});")
        if vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_AES:
        # 交给那条指令：AES 的一轮是 SubBytes/ShiftRows/MixColumns 的组合，用 C 写
        # 一遍就是重写一张只有硬件说了算的表（S 盒），而 `_mm_aesenc_si128` 就是
        # `aesenc`。非 VEX 形式只改低 128 位，VEX.128 连 255:128 一起清。
        #
        # 操作数个数不是"一族一个"：`aesimc` 一源，`aesenc`/`aesdec` 两源；VEX 的前缀
        # 形式又是三操作数（目的、源1、源2），而 legacy 的两操作数形式目的**同时是**
        # 源1。所以先看个数再决定谁是谁，不能按助记符写死。
        fn, sources = VEC_AES[stem]
        if len(ops) == 3 and vex:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        if width != 16:
            raise Refused(f"{m} width")
        # 一源的那条（`aesimc`）读的是**第二个**操作数；两源时 legacy 形式的目的
        # 同时是源1，VEX 三操作数形式的源1 是 ops[1]。所以一源取 ops[-1]。
        args = ([src_vec(ops[-1], insn)] if sources == 1
                else [src_vec(a, insn), src_vec(b, insn)])
        lines = ["{", f"    __m128i w = {fn}({', '.join(args)});",
                 f"    memcpy({lane(index)}, &w, 16);"]
        if vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("cvtps2dq", "cvtpd2dq"):
        # 按 MXCSR 舍入，不是截断，所以不能像 `cvttps2dq` 那样自己算——但也
        # 不必自己算：`_mm_cvtps_epi32` 编译出来就是这一条 `cvtps2dq`，读的是
        # 同一个线程里的同一份 MXCSR。舍入模式、越界与 NaN 的"整数不定值"
        # 因此逐位相同，而"自己按默认模式算"只是在赌没人改过 MXCSR。
        #
        # 这个移植没有 `ldmxcsr`，MXCSR 永远是 CRT 启动时的那一份；即使有，
        # 两侧读的也是同一份——这正是"让硬件再执行一次"比"把模式建模出来"更
        # 可靠的地方。
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        dst, source = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        step = 4 if stem == "cvtps2dq" else 8
        ctype = "float" if step == 4 else "double"
        fn = "_mm_cvtps_epi32" if step == 4 else "_mm_cvtpd_epi32"
        load = "_mm_loadu_ps" if step == 4 else "_mm_loadu_pd"
        sw = vec(source)[1] if is_vec(source) else ops[1].size
        pt = (f"(const void *)(uintptr_t){addr(source, insn)}"
              if source.type == capstone.x86.X86_OP_MEM
              else f"(const void *){lane(vec(source)[0])}")
        # 每个 128 位源块产出的字节数：四个单精度字，或两个字（`cvtpd2dq`
        # 只写目标的一半，上半部分是零，intrinsic 已经这样返回）。
        nbytes = 16 if step == 4 else 8
        lines = ["{", f"    const char *ps = (const char *){pt};",
                 f"    memset({lane(index)}, 0, {width}u);"]
        for off in range(0, sw, 16):
            lines.append(f"    {{ __m128i w = {fn}({load}((const {ctype} *)(ps + {off}))); "
                         f"memcpy((char *){lane(index)} + {off // 16 * nbytes}, &w, {nbytes}); }}")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("cvtdq2ps", "cvttps2dq"):
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        dst, source = ops
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        lanes = width // 4
        lines = ["{"]
        for n in range(lanes):
            if stem == "cvtdq2ps":
                # A 32-bit integer is not representable in a float exactly, and
                # C's cast rounds to nearest-even, which is what the instruction
                # does with MXCSR at its default.
                x = element(source, n, "int32_t", 4, insn)
                lines.append(f"    float w{n} = (float)({x});")
            else:
                # Truncation, and a lane that is NaN or outside int32 becomes
                # INT32_MIN — the instruction's "integer indefinite", not the C
                # cast's undefined behaviour.
                x = element(source, n, "float", 4, insn)
                lines.append(f"    float f{n} = {x};")
                # The bounds are doubles on purpose: as a float, -2147483649.0
                # rounds to -2^31 and would wrongly exclude it.
                lines.append(f"    int32_t w{n} = ((double)f{n} > -2147483649.0"
                             f" && (double)f{n} < 2147483648.0)"
                             f" ? (int32_t)f{n} : INT32_MIN;")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {4 * n}, &w{n}, 4);")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_SIGN:
        chunk = VEC_SIGN[stem]
        if len(ops) == 3:
            dst, a, sign = ops
        elif len(ops) == 2:
            dst, a, sign = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        bits = chunk * 8
        u, sg = f"uint{bits}_t", f"int{bits}_t"
        lanes = width // chunk
        lines = ["{"]
        for n in range(lanes):
            lines.append(f"    {sg} s{n} = {element(sign, n, sg, chunk, insn)};")
            lines.append(f"    {u} v{n} = {element(a, n, u, chunk, insn)};")
            # 负则取反（二补数下就是 0 - v），零则清零，正则原样。
            lines.append(f"    {u} w{n} = (s{n} == 0) ? 0"
                         f" : ((s{n} < 0) ? ({u})(0 - v{n}) : v{n});")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {chunk * n}, &w{n}, {chunk});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_DUP or stem == "movddup":
        chunk = 8 if stem == "movddup" else 4
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        dst, source = ops
        # `vmovddup xmm1, m64` is a legal form and the one this image uses: the
        # 64 bits come from memory, so the source is not a register. `element`
        # reads either kind, so the only thing to refuse is a destination that
        # is not a register.
        if not is_vec(dst) or not (is_vec(source)
                                   or source.type == capstone.x86.X86_OP_MEM):
            raise Refused(f"{m} form")
        index, width = vec(dst)
        per_lane = 16 // chunk
        picks = []
        for base in range(0, width // chunk, per_lane):
            if stem == "movddup":
                picks += [(source, base), (source, base)]
            else:
                for pair in range(per_lane // 2):
                    at = base + 2 * pair + VEC_DUP[stem]
                    picks += [(source, at), (source, at)]
        return weave(index, width, chunk, picks, insn, vex)

    if stem == "insertps":
        # imm[7:6] 选 src2 的哪个元素，imm[5:4] 选它落在哪，imm[3:0] 是最后要清零的位掩码。
        if len(ops) == 4:
            dst, a, source, sel = ops
        elif len(ops) == 3:
            dst, a, source, sel = ops[0], ops[0], ops[1], ops[2]
        else:
            raise Refused(f"{m} arity")
        if sel.type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} without an immediate")
        if not is_vec(dst) or not is_vec(a):
            raise Refused(f"{m} form")
        index, width = vec(dst)
        src_at = (sel.imm >> 6) & 3
        dst_at = (sel.imm >> 4) & 3
        lines = ["{"]
        for n in range(4):
            lines.append(f"    uint32_t w{n} = {element(a, n, 'uint32_t', 4, insn)};")
        lines.append(f"    w{dst_at} = {element(source, src_at, 'uint32_t', 4, insn)};")
        for n in range(4):
            if (sel.imm >> n) & 1:
                lines.append(f"    w{n} = 0;")
        for n in range(4):
            lines.append(f"    memcpy((char *){lane(index)} + {4 * n}, &w{n}, 4);")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in ("cvttss2si", "cvttsd2si"):
        # 向零截断。源是 NaN 或超出目标范围时结果是"整数不定值"（只有符号位），
        # C 的强制转换在那里是未定义行为，所以边界写成显式比较。
        if len(ops) != 2:
            raise Refused(f"{m} form")
        step = 4 if stem == "cvttss2si" else 8
        source = float_operand(ops[1], step, insn)
        # 目标也可以是向量寄存器：结果落在低 4/8 字节，其余清零（VEX 的 128 位
        # 形式连 255:128 一起清）。
        into_vec = is_vec(ops[0])
        out_bits = step * 8 if into_vec else ops[0].size * 8
        lines = ["{", f"    double x = (double)({source});"]
        if out_bits == 32:
            lines.append("    int64_t v = (x > -2147483649.0 && x < 2147483648.0)"
                         " ? (int64_t)x : (int64_t)INT32_MIN;")
        else:
            lines.append("    int64_t v = (x < 9223372036854775808.0)"
                         " ? (int64_t)x : INT64_MIN;")
        if into_vec:
            index, dwidth = vec(ops[0])
            lines.append(f"    memset({lane(index)}, 0, {dwidth}u);")
            lines.append(f"    memcpy({lane(index)}, &v, {step});")
            if dwidth == 16 and vex:
                lines += ["    " + z for z in upper_zeroed(index)]
        else:
            lines.append(f"    {wr(ops[0], '((uint64_t)v)', insn)}")
        lines.append("}")
        return lines

    if stem in VEC_SQRT:
        step, packed = VEC_SQRT[stem]
        ctype = "double" if step == 8 else "float"
        if len(ops) == 3:
            dst, a, source = ops
        elif len(ops) == 2:
            dst, a, source = ops[0], None, ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        lanes = width // step if packed else 1
        fn = "sqrt" if ctype == "double" else "sqrtf"
        lines = ["{"]
        for n in range(lanes):
            lines.append(f"    {ctype} d{n} = {fn}({element(source, n, ctype, step, insn)});")
        if a is not None and vec(a)[0] != index:
            # 标量形式上半部分取自 src1，与 `movss` 的三操作数形式同一条规则。
            lines.append(f"    memcpy({lane(index)}, {lane(vec(a)[0])}, {width});")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &d{n}, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_ROUND:
        step, packed = VEC_ROUND[stem]
        ctype = "double" if step == 8 else "float"
        if packed:
            if len(ops) != 3:
                raise Refused(f"{m} arity")
            dst, source, sel = ops
            a = None
        elif len(ops) == 4:
            dst, a, source, sel = ops
        elif len(ops) == 3:
            dst, a, source, sel = ops[0], None, ops[1], ops[2]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst) or sel.type != capstone.x86.X86_OP_IMM:
            raise Refused(f"{m} form")
        index, width = vec(dst)
        mode = f"{ROUND_MODE[sel.imm & 3]} | _MM_FROUND_NO_EXC"
        if source.type == capstone.x86.X86_OP_MEM:
            sp = f"(const {ctype} *)(uintptr_t){addr(source, insn)}"
        else:
            sp = f"(const {ctype} *)((const char *){lane(vec(source)[0])})"
        load = "_mm_loadu_ps" if ctype == "float" else "_mm_loadu_pd"
        store = "_mm_storeu_ps" if ctype == "float" else "_mm_storeu_pd"
        lines = ["{"]
        vtype = "__m128" if ctype == "float" else "__m128d"
        if packed:
            lines.append(f"    const {ctype} *ps = {sp};")
            lines.append(f"    for (unsigned off = 0; off < {width}u; off += 16u) {{")
            lines.append(f"        {vtype} x = {load}((const {ctype} *)((const char *)ps + off));")
            lines.append(f"        {store}(({ctype} *)((char *){lane(index)} + off), "
                         f"_{'mm_round_ps' if ctype == 'float' else 'mm_round_pd'}(x, {mode}));")
            lines.append("    }")
        else:
            lp = "_mm_load_ss" if ctype == "float" else "_mm_load_sd"
            lp2 = lp
            poke = "_mm_cvtss_f32" if ctype == "float" else "_mm_cvtsd_f64"
            rnd = "_mm_round_ss" if ctype == "float" else "_mm_round_sd"
            # 标量形式只算低一个元素：源的 r/m 形式在内存里只有 4/8 字节。
            if a is None or vec(a)[0] == index:
                ap = f"({ctype} *)((const char *){lane(index)})"
            else:
                ap = f"({ctype} *)((const char *){lane(vec(a)[0])})"
            lines.append(f"    {ctype} d = {poke}({rnd}({lp}({ap}), {lp2}({sp}), {mode}));")
            if a is not None and vec(a)[0] != index:
                lines.append(f"    memcpy({lane(index)}, {lane(vec(a)[0])}, {width}u);")
            lines.append(f"    memcpy({lane(index)}, &d, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    if stem in VEC_CMP_INT:
        chunk, rel = VEC_CMP_INT[stem]
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, a, b = ops[0], ops[0], ops[1]
        else:
            raise Refused(f"{m} arity")
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        index, width = vec(dst)
        bits = chunk * 8
        lanes = width // chunk
        # Signed, because that is what `pcmpgt` is; `pcmpeq` does not care.
        signed = f"int{bits}_t"
        lines = ["{"]
        for n in range(lanes):
            x = element(a, n, signed, chunk, insn)
            y = element(b, n, signed, chunk, insn)
            lines.append(f"    uint{bits}_t w{n} = (({x}) {rel} ({y}))"
                         f" ? UINT{bits}_MAX : 0;")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {chunk * n}, &w{n}, {chunk});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    # `cmpps`/`cmppd`/`cmpss`/`cmpsd`, with the predicate either an immediate
    # (`cmpps xmm0, xmm1, 3`) or folded into the mnemonic (`vcmpeqps`).
    named = cmp_form(stem)
    if named is not None or stem in ("cmpps", "cmppd", "cmpss", "cmpsd"):
        if named is not None:
            pred, packed, ctype = named
            if len(ops) == 3:
                dst, a, b = ops
            elif len(ops) == 2:
                dst, a, b = ops[0], ops[0], ops[1]
            else:
                raise Refused(f"{m} arity")
        else:
            ctype = "double" if stem.endswith(("pd", "sd")) else "float"
            packed = stem.endswith(("ps", "pd"))
            if len(ops) == 4:
                dst, a, b, sel = ops
            elif len(ops) == 3:
                dst, a, b, sel = ops[0], ops[0], ops[1], ops[2]
            else:
                raise Refused(f"{m} arity")
            if sel.type != capstone.x86.X86_OP_IMM:
                raise Refused(f"{m} without a predicate")
            pred = sel.imm & 31
        if not is_vec(dst):
            raise Refused(f"{m} to memory")
        form = VEC_CMP_FLOAT[pred]
        index, width = vec(dst)
        step = 8 if ctype == "double" else 4
        bits = step * 8
        lanes = width // step if packed else 1
        lines = ["{"]
        for n in range(lanes):
            x = element(a, n, ctype, step, insn)
            y = element(b, n, ctype, step, insn)
            cond = form.format(a=f"({x})", b=f"({y})")
            lines.append(f"    uint{bits}_t w{n} = ({cond}) ? UINT{bits}_MAX : 0;")
        if not packed and len(ops) == 4 and vec(a)[0] != index:
            # The VEX scalar form takes the bits above the low element from
            # src1; the legacy form leaves the destination's own in place.
            lines.append(f"    memcpy((char *){lane(index)} + {step}, "
                         f"(const char *){lane(vec(a)[0])} + {step}, {width - step});")
        for n in range(lanes):
            lines.append(f"    memcpy((char *){lane(index)} + {step * n}, &w{n}, {step});")
        if width == 16 and vex:
            lines += ["    " + z for z in upper_zeroed(index)]
        lines.append("}")
        return lines

    return None


def lift_one(insn):
    """C statements for one instruction, or Refused."""
    m = insn.mnemonic
    ops = insn.operands
    if m.startswith("lock "):
        # Capstone names the locked forms by the whole thing — `lock cmpxchg`,
        # `lock dec`, `lock xadd` — rather than as a prefix flag. The prefix
        # buys atomicity against another core and a fence; this port has one
        # thread and no other core, so what is left is the instruction. Careful
        # in `branch.py`: the flags are the inner instruction's, so the prefix
        # has to come off there too or a `jz` after `lock dec` is refused for
        # not having a compare.
        m = m[5:]
    if m.startswith("prefetch"):
        # 预取是给缓存控制器的提示：不改寄存器、不改内存、连无效地址都不报故障。
        # 架构上它什么都没做，所以这里也没有可执行的内容——但不能拒绝，否则整条
        # 函数会被一条提示废掉。
        return []
    if m in ("bswap", "movbe"):
        # 字节序反转：`bswap` 在寄存器上转，`movbe` 在访存时就转。`_byteswap_*`
        # 就是这条指令，不是"重写语义"。`movbe` 三种宽度都有（16 位带 66 前缀，
        # SDM `MOVBE`，这个映像里 71 条）；`bswap` 只有 32/64。上游 `ede9e4d3`
        # 这一轮升举的正是"16/32/64-bit MOVBE"，与这张表一一对上。
        if m == "movbe":
            if len(ops) != 2:
                raise Refused("movbe arity")
            bits = ops[0].size * 8
            fn = {16: "_byteswap_ushort", 32: "_byteswap_ulong",
                  64: "_byteswap_uint64"}.get(bits)
            if fn is None:
                raise Refused(f"movbe {bits}-bit")
            if ops[0].type == capstone.x86.X86_OP_MEM:
                return [f"*(uint{bits}_t *)(uintptr_t)({addr(ops[0], insn)}) = "
                        f"{fn}((uint{bits}_t){src(ops[1], insn)});"]
            return [wr(ops[0], f"{fn}((uint{bits}_t)(*(const uint{bits}_t *)"
                               f"(uintptr_t)({addr(ops[1], insn)})))", insn)]
        bits = ops[0].size * 8
        fn = {32: "_byteswap_ulong", 64: "_byteswap_uint64"}.get(bits)
        if fn is None:
            raise Refused(f"bswap {bits}-bit")
        return [wr(ops[0], f"{fn}((uint{bits}_t){src(ops[0], insn)})", insn)]
    if m in ("nop", "endbr64", "pause", "mfence", "lfence", "sfence"):
        # `pause` is the spin-wait hint: it changes nothing a program can read.
        # 三条栅栏管的是**别的核**看到内存的顺序。这个移植只有一个核，也没有别的
        # 核在看，所以它们没有可执行的内容——但也不能拒绝：拒绝会让整条函数作废，
        # 而它们的语义在这里本来就是空的。
        return []
    if m == "int3":
        # The only `int3` a release build emits is the one MSVC puts after a
        # call that does not return — the instruction is reached exactly when
        # the program is already aborting. A port has nothing to break into, so
        # it falls through; `ud2` is refused, because that one is deliberate.
        return []
    if m == "ud2":
        raise Refused(m)
    if m in ("blsi", "blsmsk", "blsr"):
        # BMI1, one-bit extraction: isolate the lowest set bit, mask up to and
        # including it, or clear it. All three are defined at zero (`blsi` and
        # `blsmsk` give zero, `blsr` gives zero) and all three set the flags
        # from the result, which is why `branch.py` reads them as an ALU write.
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        bits = width(ops[1])
        value = f"(uint{bits}_t)({src(ops[1], insn)})"
        expr = {"blsi": f"({value}) & (0 - ({value}))",
                "blsmsk": f"({value}) ^ (({value}) - 1)",
                "blsr": f"({value}) & (({value}) - 1)"}[m]
        return [wr(ops[0], expr, insn)]
    if m == "ret":
        return ["return;"]
    if m in FLOAT_COMPARE:
        return []  # flags only; `branch.py` is where they are tracked
    vector = vec_line(insn)
    if vector is not None:
        return vector
    if not ops and m not in ZERO_OPERAND:
        # Every handler below reads `ops[0]`, so an operand-less instruction has
        # to name itself here to get past. `cdqe` and its four siblings are the
        # ones that do: they name a register implicitly, which is why a guard on
        # `ops` silently refused 331 of them.
        raise Refused(m)
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
    if m == "xadd":
        # Exchange and add: the destination gets the sum, the source gets what
        # the destination held. Both operands are read before either is written,
        # which is the whole reason this is not two statements.
        if len(ops) != 2:
            raise Refused("xadd arity")
        bits = width(ops[0])
        wide = f"uint{bits}_t"
        return [
            "{",
            f"    {wide} before = {src(ops[0], insn)}, addend = {src(ops[1], insn)};",
            f"    {wr(ops[0], 'before + addend', insn)}",
            f"    {wr(ops[1], 'before', insn)}",
            "}",
        ]
    if m in ("tzcnt", "lzcnt"):
        # Both are defined at zero, which is what separates them from `bsf` and
        # `bsr`: the count is the operand's width, not an undefined result.
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        bits = width(ops[1])
        value = f"(uint64_t)(uint{bits}_t)({src(ops[1], insn)})"
        counter = "count_trailing" if m == "tzcnt" else "count_leading"
        return [wr(ops[0], f"{counter}({value}, {bits})", insn)]
    if m in ("add", "sub", "and", "or", "xor"):
        if len(ops) != 2:
            raise Refused("alu arity")
        op = {"add": "+", "sub": "-", "and": "&", "or": "|", "xor": "^"}[m]
        return [wr(ops[0], f"{src(ops[0], insn)} {op} {src(ops[1], insn)}", insn)]
    if m in ("mul", "imul") and len(ops) == 1:
        # 单操作数形式是**双倍宽度**的乘积：RAX 乘以操作数，128 位结果落在
        # RDX:RAX。这与 `imul r, r/m` 不是同一条指令（那条只留低半、还自带符号）。
        # `_umul128`/`_mul128` 各对应一个符号性，边界与硬件一致。
        bits = ops[0].size * 8
        if bits not in (8, 16, 32, 64):
            raise Refused(f"{m} {bits}-bit")
        x = src(ops[0], insn)
        if bits == 8:
            # `mul r/m8` 只写 AX：AL 是被乘数、AH 是结果高半，而 AX 落在 r[0]
            # 的低 16 位里，上面的位不动。
            if m == "mul":
                return ["{",
                        f"    uint32_t p = (uint32_t)(uint8_t)r[0] * (uint32_t)(uint8_t)({x});",
                        "    r[0] = (r[0] & ~(uint64_t)0xFFFF) | (uint64_t)(p & 0xFFFF);",
                        "}"]
            return ["{",
                    f"    int32_t p = (int32_t)(int8_t)r[0] * (int32_t)(int8_t)({x});",
                    "    r[0] = (r[0] & ~(uint64_t)0xFFFF) | (uint64_t)((uint32_t)p & 0xFFFF);",
                    "}"]
        if bits == 16:
            # 16 位这条没有 #DE 的风险（乘积总是装得下 32 位），所以它与 8 位
            # 一起在这里，而 `div`/`idiv` 的 16 位形式仍然拒绝。
            if m == "mul":
                return ["{",
                        f"    uint32_t p = (uint32_t)(uint16_t)r[0] * (uint32_t)(uint16_t)({x});",
                        "    r[0] = (r[0] & ~(uint64_t)0xFFFF) | (uint64_t)(p & 0xFFFF);",
                        "    r[2] = (r[2] & ~(uint64_t)0xFFFF) | (uint64_t)(p >> 16);",
                        "}"]
            return ["{",
                    f"    int32_t p = (int32_t)(int16_t)r[0] * (int32_t)(int16_t)({x});",
                    "    r[0] = (r[0] & ~(uint64_t)0xFFFF) | (uint64_t)((uint32_t)p & 0xFFFF);",
                    "    r[2] = (r[2] & ~(uint64_t)0xFFFF) | (uint64_t)(((uint32_t)p >> 16) & 0xFFFF);",
                    "}"]
        if bits == 64:
            if m == "mul":
                return ["{", "    uint64_t hi;",
                        f"    uint64_t lo = _umul128(r[0], (uint64_t)({x}), &hi);",
                        "    r[0] = lo; r[2] = hi;", "}"]
            return ["{", "    int64_t hi;",
                    f"    int64_t lo = _mul128((int64_t)r[0], (int64_t)({x}), &hi);",
                    "    r[0] = (uint64_t)lo; r[2] = (uint64_t)hi;", "}"]
        if m == "mul":
            return ["{",
                    f"    uint64_t p = (uint64_t)(uint32_t)r[0] * (uint64_t)(uint32_t)({x});",
                    "    r[0] = (uint64_t)(uint32_t)p;",
                    "    r[2] = (uint64_t)(uint32_t)(p >> 32);", "}"]
        return ["{",
                f"    int64_t p = (int64_t)(int32_t)r[0] * (int64_t)(int32_t)({x});",
                "    r[0] = (uint64_t)(uint32_t)p;",
                "    r[2] = (uint64_t)(uint32_t)((uint64_t)p >> 32);", "}"]
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
    if m in ("rol", "ror"):
        # 旋转：计数按目标宽度掩码，所以 `rol eax, 32` 是空操作。两侧拼起来，
        # `(bits - c) & (bits - 1)` 在 c 为 0 时给出 0，于是两项都是原值、或起来还是原值。
        if len(ops) != 2:
            raise Refused(f"{m} arity")
        bits = width(ops[0])
        value = f"(uint{bits}_t)({src(ops[0], insn)})"
        shift = f"(uint64_t)({src(ops[1], insn)} & {bits - 1}u)"
        other = f"(({bits}u - {shift}) & {bits - 1}u)"
        if m == "rol":
            expr = f"(uint{bits}_t)(({value} << {shift}) | ({value} >> {other}))"
        else:
            expr = f"(uint{bits}_t)(({value} >> {shift}) | ({value} << {other}))"
        return [wr(ops[0], expr, insn)]
    if m in ("shlx", "shrx", "sarx"):
        # BMI2: the count is a third operand and is masked the same way, which
        # is the whole difference from `shl`. No flags written either way.
        if len(ops) != 3:
            raise Refused(f"{m} arity")
        bits = width(ops[0])
        count = f"({src(ops[2], insn)} & {bits - 1})"
        value = src(ops[1], insn)
        if m == "shlx":
            expr = f"{value} << {count}"
        elif m == "shrx":
            expr = f"{value} >> {count}"
        else:
            expr = f"(int{bits}_t){value} >> {count}"
        return [wr(ops[0], expr, insn)]
    if m in ("cdqe", "cdq", "cqo", "cbw", "cwde"):
        # Sign extension into the fixed register, and the sign of rax into rdx
        # for the two-operand `div` that usually follows.
        if m == "cdqe":
            return ["r[0] = (uint64_t)(int64_t)(int32_t)r[0];"]
        if m == "cdq":
            return ["r[2] = (uint64_t)(uint32_t)(int32_t)((int32_t)r[0] >> 31);"]
        if m == "cqo":
            return ["r[2] = (uint64_t)((int64_t)r[0] >> 63);"]
        if m == "cbw":
            return ["r[0] = (r[0] & ~UINT64_C(0xFFFF))"
                    " | (uint64_t)(uint16_t)(int16_t)(int8_t)r[0];"]
        return ["r[0] = (uint64_t)(uint32_t)(int32_t)(int16_t)r[0];"]
    if m == "mulx":
        # BMI2: RDX times the source, high half into the first operand and low
        # into the second, with the flags left alone — which is why compilers
        # use it for a multiply that sits between a compare and a branch.
        if len(ops) != 3:
            raise Refused("mulx arity")
        bits = width(ops[0])
        if bits == 64:
            return ["{",
                    f"    uint64_t lo, hi = _umul128((uint64_t){src(ops[2], insn)}, r[2], &lo);",
                    "    " + wr(ops[1], "lo", insn),
                    "    " + wr(ops[0], "hi", insn),
                    "}"]
        return ["{",
                f"    uint64_t p = (uint64_t)(uint32_t){src(ops[2], insn)}"
                " * (uint64_t)(uint32_t)r[2];",
                "    " + wr(ops[1], "(uint32_t)p", insn),
                "    " + wr(ops[0], "(uint32_t)(p >> 32)", insn),
                "}"]
    if m == "xchg":
        if len(ops) != 2:
            raise Refused("xchg arity")
        bits = width(ops[0])
        return ["{",
                f"    uint{bits}_t x = (uint{bits}_t){src(ops[0], insn)};",
                f"    uint{bits}_t y = (uint{bits}_t){src(ops[1], insn)};",
                "    " + wr(ops[0], "y", insn),
                "    " + wr(ops[1], "x", insn),
                "}"]
    if m in ("inc", "dec", "neg", "not"):
        one = {"inc": "+ 1", "dec": "- 1"}.get(m)
        if one:
            return [wr(ops[0], f"{src(ops[0], insn)} {one}", insn)]
        if m == "neg":
            return [wr(ops[0], f"-(int{width(ops[0])}_t){src(ops[0], insn)}", insn)]
        return [wr(ops[0], f"~{src(ops[0], insn)}", insn)]
    if m == "andn":
        # BMI1: `andn dest, src1, src2` is `~src1 & src2`, three operands, and
        # the destination takes a 32-bit write's zero-extension like any other.
        if len(ops) != 3:
            raise Refused("andn arity")
        return [wr(ops[0], f"(~{src(ops[1], insn)}) & {src(ops[2], insn)}", insn)]
    if m == "bextr":
        # BMI1: the control operand packs a start bit and a length. A length
        # that runs past the top takes everything left and nothing more, and a
        # start past the top gives zero — both are the register width's doing,
        # so the mask is clamped at 64 and the result is masked by `wr`.
        if len(ops) != 3:
            raise Refused("bextr arity")
        bits = width(ops[0])
        return [
            "{",
            f"    uint32_t c = (uint32_t){src(ops[2], insn)};",
            "    uint32_t from = c & 0xFF, len = (c >> 8) & 0xFF;",
            f"    uint64_t s = (uint64_t){src(ops[1], insn)};",
            "    uint64_t m = len >= 64 ? ~UINT64_C(0) : ((UINT64_C(1) << len) - 1);",
            f"    uint64_t r0 = from >= {bits} ? 0 : ((s >> from) & m);",
            "    " + wr(ops[0], "r0", insn),
            "}",
        ]
    if m == "popcnt":
        # 数一数 1 的个数。`_popcnt64`/`_popcnt32` 就是这条指令（SSE4.2 的 POPCNT），
        # 目标宽度决定用哪一个；8/16 位操作数在硬件上也是按 16/32 位算的。
        if len(ops) != 2:
            raise Refused("popcnt arity")
        bits = width(ops[0])
        # 只数到操作数宽度为止：8/16 位的 `popcnt` 在硬件上也算 16/32 位的值，
        # 而写回时 `wr` 会截断，所以先把源遮到它自己的宽度。
        mask = ONES[bits] if bits < 64 else None
        val = f"((uint64_t){src(ops[1], insn)})"
        if mask:
            val = f"(({val}) & {mask})"
        return [wr(ops[0], f"popcount64({val})", insn)]
    if m == "rorx":
        # 循环右移但**不碰标志**。移 0 位时 C 的 `x >> 64` 是未定义行为，所以
        # 移位数单独分支写出来，与指令的"原样返回"对齐。
        if len(ops) != 3 or ops[2].type != capstone.x86.X86_OP_IMM:
            raise Refused("rorx arity")
        bits = width(ops[0])
        n = ops[2].imm & (bits - 1)
        if n == 0:
            return [wr(ops[0], src(ops[1], insn), insn)]
        return [wr(ops[0], f"((((uint{bits}_t){src(ops[1], insn)}) >> {n}) | "
                           f"(((uint{bits}_t){src(ops[1], insn)}) << {bits - n}))", insn)]
    if m == "bzhi":
        # BMI2: 从第 index 位起的高位清零。`_bzhi_u32/_u64` 就是这条指令，
        # 包括 index 大于等于宽度时"原样不动"与 index 为 0 时"全零"两个边界。
        if len(ops) != 3:
            raise Refused("bzhi arity")
        bits = width(ops[0])
        if bits == 64:
            call = (f"_bzhi_u64((uint64_t){src(ops[1], insn)}, "
                    f"(unsigned){src(ops[2], insn)})")
        else:
            call = (f"_bzhi_u32((unsigned)(uint32_t){src(ops[1], insn)}, "
                    f"(unsigned)(uint32_t){src(ops[2], insn)})")
        return [wr(ops[0], call, insn)]
    if m in ("shld", "shrd"):
        # 双精度移位：目标移 n 位，空出来的位从**源**的高位补进来。n 为 0 时结果
        # 是目标原值（也就不能写成 `x >> (bits - n)`——那是对位宽取模的移位，是
        # 未定义行为，而且在 n=0 时会给出 0）。标志按 SDM 是 CF/OF 加结果那几位，
        # 这一层只供结果，`branch` 那边照旧拒绝跨过它读标志。
        if len(ops) != 3:
            raise Refused(f"{m} arity")
        bits = width(ops[0])
        step = "<<" if m == "shld" else ">>"
        back = ">>" if m == "shld" else "<<"
        n = f"((uint32_t){src(ops[2], insn)} & {bits - 1})"
        return [
            "{",
            f"    uint{bits}_t a = (uint{bits}_t){src(ops[0], insn)};",
            f"    uint{bits}_t b = (uint{bits}_t){src(ops[1], insn)};",
            f"    uint32_t n = {n};",
            f"    uint{bits}_t d = n ? (uint{bits}_t)((a {step} n) | (b {back} ({bits} - n))) : a;",
            "    " + wr(ops[0], "d", insn),
            "}",
        ]
    if m in ("div", "idiv"):
        # The one-operand form divides RDX:RAX and leaves quotient in RAX and
        # remainder in RDX — a 128-by-64 division no C operator performs. MSVC's
        # `_udiv128`/`_div128` intrinsics are that instruction and nothing else,
        # so a divisor of zero or a quotient that does not fit raises the same
        # #DE here as it does in the original. Faults are left to propagate:
        # catching them would mean inventing behaviour the hardware does not have.
        # 这句话只对 64 位形式成立——窄形式的两条差别见下。
        #
        # 8 位、16 位与 32 位都不是那个形状：它们的被除数各自装在一个 64 位整数里
        # （AX、DX:AX、EDX:EAX），而 `_udiv128` 的头两个参数是**高 64 位**与低
        # 64 位。把 `EDX:EAX` 的 EDX 当高 64 位传进去，等于把被除数乘了 2^32——
        # 商从 2^32 那一档开始就全错，而且这个错躲得过两条 #DE（除零与商放不下
        # 都只在 EDX 相对除数很大时才触发），所以差分里看着是"对"的。
        #
        # 第二条不一致在阈值：`_udiv128` 只在商越过 **2^64** 时 #DE，而
        # `div r/m32` 在商越过 2^32 时就该 #DE。所以窄形式自己判：除数零、商越过
        # 宽度，两处都走 `fault_divide()`——一次真的调用，不会被折到分支外
        # （`_udiv128(d, 0, d)` 那种"用一次会崩的除法发 #DE"会被 if-convert，
        # 见 HELPERS 里那段注释）。除法本身用 high=0 的 64/64，它不会自己 #DE。
        if len(ops) != 1:
            raise Refused(f"{m} arity")
        bits = ops[0].size * 8
        if bits not in (8, 16, 32, 64):
            raise Refused(f"{m} {bits}-bit")
        if bits != 64:
            mask = (1 << bits) - 1
            # 被除数：8 位形式是 AX 自己（正好 16 位），16 位是 DX:AX，32 位是 EDX:EAX。
            dvd = ("(uint64_t)(r[0] & 0xFFFF)" if bits == 8 else
                   f"((uint64_t)(r[2] & {mask:#x}) << {bits})"
                   f" | (uint64_t)(r[0] & {mask:#x})")
            if bits == 32:
                # 32 位写会清零高半边，两个寄存器都是整覆盖。
                out = ["    r[0] = (uint64_t)(uint32_t)q;",
                       "    r[2] = (uint64_t)(uint32_t)rem;"]
            elif bits == 16:
                # 商写 AX、余数写 DX，两个寄存器的高半边都留着。
                out = ["    r[0] = (r[0] & ~0xFFFFull) | (uint64_t)(q & 0xFFFF);",
                       "    r[2] = (r[2] & ~0xFFFFull) | (uint64_t)(rem & 0xFFFF);"]
            else:
                # 商写 AL、余数写 AH：只有 AX 这一个 16 位窗口被动。
                out = ["    r[0] = (r[0] & ~0xFFFFull) | (uint64_t)(q & 0xFF)"
                       " | ((uint64_t)(rem & 0xFF) << 8);"]
            divisor = src(ops[0], insn)
            if m == "div":
                lines = [
                    "{",
                    f"    uint64_t dvd = {dvd};",
                    f"    uint64_t dvs = (uint64_t)(uint{bits}_t)({divisor});",
                    "    uint64_t rem, q;",
                    "    if (dvs == 0) fault_divide();",
                    # high=0 且 dvs ≥ 1：这一步的商装得进 64 位，不可能自己 #DE，
                    # 所以它可以无条件执行，也就可以被编译器随意搬动。
                    "    q = _udiv128(0, dvd, dvs, &rem);",
                    f"    if (q > {mask:#x}) fault_divide();",
                ]
            else:
                lines = [
                    "{",
                    f"    int64_t dvd = (int{bits * 2}_t)({dvd});",
                    f"    int64_t dvs = (int{bits}_t)({divisor});",
                    "    int64_t rem, q;",
                    "    if (dvs == 0) fault_divide();",
                    "    q = _div128(dvd < 0 ? -1 : 0, (uint64_t)dvd, dvs, &rem);",
                    # 两个界都要 `LL` 后缀：`-2147483648` 在 MSVC 的 C 解析里先按
                    # 无符号取字面量，减号落在一个无符号值上，于是整条比较变成
                    # **无符号**比较，`877241 < 2147483648` 恒真——每次除法都 #DE。
                    f"    if (q < {-((mask + 1) // 2)}LL || q > {mask >> 1}LL) fault_divide();",
                ]
            return lines + out + ["}"]
        # 64 位形式是唯一真的需要 128 位除数的一种：RDX 就是高 64 位，
        # `_udiv128` 的 #DE 条件（高半 ≥ 除数）与指令的完全一致。
        signed = "u" if m == "div" else ""
        wide = f"{signed}int64_t"
        return [
            "{",
            f"    {wide} d = ({wide}){src(ops[0], insn)}, rem;",
            f"    {wide} q = _{signed}div128(r[2], r[0], d, &rem);",
            "    r[0] = (uint64_t)q; r[2] = (uint64_t)rem;",
            "}",
        ]
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
        # The model's TEB base, made up but fixed: the interpreter's job is to
        # disagree with the emitter when either is wrong, and a constant base
        # reaches the same address every time.
        self.teb = ARENA + 0x3000

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
            if mo.segment:
                # The TEB, read out of the model rather than out of a register:
                # `gs:[0x30]` is the Self field, so it *is* the base.
                at += space.teb
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
        elif m == "andn":
            put(ops[0], ~get(ops[1]) & get(ops[2]))
        elif m in ("div", "idiv"):
            bits = ops[0].size * 8
            if bits not in (8, 16, 32, 64):
                raise Refused(f"{m} {bits}-bit")
            mask = (1 << bits) - 1
            # Read the divisor before touching rax/rdx: it may be `div rax`.
            raw = get(ops[0]) & mask
            if raw == 0:
                raise Refused("divide by zero")
            # 8 位形式的被除数是 AX 本身，DX 不参与；其余形式是 RDX:RAX 的低 2*bits 位。
            if bits == 8:
                dividend = regs[0] & 0xFFFF
            else:
                dividend = ((regs[2] & mask) << bits) | (regs[0] & mask)
            if m == "idiv":
                if dividend >> (2 * bits - 1):
                    dividend -= 1 << (2 * bits)
                divisor = raw - (1 << bits) if raw >> (bits - 1) else raw
                q = abs(dividend) // abs(divisor)
                if (dividend < 0) != (divisor < 0):
                    q = -q
                rem = dividend - q * divisor
                if not -(1 << (bits - 1)) <= q < (1 << (bits - 1)):
                    raise Refused("quotient overflows")
            else:
                q, rem = divmod(dividend, raw)
                if q > mask:
                    raise Refused("quotient overflows")
            if bits >= 32:
                # 32 位写会清零高半边，64 位就是全宽，两种情况都是整寄存器覆盖。
                regs[0] = q & mask
                regs[2] = rem & mask
            elif bits == 16:
                keep = MASK64 ^ 0xFFFF
                regs[0] = (regs[0] & keep) | (q & 0xFFFF)
                regs[2] = (regs[2] & keep) | (rem & 0xFFFF)
            else:
                # 商在 AL、余数在 AH：AX 的 16 位窗口被两个半字节分开写。
                regs[0] = ((regs[0] & (MASK64 ^ 0xFFFF))
                           | (q & 0xFF) | ((rem & 0xFF) << 8))
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


def uses_vector(code, rva):
    """True when the function touches the vector file."""
    for insn in MD.disasm(code, rva):
        if insn.mnemonic in ("vzeroupper", "vzeroall"):
            return True
        for op in insn.operands:
            if op.type == capstone.x86.X86_OP_REG and MD.reg_name(op.reg) in VEC:
                return True
    return False


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
    # The interpreter below models the general registers and memory, not the
    # vector file. A function that touches xmm is therefore outside what it can
    # check, and is skipped here rather than guessed at — `diff.py` is where
    # those are checked, against the hardware.
    before = len(picked)
    picked = [(rva, code, lines) for rva, code, lines in picked
              if not uses_vector(code, rva)]
    print(f"of those, {len(picked)} touch no vector register "
          f"({before - len(picked)} skipped: unmodelled here)")

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
    head = ("#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
            "#include <math.h>\n#include <intrin.h>\n"
            "#include <xmmintrin.h>\n#include <immintrin.h>\n"
            "static uint64_t v[16][4];\n" + "\n".join(HELPERS) + "\n")
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
