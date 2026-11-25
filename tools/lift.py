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

MD = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
MD.detail = True


class Refused(Exception):
    pass


def reg(op):
    if op.type != capstone.x86.X86_OP_REG:
        raise Refused("not a register")
    name = MD.reg_name(op.reg)
    if name not in IDX:
        raise Refused(f"register {name}")
    return IDX[name]


def place(op):
    """Where a register write goes: `rsp` is the stack offset variable."""
    if op.type != capstone.x86.X86_OP_REG:
        return None
    return "sp" if MD.reg_name(op.reg) == "rsp" else f"r[{reg(op)}]"


def stack_slot(op):
    """A memory operand: the guest stack is `stk`, `rsp` is an offset into it."""
    if op.type != capstone.x86.X86_OP_MEM:
        raise Refused("not memory")
    m = op.mem
    if m.segment != 0:
        raise Refused("segment")
    terms = []
    if m.base:
        base = MD.reg_name(m.base)
        if base not in IDX:
            raise Refused(f"base {base}")
        terms.append("sp" if base == "rsp" else f"r[{IDX[base]}]")
    if m.index:
        index = MD.reg_name(m.index)
        if index not in IDX:
            raise Refused(f"index {index}")
        terms.append(f"r[{IDX[index]}] * {m.scale or 1}")
    if m.disp:
        terms.append(str(m.disp))
    inner = " + ".join(terms) if terms else "0"
    return f"*(uint64_t *)(stk + (({inner}) & 255))"


def is_imm(op):
    return op.type == capstone.x86.X86_OP_IMM


def src(op):
    """An operand read by a 64-bit operation."""
    if is_imm(op):
        return f"UINT64_C({op.imm & 0xFFFFFFFFFFFFFFFF})"
    if op.type == capstone.x86.X86_OP_REG:
        return place(op)
    return stack_slot(op)


def dst(op):
    if op.type == capstone.x86.X86_OP_REG:
        return place(op)
    return stack_slot(op)


def lift_one(insn):
    """C statements for one instruction, or Refused."""
    m = insn.mnemonic
    ops = insn.operands
    if m in ("nop", "endbr64", "int3", "ud2"):
        return [] if m != "int3" else (_ for _ in ()).throw(Refused("int3"))
    if m == "ret":
        return ["r[4] = sp;", "return;"]
    if m == "push":
        return ["sp -= 8;", f"*(uint64_t *)(stk + sp) = {src(ops[0])};"]
    if m == "pop":
        return [f"{dst(ops[0])} = *(uint64_t *)(stk + sp);", "sp += 8;"]
    if m == "mov":
        if len(ops) != 2:
            raise Refused("mov arity")
        return [f"{dst(ops[0])} = {src(ops[1])};"]
    if m in ("movzx", "movsx", "movsxd"):
        if len(ops) != 2 or ops[1].type != capstone.x86.X86_OP_REG:
            raise Refused("movzx form")
        width = {"movzx": "Z", "movsx": "S", "movsxd": "S"}[m]
        size = ops[1].size * 8
        inner = src(ops[1])
        if width == "Z":
            return [f"{dst(ops[0])} = (uint64_t)(uint{size}_t)({inner});"]
        return [f"{dst(ops[0])} = (uint64_t)(int64_t)(int{size}_t)({inner});"]
    if m == "lea":
        if len(ops) != 2 or ops[1].type != capstone.x86.X86_OP_MEM:
            raise Refused("lea form")
        mem = ops[1].mem
        if mem.segment != 0:
            raise Refused("lea segment")
        parts = []
        if mem.base:
            base = MD.reg_name(mem.base)
            if base not in IDX:
                raise Refused(f"lea base {base}")
            parts.append("sp" if base == "rsp" else f"r[{IDX[base]}]")
        if mem.index:
            index = MD.reg_name(mem.index)
            if index not in IDX:
                raise Refused(f"lea index {index}")
            parts.append(f"r[{IDX[index]}] * {mem.scale or 1}")
        if mem.disp or not parts:
            parts.append(f"UINT64_C({mem.disp & 0xFFFFFFFFFFFFFFFF})")
        return [f"{dst(ops[0])} = {' + '.join(parts)};"]
    if m in ("add", "sub", "and", "or", "xor", "imul", "shl", "shr", "sar"):
        if len(ops) < 2:
            raise Refused("alu arity")
        op = {"add": "+", "sub": "-", "and": "&", "or": "|", "xor": "^"}.get(m)
        if m == "imul":
            if len(ops) == 2:
                return [f"{dst(ops[0])} = {dst(ops[0])} * {src(ops[1])};"]
            return [f"{dst(ops[0])} = {src(ops[1])} * {src(ops[2])};"]
        if m in ("shl", "shr", "sar"):
            count = src(ops[1])
            cast = {"shl": f"{dst(ops[0])} = {dst(ops[0])} << ({count});",
                    "shr": f"{dst(ops[0])} = {dst(ops[0])} >> ({count});",
                    "sar": f"{dst(ops[0])} = (uint64_t)((int64_t){dst(ops[0])} >> ({count}));"}[m]
            return [cast]
        if op is None:
            raise Refused(m)
        return [f"{dst(ops[0])} = {dst(ops[0])} {op} {src(ops[1])};"]
    if m in ("inc", "dec", "neg", "not"):
        one = {"inc": "+ 1", "dec": "- 1"}.get(m)
        if one:
            return [f"{dst(ops[0])} = {dst(ops[0])} {one};"]
        if m == "neg":
            return [f"{dst(ops[0])} = (uint64_t)(-(int64_t){dst(ops[0])});"]
        return [f"{dst(ops[0])} = ~{dst(ops[0])};"]
    if m in ("cmp", "test"):
        # Nothing in this subset reads the flags, so nothing to keep.
        return []
    raise Refused(m)


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


def interpret(code, base_rva, regs, stk):
    for insn in MD.disasm(code, base_rva):
        m = insn.mnemonic
        ops = insn.operands

        def address(op):
            m_ = op.mem
            at = m_.disp
            if m_.base:
                at += regs[IDX[MD.reg_name(m_.base)]]
            if m_.index:
                at += regs[IDX[MD.reg_name(m_.index)]] * (m_.scale or 1)
            return at & 255

        def get(op):
            if op.type == capstone.x86.X86_OP_IMM:
                return op.imm & 0xFFFFFFFFFFFFFFFF
            if op.type == capstone.x86.X86_OP_REG:
                return regs[IDX[MD.reg_name(op.reg)]]
            return struct.unpack_from("<Q", stk, address(op))[0]

        def put(op, value):
            value &= 0xFFFFFFFFFFFFFFFF
            if op.type == capstone.x86.X86_OP_REG:
                regs[IDX[MD.reg_name(op.reg)]] = value
            else:
                struct.pack_into("<Q", stk, address(op), value)

        if m in ("nop", "endbr64", "cmp", "test"):
            continue
        if m == "ret":
            return
        if m == "push":
            regs[4] -= 8
            struct.pack_into("<Q", stk, regs[4], get(ops[0]))
            continue
        if m == "pop":
            put(ops[0], struct.unpack_from("<Q", stk, regs[4])[0])
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
            value = m_.disp
            if m_.base:
                value += regs[IDX[MD.reg_name(m_.base)]]
            if m_.index:
                value += regs[IDX[MD.reg_name(m_.index)]] * (m_.scale or 1)
            put(ops[0], value)
        elif m in ("add", "sub", "and", "or", "xor"):
            a, b = get(ops[0]), get(ops[1])
            put(ops[0], {"add": a + b, "sub": a - b, "and": a & b, "or": a | b, "xor": a ^ b}[m])
        elif m == "imul":
            a, b = get(ops[0]), get(ops[-1])
            put(ops[0], a * b)
        elif m in ("shl", "shr", "sar"):
            count = get(ops[1]) & 63
            a = get(ops[0])
            if m == "shl":
                put(ops[0], a << count)
            elif m == "shr":
                put(ops[0], a >> count)
            else:
                signed = a - (1 << 64) if a >> 63 else a
                put(ops[0], signed >> count)
        elif m in ("inc", "dec", "neg", "not"):
            a = get(ops[0])
            put(ops[0], {"inc": a + 1, "dec": a - 1, "neg": -a, "not": ~a}[m])
        else:
            raise Refused(m)


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
        body.append(f"    uint64_t sp = {STACK // 2};")
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
    whole = head + "\n".join(body) + "\n" + driver
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
    run = subprocess.run([exe, os.path.join(work, "in.bin"), os.path.join(work, "out.bin")], capture_output=True)
    if run.returncode != 0:
        print(f"run failed: {run.returncode}")
        return

    data = open(os.path.join(work, "out.bin"), "rb").read()
    bad = 0
    for i, (rva, code, _) in enumerate(picked):
        regs = list(inputs[i])
        stk = bytearray(stacks[i])
        try:
            interpret(code, rva, regs, stk)
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
