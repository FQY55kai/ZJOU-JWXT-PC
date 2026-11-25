"""Branches, without a flag register.

A full lifter models CF/OF/SF/ZF/AF/PF and every instruction that writes them.
That is the correct end state and it is also three weeks of work that does not
have to come first: in real code the overwhelming majority of branches read the
flags of a `cmp`/`test` that is *right there*, one instruction earlier. So this
pass recognises that shape and turns it into a comparison in C, and refuses
everything else — `setcc`, `cmovcc`, `adc`, a `jcc` whose flags were written
further back, a branch out of the function.

What it does cover is the control flow that makes a function a function: labels,
`goto`, loops, early returns — plus the straight-line body `lift` already knew.
"""

import capstone

import lift as module

# `jcc` → the C relation, per operand size and signedness.
UNSIGNED = {"a": ">", "ae": ">=", "b": "<", "be": "<="}
SIGNED = {"g": ">", "ge": ">=", "l": "<", "le": "<="}
EQUALITY = {"e": "==", "ne": "!="}

# Instructions that leave the flags alone, so a compare stays live across them.
# Everything else that is not a consumer invalidates it: a `cmp` followed by an
# `add` and then a `jz` branches on the add, not on the compare.
FLAG_SAFE = {
    "mov", "movzx", "movsx", "movsxd", "movabs", "lea", "not", "bswap", "nop",
    "push", "pop", "cbw", "cwde", "cdqe", "cdq", "cqo", "xchg", "endbr64",
}


class Refused(Exception):
    pass


# Call targets seen while lifting one function, so a closure can be walked
# without decoding everything twice.
CALLS: dict[int, set[int]] = {}


def condition(pending, suffix):
    """The C condition for a consumer of `pending`, or Refused."""
    kind, a, b, bits = pending
    if kind == "test":
        if suffix == "e":
            return f"(({a}) == 0)"
        if suffix == "ne":
            return f"(({a}) != 0)"
        if suffix == "s":
            return f"((({a}) >> {bits - 1}) & 1)"
        if suffix == "ns":
            return f"!((({a}) >> {bits - 1}) & 1)"
        raise Refused(f"test {suffix}")
    if suffix in EQUALITY:
        return f"(({a}) {EQUALITY[suffix]} ({b}))"
    if suffix in UNSIGNED:
        return f"(({a}) {UNSIGNED[suffix]} ({b}))"
    if suffix in SIGNED:
        return f"((int{bits}_t)({a}) {SIGNED[suffix]} (int{bits}_t)({b}))"
    # Overflow and parity conditions need the producer's carry chain, which this
    # pass does not carry; they are refused rather than guessed.
    raise Refused(f"condition {suffix}")


def lift_function(code, rva):
    """C statements for a function with branches, or Refused."""
    insns = list(module.MD.disasm(code, rva))
    while insns and insns[-1].mnemonic in ("int3", "nop", "ud2"):
        insns.pop()
    if not insns:
        raise Refused("empty")
    if insns[-1].mnemonic not in ("ret", "jmp"):
        raise Refused("does not end in ret")

    calls: set[int] = set()
    start, end = rva, rva + len(code)
    targets = set()
    for insn in insns:
        m = insn.mnemonic
        if m.startswith("j"):
            if not insn.operands or insn.operands[0].type != capstone.x86.X86_OP_IMM:
                raise Refused("indirect branch")
            target = insn.operands[0].imm
            if not start <= target < end:
                # Out of range is a tail call, not an error: the function hands
                # its frame to another one and returns when that returns.
                if insn.mnemonic == "jmp":
                    calls.add(target)
                    continue
                raise Refused("branch out of the function")
            targets.add(target)
        if m == "call":
            aops = insn.operands
            if aops and aops[0].type == capstone.x86.X86_OP_IMM:
                calls.add(aops[0].imm)
            else:
                raise Refused("indirect call")

    out = []
    # `cmp`/`test` seen but not yet consumed by a branch: the operands as C.
    pending = None
    # Comparands are snapshotted into locals at the compare, because a
    # flag-neutral `mov` in between can overwrite a register the compare read:
    # `cmp eax, 0 / mov eax, [rbx] / jz L` branches on the *old* eax.
    snap = 0
    for insn in insns:
        m, ops = insn.mnemonic, insn.operands
        if insn.address in targets:
            out.append(f"L{insn.address:#x}: ;")
        if m in ("cmp", "test"):
            snap += 1
            a, b = module.src(ops[0], insn), module.src(ops[1], insn)
            out.append(f"uint64_t a{snap} = {a}, b{snap} = {b};")
            pending = (m, f"a{snap}", f"b{snap}", ops[0].size * 8)
            continue
        if m.startswith("j") and m != "jmp":
            if pending is None:
                raise Refused(f"{m} without a compare")
            kind, a, b, bits = pending
            pending = None
            suffix = m[1:]
            target = ops[0].imm
            cond = condition((kind, a, b, bits), suffix)
            out.append(f"if ({cond}) goto L{target:#x};")
            continue
        if m == "jmp":
            target = ops[0].imm
            if start <= target < end:
                out.append(f"goto L{target:#x};")
            else:
                out.append(f"f_{target:#x}(r, stk);")
                out.append("return;")
            continue
        if m.startswith("set") or m.startswith("cmov"):
            suffix = m[3:] if m.startswith("set") else m[4:]
            if pending is None:
                raise Refused(f"{m} without a compare")
            cond = condition(pending, suffix)
            pending = None
            if m.startswith("set"):
                out.append(module.wr(ops[0], f"({cond}) ? 1 : 0", insn))
            else:
                out.append(f"if ({cond}) {module.wr(ops[0], module.src(ops[1], insn), insn)}")
            continue
        if m in ("adc", "sbb", "rcl", "rcr"):
            raise Refused(m)
        if m not in FLAG_SAFE:
            # Nothing here reads the flags, but this writes them: a later
            # branch must not use the compare before it.
            pending = None
        if m == "call":
            target = ops[0].imm
            out.append(f"f_{target:#x}(r, stk);")
            pending = None  # a call does not preserve the flags
            continue

        out.extend(module.lift_one(insn))
        # A write to a flag-setting instruction invalidates the pending compare
        # only if something later reads flags we did not model; since nothing
        # does (any reader is refused), the pending compare can stay until a
        # branch consumes it.
    CALLS[rva] = calls
    return out
