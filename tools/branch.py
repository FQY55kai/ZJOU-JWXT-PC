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


def lift_function(code, rva):
    """C statements for a function with branches, or Refused."""
    insns = list(module.MD.disasm(code, rva))
    while insns and insns[-1].mnemonic in ("int3", "nop", "ud2"):
        insns.pop()
    if not insns:
        raise Refused("empty")
    if insns[-1].mnemonic == "jmp":
        raise Refused("ends in a tail call")
    if insns[-1].mnemonic != "ret":
        raise Refused("does not end in ret")

    start, end = rva, rva + len(code)
    targets = set()
    for insn in insns:
        m = insn.mnemonic
        if m.startswith("j"):
            if not insn.operands or insn.operands[0].type != capstone.x86.X86_OP_IMM:
                raise Refused("indirect branch")
            target = insn.operands[0].imm
            if not start <= target < end:
                raise Refused("branch out of the function")
            targets.add(target)
        if m == "call":
            raise Refused("calls")

    out = []
    # `cmp`/`test` seen but not yet consumed by a branch: the operands as C.
    pending = None
    for insn in insns:
        m, ops = insn.mnemonic, insn.operands
        if insn.address in targets:
            out.append(f"L{insn.address:#x}: ;")
        if m in ("cmp", "test"):
            pending = (m, module.src(ops[0]), module.src(ops[1]), ops[0].size * 8)
            continue
        if m.startswith("j") and m != "jmp":
            if pending is None:
                raise Refused(f"{m} without a compare")
            kind, a, b, bits = pending
            pending = None
            suffix = m[1:]
            target = ops[0].imm
            if kind == "test":
                # `test a, a` / `test a, imm` only ever asks for zero or sign.
                if suffix == "e":
                    cond = f"(({a}) == 0)"
                elif suffix == "ne":
                    cond = f"(({a}) != 0)"
                elif suffix == "s":
                    cond = f"((({a}) >> {bits - 1}) & 1)"
                elif suffix == "ns":
                    cond = f"!((({a}) >> {bits - 1}) & 1)"
                else:
                    raise Refused(f"test {m}")
            elif suffix in EQUALITY:
                cond = f"(({a}) {EQUALITY[suffix]} ({b}))"
            elif suffix in UNSIGNED:
                cond = f"(({a}) {UNSIGNED[suffix]} ({b}))"
            elif suffix in SIGNED:
                cond = f"((int{bits}_t)({a}) {SIGNED[suffix]} (int{bits}_t)({b}))"
            else:
                raise Refused(f"jcc {m}")
            out.append(f"if ({cond}) goto L{target:#x};")
            continue
        if m == "jmp":
            out.append(f"goto L{ops[0].imm:#x};")
            continue
        if m in ("setcc", "cmovcc") or m.startswith("set") or m.startswith("cmov"):
            raise Refused(m)
        if m in ("adc", "sbb", "rcl", "rcr"):
            raise Refused(m)
        if m not in FLAG_SAFE:
            # Nothing here reads the flags, but this writes them: a later
            # branch must not use the compare before it.
            pending = None
        out.extend(module.lift_one(insn))
        # A write to a flag-setting instruction invalidates the pending compare
        # only if something later reads flags we did not model; since nothing
        # does (any reader is refused), the pending compare can stay until a
        # branch consumes it.
    return out
    return out
