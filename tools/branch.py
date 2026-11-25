"""Branches, jump tables and the flags they read, over a walked control flow.

A full lifter models CF/OF/SF/ZF/AF/PF and every instruction that writes them.
That is the correct end state and it is also three weeks of work that does not
have to come first: in real code the overwhelming majority of branches read the
flags of a `cmp`/`test` that is *right there*.

Two rules decide what is accepted, both taken from Xbox2Windows
(`tools/cpu_switch_aot.py`, see docs/reference-clones.md):

1. **Flags are modelled only between a producer and its consumers inside one
   straight-line run that no other edge enters.** `Jcc`, `SETcc` and `CMOVcc`
   consume; `mov`/`lea`/`not`/`bswap`/`push`/`pop`/`cbw`/`cwde`/`cdqe`/`cqo`
   leave them alone; anything else that writes them ends the run. The producer
   is `cmp`/`test`, or the result of an ALU op (ZF and SF are the result's;
   carry and overflow are not the comparison a condition names, so those
   conditions are refused).
2. **Instructions are reached, not scanned.** MSVC puts a jump table after the
   `jmp` that reads it, so a linear sweep decodes the table's own bytes as
   instructions. The walk here starts at the entry and follows edges; whatever
   no edge reaches is data.

An indirect `jmp` is accepted only as the dispatch of one of these sequences,
with the base the same RIP-relative address on the path and the index the one
the guard compared:

    mov E32, [B+I*4+table]; add E,B; jmp E   B = image base
    movsxd E,[B+I*4];       add E,B; jmp E   B = table

The dispatch becomes a `switch` over the table's targets. Everything else is
refused: `adc`/`sbb`, a `jcc` whose flags were written further back.

A call or a jump whose target is not an immediate is not refused, because the
target is a value and a value can be looked up. `call rax`, `call [rcx+0x18]`,
`jmp [rip+IAT]` and a tail call through a vtable slot all become the same shape
— read the operand, hand it to a dispatch that knows the function table — and
the indirect `jmp` shapes above are tried first, so a recognised jump table is
still a `switch` rather than a lookup. The site becomes

    uint64_t t = <the operand, read before the push>;
    *(uint64_t *)(uintptr_t)(r[4] - 8) = (uint64_t)(img + <return>);
    r[4] -= 8;
    call_indirect(r, t);
    r[4] += 8;

and the translation unit owes this module two hooks:

    void call_indirect(uint64_t *r, uint64_t target);
    void jump_indirect(uint64_t *r, uint64_t target);

The contract is the same for both: `target` is an address in the guest's own
space, `r` is the register file, the return address (for `call_indirect`) has
already been pushed and `r[4]` already lowered. The hook routes to the lifted
body if this unit has one at that address and otherwise runs the original bytes
there — which is not a fallback in the sense of being worse, it is the same code
the console ran.
"""

import capstone

import lift as module

# `jcc` → the C relation, per operand size and signedness.
UNSIGNED = {"a": ">", "ae": ">=", "b": "<", "be": "<="}
SIGNED = {"g": ">", "ge": ">=", "l": "<", "le": "<="}
EQUALITY = {"e": "==", "ne": "!="}

# Instructions whose ZF and SF the next consumer can read off the result: the
# destination holds it, so a snapshot of the destination right after the write
# is the flag's source. CF and OF are *not* covered — after `add` they mean
# carry and signed overflow, not the comparison the condition names — so only
# e/ne/s/ns may read these.
FLAG_FROM_RESULT = {"add", "sub", "and", "or", "xor", "inc", "dec", "neg",
                    "andn", "bextr", "xadd", "blsi", "blsmsk", "blsr"}

# The vector unit's moves and bitwise ops. None of them write the arithmetic
# flags — the ones that do (`ucomiss`, `ptest`) are not translated at all — so a
# `cmp` in front of a spill of xmm6 is still live after it. Without this the
# prologue of every function that touches an xmm register would invalidate the
# compare in front of its first branch.
VEC_FLAGLESS = {
    "movss", "movsd", "movd", "movq", "movaps", "movups", "movapd", "movupd",
    "movdqa", "movdqu", "lddqu", "xorps", "xorpd", "pxor", "andps", "andpd",
    "pand", "orps", "orpd", "por", "andnps", "andnpd", "pandn",
    "vzeroupper", "vzeroall",
}

# ... and the same for every other vector stem this lifter knows. Built from
# `lift`'s tables rather than written out again, so a family added there is
# flag-neutral here without a second edit — which is the kind of omission that
# refuses a function for a reason that has nothing to do with the instruction
# that actually stopped it.
FLAGLESS_STEMS = VEC_FLAGLESS | module.VEC_STEMS


def flag_neutral(mnemonic):
    """True when the instruction leaves the arithmetic flags alone."""
    stem = mnemonic[1:] if mnemonic.startswith("v") else mnemonic
    return stem in FLAGLESS_STEMS or module.cmp_form(stem) is not None


# Instructions that leave the flags alone, so a compare stays live across them.
# Everything else that is not a consumer invalidates it: a `cmp` followed by an
# `add` and then a `jz` branches on the add, not on the compare.
FLAG_SAFE = {
    "mov", "movabs", "movzx", "movsx", "movsxd", "lea", "not", "bswap", "nop",
    "push", "pop", "cbw", "cwde", "cdqe", "cdq", "cqo", "xchg", "endbr64",
    "shlx", "shrx", "sarx", "mulx",
    "sete", "setne", "sets", "setns", "seta", "setae", "setb", "setbe",
    "setg", "setge", "setl", "setle", "setc", "setnc", "seto", "setno",
    "setp", "setnp",
}


class Refused(Exception):
    pass


# Call targets seen while lifting one function, so a closure can be walked
# without decoding everything twice.
CALLS: dict[int, set[int]] = {}

# The image, for reading jump tables. Set once by whoever holds the PE.
_IMAGE = None


def set_image(pe):
    """Bind the image so a dispatch's table can be read. Optional: without it a
    jump table is refused rather than guessed."""
    global _IMAGE
    _IMAGE = pe


def image_bytes(rva, length):
    if _IMAGE is None:
        raise Refused("jump table without the image")
    data = _IMAGE.get_data(rva, length)
    if len(data) != length:
        raise Refused("jump table outside the image")
    return data


def condition(pending, suffix):
    """The C condition for a consumer of `pending`, or Refused."""
    kind, a, b, bits = pending
    if kind == "cf":
        # `BT` selects one bit into CF and leaves the rest untouched, so only
        # the carry conditions may read it.
        if suffix in ("c", "b"):
            return f"(({a}) != 0)"
        if suffix in ("nc", "ae"):
            return f"(({a}) == 0)"
        raise Refused(f"cf only for carry, not {suffix}")
    if kind == "alu":
        if suffix == "e":
            return f"(({a}) == 0)"
        if suffix == "ne":
            return f"(({a}) != 0)"
        if suffix == "s":
            return f"((({a}) >> {bits - 1}) & 1)"
        if suffix == "ns":
            return f"!((({a}) >> {bits - 1}) & 1)"
        # Carry, overflow and parity after an ALU op are not the comparison the
        # condition names (`jb` after `add` reads the carry-out).
        raise Refused(f"alu result only for zf/sf, not {suffix}")
    if kind == "test":
        if suffix == "e":
            return f"(({a}) == 0)"
        if suffix == "ne":
            return f"(({a}) != 0)"
        if suffix == "s":
            return f"((({a}) >> {bits - 1}) & 1)"
        if suffix == "ns":
            return f"!((({a}) >> {bits - 1}) & 1)"
        # `TEST` clears CF and OF, so the signed conditions are the sign bit's
        # and the carry conditions are constants — `jb` after `test` never
        # branches, `jae` always does.
        sign = f"((({a}) >> {bits - 1}) & 1)"
        if suffix == "l":
            return sign
        if suffix == "ge":
            return f"!{sign}"
        if suffix == "le":
            return f"((({a}) == 0) || {sign})"
        if suffix == "g":
            return f"((({a}) != 0) && !{sign})"
        if suffix == "b":
            return "0"
        if suffix == "ae":
            return "1"
        raise Refused(f"test {suffix}")
    if kind == "fcom":
        # `ucomiss`/`ucomisd` and their ordered twins, as the four conditions
        # the SDM defines: unordered sets ZF=PF=CF=1, greater clears all three,
        # less sets CF, equal sets ZF. C's relational operators are false for a
        # NaN, which is what makes the negated forms below exactly right: `jb`
        # is `!(a >= b)` and so is true when either side is NaN, just as CF=1
        # is. Getting this wrong by writing `(a) < (b)` would lose every
        # unordered case, and compilers emit this pair precisely to catch them.
        unordered = f"(({a}) != ({a}) || ({b}) != ({b}))"
        if suffix in ("e", "z"):
            return f"(!(({a}) < ({b})) && !(({b}) < ({a})))"
        if suffix in ("ne", "nz"):
            return f"(({a}) < ({b}) || ({b}) < ({a}))"
        if suffix == "a":
            return f"(({a}) > ({b}))"
        if suffix == "ae":
            return f"(({a}) >= ({b}))"
        if suffix == "b" or suffix == "c":
            return f"(!(({a}) >= ({b})))"
        if suffix == "be":
            return f"(!(({a}) > ({b})))"
        if suffix == "nc":
            return f"(({a}) >= ({b}))"
        if suffix == "p":
            return unordered
        if suffix == "np":
            return f"!{unordered}"
        raise Refused(f"float compare {suffix}")
    if suffix in EQUALITY:
        # The same truncation as the ordering conditions below, and for the same
        # reason. `cmp eax, -1` compares 0xFFFFFFFF against 0xFFFFFFFF and finds
        # them equal; comparing the 64-bit snapshots would compare
        # 0x00000000FFFFFFFF against 0xFFFFFFFFFFFFFFFF and find them different.
        # Found by the harness on `0x7c62b0`, a function whose whole job is
        # `cmp eax, -1; setne al`, and it is the same one-off signature as
        # `0x272070` from the previous round — one bug, two functions, and
        # neither the interpreter nor the emitter could see it.
        return f"((uint{bits}_t)({a}) {EQUALITY[suffix]} (uint{bits}_t)({b}))"
    if suffix in UNSIGNED:
        # Both sides are truncated to the operand's width first. The snapshots
        # are 64-bit, and an immediate arrives sign-extended to 64 bits, so
        # `cmp ecx, -2` would otherwise compare 0x00000000_FFFFFFFE against
        # 0xFFFFFFFF_FFFFFFFE and report "below" where the hardware reports
        # "not below" — which is exactly the boundary case the compare exists
        # to test. Found by the differential harness on `0x2839d0`, a function
        # whose whole job is `(x - 2) > 1`.
        return f"((uint{bits}_t)({a}) {UNSIGNED[suffix]} (uint{bits}_t)({b}))"
    if suffix in SIGNED:
        return f"((int{bits}_t)({a}) {SIGNED[suffix]} (int{bits}_t)({b}))"
    # Overflow and parity conditions need the producer's carry chain, which this
    # pass does not carry; they are refused rather than guessed.
    raise Refused(f"condition {suffix}")


def _reg_index(op):
    """The index of an operand's parent register, or None."""
    if op.type != capstone.x86.X86_OP_REG:
        return None
    name = module.MD.reg_name(op.reg)
    return module.SUBS[name][0] if name in module.SUBS else None


def _guard_bound(before, index):
    """`cmp I, n; ja L` in front of the dispatch: the table holds n+1 entries."""
    for insn in reversed(before[-4:]):
        if insn.mnemonic in ("ja", "jae", "jb", "jbe"):
            continue
        if insn.mnemonic == "cmp" and len(insn.operands) == 2:
            if _reg_index(insn.operands[0]) == index and insn.operands[1].type == capstone.x86.X86_OP_IMM:
                return insn.operands[1].imm
    return None


def _dispatch(recent, jmp, start, end):
    """The jump table a `jmp reg` dispatches through, or None.

    `recent` holds the last few instructions before the `jmp`, oldest first. The
    shape is `add E,B; jmp E` on top of `movsxd E,[B+I*4]` and a `lea B,[rip+t]`.
    """
    if len(recent) < 3 or not jmp.operands:
        return None
    add, load, lea = recent[-1], recent[-2], recent[-3]
    target = _reg_index(jmp.operands[0])
    if target is None or add.mnemonic != "add" or len(add.operands) != 2:
        return None
    if _reg_index(add.operands[0]) != target:
        return None
    if add.operands[1].type != capstone.x86.X86_OP_REG:
        return None
    base = _reg_index(add.operands[1])
    if base is None or base == target:
        return None
    if load.mnemonic not in ("movsxd", "mov") or len(load.operands) != 2:
        return None
    if _reg_index(load.operands[0]) != target:
        return None
    mem = load.operands[1]
    if mem.type != capstone.x86.X86_OP_MEM or mem.mem.scale != 4 or mem.mem.base == 0:
        return None
    if module.SUBS[module.MD.reg_name(mem.mem.base)][0] != base or mem.mem.index == 0:
        return None
    index = module.SUBS[module.MD.reg_name(mem.mem.index)][0]
    if lea.mnemonic != "lea" or len(lea.operands) != 2 or _reg_index(lea.operands[0]) != base:
        return None
    if lea.operands[1].type != capstone.x86.X86_OP_MEM:
        return None
    if lea.operands[1].mem.base == 0 or module.MD.reg_name(lea.operands[1].mem.base) != "rip":
        return None
    table = module.rip_target(lea, lea.operands[1].mem.disp) + mem.mem.disp

    # How many entries: the guard's bound when one is in reach, else up to the
    # first entry that leaves the function (the table is the only thing there).
    bound = _guard_bound(recent[:-3], index)
    entries = []
    limit = bound + 1 if bound is not None else 1024
    for k in range(limit):
        entry = int.from_bytes(image_bytes(table + 4 * k, 4), "little", signed=True)
        case = table + entry
        if not start <= case < end:
            if bound is None:
                break
            raise Refused("jump table entry leaves the function")
        entries.append(case)
    if not entries:
        return None
    return {"index": index, "entries": entries, "table": table}


def walk(code, rva):
    """Decode every instruction an edge reaches, and the dispatches met.

    Straight-line runs are decoded forward; a branch enqueues its target and a
    `jmp` ends the run. A jump table enqueues its cases, so the table's own
    bytes (which sit inside the function) are never decoded as code.
    """
    end = rva + len(code)
    insns = {}
    starts = set()
    calls = set()
    dispatches = {}
    todo = [rva]
    while todo:
        at = todo.pop()
        if at in starts or not rva <= at < end:
            continue
        starts.add(at)
        recent = []
        while True:
            if not rva <= at < end or at in insns:
                break
            insn = next(module.MD.disasm(code[at - rva:at - rva + 15], at), None)
            if insn is None:
                raise Refused("undecodable")
            insns[at] = insn
            m = insn.mnemonic
            nxt = at + insn.size
            if m == "call":
                if not insn.operands:
                    raise Refused("call without an operand")
                # `call rax` / `call [rcx+0x18]` names no target at translate
                # time: it is a value. It is not refused — the site is emitted
                # as `call_indirect` and resolved against the dispatch at run
                # time — but it adds no edge to `calls`, so nothing static here
                # can see what it reaches.
                if insn.operands[0].type == capstone.x86.X86_OP_IMM:
                    calls.add(insn.operands[0].imm)
            elif m in ("ret", "retf"):
                break
            elif m == "jmp":
                op = insn.operands[0] if insn.operands else None
                if op is None:
                    raise Refused("jmp without an operand")
                if op.type == capstone.x86.X86_OP_IMM:
                    if rva <= op.imm < end:
                        todo.append(op.imm)
                    else:
                        calls.add(op.imm)  # a tail call: the frame goes with it
                    break
                table = _dispatch(recent, insn, rva, end)
                if table is None:
                    # Not one of the two dispatch shapes — an import thunk
                    # (`jmp [rip+IAT]`) or a tail call through a vtable slot.
                    # The target is a value, so it is resolved at run time the
                    # same way an indirect call is, and the two entries of the
                    # function table are known here even if the one this jump
                    # takes is not.
                    break
                dispatches[at] = table
                for case in table["entries"]:
                    if rva <= case < end:
                        todo.append(case)
                    else:
                        raise Refused("jump table entry out of the function")
                break
            elif m.startswith("j"):
                op = insn.operands[0]
                if op.type != capstone.x86.X86_OP_IMM:
                    raise Refused("indirect branch")
                if rva <= op.imm < end:
                    todo.append(op.imm)
                else:
                    raise Refused("branch out of the function")
            recent.append(insn)
            if len(recent) > 4:
                recent.pop(0)
            at = nxt
    return insns, starts, calls, dispatches


def lift_function(code, rva):
    """C statements for a function with branches, or Refused."""
    insns, starts, calls, dispatches = walk(code, rva)
    if not insns:
        raise Refused("empty")
    start, end = rva, rva + len(code)

    out = []
    # `cmp`/`test` seen but not yet consumed by a branch: the operands as C.
    pending = None
    # Comparands are snapshotted into locals at the compare, because a
    # flag-neutral `mov` in between can overwrite a register the compare read:
    # `cmp eax, 0 / mov eax, [rbx] / jz L` branches on the *old* eax.
    snap = 0
    for at in sorted(insns):
        insn = insns[at]
        # A run that an edge enters cannot borrow the flags of the run before it.
        if at in starts:
            pending = None
            out.append(f"L{at:#x}: ;")
        m, ops = insn.mnemonic, insn.operands
        if m.startswith("lock "):
            # Capstone names the locked forms by the whole instruction. The
            # prefix is atomicity against another core, which this port does not
            # have, so the body is the inner instruction's — but the *flags* are
            # the inner instruction's too, and `jz` after `lock dec` reads them.
            m = m[5:]
        if at in dispatches:
            table = dispatches[at]
            snap += 1
            out.append(f"uint64_t i{snap} = r[{table['index']}];")
            out.append(f"switch (i{snap}) {{")
            for k, case in enumerate(table["entries"]):
                out.append(f"case {k}: goto L{case:#x};")
            # Unreachable on the console: the guard in front of the dispatch
            # bounds the index. Ending the function is what a stray index would
            # do here anyway, without running off the end of the switch.
            out.append("default: return;")
            out.append("}")
            continue
        if m == "cmpxchg":
            # The comparison is the destination against rax, taken before the
            # store, and ZF *is* its equality — so a `jne` after `lock cmpxchg`
            # is a `jne` on `dest - rax`, which is exactly what a `cmp` would
            # have left behind. The store then depends on the same comparison
            # the other way round: equal writes the source, unequal writes the
            # old value into rax.
            if len(ops) != 2:
                raise Refused("cmpxchg arity")
            bits = module.width(ops[0])
            wide = f"uint{bits}_t"
            snap += 1
            out.append(f"{wide} o{snap} = {module.src(ops[0], insn)};")
            out.append(f"{wide} v{snap} = {module.src(ops[1], insn)};")
            out.append(f"{wide} x{snap} = (uint{bits}_t)r[0];")
            if bits == 64:
                back = f"r[0] = (uint64_t)o{snap};"
            elif bits == 32:
                back = f"r[0] = (uint64_t)(uint32_t)o{snap};"
            else:
                mask = module.ONES[bits]
                back = f"r[0] = (r[0] & ~{mask}) | ((uint64_t)o{snap} & {mask});"
            out.append(f"if (x{snap} == o{snap}) {module.wr(ops[0], f'v{snap}', insn)}"
                       f" else {back}")
            pending = ("cmp", f"o{snap}", f"x{snap}", bits)
            continue
        if m in ("cmp", "test"):
            snap += 1
            a, b = module.src(ops[0], insn), module.src(ops[1], insn)
            out.append(f"uint64_t a{snap} = {a}, b{snap} = {b};")
            pending = (m, f"a{snap}", f"b{snap}", ops[0].size * 8)
            continue
        if m in module.FLOAT_COMPARE:
            # The two low lanes, snapshotted as C floats: a `mov` between the
            # compare and its branch could otherwise overwrite the register.
            if len(ops) != 2:
                raise Refused(f"{m} arity")
            snap += 1
            size = module.FLOAT_COMPARE[m]
            ctype = "float" if size == 4 else "double"
            out.append(f"{ctype} f{snap} = {module.float_operand(ops[0], size, insn)}"
                       f", g{snap} = {module.float_operand(ops[1], size, insn)};")
            pending = ("fcom", f"f{snap}", f"g{snap}", size * 8)
            continue
        if m.startswith("j") and m != "jmp":
            if pending is None:
                raise Refused(f"{m} without a compare")
            suffix = m[1:]
            cond = condition(pending, suffix)
            pending = None
            out.append(f"if ({cond}) goto L{ops[0].imm:#x};")
            continue
        if m == "jmp":
            if ops[0].type != capstone.x86.X86_OP_IMM:
                # `jmp rax` / `jmp [rax]`: a tail call whose target is a value.
                # Read it before anything else writes a register, then hand it
                # to the dispatch — which is the same shape as a lifted `j_`,
                # with the original bytes as the fallback. `rsp` already holds
                # the caller's return address, so there is nothing to push and
                # the callee's `ret` leaves for *our* caller.
                snap += 1
                out.append(f"uint64_t t{snap} = {module.src(ops[0], insn)};")
                out.append(f"jump_indirect(r, t{snap});")
                out.append("return;")
                continue
            target = ops[0].imm
            if start <= target < end:
                out.append(f"goto L{target:#x};")
            else:
                # A tail jump passes its caller's return address through: no
                # push, no frame, and the callee's `ret` leaves for the
                # caller of this function, not for us.
                out.append(f"j_{target:#x}(r);")
                out.append("return;")
            continue
        if m == "bt":
            if len(ops) != 2:
                raise Refused("bt arity")
            bits = module.width(ops[0])
            if ops[0].type == capstone.x86.X86_OP_MEM and ops[1].type != capstone.x86.X86_OP_IMM:
                raise Refused("bt memory with a register offset")
            snap += 1
            out.append(f"uint64_t a{snap} = {module.src(ops[0], insn)};")
            offset = str(ops[1].imm) if ops[1].type == capstone.x86.X86_OP_IMM else module.src(ops[1], insn)
            out.append(f"uint64_t b{snap} = (a{snap} >> ({offset} & {bits - 1})) & 1;")
            pending = ("cf", f"b{snap}", None, bits)
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
        if m in ("adc", "sbb", "rcl", "rcr", "bts", "btr", "btc"):
            raise Refused(m)
        if m == "call":
            # The hardware pushes a return address at `rsp - 8` and enters the
            # callee with `rsp` there. Reproduce both, or every frame the callee
            # builds sits eight bytes off and the stack it leaves behind is not
            # the stack the original left behind.
            ret = at + insn.size
            if ops[0].type != capstone.x86.X86_OP_IMM:
                # The target is a value, so it has to be read before the push:
                # `call qword ptr [rsp+0x28]` reads the slot *above* the return
                # address it is about to write, and the hardware computes the
                # operand before it writes anything. `t` is that value, and
                # `call_indirect` is the dispatch — the same table a lifted
                # `f_` would be reached through, with the original bytes as the
                # fallback for a target the unit does not know.
                snap += 1
                out.append(f"uint64_t t{snap} = {module.src(ops[0], insn)};")
                out.append(f"*(uint64_t *)(uintptr_t)(r[4] - 8) = (uint64_t)(img + {ret:#x});")
                out.append("r[4] -= 8;")
                out.append(f"call_indirect(r, t{snap});")
                out.append("r[4] += 8;")
                pending = None
                continue
            out.append(f"*(uint64_t *)(uintptr_t)(r[4] - 8) = (uint64_t)(img + {ret:#x});")
            out.append("r[4] -= 8;")
            out.append(f"f_{ops[0].imm:#x}(r);")
            out.append("r[4] += 8;")
            pending = None  # a call does not preserve the flags
            continue
        if m not in FLAG_SAFE and not flag_neutral(m):
            pending = None
        out.extend(module.lift_one(insn))
        if m in FLAG_FROM_RESULT:
            # ZF and SF are the result's, and the result is in the destination
            # now — but a later `mov` may overwrite it before the branch reads
            # the flag, so it is snapshotted here.
            snap += 1
            out.append(f"uint64_t a{snap} = (uint64_t)({module.src(ops[0], insn)});")
            pending = ("alu", f"a{snap}", None, module.width(ops[0]))
    CALLS[rva] = calls
    return out
