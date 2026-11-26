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
                    "andn", "bextr", "xadd", "blsi", "blsmsk", "blsr", "popcnt"}

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
        # 进位与溢出：`add`/`sub` 的 CF 是"结果往回退不回去"，OF 是两个源与结果的
        # 符号位组合。手头只有结果和**源**操作数，但另一个操作数可以从这两者反推：
        # `add` 的被加数是 `结果 - 源`，`sub` 的被减数是 `结果 + 源`（都在模 2^bits
        # 下）。所以不必在 ALU 之前多存一份目标值。
        if b is None:
            raise Refused(f"alu result only for zf/sf, not {suffix}")
        opk, bv = b
        u = f"uint{bits}_t"
        zf = f"(({a}) == 0)"
        sign = f"((({a}) >> {bits - 1}) & 1)"
        if opk == "add":
            av = f"(({u})(({a}) - ({bv})))"
            cf = f"((({u})({a})) < ({av}))"
            of = f"((((({av}) ^ ({a})) & (({bv}) ^ ({a}))) >> {bits - 1}) & 1)"
        elif opk == "sub":
            av = f"(({u})(({a}) + ({bv})))"
            cf = f"((({av}) < ({u})({bv})))"
            of = f"((((({av}) ^ ({bv})) & (({av}) ^ ({a}))) >> {bits - 1}) & 1)"
        elif opk == "inc":
            # `inc`/`dec` 不碰 CF，所以凡是要进位的条件都没法回答；溢出则仍是结果的
            # 事：加一溢出当且仅当结果是最小值，减一溢出当且仅当结果是最大值。
            cf = None
            of = f"((({a}) == (({u})1 << {bits - 1})) ? 1 : 0)"
        elif opk == "dec":
            cf = None
            of = f"((({a}) == ((({u})~0) >> 1)) ? 1 : 0)"
        elif opk == "zero":
            # `popcnt` 只让 ZF 跟着结果走，其余三位一律清零。
            cf = "0"
            of = "0"
        else:
            # 逻辑运算把 CF 与 OF 清零，所以进位条件成了常量。
            cf = "0"
            of = "0"
        if suffix in ("c", "b", "nc", "ae", "be", "a", "nbe") and cf is None:
            raise Refused(f"{opk} keeps the carry, so {suffix} is not the result's")
        if suffix == "o":
            return of if opk != "logic" else "0"
        if suffix == "no":
            return f"!({of})" if opk != "logic" else "1"
        if suffix in ("c", "b"):
            return cf
        if suffix in ("nc", "ae"):
            return f"!({cf})"
        if suffix == "be":
            return f"(({cf}) || {zf})"
        if suffix in ("a", "nbe"):
            return f"(!({cf}) && !{zf})" if opk != "logic" else f"(!{zf})"
        if opk == "zero":
            sign = "0"
        if suffix in ("l", "nge"):
            return f"({sign} != ({of}))"
        if suffix in ("ge", "nl"):
            return f"({sign} == ({of}))"
        if suffix in ("le", "ng"):
            return f"({zf} || ({sign} != ({of})))"
        if suffix in ("g", "nle"):
            return f"(!{zf} && ({sign} == ({of})))"
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
    if kind == "vtest":
        # PTEST/PTEST-like flags, per SDM: the first operand is the ModRM.reg one
        # (`a`) and the second the r/m one (`b`).
        #
        #   ZF = ((a AND b) == 0)
        #   CF = ((b AND NOT a) == 0)
        #
        # and OF/SF/AF/PF cleared. Not the pair the names suggest: the AND lands
        # in ZF, the ANDNOT in CF. Written the other way round, every `jz` after
        # a `vtestps` is inverted — and it looks right on the common
        # `vtestps x, x` because both forms then agree.
        if suffix in ("e", "z"):
            return f"(({a}) == 0)"
        if suffix in ("ne", "nz"):
            return f"(({a}) != 0)"
        if suffix in ("c", "b"):
            return f"(({b}) == 0)"
        if suffix in ("nc", "ae"):
            return f"(({b}) != 0)"
        if suffix == "be":
            return f"((({b}) == 0) || (({a}) == 0))"
        if suffix == "a":
            return f"((({b}) != 0) && (({a}) != 0))"
        # The rest were cleared, so they are constants rather than refusals:
        # `js` after a ptest never branches, `jns` always does, and parity is
        # even-because-zero.
        if suffix in ("s", "o", "p", "pe"):
            return "0"
        if suffix in ("ns", "no", "np", "po"):
            return "1"
        raise Refused(f"vtest {suffix}")
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
    if kind in ("cmp", "cmp_nocf") and suffix in ("s", "ns"):
        # `js` after a compare is the sign of the difference, which is not
        # either operand's sign: `cmp 3, 5` is negative although both are
        # positive. Computed at the operand width, like every other compare.
        diff = f"(uint{bits}_t)((uint{bits}_t)({a}) - (uint{bits}_t)({b}))"
        sign = f"((({diff}) >> {bits - 1}) & 1)"
        return sign if suffix == "s" else f"!{sign}"
    if kind == "cmp_nocf":
        # A `rol`/`ror` in between wrote CF and OF and left ZF/SF alone: only the
        # equality and the sign bit are still the compare's.
        raise Refused(f"rotate between the compare and {suffix}")
    if kind == "cmp" and suffix in ("o", "no"):
        # Signed overflow of `a - b`: it happens exactly when the operands differ
        # in sign and the difference disagrees with `a`. Two xors and an and,
        # which is why it is computed here rather than refused.
        diff = f"(uint{bits}_t)((uint{bits}_t)({a}) - (uint{bits}_t)({b}))"
        ovf = (f"(((((uint{bits}_t)({a}) ^ (uint{bits}_t)({b}))"
               f" & ((uint{bits}_t)({a}) ^ {diff})) >> {bits - 1}) & 1)")
        return ovf if suffix == "o" else f"!{ovf}"
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


def alu_shape(mnemonic):
    """How an ALU op produces CF/OF, or None when this model cannot say.

    `add` 与 `sub` 的进位方向相反，逻辑运算把两位都清零，其余（`inc`/`dec` 不动
    CF，`blsi`/`bextr`/`andn` 从别处取 CF）不在这条规则的覆盖里——认不出来就退回
    只支持 ZF/SF，而不是给一个看着像的答案。
    """
    m = mnemonic[5:] if mnemonic.startswith("lock ") else mnemonic
    if m in ("add", "xadd"):
        return "add"
    if m == "sub":
        return "sub"
    if m in ("and", "or", "xor"):
        return "logic"
    if m == "popcnt":
        return "zero"       # CF/OF/SF 全清，只有 ZF 是结果的事
    if m == "inc":
        return "inc"
    if m == "dec":
        return "dec"
    return None


def flag_step(pending, insn):
    """What `insn` leaves behind for the flags: a token, or None.

    Mirrors the bookkeeping inside `lift_function`, and exists separately so the
    block dataflow below can run *before* anything is emitted. The tokens name
    their operands by the address of the producing instruction (`pa_<at>`), not
    by a running counter, so a consumer three blocks away can name them too.
    """
    m = insn.mnemonic
    at = insn.address
    if m.startswith("lock "):
        m = m[5:]
    if m == "cmpxchg":
        return ("cmp", f"pa_{at:x}", f"pb_{at:x}", module.width(insn.operands[0]))
    if m == "cmp":
        return ("cmp", f"pa_{at:x}", f"pb_{at:x}", insn.operands[0].size * 8)
    if m == "test":
        return ("test", f"pa_{at:x}", "0", insn.operands[0].size * 8)
    if m in module.FLOAT_COMPARE:
        return ("fcom", f"pf_{at:x}", f"pg_{at:x}", module.FLOAT_COMPARE[m] * 8)
    if m in ("ptest", "vptest", "vtestps", "vtestpd"):
        return ("vtest", f"pz_{at:x}", f"pc_{at:x}", 64)
    if m == "bt":
        return ("cf", f"pb_{at:x}", None, module.width(insn.operands[0]))
    if m in ("rol", "ror"):
        # A rotate writes CF and OF and keeps the rest, so equality and the sign
        # are still the compare's; carry and the signed ordering are not.
        if pending is None:
            return None
        if pending[0] in ("cmp", "test"):
            return ("cmp_nocf",) + tuple(pending[1:])
        return pending if pending[0] == "alu" else None
    if m.startswith("set"):
        return pending           # reads a flag, writes a byte
    if m.startswith("cmov"):
        return pending           # reads the flags and writes a register
    if m == "call":
        return None              # a call does not preserve them
    if m in ("bts", "btr", "btc"):
        # 改一位并把改动前的那一位送进 CF，其余标志不动——与 `bt` 同一个 token。
        if not insn.operands or insn.operands[0].type != capstone.x86.X86_OP_REG:
            return None
        return ("cf", f"pcb_{at:x}", None, module.width(insn.operands[0]))
    if m in ("adc", "sbb", "rcl", "rcr"):
        return None              # refused later, with the instruction's name
    if m.startswith("j") or m == "ret" or m == "retf":
        return pending           # a branch reads the flags and writes none
    if m not in FLAG_SAFE and not flag_neutral(m):
        return None
    if m in FLAG_FROM_RESULT:
        shape = alu_shape(m)
        if shape is not None and len(insn.operands) < 2 and shape not in ("inc", "dec"):
            shape = None
        # `inc`/`dec` 只有一个操作数，条件里也从不读第二个源，占位写 "1"。
        bv = f"pb_{at:x}" if len(insn.operands) >= 2 else "1"
        return ("alu", f"pa_{at:x}", (shape, bv) if shape else None,
                module.width(insn.operands[0]))
    return pending


def flag_flow(insns, block_starts, preds, rva):
    """The pending token at each block entry, from the entry block onwards.

    A block inherits the flags of its predecessor only when it has exactly one
    way in: with two, the paths may have left different flags and choosing one
    is a guess. This is what a linear scan cannot do — `cmp eax,3 / jae L1 /
    … / L1: jne L0` puts the merge point *between* the compare and the branch
    that reads it, so the linear state is gone by the time the branch arrives,
    even though the branch itself has a single way in.
    """
    starts = set(block_starts)
    owner, cur = {}, None
    for at in sorted(insns):
        if at in starts:
            cur = at
        owner[at] = cur
    pending_in = {b: None for b in block_starts}
    settled = False
    for _ in range(len(block_starts) + 2):
        out, prev = {}, {}
        for at in sorted(insns):
            b = owner[at]
            seed = pending_in[b] if at == b else prev.get(b)
            prev[b] = flag_step(seed, insns[at])
            out[at] = prev[b]
        changed = set()
        for b in block_starts:
            if b == rva:
                continue
            comes = preds.get(b)
            if not comes or len(comes) != 1:
                continue
            want = out.get(next(iter(comes)))
            if want != pending_in[b]:
                pending_in[b] = want
                changed.add(b)
        if not changed:
            settled = True
            break
    if not settled:
        # A cycle that never settles: the flags at its entry depend on the
        # iteration, so there is no answer here. Refusing to choose is the only
        # honest option left.
        for b in block_starts:
            pending_in[b] = None
    return pending_in


def predecessors(insns, dispatches):
    """(target -> {source addresses}) over the edges this function has.

    `walk` knows which addresses are entered by an edge but not by whom, and the
    difference matters for the flags: a label with one way in can inherit the
    flags of the instruction that leads to it, a label with two cannot — the two
    paths may have left different flags behind, and picking either one is a
    guess that a differential test would have to catch later, at run time, on
    some input nobody chose.
    """
    preds = {}
    for at, insn in insns.items():
        m = insn.mnemonic
        nxt = at + insn.size
        if m in ("ret", "retf"):
            continue
        if at in dispatches:
            # Every case of the table is an edge in, so the cases are merged
            # points by construction.
            for case in dispatches[at]["entries"]:
                preds.setdefault(case, set()).add(at)
            continue
        if m == "jmp":
            op = insn.operands[0] if insn.operands else None
            if op is not None and op.type == capstone.x86.X86_OP_IMM:
                preds.setdefault(op.imm, set()).add(at)
            continue
        if m.startswith("j"):
            op = insn.operands[0]
            if op.type == capstone.x86.X86_OP_IMM:
                preds.setdefault(op.imm, set()).add(at)
        if nxt in insns:
            preds.setdefault(nxt, set()).add(at)
    return preds


def lift_function(code, rva):
    """C statements for a function with branches, or Refused."""
    insns, starts, calls, dispatches = walk(code, rva)
    if not insns:
        raise Refused("empty")
    start, end = rva, rva + len(code)

    preds = predecessors(insns, dispatches)
    block_starts = sorted(set(starts) | {rva})
    arriving = flag_flow(insns, block_starts, preds, rva)

    out = []
    # `cmp`/`test` seen but not yet consumed by a branch: the operands as C,
    # and the address of the instruction that produced them — a label may only
    # inherit them when that producer is at or before its one and only way in.
    pending = None
    # Comparands are snapshotted into locals at the compare, because a
    # flag-neutral `mov` in between can overwrite a register the compare read:
    # `cmp eax, 0 / mov eax, [rbx] / jz L` branches on the *old* eax.
    snap = 0
    for at in sorted(insns):
        insn = insns[at]
        # A run that an edge enters cannot borrow the flags of the run before
        # it; it takes the ones the dataflow says arrive here, which is a token
        # only where every way in left the same compare behind.
        if at in starts:
            pending = arriving.get(at)
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
            # The flag operands are named after the address of the instruction
            # that produced them, not after a running counter: a consumer in
            # another block has to name them before this block is emitted.
            out.append(f"{wide} pa_{at:x} = {module.src(ops[0], insn)};")
            out.append(f"{wide} v{snap} = {module.src(ops[1], insn)};")
            out.append(f"{wide} pb_{at:x} = (uint{bits}_t)r[0];")
            if bits == 64:
                back = f"r[0] = (uint64_t)pa_{at:x};"
            elif bits == 32:
                back = f"r[0] = (uint64_t)(uint32_t)pa_{at:x};"
            else:
                mask = module.ONES[bits]
                back = f"r[0] = (r[0] & ~{mask}) | ((uint64_t)pa_{at:x} & {mask});"
            out.append(f"if (pb_{at:x} == pa_{at:x}) {module.wr(ops[0], f'v{snap}', insn)}"
                       f" else {back}")
            pending = ("cmp", f"pa_{at:x}", f"pb_{at:x}", bits)
            continue
        if m in ("cmp", "test"):
            snap += 1
            a, b = module.src(ops[0], insn), module.src(ops[1], insn)
            if m == "test":
                # `test` 的 ZF/SF 来自 `a & b`，不是来自 `a`。`test byte [rcx+0xb0], 1`
                # 在那一字节是 2 时 ZF 为 1——只取第一个操作数就是把每一条位测试都做成
                # "非零即真"，方向正好反了。截到操作数宽度再与，符号位与奇偶位随之正确。
                bits = ops[0].size * 8
                wide = f"uint{bits}_t"
                out.append(f"uint64_t pa_{at:x} = (uint64_t)"
                           f"(({wide})({a}) & ({wide})({b}));")
                pending = ("test", f"pa_{at:x}", "0", bits)
            else:
                out.append(f"uint64_t pa_{at:x} = {a}, pb_{at:x} = {b};")
                pending = ("cmp", f"pa_{at:x}", f"pb_{at:x}", ops[0].size * 8)
            continue
        if m in module.FLOAT_COMPARE:
            # The two low lanes, snapshotted as C floats: a `mov` between the
            # compare and its branch could otherwise overwrite the register.
            if len(ops) != 2:
                raise Refused(f"{m} arity")
            snap += 1
            size = module.FLOAT_COMPARE[m]
            ctype = "float" if size == 4 else "double"
            out.append(f"{ctype} pf_{at:x} = {module.float_operand(ops[0], size, insn)}"
                       f", pg_{at:x} = {module.float_operand(ops[1], size, insn)};")
            pending = ("fcom", f"pf_{at:x}", f"pg_{at:x}", size * 8)
            continue
        if m in ("ptest", "vptest", "vtestps", "vtestpd"):
            # 它写 CF 与 ZF，所以不能只当一条 SIMD 指令塞进 `vec_line`——那条路的
            # 契约是"不碰标志"。
            #
            # 而且 VTESTPS/VTESTPD 只比**符号位**：SDM 说的是 "bitwise comparison
            # of all the sign bits of the packed elements"，不是整字的与。小端下
            # 每个 32 位元素的符号位落在该元素第 4 个字节的最高位，pd 是第 8 个
            # 字节。按整字算（`ptest` 的算法）会在两个同号数上给出相反答案。
            if len(ops) != 2:
                raise Refused(f"{m} arity")
            snap += 1
            size = 32 if ops[0].size == 32 else 16
            if m == "vtestpd":
                sign = "(((k) & 7u) == 7u ? 0x80 : 0)"
            elif m == "vtestps":
                sign = "(((k) & 3u) == 3u ? 0x80 : 0)"
            else:
                sign = "0xFF"      # `ptest` 是整字的与，没有符号位可挑
            # `lane_ptr`, not `lane_at`: a byte-wise reader needs the lane's
            # address, and `lane_at` is its value.
            left = module.lane_ptr(ops[0], 0, insn)
            right = module.lane_ptr(ops[1], 0, insn)
            out.append(f"const uint8_t *lp{snap} = {left}, *rp{snap} = {right};")
            out.append(f"uint64_t pz_{at:x} = 0, pc_{at:x} = 0;")
            out.append(f"for (unsigned k = 0; k < {size}u; k++) {{")
            out.append(f"    const uint8_t mk = (uint8_t)({sign});")
            out.append(f"    pz_{at:x} |= (uint64_t)(lp{snap}[k] & rp{snap}[k] & mk);")
            out.append(f"    pc_{at:x} |= (uint64_t)(rp{snap}[k] & (uint8_t)~lp{snap}[k] & mk);")
            out.append("}")
            pending = ("vtest", f"pz_{at:x}", f"pc_{at:x}", 64)
            continue
        if m in ("rol", "ror"):
            # 旋转写 CF 与 OF，保留 ZF/SF/PF/AF。所以夹一个旋转之后，相等与符号位
            # 还是比较的，无符号条件与有符号序关系不再是——它们读的是旋转的进位与溢出。
            if pending is not None and pending[0] in ("cmp", "test"):
                pending = ("cmp_nocf",) + tuple(pending[1:])
            elif pending is not None and pending[0] != "alu":
                pending = None
            out.extend(module.lift_one(insn))
            continue
        if m.startswith("j") and m != "jmp":
            if pending is None:
                raise Refused(f"{m} without a compare")
            suffix = m[1:]
            cond = condition(pending, suffix)
            # A `jcc` reads the flags and writes none, so the compare is still
            # the live one for the next consumer — the fall-through included.
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
            out.append(f"uint64_t pb_{at:x} = (a{snap} >> ({offset} & {bits - 1})) & 1;")
            pending = ("cf", f"pb_{at:x}", None, bits)
            continue
        if m.startswith("set") or m.startswith("cmov"):
            suffix = m[3:] if m.startswith("set") else m[4:]
            if pending is None:
                raise Refused(f"{m} without a compare")
            cond = condition(pending, suffix)
            # Neither writes a flag: `setcc` writes a byte, `cmovcc` a register.
            # So `cmp / setcc / setcc` and `cmp / cmovae / cmovb` — both are
            # ordinary compiler output, the second reading the same comparison —
            # stay legal until an instruction that really writes the flags runs.
            if m.startswith("set"):
                out.append(module.wr(ops[0], f"({cond}) ? 1 : 0", insn))
            else:
                out.append(f"if ({cond}) {module.wr(ops[0], module.src(ops[1], insn), insn)}")
            continue
        if m in ("bts", "btr", "btc"):
            # 位测试并置位/清位/取反：CF 拿到**改动之前**那一位，其余标志不动。
            # 与 `bt` 同一条规则，所以用的是同一个 token 类型。
            # 内存目标是另一回事：位号会跨字节寻址，这里只做寄存器。
            if len(ops) != 2 or ops[0].type != capstone.x86.X86_OP_REG:
                raise Refused(f"{m} into memory")
            bits = module.width(ops[0])
            idx = ops[1].imm & (bits - 1) if ops[1].type == capstone.x86.X86_OP_IMM \
                else None
            snap += 1
            out.append(f"uint64_t pcb_{at:x};")
            out.append("{")
            out.append(f"    uint64_t x = (uint64_t)({module.src(ops[0], insn)});")
            if idx is None:
                out.append(f"    uint64_t b = (uint64_t)({module.src(ops[1], insn)}) "
                           f"& {bits - 1};")
            else:
                out.append(f"    uint64_t b = {idx};")
            out.append(f"    pcb_{at:x} = (x >> b) & 1;")
            out.append(f"    x = {'x | ((uint64_t)1 << b)' if m == 'bts' else ''}"
                       f"{'x & ~((uint64_t)1 << b)' if m == 'btr' else ''}"
                       f"{'x ^ ((uint64_t)1 << b)' if m == 'btc' else ''};")
            out.append("    " + module.wr(ops[0], "x", insn))
            out.append("}")
            pending = ("cf", f"pcb_{at:x}", None, bits)
            continue
        if m in ("adc", "sbb", "rcl", "rcr"):
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
        # 进位与溢出要用到源操作数，而它马上就没了，所以在指令执行前存一份。
        # 只在认得出来的 add/sub/and/or/xor 上存：其余形态的 CF 另有来处。
        shape = alu_shape(m) if m in FLAG_FROM_RESULT else None
        if shape is not None and len(ops) >= 2:
            out.append(f"uint64_t pb_{at:x} = (uint64_t)({module.src(ops[1], insn)});")
        elif shape not in ("inc", "dec"):
            shape = None
        out.extend(module.lift_one(insn))
        if m in FLAG_FROM_RESULT:
            # ZF and SF are the result's, and the result is in the destination
            # now — but a later `mov` may overwrite it before the branch reads
            # the flag, so it is snapshotted here.
            snap += 1
            out.append(f"uint64_t pa_{at:x} = (uint64_t)({module.src(ops[0], insn)});")
            pending = ("alu", f"pa_{at:x}",
                       (shape, f"pb_{at:x}" if len(ops) >= 2 else "1") if shape else None,
                       module.width(ops[0]))
    CALLS[rva] = calls
    return out
