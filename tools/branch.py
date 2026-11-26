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
                    "andn", "bextr", "xadd", "blsi", "blsmsk", "blsr", "popcnt",
                    "tzcnt", "lzcnt"}

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


def flags_condition(zf, sign, cf, of, suffix):
    """四个标志各自的表达式 → 一个条件表达式。

    `adc`/`sbb` 与 ALU 的结果走同一张映射表：差别只在 `cf`/`of` 怎么算出来，
    而不是分支怎么读它。把这一段抽出来，是为了让"进位从哪来"和"条件怎么读"分开，
    否则每加一种标志生产者就要复制一遍十四个后缀。
    """
    if suffix in ("e", "z"):
        return zf
    if suffix in ("ne", "nz"):
        return f"!({zf})"
    if suffix == "s":
        return sign
    if suffix == "ns":
        return f"!({sign})"
    if suffix == "o":
        return of
    if suffix == "no":
        return f"!({of})"
    if suffix in ("c", "b"):
        return cf
    if suffix in ("nc", "ae"):
        return f"!({cf})"
    if suffix == "be":
        return f"(({cf}) || {zf})"
    if suffix in ("a", "nbe"):
        return f"(!({cf}) && !({zf}))"
    if suffix in ("l", "nge"):
        return f"({sign} != ({of}))"
    if suffix in ("ge", "nl"):
        return f"({sign} == ({of}))"
    if suffix in ("le", "ng"):
        return f"({zf} || ({sign} != ({of})))"
    if suffix in ("g", "nle"):
        return f"(!({zf}) && ({sign} == ({of})))"
    raise Refused(f"{suffix} is not a condition these flags name")


# 每个条件后缀读哪几位标志。合并点把四位各存成一个 0/1 变量，所以这张表在这里
# 有两个用处：算出"这个后缀需要哪几个变量"，以及把 `condition` 的四个入参接上。
SUFFIX_FLAGS = {
    "e": "z", "z": "z", "ne": "z", "nz": "z",
    "s": "s", "ns": "s",
    "o": "o", "no": "o",
    "c": "c", "b": "c", "nc": "c", "ae": "c",
    "be": "cz", "a": "cz", "nbe": "cz",
    "l": "so", "nge": "so", "ge": "so", "nl": "so",
    "le": "zso", "ng": "zso", "g": "zso", "nle": "zso",
}

# 反过来：想把某一位存成变量，要问 `condition` 是哪个后缀。进位有两个名字，而
# `cmp` 只认 `b` 那个（`c` 与 `b` 是同一位，但 `UNSIGNED` 表里只有 `b`），所以每
# 一位给一串别名，按顺序试第一个能答的。
FLAG_SUFFIX = {"z": ("e", "z"), "s": ("s",), "c": ("b", "c"), "o": ("o", "no")}


def flag_expr(token, flag):
    """某一位标志的 0/1 表达式，或者 Refused。

    别名要一个个试：`bt` 的进位只能按 `c`/`b` 问，`cmp` 的只能按 `b` 问，`test`
    的按 `b` 问出来是常量 0——三种都对，但只有试过才知道是哪一个。
    """
    for suffix in FLAG_SUFFIX[flag]:
        try:
            return condition(token, suffix)
        except Refused:
            continue
    raise Refused(f"this producer does not name {flag}")


def carry_expression(pending):
    """CF 作为**值**（0 或 1 的表达式），给 `adc`/`sbb` 读。

    标志模型里 CF 一直只是"分支能读的一个条件"。`adc`/`sbb` 要把它当加数，
    所以这里把每种生产者的 CF 写成一个 0/1 表达式。**算不出来的如实拒绝**：
    `inc`/`dec` 不动 CF、`rol`/`ror` 拿不到旧 CF——这几种不猜。
    """
    if pending is None:
        raise Refused("adc/sbb without a producer for the carry")
    kind, a, b, bits = pending
    u = f"uint{bits}_t"
    if kind == "cf":
        return f"(({a}) != 0)"
    if kind == "test":
        return "0"                       # `test` 清 CF
    if kind == "cmp":
        return f"((({u})({a})) < (({u})({b})))"
    if kind == "vtest":
        return f"(({b}) == 0)"           # CF = (SRC2 AND NOT SRC1) == 0
    if kind == "adcsbb":
        return f"(({b[3]}) & 1)"
    if kind == "alu":
        if b is None:
            raise Refused("adc/sbb after a producer with no carry")
        opk, bv = b[0], b[1]
        if opk == "add":
            av = f"(({u})(({a}) - ({bv})))"
            return f"((({u})({a})) < ({av}))"
        if opk == "sub":
            av = f"(({u})(({a}) + ({bv})))"
            return f"((({av}) < ({u})({bv})))"
        if opk in ("logic", "zero"):
            return "0"
        if opk == "zcf":
            return f"(({bv}) == 0)"
        if opk == "bextr":
            return "0"
        if opk == "bls":
            raise Refused("bls leaves the carry undefined")
        if opk == "neg":
            return f"(({bv}) != 0)"
        if opk in ("shl", "shr", "sar"):
            # 移出去的最后一位，和 `condition` 里同一个式子。
            n = b[2]
            if opk == "shl":
                return f"((({u})({bv}) >> {bits - n}) & 1)"
            return f"((({u})({bv}) >> {n - 1}) & 1)"
        raise Refused(f"adc/sbb after {opk}, which keeps the carry")
    if kind == "cmp_nocf":
        raise Refused("adc/sbb after a rotate")
    if kind == "fcom":
        # `ucomiss` 的 CF 就是"无序或小于"那一位，与 `jb` 读的是同一位——所以
        # `adc` 接在浮点比较后面时，进位数是 `condition(..., "b")`。
        return condition(pending, "b")
    if kind == "merge":
        # 合并点的进位是块首那个 0/1 变量——两条路各自在离开时写它。
        if "c" not in b:
            raise Refused("adc/sbb at a join that did not keep the carry")
        return f"(fc_{a:x} != 0)"
    raise Refused(f"adc/sbb after {kind}")


def condition(pending, suffix):
    """The C condition for a consumer of `pending`, or Refused."""
    kind, a, b, bits = pending
    if kind == "merge":
        # 两条路留下了不同的标志，谁都没法拿来当答案。能做的只有一件事：在每个
        # 出口把标志**存成变量**，在块首按变量读。硬件上标志本来就是状态，出口存
        # 一次就是它的行为，所以这不是猜；`a` 是块首地址，`b` 是这条路真的存下来
        # 的那几位（四个出口的可用集合的交集）。
        need = SUFFIX_FLAGS.get(suffix)
        if need is None or not set(need) <= set(b):
            raise Refused(f"{suffix} is not one of the flags this join kept")
        return flags_condition(f"(fz_{a:x} != 0)", f"(fs_{a:x} != 0)",
                               f"(fc_{a:x} != 0)", f"(fo_{a:x} != 0)", suffix)
    if kind == "cf":
        # `BT` selects one bit into CF and leaves the rest untouched, so only
        # the carry conditions may read it.
        if suffix in ("c", "b"):
            return f"(({a}) != 0)"
        if suffix in ("nc", "ae"):
            return f"(({a}) == 0)"
        raise Refused(f"cf only for carry, not {suffix}")
    if kind == "alu":
        if b is not None and b[0] == "zcf":
            # `tzcnt`/`lzcnt`：ZF 与 CF 都在说"源是零"，但**结果**不是——源为零时
            # 结果是位宽（32/64）而不是 0，所以 equality 不能按结果写，得按源写。
            # SF/OF/AF/PF 在 SDM 里未定义，凡是要它们的条件一律拒绝。
            zero = f"(({b[1]}) == 0)"
            if suffix in ("e", "z", "c", "b", "be"):
                return zero
            if suffix in ("ne", "nz", "nc", "ae"):
                return f"!{zero}"
            if suffix == "a":
                return f"!{zero}"
            raise Refused(f"tzcnt/lzcnt leaves {suffix} undefined")
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
        opk, bv = b[0], b[1]
        u = f"uint{bits}_t"
        zf = f"(({a}) == 0)"
        sign = f"((({a}) >> {bits - 1}) & 1)"
        if opk in ("shl", "shr", "sar"):
            # 移位量是立即数（`shift_shape` 只在那种情况下认它）：CF 是移出去的最后
            # 一位，OF 只在移一位时有意义，那一位是"符号位变了吗"。ZF/SF 还是结果
            # 的，PF 与 AF 不动（这里用不到）。
            n = b[2]
            if opk == "shl":
                cf = f"((({u})({bv}) >> {bits - n}) & 1)"
                of = f"((((({a}) >> {bits - 1}) & 1) ^ {cf}) & 1)"
            elif opk == "shr":
                cf = f"((({u})({bv}) >> {n - 1}) & 1)"
                of = f"((({u})({bv}) >> {bits - 1}) & 1)"
            else:
                cf = f"((({u})({bv}) >> {n - 1}) & 1)"
                of = "0"
        elif opk == "add":
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
        elif opk == "bextr":
            # `bextr`：CF/OF 清零，SF/AF/PF 未定义——只留 equality 与进位条件。
            cf = "0"
            of = "0"
            sign = None
        elif opk == "bls":
            cf = None       # 未定义：见 `alu_shape` 里那一条
            of = "0"
        elif opk == "neg":
            # `neg`：借位当且仅当源不是零；溢出当且仅当结果是那个最小值。
            cf = f"(({bv}) != 0)"
            of = f"((({a}) == (({u})1 << {bits - 1})) ? 1 : 0)"
        elif opk == "zcf":
            # `tzcnt`/`lzcnt`：ZF 与 CF 是同一个谓词（源是零），SF/OF/AF/PF 在
            # SDM 里是未定义——凡是需要它们的条件一律拒绝，不拿结果去凑。
            cf = f"(({bv}) == 0)"
            zf = f"(({bv}) == 0)"
            sign = None
            of = None
        elif opk == "zero":
            # `popcnt` 只让 ZF 跟着结果走，其余三位一律清零。
            cf = "0"
            of = "0"
        else:
            # 逻辑运算把 CF 与 OF 清零，所以进位条件成了常量。
            cf = "0"
            of = "0"
        if opk == "zero":
            sign = "0"
        # 四位里缺哪一位，就拒绝读那一位的条件——`inc` 的进位、`tzcnt` 的符号都是
        # 这样。缺的写成常量只是为了让调用方参数齐全，读到它的后缀上面已经拦掉了。
        marks = SUFFIX_FLAGS.get(suffix)
        if marks is None:
            raise Refused(f"{opk} does not name {suffix}")
        if (("s" in marks and sign is None) or ("c" in marks and cf is None)
                or ("o" in marks and of is None)):
            raise Refused(f"{opk} leaves {suffix} undefined")
        return flags_condition(zf or "0", sign or "0", cf or "0", of or "0", suffix)
    if kind == "adcsbb":
        # `adc`/`sbb` 的两个源与进位都是**变量**：四个标志全都能直接写出来，
        # 不用像 `add`/`sub` 那样从"结果 + 源"反推，所以进位链能一直连下去。
        opk, av, bv, cfv = b
        zf = f"(({a}) == 0)"
        sign = f"((({a}) >> {bits - 1}) & 1)"
        if opk == "adc":
            of = f"((((({av}) ^ ({a})) & (({bv}) ^ ({a}))) >> {bits - 1}) & 1)"
        else:
            of = f"((((({av}) ^ ({bv})) & (({av}) ^ ({a}))) >> {bits - 1}) & 1)"
        return flags_condition(zf, sign, f"(({cfv}) & 1)", of, suffix)
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


def kept_flags(token):
    """Which of the four flags `token` can be asked for at all.

    "Can be asked" is the same question as "does `condition` answer", so it is
    asked by calling it: one probe per flag, and the ones that raise are the
    ones this producer does not name. A join may then keep only the intersection
    of what every way in can produce — and refuse a consumer that wants one the
    paths disagree about, rather than picking a path.
    """
    keep = set()
    if token is not None:
        for flag in FLAG_SUFFIX:
            try:
                flag_expr(token, flag)
            except Refused:
                continue
            keep.add(flag)
    return frozenset(keep)


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
    if m in ("and", "or", "xor", "andn"):
        return "logic"      # `andn` 同样把 CF/OF 清零，SF/ZF 是结果的
    if m == "bextr":
        return "bextr"      # CF/OF 清零，SF 在 SDM 里未定义
    if m in ("blsi", "blsmsk", "blsr"):
        # 溢出清零是文档写明的；进位这三位在 SDM 里不都定义得一样，而片段差分手里
        # 那台机器跑完 `blsi` 之后 CF 是 1（源并不是零），说明它不是"源是零"那一位
        # ——所以这里只承认 OF，进位照旧拒绝，不拿一个看着像的式子顶上。
        return "bls"
    if m == "neg":
        return "neg"        # 借位是"源不是零"，溢出是"结果是那个最小值"
    if m == "popcnt":
        return "zero"       # CF/OF/SF 全清，只有 ZF 是结果的事
    if m in ("tzcnt", "lzcnt"):
        return "zcf"        # ZF 与 CF 都说"源是零"，SF/OF 未定义
    if m == "inc":
        return "inc"
    if m == "dec":
        return "dec"
    return None


def shift_shape(insn):
    """(形状, 位移量) for a scalar shift, or None when it cannot be pinned down.

    `shl`/`shr`/`sar` leave all four flags defined, but only for a count that is
    known and non-zero: SDM says a count of 0 leaves every flag alone, so a
    static "the result decides ZF" is wrong there, and a count in `cl` is not
    known at translate time at all. Both cases are refused rather than answered
    with the wrong one — a shift by a register is common, but a wrong branch is
    worse than a missing function.
    """
    m = insn.mnemonic
    if m not in ("shl", "sal", "shr", "sar") or len(insn.operands) != 2:
        return None
    count = insn.operands[1]
    if count.type != capstone.x86.X86_OP_IMM:
        return None
    bits = insn.operands[0].size * 8
    n = count.imm & (bits - 1)
    if n == 0:
        return None
    return ("shl" if m == "sal" else m), n


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
    if m in ("shl", "sal", "shr", "sar"):
        shift = shift_shape(insn)
        if shift is None:
            return None
        return ("alu", f"pa_{at:x}", (shift[0], f"pb_{at:x}", shift[1]),
                module.width(insn.operands[0]))
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
        # 内存目标也一样：CF 是一位，地址怎么算与它无关。
        if not insn.operands:
            return None
        return ("cf", f"pb_{at:x}", None, 64)
    if m in ("adc", "sbb"):
        # 能算出进位就留下一个带进位的 token，否则交给发射侧如实拒绝。
        try:
            carry_expression(pending)
        except Refused:
            return None
        return ("adcsbb", f"pa_{at:x}",
                (m, f"pd_{at:x}", f"pb_{at:x}", f"pcf_{at:x}"),
                module.width(insn.operands[0]))
    if m in ("rcl", "rcr"):
        return None              # refused later, with the instruction's name
    if m.startswith("j") or m == "ret" or m == "retf":
        return pending           # a branch reads the flags and writes none
    # 产生标志的 ALU 指令排在"动没动标志"这道门之前：`add`/`xor` 当然动了标志，
    # 只是动的不是上一个比较的那些，所以它不是 `FLAG_SAFE`，而是**新的**生产者。
    # 顺序写反的话，`xor eax,eax` 之后过一条边就查不到任何 token，块首的 `cmove`
    # 会被记成"without a compare"——真实原因却是查表把产生者当成了破坏者。
    if m in FLAG_FROM_RESULT:
        shape = alu_shape(m)
        if shape is not None and len(insn.operands) < 2                 and shape not in ("inc", "dec", "neg"):
            shape = None
        # `inc`/`dec` 只有一个操作数，条件里也从不读第二个源，占位写 "1"。
        # `neg` 也只有一个操作数，但它的借位与溢出都要"取负之前"的那个值，发射侧
        # 会替它存一份 `pb_`——所以它读的是 `pb_`，不是占位的 "1"。
        bv = f"pb_{at:x}" if len(insn.operands) >= 2 or m == "neg" else "1"
        return ("alu", f"pa_{at:x}", (shape, bv) if shape else None,
                module.width(insn.operands[0]))
    if m not in FLAG_SAFE and not flag_neutral(m):
        return None
    return pending


def flag_flow(insns, block_starts, preds, rva):
    """The pending token at each block entry, and the token leaving each insn.

    A block inherits the flags of its predecessors when they all leave the
    *same* token. One way in is the common case; several ways in are fine too as
    long as the answer is the same from every one, because the token names its
    operands by the address of the producing instruction — two paths carrying
    the same token means the same instruction ran and its snapshots are the same
    variables, so no value is being guessed.

    Different tokens are a real join: `cmp eax, 3` on one path and `test r9, r9`
    on the other both leave a ZF, and the block that reads it needs whichever
    one ran. That is not a guess either, as long as the flags are made into
    state — each way out writes them, the block reads the variable. `merge` is
    that token; `lift_function` emits the writes.

    This is what a linear scan cannot do — `cmp eax,3 / jae L1 / … / L1: jne L0`
    puts the merge point *between* the compare and the branch that reads it, so
    the linear state is gone by the time the branch arrives, even though the
    branch itself has a well-defined way in.
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
            if not comes:
                continue
            # 所有入边留下**同一个** token 就直接继承：同一个 token 意味着同一条
            # 产生指令跑过、快照是同一批变量，取哪条路都是同一个值。
            wants = {out.get(p) for p in comes}
            if len(wants) == 1:
                want = wants.pop()
            elif None in wants:
                # 有一条路连标志都没留下（`call` 之后、或者根本没产生过），
                # 另外几条留下的对它没有意义——这个合并点没有答案。
                continue
            else:
                # 各条路留的标志不同。四位的可用集合取交集，能把"每条路都说得出
                # 来的那几位"存成变量；交集为空就没什么可合并的。
                keep = frozenset.intersection(*(kept_flags(t) for t in wants))
                if not keep:
                    continue
                want = ("merge", b, keep, 64)
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
    return pending_in, out


def predecessors(insns, dispatches):
    """(target -> {source addresses}) over the edges this function has.

    `walk` knows which addresses are entered by an edge but not by whom, and the
    difference matters for the flags: a label with one way in can inherit the
    flags of the instruction that leads to it; a label with more than one has to
    either agree with all of them or have the flags written out on the way.
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


def merge_needs(insns, block_starts, arriving, leaving, preds):
    """Which flags each join actually has to keep, and nobody else.

    A join token is four variables written at every way out. Emitting all four
    at every crossing would put the work in functions that never read a flag
    across a join, so the demand is worked out first: walk the blocks as the
    emitter does, note what each consumer asks for, then pass that demand back
    along the joins' own in-edges — a join whose variable is only stored into
    another join's variable is still read, one hop away.
    """
    need = {}
    starts_set = set(block_starts)
    pending = None
    for at in sorted(insns):
        if at in starts_set:
            pending = arriving.get(at)
        m = insns[at].mnemonic
        if m.startswith("lock "):
            m = m[5:]
        if pending is not None and pending[0] == "merge":
            flags = None
            if m.startswith("set") and len(m) > 3:
                flags = SUFFIX_FLAGS.get(m[3:])
            elif m.startswith("cmov") and len(m) > 4:
                flags = SUFFIX_FLAGS.get(m[4:])
            elif m.startswith("j") and m != "jmp":
                flags = SUFFIX_FLAGS.get(m[1:])
            elif m in ("adc", "sbb", "rcl", "rcr"):
                flags = "c"
            if flags:
                need.setdefault(pending[1], set()).update(flags)
        pending = flag_step(pending, insns[at])
    changed = True
    while changed:
        changed = False
        for b in block_starts:
            tok = arriving.get(b)
            if tok is None or tok[0] != "merge":
                continue
            want = need.get(b)
            if not want:
                continue
            for p in preds.get(b, ()):
                t = leaving.get(p)
                if t is None or t[0] != "merge":
                    continue
                add = want & t[2]
                if add and not add <= need.setdefault(t[1], set()):
                    need[t[1]] |= add
                    changed = True
    return need


def lift_function(code, rva):
    """C statements for a function with branches, or Refused."""
    insns, starts, calls, dispatches = walk(code, rva)
    if not insns:
        raise Refused("empty")
    start, end = rva, rva + len(code)

    preds = predecessors(insns, dispatches)
    block_starts = sorted(set(starts) | {rva})
    arriving, leaving = flag_flow(insns, block_starts, preds, rva)

    # 合并点的四位标志是**状态**：每条出口在离开时把要用的那几位写成 0/1 变量，
    # 块首按变量读。没人读的合并点整个撤掉，否则每个交叉口都多出四条赋值。
    need = merge_needs(insns, block_starts, arriving, leaving, preds)
    for b, tok in arriving.items():
        if tok is not None and tok[0] == "merge" and not need.get(b):
            arriving[b] = None

    # 每个出口要替哪几个合并点存哪几位。出口是终结指令时插在它那几行代码之前
    # （两条路都要走过这些赋值），否则插在这条指令之后。
    per_block, feeds = {}, {}
    for b, tok in arriving.items():
        if tok is None or tok[0] != "merge":
            continue
        keep = (need.get(b) or set()) & set(tok[2])
        if not keep:
            continue
        per_block[b] = keep
        for p in preds.get(b, ()):
            feeds.setdefault(p, {})[b] = keep

    out = []
    # 声明放在最前面：一个块在文本上可能先于它的某条出口（回边），而 C 要求先
    # 声明后使用。初值 0 只是让编译器别为"可能未初始化"报警，每条边都会在进入
    # 之前赋值。
    for b in sorted(per_block):
        out.append("uint8_t "
                   + ", ".join(f"f{f}_{b:x} = 0" for f in sorted(per_block[b]))
                   + ";")
    marks = {}
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
        marks[at] = len(out)
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
        if m in ("bt", "bts", "btr", "btc"):
            if len(ops) != 2:
                raise Refused(f"{m} arity")
            if ops[1].type not in (capstone.x86.X86_OP_REG, capstone.x86.X86_OP_IMM):
                raise Refused(f"{m} offset form")
            bits = module.width(ops[0])
            snap += 1
            out.append(f"uint64_t pb_{at:x};")
            if ops[0].type == capstone.x86.X86_OP_MEM:
                # 内存目标的位号会跨字节：地址是 `基地 + (位号 >> 3)`，位是 `位号 & 7`。
                # 位号按**有符号**解释，所以负位号落到基地前面那些字节——这就是
                # `bt [mem], reg` 与 `bt reg, reg` 不是同一件事的地方：后者在宽度内
                # 取模，前者只取字节和位两段，不取模。
                #
                # 立即数形式反过来，它**不跨字节**：位号在操作数宽度内取模，地址停
                # 在原处（汇编器把大位号编成位移，那是另一条编码）。
                code = f"(uint8_t *)(uintptr_t)({module.addr(ops[0], insn)})"
                if ops[1].type == capstone.x86.X86_OP_IMM:
                    off = ops[1].imm & (bits - 1)
                    out.append("{")
                    out.append(f"    uint8_t *p{snap} = {code} + {off >> 3};")
                    out.append(f"    uint64_t b{snap} = {off & 7};")
                else:
                    # 位号寄存器按**它自己的宽度**看，再符号扩展成 64 位——内存
                    # 形式的位号是有符号值（负数落到基地之前），所以 32 位寄存器
                    # 里的 0xFFFFFFF8 是 -8，不是 +4294967288。
                    osz = ops[1].size * 8
                    ext = "(int64_t)(uint64_t)" if osz == 64 \
                        else f"(int64_t)(int{osz}_t)"
                    out.append("{")
                    out.append(f"    int64_t o{snap} = {ext}"
                               f"({module.src(ops[1], insn)});")
                    out.append(f"    uint8_t *p{snap} = {code} + (o{snap} >> 3);")
                    out.append(f"    uint64_t b{snap} = (uint64_t)o{snap} & 7;")
                out.append(f"    pb_{at:x} = (uint64_t)(*p{snap} >> b{snap}) & 1;")
                if m == "bts":
                    out.append(f"    *p{snap} = (uint8_t)(*p{snap} | (uint8_t)(1u << b{snap}));")
                elif m == "btr":
                    out.append(f"    *p{snap} = (uint8_t)(*p{snap} & (uint8_t)~(1u << b{snap}));")
                elif m == "btc":
                    out.append(f"    *p{snap} = (uint8_t)(*p{snap} ^ (uint8_t)(1u << b{snap}));")
                out.append("}")
                pending = ("cf", f"pb_{at:x}", None, 64)
                continue
            b = f"{ops[1].imm & (bits - 1)}" if ops[1].type == capstone.x86.X86_OP_IMM \
                else f"((uint64_t)({module.src(ops[1], insn)}) & {bits - 1})"
            if m == "bt":
                out.append(f"pb_{at:x} = ((uint64_t)({module.src(ops[0], insn)}) >> {b}) & 1;")
                pending = ("cf", f"pb_{at:x}", None, bits)
                continue
            out.append("{")
            out.append(f"    uint64_t x = (uint64_t)({module.src(ops[0], insn)});")
            out.append(f"    uint64_t b = {b};")
            out.append(f"    pb_{at:x} = (x >> b) & 1;")
            out.append(f"    x = {'x | ((uint64_t)1 << b)' if m == 'bts' else ''}"
                       f"{'x & ~((uint64_t)1 << b)' if m == 'btr' else ''}"
                       f"{'x ^ ((uint64_t)1 << b)' if m == 'btc' else ''};")
            out.append("    " + module.wr(ops[0], "x", insn))
            out.append("}")
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
            # 走到这里说明操作数形式在上面那道门里已经发射过了——寄存器与内存
            # 两条地址规则都在那里，留这一条只是为了不让新的形式静默通过。
            raise Refused(f"{m} form")
        if m in ("adc", "sbb"):
            # 进位先落成一个变量（它读的是**上一条**生产者的结果，此刻还没被覆盖），
            # 再执行加法/减法。64 位用 `_addcarry_u64`/`_subborrow_u64`——那两条
            # intrinsic 就是 ADC/SBB 本身，进位出得来；窄操作数在 64 位里算，
            # 进位看第 bits 位，不需要第二套公式。
            if len(ops) != 2:
                raise Refused(f"{m} arity")
            cf = carry_expression(pending)
            bits = module.width(ops[0])
            snap += 1
            out.append(f"uint64_t pd_{at:x} = (uint64_t)({module.src(ops[0], insn)});")
            out.append(f"uint64_t pb_{at:x} = (uint64_t)({module.src(ops[1], insn)});")
            out.append(f"uint64_t pcy_{at:x} = (uint64_t)({cf}) & 1;")
            out.append(f"uint64_t pa_{at:x}, pcf_{at:x};")
            if bits == 64:
                fn = "_addcarry_u64" if m == "adc" else "_subborrow_u64"
                out.append(f"pcf_{at:x} = (uint64_t){fn}((unsigned char)pcy_{at:x}, "
                           f"pd_{at:x}, pb_{at:x}, &pa_{at:x});")
            elif m == "adc":
                out.append(f"pa_{at:x} = pd_{at:x} + pb_{at:x} + pcy_{at:x};")
                out.append(f"pcf_{at:x} = (pa_{at:x} >> {bits}) & 1;")
                out.append(f"pa_{at:x} &= {module.ONES[bits]};")
            else:
                out.append(f"pcf_{at:x} = pd_{at:x} < pb_{at:x} + pcy_{at:x};")
                out.append(f"pa_{at:x} = (pd_{at:x} - pb_{at:x} - pcy_{at:x}) "
                           f"& {module.ONES[bits]};")
            out.append("    " + module.wr(ops[0], f"pa_{at:x}", insn))
            pending = ("adcsbb", f"pa_{at:x}",
                       (m, f"pd_{at:x}", f"pb_{at:x}", f"pcf_{at:x}"), bits)
            continue
        if m in ("rcl", "rcr"):
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
        # 移位是唯一的例外——它的 `pb_` 存的是**被移的那个值**（也就是目标在移位
        # 之前的样子），因为 CF 与 OF 都是从它身上移出去的那几位算的。
        shift = shift_shape(insn) if m in ("shl", "sal", "shr", "sar") else None
        # `neg` 的借位是"源不是零"，而源就是目标在取负之前的值——与移位同一个
        # 问题，同一个解法：写之前存一份。
        pre = m == "neg"
        shape = alu_shape(m) if m in FLAG_FROM_RESULT else None
        if shift is not None or pre:
            out.append(f"uint64_t pb_{at:x} = (uint64_t)({module.src(ops[0], insn)});")
        elif shape is not None and len(ops) >= 2:
            out.append(f"uint64_t pb_{at:x} = (uint64_t)({module.src(ops[1], insn)});")
        elif shape not in ("inc", "dec"):
            shape = None
        out.extend(module.lift_one(insn))
        if shift is not None:
            snap += 1
            out.append(f"uint64_t pa_{at:x} = (uint64_t)({module.src(ops[0], insn)});")
            pending = ("alu", f"pa_{at:x}", (shift[0], f"pb_{at:x}", shift[1]),
                       module.width(ops[0]))
        elif m in FLAG_FROM_RESULT:
            # ZF and SF are the result's, and the result is in the destination
            # now — but a later `mov` may overwrite it before the branch reads
            # the flag, so it is snapshotted here.
            snap += 1
            out.append(f"uint64_t pa_{at:x} = (uint64_t)({module.src(ops[0], insn)});")
            pending = ("alu", f"pa_{at:x}",
                       (shape, f"pb_{at:x}" if len(ops) >= 2 or pre else "1") if shape else None,
                       module.width(ops[0]))
    # 每个合并点的变量由每条入边在离开时写一次。位置是事后插的：一条指令发出的
    # 行数不定（一段 goto、一个 switch、或者一整段向量运算），插在"下一条指令的
    # 第一行之前"比在循环里到处记位置可靠；终结指令则插在它自己第一行之前，因为
    # 那几行就是那个 `goto`，赋值必须走在它前面。
    order = sorted(insns)
    puts = {}
    for i, at in enumerate(order):
        if at not in feeds:
            continue
        m = insns[at].mnemonic
        ends = (at in dispatches or m in ("jmp", "ret", "retf")
                or m.startswith("j"))
        if ends:
            pos = marks[at]
        else:
            nxt = order[i + 1] if i + 1 < len(order) else None
            if nxt is None:
                pos = len(out)
            else:
                # 落在下一条指令之前，而且要落在它的**标签之前**：那道标签是别人
                # 跳进来的入口，跳进来的那条边有自己的赋值，不能被这一条覆盖。
                pos = marks[nxt] - (1 if nxt in starts else 0)
        for b, keep in sorted(feeds[at].items()):
            for f in sorted(keep):
                expr = flag_expr(leaving[at], f)
                puts.setdefault(pos, []).append(f"f{f}_{b:x} = (uint8_t)({expr});")
    for pos in sorted(puts, reverse=True):
        # 从后往前插：插在前面不会挪动后面已经算好的位置。
        out[pos:pos] = puts[pos]
    CALLS[rva] = calls
    return out
