"""What shapes do the refused instructions actually have?

A refusal histogram says `vmovaps` stops 6,998 functions. It does not say which
`vmovaps`: `xmm <- xmm` and `ymm <- [rsp+0x20]` and `[rax] <- ymm` are three
different emissions, and adding one of them does nothing for the other two. So
before writing an emitter, count the operand shapes the image actually contains,
restricted to functions that fail on the family under study.

    python tools/shape.py --of vmovaps --sample 40000
    python tools/shape.py --of segment,div,cmpxchg
"""

import argparse
import collections
import os
import re
import struct
import sys

import capstone
import pefile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lift  # noqa: E402

FAMILIES = {
    "simd": re.compile(r"^(v?)(mov|movd|movq|xor|and|or|add|sub|mul|div|min|max|"
                       r"ucomis|comis|pinsr|pextr|pshuf|punpck|shuf|broadcast|"
                       r"unpck|cvtsi|cvtss|cvtps|cvttss|cvtdq|cvtpi|blend|"
                       r"permil|perm|insert|extract|zeroupper|pxor|pcmpeq|"
                       r"pcmpgt|ptest|pack|padd|psub|pmull|pmulh|psll|psrl|psra)"),
    "segment": re.compile(r"^$"),
}


def shape(insn):
    """`reg<-reg`, `reg<-[mem]`, `[mem]<-reg`, ... at each operand's width."""
    out = []
    for op in insn.operands:
        if op.type == capstone.x86.X86_OP_REG:
            name = lift.MD.reg_name(op.reg)
            out.append(name)
        elif op.type == capstone.x86.X86_OP_IMM:
            out.append("imm")
        else:
            m = op.mem
            s = ""
            if m.segment:
                s += lift.MD.reg_name(m.segment) + ":"
            s += "["
            if m.base:
                s += lift.MD.reg_name(m.base)
            if m.index:
                s += "+" + lift.MD.reg_name(m.index) + f"*{m.scale}"
            if m.disp:
                s += "+d"
            s += "]"
            out.append(s)
    return f"{insn.mnemonic} " + ",".join(out)


def walk(pe):
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    raw = pe.get_data(entry.VirtualAddress, entry.Size)
    for at in range(0, len(raw) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", raw, at)
        if begin and end > begin:
            yield begin, pe.get_data(begin, end - begin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
    ap.add_argument("--of", default="simd", help="comma-separated mnemonic prefixes, or 'simd'")
    ap.add_argument("--sample", type=int, default=40000)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    pe = pefile.PE(args.exe, fast_load=True)
    lift.MD = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    lift.MD.detail = True

    want = args.of.split(",")
    if "simd" in want:
        want = None

    def interesting(mnemonic):
        if want is not None:
            return any(mnemonic.startswith(w) for w in want)
        return bool(FAMILIES["simd"].match(mnemonic))

    # Only instructions inside functions that die on this family matter: an
    # instruction in a function that already lifts is already handled.
    tally = collections.Counter()
    funcs = collections.Counter()
    seen = 0
    for rva, code in walk(pe):
        seen += 1
        if args.sample and seen > args.sample:
            break
        hits = []
        try:
            for insn in lift.MD.disasm(code, rva):
                if interesting(insn.mnemonic) and insn.mnemonic not in (
                        "mov", "movzx", "movsx", "movsxd"):
                    hits.append(shape(insn))
        except capstone.CsError:
            continue
        if not hits:
            continue
        # Is this function refused *because* of them? Ask the lifter.
        try:
            import branch
            branch.lift_function(code, rva)
        except Exception:
            pass
        else:
            continue
        for h in set(hits):
            funcs[h] += 1
        for h in hits:
            tally[h] += 1

    print(f"{seen:,} functions scanned")
    print(f"\n{'occurrences':>12} {'functions':>10}  shape")
    for s, n in tally.most_common(args.top):
        print(f"{n:12,} {funcs[s]:10,}  {s}")


if __name__ == "__main__":
    main()
