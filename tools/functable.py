"""The function table, grown past `.pdata`.

`.pdata` names every function that can unwind, which is 92% of `.text`. The
missing 8% are leaf functions with no unwind information — and they are exactly
the ones a lifter keeps meeting, because every `call rel32` that leaves the
table lands in one of them (94,389 such edges in the first 20,000 functions
decoded).

The fix is not a heuristic sweep: a call target *is* a function's first
instruction. So: take the calls, add their targets, decode what was added, take
the new calls, repeat until nothing new appears.

    python tools/functable.py --limit 20000
    python tools/functable.py --full          # every function, slower
"""

import argparse
import struct
import sys

import capstone
import pefile

MD = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
MD.detail = True


def pdata(pe):
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    table = pe.get_data(entry.VirtualAddress, entry.Size)
    out = []
    for at in range(0, len(table) - 11, 12):
        begin, end, unwind = struct.unpack_from("<III", table, at)
        if begin and end > begin:
            out.append((begin, end, unwind))
    return out


def text_range(pe):
    sec = next(s for s in pe.sections if s.Name.rstrip(bytes([0])) == b".text")
    return sec.VirtualAddress, sec.VirtualAddress + sec.Misc_VirtualSize


def grow(pe, known, lo, hi, rounds=3):
    """Adds call targets to the function table until nothing new turns up."""
    added = {}
    frontier = list(known)
    for round_no in range(rounds):
        fresh = set()
        for begin in frontier:
            end = known.get(begin) or (begin + 256)
            code = pe.get_data(begin, min(end - begin, 2048))
            for insn in MD.disasm(code, begin):
                if insn.mnemonic not in ("call", "jmp"):
                    continue
                ops = insn.operands
                if ops and ops[0].type == capstone.x86.X86_OP_IMM:
                    target = ops[0].imm
                    if lo <= target < hi and target not in known and target not in added:
                        fresh.add(target)
        if not fresh:
            break
        for target in fresh:
            added[target] = target + 64  # a bound until something better is known
        known.update(added)
        frontier = sorted(fresh)
        print(f"  round {round_no + 1}: +{len(fresh):,} function starts from call targets")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
    ap.add_argument("--limit", type=int, default=20000, help="functions decoded in the first round")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    pe = pefile.PE(args.exe, fast_load=True)
    named = pdata(pe)
    lo, hi = text_range(pe)
    known = {begin: end for begin, end, _ in named}
    print(f".pdata           {len(named):,} functions")
    print(f".text            {lo:#x}..{hi:#x} ({(hi - lo)/2**20:.1f} MB)")

    if not args.full:
        # Only a slice is decoded in the first round; the rest still counts in
        # the table, which is what the lifter will walk.
        seeds = {b: e for b, e in list(known.items())[: args.limit]}
        added = grow(pe, seeds, lo, hi, rounds=2)
        known.update(added)
    else:
        known.update(grow(pe, known, lo, hi, rounds=4))

    covered = sum(max(0, e - b) for b, e in known.items())
    print(f"function table   {len(known):,} functions, {covered/2**20:.1f} MB named")
    print(f"coverage         {covered/(hi - lo):.0%} of .text")
    missing = len(known) - len(named)
    print(f"grown by         {missing:,} starts that `.pdata` does not name")


if __name__ == "__main__":
    sys.exit(main())
