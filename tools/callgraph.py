"""The call graph, and what a closure lift would have to cover.

The branch pass accepted 121 functions out of 207,811, and the reason was not
branches: it was calls. Two thirds of the functions in this image end in a tail
call and the rest call out from the middle. A recompiler that lifts functions
one at a time can never cover them; one that lifts a *closure* — the entry and
everything it reaches — can, because a call between two lifted functions is just
a call.

This script answers the questions that decide whether that is worth building:
how many edges are internal, how many leave through the import table, how many
are indirect, and how big a closure is from a typical entry.

    python tools/callgraph.py --limit 40000
"""

import argparse
import collections
import struct
import sys

import capstone
import pefile

MD = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
MD.detail = True


def functions(pe):
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    table = pe.get_data(entry.VirtualAddress, entry.Size)
    out = []
    for at in range(0, len(table) - 11, 12):
        begin, end, unwind = struct.unpack_from("<III", table, at)
        if begin and end > begin:
            out.append((begin, end, unwind))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
    ap.add_argument("--limit", type=int, default=40000)
    args = ap.parse_args()

    pe = pefile.PE(args.exe, fast_load=True)
    funcs = functions(pe)
    known = {begin for begin, _, _ in funcs}
    print(f"functions in .pdata: {len(funcs):,}")

    internal = collections.defaultdict(set)   # caller rva -> callee rvas
    unseen = set()                            # call targets .pdata does not name
    imports = collections.Counter()
    indirect = collections.Counter()
    tails = 0
    seen = 0
    for begin, end, _ in funcs[: args.limit]:
        code = pe.get_data(begin, min(end - begin, 4096))
        seen += 1
        for insn in MD.disasm(code, begin):
            if insn.mnemonic not in ("call", "jmp"):
                continue
            ops = insn.operands
            if not ops:
                continue
            if ops[0].type == capstone.x86.X86_OP_IMM:
                target = ops[0].imm
                if target in known:
                    internal[begin].add(target)
                    if insn.mnemonic == "jmp":
                        tails += 1
                else:
                    imports["out of range"] += 1
                    unseen.add(target)
            elif ops[0].type == capstone.x86.X86_OP_MEM:
                mem = ops[0].mem
                if mem.base and MD.reg_name(mem.base) == "rip":
                    imports["via RIP (import or vtable)"] += 1
                else:
                    indirect[MD.reg_name(mem.base) if mem.base else "?"] += 1
            else:
                indirect["register"] += 1

    # Every `call rel32` target that `.pdata` does not list is still a function:
    # those without unwind information are the 8% the table misses.
    text = next((sec.VirtualAddress, sec.Misc_VirtualSize) for sec in pe.sections
                if sec.Name.rstrip(bytes([0])) == b".text")
    unknown = [t for t in unseen if text[0] <= t < text[0] + text[1]]
    edges = sum(len(v) for v in internal.values())
    callers = len(internal)
    with_edges = sum(1 for v in internal.values() if v)
    print(f"decoded            {seen:,} functions ({seen/len(funcs):.0%})")
    print(f"functions calling   {with_edges:,} ({with_edges/seen:.1%} of those decoded)")
    print(f"internal edges      {edges:,} (mean {edges/max(with_edges,1):.1f} per calling function)")
    print(f"tail calls internal {tails:,} ({tails/max(edges,1):.0%} of edges)")
    print(f"call targets not in .pdata, inside .text: {len(unknown):,}"
          f" -> the function table can be grown to {len(known) + len(unknown):,}")
    print()
    print("edges that leave the lifted set:")
    for why, n in imports.most_common(4):
        print(f"  {n:7,}  {why}")
    print("indirect calls, by base register:")
    total_indirect = sum(indirect.values())
    for reg, n in indirect.most_common(5):
        print(f"  {n:7,}  [{reg}]")
    print(f"  {total_indirect:7,}  total indirect ({total_indirect/max(edges+total_indirect,1):.0%} of all calls)")

    # How big is a closure? Breadth-first from the most-connected entries.
    ranked = sorted(internal.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    for root, _ in ranked:
        seen_set = {root}
        frontier = [root]
        for _depth in range(4):
            nxt = []
            for fn in frontier:
                for callee in internal.get(fn, ()):
                    if callee not in seen_set:
                        seen_set.add(callee)
                        nxt.append(callee)
            frontier = nxt
            if not frontier:
                break
        print(f"closure from {root:#x}: {len(seen_set):,} functions at depth 4")


if __name__ == "__main__":
    sys.exit(main())
