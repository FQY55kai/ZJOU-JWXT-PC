"""Why was this function refused? The instruction, not the message.

`cover.py` reports *which category* stopped a function, and a category is a
message — `cmovbe without a compare` names the instruction that could not be
lifted, which is not the same thing as the instruction that is wrong. When the
fault is a missing entry in one of `branch.py`'s two flag tables, the consumer
is the victim and the producer is the culprit, and the message points at the
victim.

So this walks `.pdata`, lifts each function, and for the ones refused with a
matching reason it reads the **innermost instruction out of the traceback** —
the frame where the refusal was raised still has `insn` in its locals. Then it
prints the instruction, the four before it, and a histogram of
(predecessor, refused instruction) so thirty refusals sort themselves into
three or four causes.

    python tools/why.py --limit 40001 --reason "without a compare"
    python tools/why.py --reason undecodable --show 4
    python tools/why.py --limit 2000 --reason "" --show 20      # everything

The first version of this matched the mnemonic in the message text instead of
reading the traceback, so it happily printed the first `je` in a function that
had four of them — and the context it showed was `test r8, r8 / je`, which is
plainly liftable. That is why the traceback is the only honest source here.
"""

import argparse
import collections
import os
import struct
import sys

import pefile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import branch  # noqa: E402
import lift  # noqa: E402

DEFAULT_EXE = (r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK"
               r"\game_scarlett_master.exe")


def raising_frame(why):
    """(instruction, address) of the innermost frame that names one."""
    insn = site = None
    tb = why.__traceback__
    while tb is not None:
        local = tb.tb_frame.f_locals
        candidate = local.get("insn")
        if candidate is not None and hasattr(candidate, "address"):
            insn = candidate
        # `undecodable` is raised before there is an instruction: the frame's
        # `at` is the address it failed to decode, which is the useful thing.
        if isinstance(local.get("at"), int) and local.get("insn") is None:
            site = local["at"]
        tb = tb.tb_next
    return insn, site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=DEFAULT_EXE)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N .pdata entries (0 = whole table)")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--reason", default="without a compare",
                    help="substring of the refusal message ('' = every refusal)")
    ap.add_argument("--show", type=int, default=8,
                    help="how many functions to print with context")
    ap.add_argument("--context", type=int, default=4,
                    help="instructions to print before the refused one")
    args = ap.parse_args()

    pe = pefile.PE(args.exe, fast_load=True)
    branch.set_image(pe)
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    table = pe.get_data(entry.VirtualAddress, entry.Size)

    seen = collections.Counter()
    shown = total = 0
    for at in range(0, len(table) - 11, 12):
        if args.limit and at // 12 >= args.limit:
            break
        if at // 12 < args.skip:
            continue
        begin, end, _ = struct.unpack_from("<III", table, at)
        if not begin or end <= begin or end - begin > 4096:
            continue
        code = pe.get_data(begin, end - begin)
        try:
            branch.lift_function(code, begin)
        except (branch.Refused, lift.Refused) as why:
            text = str(why)
            err = why
        else:
            continue
        if args.reason and args.reason not in text:
            continue
        total += 1
        insn, site = raising_frame(err)
        if insn is None:
            key = (f"<no insn> {site:#x}" if site else f"<no insn> at {begin:#x}",
                   text[:48])
            seen[key] += 1
            continue
        disasm = list(lift.MD.disasm(code, begin))
        idx = next((n for n, i in enumerate(disasm) if i.address == insn.address), None)
        prev = disasm[idx - 1].mnemonic if idx else "<first>"
        seen[(prev, insn.mnemonic, text[:48])] += 1
        if shown < args.show and idx is not None:
            shown += 1
            print(f"\n{begin:#x}  {text}")
            for one in disasm[max(0, idx - args.context):idx + 1]:
                print(f"    {one.address:#x}: {one.mnemonic} {one.op_str}")
    print(f"\n{total} function(s) refused"
          + (f" with {args.reason!r}" if args.reason else "") + "; 前一条 × 计数：")
    for key, n in seen.most_common(30):
        print(f"{n:5d}  " + "  ".join(key))


if __name__ == "__main__":
    main()
