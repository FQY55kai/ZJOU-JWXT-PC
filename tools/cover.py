"""How much of the image lifts, and what stops the rest.

Acceptance counts are the roadmap's only real progress metric: "we added a
feature" means nothing, "this many more of the title's own functions now
translate" is measurable. This walks `.pdata` and reports both passes with a
histogram of the refusal at the first instruction that stopped each function.

    python tools/cover.py                    # whole table, both passes
    python tools/cover.py --sample 20000     # a prefix, for a quick number
    python tools/cover.py --pass branch --reasons 20
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


def walk(pe):
    """(rva, code) for every function in `.pdata`, in table order."""
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
    raw = pe.get_data(entry.VirtualAddress, entry.Size)
    for at in range(0, len(raw) - 11, 12):
        begin, end, _ = struct.unpack_from("<III", raw, at)
        if begin and end > begin:
            yield begin, pe.get_data(begin, end - begin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
    ap.add_argument("--sample", type=int, default=0, help="stop after this many functions")
    ap.add_argument("--pass", dest="passes", action="append", choices=["plain", "branch"],
                    help="passes to report on (default: both)")
    ap.add_argument("--reasons", type=int, default=10)
    ap.add_argument("--max-bytes", type=int, default=0, help="skip functions larger than this")
    args = ap.parse_args()
    passes = args.passes or ["plain", "branch"]

    pe = pefile.PE(args.exe, fast_load=True)
    branch.set_image(pe)
    tally = {name: [0, collections.Counter()] for name in passes}
    seen = 0
    for rva, code in walk(pe):
        seen += 1
        if args.sample and seen > args.sample:
            break
        if args.max_bytes and len(code) > args.max_bytes:
            continue
        for name in passes:
            try:
                if name == "plain":
                    lift.lift_function(code, rva)
                else:
                    branch.lift_function(code, rva)
            except (lift.Refused, branch.Refused) as why:
                tally[name][1][str(why).split("(")[0].strip()] += 1
            else:
                tally[name][0] += 1

    print(f"{seen:,} functions in .pdata")
    for name in passes:
        ok, reasons = tally[name]
        print(f"  {name:6s}: {ok:,} lift ({ok/seen:.1%})")
        for why, n in reasons.most_common(args.reasons):
            print(f"      {n:7,}  {why}")


if __name__ == "__main__":
    main()
