"""Which build are these numbers about, and which rung of the ladder are we on.

Two separate mistakes this exists to stop.

**One: a number with no build behind it.** `cover.py` says 99.9%; 99.9% of
what, on whose executable? A title's `.pdata` table changes with every build,
so a coverage figure without an identity is not a fact, it is a memory. This
reads the three things that identify a PE without shipping it: the file size,
the SHA-256 of its executable sections, and the CodeView record (`RSDS` GUID +
age + the PDB path the linker wrote). That is enough to say "these figures are
about this build and no other".

**Two: a ruler that measures the wrong thing.** Translation coverage counts
functions that compile. It does not count functions that *run* — there is no
loader in this tree at all, so "99.9% translated" and "0% of the ladder" are
both true at once, and only one of them sounds like progress. The ladder below
is the other ruler: six rungs from "the code translates" to "a frame is drawn".

    python tools/progress.py                     # the ladder + the last row
    python tools/progress.py --run --tests       # measure, then record
    python tools/progress.py --functions 124/125 --closure 1 --note "..."

A row is appended only when something moved. Any number going down against the
same build is a regression and exits non-zero: the point of a ledger is to make
that impossible to miss.
"""

import argparse
import datetime
import hashlib
import os
import re
import struct
import subprocess
import sys

import pefile

DEFAULT_EXE = (r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK"
               r"\game_scarlett_master.exe")
LEDGER = os.path.join("docs", "progress.tsv")
NUMBERS = ("translation", "total", "snippets", "snippet_total", "closure",
           "hle_left")
COLUMNS = ("date", "sha", "size", "guid", "age", "translation", "total",
           "snippets", "snippet_total", "functions", "closure", "hle_left",
           "note")


def codeview(exe):
    """(GUID, age, PDB path) from the debug directory, or None for each."""
    pe = pefile.PE(exe, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DEBUG"]])
    directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[6]
    if not directory.Size:
        return None, None, None
    raw = pe.get_data(directory.VirtualAddress, directory.Size)
    for at in range(0, len(raw), 28):
        _, _, _, _, kind, _, _, pointer = struct.unpack_from("<IIHHIIII", raw, at)
        if kind != 2:                      # IMAGE_DEBUG_TYPE_CODEVIEW
            continue
        # PointerToRawData 是**文件偏移**，不是 RVA——pefile 的 get_data 只认 RVA，
        # 这里要自己 seek。
        with open(exe, "rb") as f:
            f.seek(pointer)
            record = f.read(24)
            if record[:4] != b"RSDS":
                return None, None, None
            age = struct.unpack_from("<I", record, 20)[0]
            pdb = f.read(260).split(b"\x00")[0].decode("latin1")
        return record[4:20].hex(), age, pdb
    return None, None, None


def identity(exe):
    """The build this run is about."""
    pe = pefile.PE(exe, fast_load=True)
    digest = hashlib.sha256()
    for section in pe.sections:
        if section.Characteristics & 0x20000000:
            digest.update(pe.get_data(section.VirtualAddress,
                                      section.SizeOfRawData))
    guid, age, pdb = codeview(exe)
    return {"size": os.path.getsize(exe), "sha": digest.hexdigest()[:16],
            "guid": guid, "age": age, "pdb": pdb}


def mine(text, *patterns):
    """The first captured number of the first pattern that matches."""
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def measure(exe, run, tests):
    """Translation and snippet counts, re-measured if asked."""
    out = dict.fromkeys(NUMBERS[:4])
    tools = os.path.dirname(os.path.abspath(__file__))

    def call(tool, *args):
        result = subprocess.run([sys.executable, os.path.join(tools, tool), exe]
                                + list(args), capture_output=True, text=True,
                                encoding="mbcs", errors="replace")
        return result.stdout

    if run:
        text = call("cover.py", "--sample", "40001", "--pass", "branch")
        out["total"] = mine(text, r"([\d,]+) functions in \.pdata")
        out["translation"] = mine(text, r"branch:\s*([\d,]+) lift")
    if tests:
        text = call("diff.py", "--snippets")
        out["snippets"] = mine(text, r"against hardware:\s*([\d,]+) case")
        differ = mine(text, r"([\d,]+) differ")
        out["snippet_total"] = (None if out["snippets"] is None
                                else out["snippets"] + (differ or 0))
    return out


def stubs_left():
    """How many HLE stubs still only name themselves."""
    path = os.path.join("hle", "stubs.c")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().count('hle_unimplemented("')


# 每级一个判据，判据只用台账里有的数：算不出来的就不算数（None 一律判否）。
LADDER = (
    ("L0", "静态覆盖", "接受率 ≥ 99%，且片段差分无一例不一致",
     lambda m: (m["translation"] and m["total"] and m["snippets"] is not None
                and m["translation"] / m["total"] >= 0.99
                and m["snippets"] == m["snippet_total"])),
    ("L1", "闭包能编译", "closure.py 把一组互相调用的函数编成一个能链的 .c",
     lambda m: m["closure"] >= 1),
    ("L2", "导入面闭合", "37 个 ours 全有实现，一个只报姓名的桩都不剩",
     lambda m: m["hle_left"] == 0),
    ("L3", "映像能映射", "loader 把 PE 映射进进程，按 .pdata 注册展开信息",
     lambda m: m["closure"] >= 2),
    ("L4", "入口真的跑了", "映像里的入口执行了第一条指令（不是片段）",
     lambda m: m["closure"] >= 3),
    ("L5", "首帧", "GPU 与内存一致性可用，画面出来了",
     lambda m: m["closure"] >= 4),
)


def as_int(value, default=None):
    text = str(value if value is not None else "").strip()
    return int(text) if text.lstrip("-").isdigit() else default


def metrics(row):
    """The ledger row as numbers, None where it has nothing to say."""
    m = {key: as_int(row.get(key)) for key in NUMBERS}
    m["functions"] = row.get("functions") or ""
    m["closure"] = m["closure"] or 0
    return m


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    with open(LEDGER, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def show(ident, rows):
    mine_rows = [r for r in rows if r.get("sha") == ident["sha"]]
    last = mine_rows[-1] if mine_rows else {}
    print(f"build  {ident['sha']}  {ident['size']:,} bytes")
    print(f"       CodeView {ident['guid']}/{ident['age']}")
    if ident["pdb"]:
        print(f"       {ident['pdb']}")
    if not mine_rows:
        print(f"\nno row for this build in {LEDGER} — record one with --run")
    else:
        print(f"\nlast row {last['date']}: translation {last.get('translation')}"
              f"/{last.get('total')}, snippets {last.get('snippets')}"
              f"/{last.get('snippet_total')}, functions {last.get('functions')}"
              f"  ({len(mine_rows)} row(s) for this build)")
    m = metrics(last)
    print()
    for name, title, why, test in LADDER:
        try:
            done = bool(test(m))
        except (TypeError, ZeroDivisionError):
            done = False
        print(f"  {'[x]' if done else '[ ]'} {name}  {title:<14} {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=DEFAULT_EXE)
    ap.add_argument("--run", action="store_true", help="re-measure translation")
    ap.add_argument("--tests", action="store_true", help="re-measure snippets")
    ap.add_argument("--functions", default=None, help="whole-function diff, 124/125")
    ap.add_argument("--closure", type=int, default=None,
                    help="0 nothing, 1 a closure compiles, 2 a loader maps the "
                         "image, 3 the entry runs, 4 a frame is drawn")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    ident = identity(args.exe)
    rows = read_ledger()
    if not (args.run or args.tests or args.functions or args.closure is not None):
        show(ident, rows)
        return 0

    counts = measure(args.exe, args.run, args.tests)
    same = [r for r in rows if r.get("sha") == ident["sha"]]
    previous = same[-1] if same else None
    left = stubs_left()

    row = {
        "date": datetime.date.today().isoformat(),
        "sha": ident["sha"],
        "size": str(ident["size"]),
        "guid": ident["guid"] or "",
        "age": str(ident["age"] or ""),
        "functions": args.functions or (previous or {}).get("functions", ""),
        "closure": str(args.closure if args.closure is not None
                       else (previous or {}).get("closure", "0")),
        "hle_left": "" if left is None else str(left),
        "note": args.note,
    }
    for key in NUMBERS[:4]:                # 没重测的沿用上次，一个空值都不写
        row[key] = str(counts[key] if counts[key] is not None
                       else (previous or {}).get(key, ""))

    bad = []
    if previous:
        for key in ("translation", "snippets", "snippet_total"):
            was, now = as_int(previous.get(key)), as_int(row[key])
            if was is not None and now is not None and now < was:
                bad.append(f"{key} {was} -> {now}")
        was = (previous.get("functions") or "").split("/")[0]
        now = (row["functions"] or "").split("/")[0]
        if was.isdigit() and now.isdigit() and int(now) < int(was):
            bad.append(f"functions {previous['functions']} -> {row['functions']}")
        if all(previous.get(k, "") == row[k] for k in COLUMNS
               if k not in ("date", "note")):
            print("unchanged; nothing appended")
            show(ident, rows)
            return 2 if bad else 0

    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    fresh = not os.path.exists(LEDGER)
    with open(LEDGER, "a", encoding="utf-8", newline="\n") as f:
        if fresh:
            f.write("\t".join(COLUMNS) + "\n")
        f.write("\t".join(row[k] for k in COLUMNS) + "\n")
    print(f"appended to {LEDGER}")
    if bad:
        print("REGRESSION on the same build: " + ", ".join(bad))
    show(ident, read_ledger())
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
