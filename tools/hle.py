"""Who provides each import: the surface HLE has to cover, and nothing more.

`cover.py` counts what can be *translated*; this counts what has to be *replaced*.
They are different walls. A title's own code can be lifted function by function
(99.9% of 40,002 here), and the result still cannot run, because 37 of the 587
names it imports do not exist on Windows in any form.

The classification is data, not a list someone typed from memory:

    api-ms-win-* / ext-ms-win-*   -> host    the API sets; Windows resolves them
    ships beside the exe, x64 PE   -> title  load that copy, mapping it like the exe
    a known in-box module          -> host    WS2_32, WINHTTP, bcrypt, ...
    otherwise                      -> ours    it is the console's, HLE writes it

The order matters and is the same order the loader must try in: a copy that
ships beside the executable beats the one Windows would pick, which beats ours,
which beats a stub. Getting that order wrong is how a title ends up calling the
host's `XMemAlloc` — there is no such function, so it calls ours anyway; but
`BinkOpenWithOptions` exists in both and the console's copy is the right one.

    python tools/hle.py                      # the census
    python tools/hle.py --emit hle           # table.c + stubs.c + policy.h
"""

import argparse
import collections
import os
import re
import sys

import pefile

DEFAULT_EXE = (r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK"
               r"\game_scarlett_master.exe")

# 前缀匹配：API set 由 Windows 自己解析，谁都不用实现。
API_SET = ("api-ms-win-", "ext-ms-win-")

# Windows 自带（或 redist 一定装到 System32）的模块。**故意不写 msvcp140 /
# vcruntime140**：这个标题把它们放在 exe 旁边，那就该用旁边那一份——同一份
# 头文件编出来的布局，比 System32 里版本不确定的那份更可能是它期望的。
IN_BOX = {
    "kernel32.dll", "user32.dll", "advapi32.dll", "ole32.dll", "oleaut32.dll",
    "shell32.dll", "shlwapi.dll", "ws2_32.dll", "winhttp.dll", "wininet.dll",
    "bcrypt.dll", "crypt32.dll", "ncrypt.dll", "iphlpapi.dll", "dnsapi.dll",
    "d3d12.dll", "dxgi.dll", "d3d11.dll", "dstorage.dll", "gameinput.dll",
    "mfplat.dll", "mfreadwrite.dll", "mf.dll", "mfuuid.dll", "wmcodecdspuuid.dll",
    "setupapi.dll", "version.dll", "psapi.dll", "dbghelp.dll", "powrprof.dll",
    "winmm.dll", "xinput1_4.dll", "dinput8.dll", "dsound.dll", "combase.dll",
    "rpcrt4.dll", "sechost.dll", "ntdll.dll", "ucrtbase.dll",
}

KIND_ORDER = ("ours", "title", "host")


def is_x64_pe(path):
    """A copy that Windows can map: x64, and not a console-only image."""
    try:
        pe = pefile.PE(path, fast_load=True)
    except Exception:
        return False
    oh = pe.OPTIONAL_HEADER
    # 子系统 2 是 Windows GUI，3 是 CUI；Xbox 的映像同样是 2，所以这一条只能
    # 排除明显的非 Windows 映像，真正的判据是"它能不能被 LoadLibrary 加载"，
    # 而那个只有跑一次才知道。这里只回答"它是不是一个 x64 PE"。
    return pe.FILE_HEADER.Machine == 0x8664 and oh.Subsystem in (2, 3)


def imports(exe):
    """[(dll, [name, ...]), ...] in the order the directory lists them."""
    pe = pefile.PE(exe, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
    out = []
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        dll = entry.dll.decode()
        names = [i.name.decode() if i.name else f"#{i.ordinal}"
                 for i in entry.imports]
        out.append((dll, names))
    return out


# `FF 15` = call [rip+d]，`FF 25` = jmp [rip+d]。
CALL_RIP = bytes((0xFF, 0x15))
JMP_RIP = bytes((0xFF, 0x25))


def references(exe):
    """{(dll, name): (calls, jumps)} -- the direct references to one thunk.

    只数**直接**引用，而这一步给出的答案比预期重要：这个标题的 37 个主机 API
    **一个 direct call 都没有**，每个只有一处 `jmp [rip+iat]`。也就是说导入表在
    启动时被解析成函数指针，之后全是间接调用 —— 静态扫调用点这条路对它们无效，
    HLE 该换的不是 IAT 里的值，是"解析那一刻拿到的指针"。
    """
    pe = pefile.PE(exe, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
    target = {}
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        dll = entry.dll.decode()
        for i in entry.imports:
            name = i.name.decode() if i.name else f"#{i.ordinal}"
            target[i.address] = (dll, name)
    hits = collections.defaultdict(lambda: [0, 0])
    for sec in pe.sections:
        if not sec.Characteristics & 0x20000000:
            continue
        base = pe.OPTIONAL_HEADER.ImageBase + sec.VirtualAddress
        data = pe.get_data(sec.VirtualAddress, sec.SizeOfRawData)
        for slot, pattern in ((0, CALL_RIP), (1, JMP_RIP)):
            at = data.find(pattern)
            while at >= 0:
                disp = int.from_bytes(data[at + 2:at + 6], "little", signed=True)
                va = base + at + 6 + disp
                if va in target:
                    hits[target[va]][slot] += 1
                at = data.find(pattern, at + 1)
    return hits



def classify(dll, game_dir):
    """Which of ours / title / host / stub provides `dll`."""
    low = dll.lower()
    if low.startswith(API_SET):
        return "host"
    beside = os.path.join(game_dir, dll)
    if os.path.exists(beside) and is_x64_pe(beside):
        return "title"
    if low in IN_BOX:
        return "host"
    return "ours"


def census(exe, game_dir):
    """{dll: (kind, {name: kind})} — one row per DLL, kinds already resolved."""
    rows = {}
    for dll, names in imports(exe):
        kind = classify(dll, game_dir)
        rows[dll] = (kind, names)
    return rows


def report(rows, sites=None):
    by_kind = collections.Counter()
    for kind, names in rows.values():
        by_kind[kind] += len(names)
    total = sum(by_kind.values())
    print(f"{len(rows)} DLLs, {total:,} imports")
    for kind in KIND_ORDER:
        ndll = sum(1 for k, _ in rows.values() if k == kind)
        print(f"  {kind:6s} {by_kind[kind]:>5,}  over {ndll:>2} DLL(s)")

    print("\n--- ours: the whole HLE surface")
    for dll, (kind, names) in sorted(rows.items(), key=lambda r: -len(r[1][1])):
        if kind != "ours":
            continue
        print(f"\n  {len(names):>3}  {dll}")
        for name in sorted(names):
            calls, jumps = (sites or {}).get((dll, name), (0, 0))
            print(f"       {name:<40} call={calls:<3} jmp={jumps}")

    print("\n--- title: a copy ships beside the exe, map that one")
    for dll, (kind, names) in sorted(rows.items(), key=lambda r: -len(r[1][1])):
        if kind == "title":
            print(f"  {len(names):>4}  {dll}")

    print("\n--- host: nothing to write")
    print("   " + ", ".join(f"{d}({len(n)})" for d, (k, n)
                            in sorted(rows.items(), key=lambda r: -len(r[1][1]))
                            if k == "host"))


# 生成的 C 只用 ASCII。cl 在 936 代码页下读 UTF-8 注释会报 C4819（"该文件包含
# 不能在当前代码页中表示的字符"），而这份文件是要被别的工具链也吃进去的。
C_HEADER = """\
/* Generated by tools/hle.py from the import directory of the title. Do not edit.
 *
 * The table answers one question: who provides `dll!name`. The order is the
 * order the loader must try -- ours, then the copy beside the executable, then
 * Windows, then a stub that names itself when called. */
"""


def emit(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    table = [C_HEADER,
             "#include <stddef.h>\n",
             "typedef enum { HLE_HOST, HLE_TITLE, HLE_OURS, HLE_STUB } hle_kind;\n",
             "typedef struct { const char *dll, *name; hle_kind kind; } hle_import;\n",
             "const hle_import HLE_IMPORTS[] = {"]
    stubs = [C_HEADER,
             "#include <stdint.h>\n",
             "/* Every stub names itself when called: a stub that returns 0 moves the",
             " * title's failure somewhere else. */",
             "intptr_t hle_unimplemented(const char *what);\n"]
    ours = []
    for dll, (kind, names) in sorted(rows.items()):
        for name in sorted(names):
            if kind == "ours":
                table.append(f'    {{ "{dll}", "{name}", HLE_OURS }},')
                ours.append((dll, name))
            elif kind == "title":
                table.append(f'    {{ "{dll}", "{name}", HLE_TITLE }},')
            else:
                table.append(f'    {{ "{dll}", "{name}", HLE_HOST }},')
    table.append("};\n")
    table.append(f"const size_t HLE_IMPORTS_COUNT = "
                 f"sizeof(HLE_IMPORTS) / sizeof(HLE_IMPORTS[0]);")
    with open(os.path.join(out_dir, "table.c"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(table) + "\n")

    # 名字里的 C++ 修饰名一个都不该出现在 ours 里：如果出现了，说明分类错了。
    bad = [n for d, n in ours if not n.replace("_", "").isalnum()]
    for dll, name in ours:
        stubs.append(f"\nintptr_t {name}(void)\n"
                     f"{{\n    return hle_unimplemented(\"{dll}!{name}\");\n}}")
    with open(os.path.join(out_dir, "stubs.c"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("\n".join(stubs) + "\n")

    print(f"\nwrote {out_dir}/table.c ({sum(len(n) for _, n in rows.values())} rows), "
          f"{out_dir}/stubs.c ({len(ours)} stubs)")
    if bad:
        print(f"  ! {len(bad)} name(s) are not plain identifiers, so a stub for "
              f"them cannot link: {', '.join(bad[:6])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exe", nargs="?", default=DEFAULT_EXE)
    ap.add_argument("--game-dir", default=None,
                    help="where the title's own DLLs sit (default: beside the exe)")
    ap.add_argument("--emit", default=None, help="directory for table.c / stubs.c")
    args = ap.parse_args()

    game_dir = args.game_dir or os.path.dirname(args.exe)
    rows = census(args.exe, game_dir)
    sites = references(args.exe)
    report(rows, sites)
    if args.emit:
        emit(rows, args.emit)


if __name__ == "__main__":
    sys.exit(main())
