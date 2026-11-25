"""Probe the title's PE: how many functions .pdata names, how big they are, and
where the graphics boundary sits.

A static recompiler needs three facts before anything else: function boundaries
(no symbols required, `.pdata` has them for every function that can unwind),
how much code that is, and which imports the lifted code will meet. This script
reports all three from the binary alone.

    python tools/recomp/probe.py <game_scarlett_master.exe>
"""

import collections
import struct
import sys

import pefile


def sections(pe):
    return [(s.Name.rstrip(b"\0").decode("ascii", "replace"), s.VirtualAddress, s.Misc_VirtualSize) for s in pe.sections]


def directory(pe, index):
    entry = pe.OPTIONAL_HEADER.DATA_DIRECTORY[index]
    return entry.VirtualAddress, entry.Size


def runtime_functions(pe, rva, size):
    """`.pdata`'s RUNTIME_FUNCTION table: 12 bytes each, RVAs in ascending order."""
    data = pe.get_data(rva, size)
    out = []
    for at in range(0, len(data) - 11, 12):
        begin, end, unwind = struct.unpack_from("<III", data, at)
        if begin == 0 or end <= begin:
            continue
        out.append((begin, end, unwind))
    return out


def main(path):
    pe = pefile.PE(path, fast_load=False)
    print(f"file           {path}")
    print(f"image base     {pe.OPTIONAL_HEADER.ImageBase:#x}")
    print(f"sections       {len(pe.sections)}")
    for name, va, vsize in sections(pe):
        print(f"  {name:<10} rva {va:#010x} size {vsize/2**20:8.2f} MB")

    pdata_rva, pdata_size = directory(pe, 3)
    funcs = runtime_functions(pe, pdata_rva, pdata_size)
    sizes = [end - begin for begin, end, _ in funcs]
    total = sum(sizes)
    sizes.sort()
    print()
    print(f"functions      {len(funcs):,} (from .pdata at {pdata_rva:#x}, {pdata_size/2**20:.1f} MB of table)")
    print(f"code           {total/2**20:.1f} MB, mean {total/max(len(sizes),1):.0f} B")
    if sizes:
        def at(q):
            return sizes[min(len(sizes) - 1, int(len(sizes) * q))]
        print(f"percentiles    10% {at(.1)} B   50% {at(.5)} B   90% {at(.9)} B   99% {at(.99)} B   max {sizes[-1]} B")
        print(f"tiny (<64 B)   {sum(1 for s in sizes if s < 64):,}  ({sum(1 for s in sizes if s < 64)/len(sizes):.0%})")
        print(f"huge (>8 KB)   {sum(1 for s in sizes if s > 8192):,}")

    text = next((va, vsize) for name, va, vsize in sections(pe) if name == ".text")
    print(f"covered by .pdata: {total/max(text[1],1):.0%} of .text ({text[1]/2**20:.1f} MB)")

    imports = collections.Counter()
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        name = entry.dll.decode("ascii", "replace")
        for imp in entry.imports:
            imports[name] += 1
    print()
    print("imports by module (the runtime layer a port must provide):")
    for name, n in imports.most_common(14):
        print(f"  {name:<24} {n:4} symbols")

    print()
    print("graphics boundary:")
    for dll in ("d3d12.dll", "D3D12Core.dll", "dxgi.dll", "dstorage.dll"):
        hits = [i.name.decode() for e in pe.DIRECTORY_ENTRY_IMPORT if e.dll.decode().lower() == dll.lower() for i in e.imports if i.name]
        if hits:
            print(f"  {dll}: {', '.join(sorted(hits)[:8])}{' …' if len(hits) > 8 else ''}")
    # A D3D12 title built on the GDK reaches the device through a vtable, so the
    # calls a recompiler must intercept are indirect: report how many indirect
    # calls the binary holds, as a share of all calls.
    return funcs


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Games\BaiduNetdiskDownload\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe")
