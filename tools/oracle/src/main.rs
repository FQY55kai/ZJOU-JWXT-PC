//! The hardware oracle: run a function's original bytes and say what it did.
//!
//! A port that recompiles code has to answer "is the lifted function the same
//! function?" at scale. Every earlier project in this genre paid for the answer
//! with a whole emulator (N64/PS2/PSP) or a whole console runtime. Here the
//! guest is x86-64 and so is the host, so the oracle is a call: set the sixteen
//! registers, call the original bytes where the image already has them, read
//! the registers back.
//!
//! Input (little endian, repeated):
//!     u32 rva, u32 len, u64 regs[16], u8 arena[0x4000], u64 vec[64]
//! Output, one record per input:
//!     u64 regs[16], u8 arena[0x4000], u64 vec[64]
//!
//! Guest addresses are host addresses. The image is mapped at the base it asks
//! for, the arena is mapped at a fixed address, and `rsp` in a record is a real
//! pointer into that arena — so a function's stores land where they would have
//! landed, and comparing the arena compares the memory it touched, not just the
//! registers it left behind.
//!
//!     cargo build --release && ./oracle <game.exe> cases.bin out.bin

use std::ffi::c_void;
use std::io::{Read, Write};

use oracle_guest::run_case;

const PAGE: usize = 64 * 1024;

/// The arena, at a fixed address both sides agree on: below the Win64 user
/// limit (0x7FFF_FFFF_FFFF) and far from the image at 0x1_4000_0000, so a wild
/// displacement faults instead of quietly landing in the other region.
const ARENA: usize = 0x20_0000_0000;
const ARENA_SIZE: usize = 0x4000;
/// Sixteen 256-bit vector registers, four 64-bit lanes each.
const VEC_BYTES: usize = 16 * 32;
const RECORD: usize = 8 + 16 * 8 + ARENA_SIZE + VEC_BYTES;

#[link(name = "kernel32")]
extern "system" {
    fn VirtualAlloc(at: *const c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(at: *mut c_void, size: usize, kind: u32) -> i32;
    fn LoadLibraryA(name: *const u8) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    fn AddVectoredExceptionHandler(first: u32, f: extern "system" fn(*mut c_void) -> i32) -> *mut c_void;
}

extern "C" {
    #[link_name = "_setjmp"]
    fn setjmp(buf: *mut c_void) -> i32;
    fn longjmp(buf: *mut c_void, value: i32) -> !;
}

/// One case's fault is that case's own business.
///
/// Half the functions in this image read a global the file leaves zero, so
/// `mov rax, [rip+G]; mov ecx, [rax+0xBF4]` is an access violation before any
/// translation is involved. Without this the whole batch dies with the first
/// one, and the driver has to bisect — six re-runs per offending case, each one
/// mapping a 176 MB image again. That turned a two-minute comparison into a
/// fifteen-minute one, which is its own kind of wrong answer.
///
/// `_JUMP_BUFFER` on x64 is 256 bytes and `setjmp`/`longjmp` are the CRT's own;
/// the handler runs on the guest's stack, but `longjmp` restores rsp and rip
/// from the buffer, which was filled on the host's.
static mut JUMP: [u64; 32] = [0; 32];
static mut ARMED: bool = false;

extern "system" fn on_fault(_info: *mut c_void) -> i32 {
    unsafe {
        if !ARMED {
            return 0; // EXCEPTION_CONTINUE_SEARCH: not ours, let it crash
        }
        ARMED = false;
        longjmp(std::ptr::addr_of_mut!(JUMP) as *mut c_void, 1);
    }
}


/// Calls a stub written here (`mov rax, rcx; add rax, rdx; ret`) so that a
/// fault in the harness is told apart from a fault in the guest.
fn selftest() {
    unsafe {
        let page = VirtualAlloc(std::ptr::null(), PAGE, 0x3000, 0x40) as *mut u8;
        assert!(!page.is_null());
        let code = [0x48u8, 0x89, 0xC8, 0x48, 0x01, 0xD0, 0xC3]; // mov rax,rcx; add rax,rdx; ret
        std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
        map_arena();
        let mut regs = [0u64; 16];
        regs[1] = 7; // rcx
        regs[2] = 35; // rdx
        regs[4] = ARENA as u64 + 0x3000;
        let mut vec = [0u64; 64];
        vec[0] = 0x3FF0_0000_0000_0000; // 1.0 in xmm0's low lane
        let out = run_case(&mut regs, &mut vec, page);
        println!("selftest rax = {} (want 42), xmm0 {:#x} (want unchanged), guest rsp {:#x} -> {:#x} (call pushed, ret popped: {})",
                 out[0], vec[0], regs[4], out[4] - 8, out[4] - 8 == regs[4]);
    }
}

fn map_arena() -> *mut u8 {
    unsafe {
        let got = VirtualAlloc(ARENA as *const c_void, ARENA_SIZE, 0x3000, 0x40);
        assert_eq!(got as usize, ARENA, "the arena's address is taken");
        got as *mut u8
    }
}

/// Maps the image at the base it asks for, section by section.
///
/// A function copied to an arbitrary page is only runnable if it never touches
/// its own image: every `mov rax, [rip+N]` would resolve against the wrong
/// base. Mapping the image where it wants to be makes those reads land on the
/// bytes the compiler meant, which is most of what a real function does.
fn map_image(image: &[u8]) -> (*mut u8, usize) {
    let at = u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
    let sections = u16::from_le_bytes(image[at + 6..at + 8].try_into().unwrap()) as usize;
    let optional = u16::from_le_bytes(image[at + 20..at + 22].try_into().unwrap()) as usize;
    let opt = at + 24;
    let base = u64::from_le_bytes(image[opt + 24..opt + 32].try_into().unwrap()) as usize;
    let size = u32::from_le_bytes(image[opt + 56..opt + 60].try_into().unwrap()) as usize;
    unsafe {
        let got = VirtualAlloc(base as *const c_void, size, 0x3000, 0x40) as usize;
        assert_eq!(got, base, "the image's base is taken ({got:#x})");
        let table = opt + optional;
        for i in 0..sections {
            let e = &image[table + i * 40..table + i * 40 + 40];
            let virtual_size = u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
            let virtual_address = u32::from_le_bytes(e[12..16].try_into().unwrap()) as usize;
            let raw_size = u32::from_le_bytes(e[16..20].try_into().unwrap()) as usize;
            let raw_pointer = u32::from_le_bytes(e[20..24].try_into().unwrap()) as usize;
            let n = raw_size.min(virtual_size.max(raw_size));
            if raw_pointer + n <= image.len() && virtual_address + n <= size {
                std::ptr::copy_nonoverlapping(
                    image.as_ptr().add(raw_pointer),
                    (base + virtual_address) as *mut u8,
                    n,
                );
            }
        }
        (base as *mut u8, size)
    }
}

/// Resolve the import table in the mapped image.
///
/// The file's IAT holds a hint/name pair per import, not an address — the
/// loader writes the addresses. Without this every `jmp [__imp_Foo]` in the
/// original code jumps into a string and runs whatever the bytes say, which is
/// the same non-answer on both sides only by accident. A recompiled binary
/// needs this for the same reason a loader does.
unsafe fn fix_imports(base: usize, image: &[u8]) {
    let at = u32::from_le_bytes(image[0x3c..0x40].try_into().unwrap()) as usize;
    let opt = at + 24;
    let dirs = opt + 112; // PE32+: the data directories follow the optional header
    let import = dirs + 8;
    let table = u32::from_le_bytes(image[import..import + 4].try_into().unwrap()) as usize;
    if table == 0 {
        return;
    }
    let mut descriptor = table;
    loop {
        let d = &image[descriptor..descriptor + 20];
        let name = u32::from_le_bytes(d[12..16].try_into().unwrap()) as usize;
        let first = u32::from_le_bytes(d[16..20].try_into().unwrap()) as usize;
        let original = u32::from_le_bytes(d[0..4].try_into().unwrap()) as usize;
        if name == 0 || first == 0 {
            break;
        }
        let module = LoadLibraryA(image.as_ptr().add(name));
        if !module.is_null() {
            let source = if original != 0 { original } else { first };
            for i in 0.. {
                let slot = source + i * 8;
                let thunk = u64::from_le_bytes(image[slot..slot + 8].try_into().unwrap());
                if thunk == 0 {
                    break;
                }
                let fn_at = if thunk >> 63 != 0 {
                    GetProcAddress(module, (thunk & 0xFFFF) as *const u8)
                } else {
                    GetProcAddress(module, image.as_ptr().add(thunk as usize + 2))
                };
                let into = (base + first + i * 8) as *mut u64;
                *into = fn_at as u64;
            }
        }
        descriptor += 20;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "--selftest") {
        selftest();
        return;
    }
    if args.len() < 4 {
        eprintln!("usage: oracle <game.exe> <cases.bin> <out.bin>");
        std::process::exit(2);
    }
    let image = std::fs::read(&args[1]).expect("reading the image");
    let mut cases = Vec::new();
    std::fs::File::open(&args[2])
        .expect("cases")
        .read_to_end(&mut cases)
        .expect("cases");
    let mut out = Vec::with_capacity(cases.len() / RECORD * (128 + ARENA_SIZE + VEC_BYTES + 4));
    unsafe { AddVectoredExceptionHandler(1, on_fault) };

    let (mapped, image_size) = map_image(&image);
    unsafe { fix_imports(mapped as usize, &image) };
    let arena = map_arena();

    let mut at = 0usize;
    let mut ran = 0u32;
    while at + RECORD <= cases.len() {
        let rva = u32::from_le_bytes(cases[at..at + 4].try_into().unwrap()) as usize;
        let len = u32::from_le_bytes(cases[at + 4..at + 8].try_into().unwrap()) as usize;
        let mut regs = [0u64; 16];
        for (i, slot) in regs.iter_mut().enumerate() {
            let from = at + 8 + i * 8;
            *slot = u64::from_le_bytes(cases[from..from + 8].try_into().unwrap());
        }
        let arena_at = at + 8 + 128;
        let vec_at = arena_at + ARENA_SIZE;
        let mut vec = [0u64; 64];
        for (i, slot) in vec.iter_mut().enumerate() {
            let from = vec_at + i * 8;
            *slot = u64::from_le_bytes(cases[from..from + 8].try_into().unwrap());
        }
        at += RECORD;

        unsafe {
            std::ptr::copy_nonoverlapping(cases.as_ptr().add(arena_at), arena, ARENA_SIZE);
        }

        let mut landed = false;
        if len != 0 && rva + len <= image_size {
            let mut back = [0u64; 16];
            let faulted = unsafe {
                ARMED = true;
                if setjmp(std::ptr::addr_of_mut!(JUMP) as *mut c_void) == 0 {
                    back = run_case(&mut regs, &mut vec, (mapped as usize + rva) as *const u8);
                    ARMED = false;
                    false
                } else {
                    true
                }
            };
            if !faulted {
                // The `ret` popped the return address the `call` pushed, so the
                // guest's own `rsp` is this minus eight. Report it where the
                // lifted C will be, which is the point before the call.
                back[4] = back[4].wrapping_sub(8);
                out.extend_from_slice(&back.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
                unsafe { out.extend_from_slice(std::slice::from_raw_parts(arena, ARENA_SIZE)) };
                out.extend_from_slice(&vec.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
                out.extend_from_slice(&1u32.to_le_bytes());
                ran += 1;
                landed = true;
            }
        }
        if !landed {
            // Faulted, or not a real function: a record either way, so the
            // driver's records line up with its cases one to one.
            out.extend_from_slice(&regs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            out.extend_from_slice(&cases[arena_at..arena_at + ARENA_SIZE]);
            out.extend_from_slice(&cases[vec_at..vec_at + VEC_BYTES]);
            out.extend_from_slice(&0u32.to_le_bytes());
        }
    }
    unsafe { VirtualFree(mapped as *mut c_void, 0, 0x8000) };
    std::fs::File::create(&args[3])
        .expect("out")
        .write_all(&out)
        .expect("out");
    eprintln!("oracle: {ran} functions ran");
}
