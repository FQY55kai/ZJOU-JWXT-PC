//! The hardware oracle: run a function's original bytes and say what it did.
//!
//! A port that recompiles code has to answer "is the lifted function the same
//! function?" at scale. Every earlier project in this genre paid for the answer
//! with a whole emulator (N64/PS2/PSP) or a whole console runtime. Here the
//! guest is x86-64 and so is the host, so the oracle is a call: copy the bytes
//! somewhere executable, set the sixteen registers, call, read them back.
//!
//! Input (little endian, repeated):
//!     u32 rva, u32 len, u64 regs[16], u8 stack[256]
//! Output, one record per input:
//!     u64 regs[16], u8 stack[256]
//!
//! The stack is the guest's: `rsp` in the input record is an offset into it,
//! and only functions whose every memory operand stays inside 256 bytes are
//! safe to run this way (the caller filters those).
//!
//!     cargo run --release -- <game.exe> cases.bin out.bin

use std::arch::asm;
use std::ffi::c_void;
use std::io::{Read, Write};

const PAGE: usize = 64 * 1024;

#[link(name = "kernel32")]
extern "system" {
    fn VirtualAlloc(at: *const c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(at: *mut c_void, size: usize, kind: u32) -> i32;
}

static mut REGS: *mut u64 = std::ptr::null_mut();
static mut STK: *mut u8 = std::ptr::null_mut();
static mut CODE: *const u8 = std::ptr::null();

static mut HOST_RSP: u64 = 0;
static mut SAVE_RBX: u64 = 0;
static mut SAVE_RBP: u64 = 0;
static mut GUEST_RSP: u64 = 0;
static mut OUT: [u64; 16] = [0; 16];

/// Calls `CODE` with the sixteen registers in `REGS` and the stack in `STK`,
/// then writes what they hold afterwards back into `REGS`.
///
/// The guest is the same ABI as the host, so it preserves rbx/rbp/rsi/rdi and
/// r12-r15 by itself: only the caller-saved half needs care, and the stores use
/// rip-relative targets so that no scratch register is needed to keep them.
#[inline(never)]
unsafe fn call_guest() {
    asm!(
        // Host stack, to come back to; rbx and rbp cannot be declared as
        // clobbers (LLVM reserves them), so they are saved by hand.
        "mov [rip + {host}], rsp",
        "mov [rip + {srbx}], rbx",
        "mov [rip + {srbp}], rbp",
        // Guest stack: the stack buffer plus the offset the record gives.
        "mov rax, [rip + {regs}]",
        "mov rax, [rax + 32]",
        "add rax, [rip + {stk}]",
        "mov [rip + {guest}], rax",
        // All sixteen, each through the static so no register is a middleman.
        "mov rax, [rip + {regs}]", "mov rax, [rax + 0]",
        "mov rcx, [rip + {regs}]", "mov rcx, [rcx + 8]",
        "mov rdx, [rip + {regs}]", "mov rdx, [rdx + 16]",
        "mov rbx, [rip + {regs}]", "mov rbx, [rbx + 24]",
        "mov rbp, [rip + {regs}]", "mov rbp, [rbp + 40]",
        "mov rsi, [rip + {regs}]", "mov rsi, [rsi + 48]",
        "mov rdi, [rip + {regs}]", "mov rdi, [rdi + 56]",
        "mov r8,  [rip + {regs}]", "mov r8,  [r8 + 64]",
        "mov r9,  [rip + {regs}]", "mov r9,  [r9 + 72]",
        "mov r10, [rip + {regs}]", "mov r10, [r10 + 80]",
        "mov r11, [rip + {regs}]", "mov r11, [r11 + 88]",
        "mov r12, [rip + {regs}]", "mov r12, [r12 + 96]",
        "mov r13, [rip + {regs}]", "mov r13, [r13 + 104]",
        "mov r14, [rip + {regs}]", "mov r14, [r14 + 112]",
        "mov r15, [rip + {regs}]", "mov r15, [r15 + 120]",
        "mov rsp, [rip + {guest}]",
        "call [rip + {code}]",
        // Straight after the call: the guest's own rsp, then every register
        // into the static array (rip-relative stores need no scratch).
        "mov [rip + {out} + 32], rsp",
        "mov [rip + {out} + 0], rax",
        "mov [rip + {out} + 8], rcx",
        "mov [rip + {out} + 16], rdx",
        "mov [rip + {out} + 24], rbx",
        "mov [rip + {out} + 40], rbp",
        "mov [rip + {out} + 48], rsi",
        "mov [rip + {out} + 56], rdi",
        "mov [rip + {out} + 64], r8",
        "mov [rip + {out} + 72], r9",
        "mov [rip + {out} + 80], r10",
        "mov [rip + {out} + 88], r11",
        "mov [rip + {out} + 96], r12",
        "mov [rip + {out} + 104], r13",
        "mov [rip + {out} + 112], r14",
        "mov [rip + {out} + 120], r15",
        "mov rbx, [rip + {srbx}]",
        "mov rbp, [rip + {srbp}]",
        "mov rsp, [rip + {host}]",
        host = sym HOST_RSP,
        srbx = sym SAVE_RBX,
        srbp = sym SAVE_RBP,
        guest = sym GUEST_RSP,
        regs = sym REGS,
        stk = sym STK,
        code = sym CODE,
        out = sym OUT,
        // The guest owns all sixteen; the Win64 clobber list covers the
        // volatile half, and these cover the rest. Without them the compiler is
        // free to keep a local in rbx/rbp and find it gone on return.
        out("rsi") _,
        out("rdi") _,
        out("r12") _,
        out("r13") _,
        out("r14") _,
        out("r15") _,
        clobber_abi("win64"),
    );
}

/// Calls a stub written here (`mov rax, rcx; add rax, rdx; ret`) so that a
/// fault in the harness is told apart from a fault in the guest.
fn selftest() {
    unsafe {
        let page = VirtualAlloc(std::ptr::null(), PAGE, 0x3000, 0x40) as *mut u8;
        assert!(!page.is_null());
        let code = [0x48u8, 0x89, 0xC8, 0x48, 0x01, 0xD0, 0xC3]; // mov rax,rcx; add rax,rdx; ret
        std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
        let mut regs = [0u64; 16];
        regs[1] = 7; // rcx
        regs[2] = 35; // rdx
        regs[4] = STACK_OFF;
        let mut stk = Aligned([0u8; 256]);
        REGS = regs.as_mut_ptr();
        STK = stk.0.as_mut_ptr();
        CODE = page;
        call_guest();
        println!("selftest rax = {} (want 42), rsp offset = {}", OUT[0], OUT[4].wrapping_sub(stk.0.as_ptr() as u64));
    }
}

const STACK_OFF: u64 = 128;

#[repr(align(16))]
struct Aligned([u8; 256]);

/// RVA to file offset: a virtual address is not a file position, and the
/// difference is the whole of a section's raw data versus its image size.
fn rva_to_offset(image: &[u8], rva: usize) -> Option<usize> {
    let at = u32::from_le_bytes(image.get(0x3c..0x40)?.try_into().ok()?) as usize;
    let sections = u16::from_le_bytes(image.get(at + 6..at + 8)?.try_into().ok()?) as usize;
    let optional = u16::from_le_bytes(image.get(at + 20..at + 22)?.try_into().ok()?) as usize;
    let table = at + 24 + optional;
    for i in 0..sections {
        let entry = image.get(table + i * 40..table + i * 40 + 40)?;
        let virtual_size = u32::from_le_bytes(entry[8..12].try_into().ok()?) as usize;
        let virtual_address = u32::from_le_bytes(entry[12..16].try_into().ok()?) as usize;
        let raw_size = u32::from_le_bytes(entry[16..20].try_into().ok()?) as usize;
        let raw_pointer = u32::from_le_bytes(entry[20..24].try_into().ok()?) as usize;
        let span = virtual_size.max(raw_size);
        if rva >= virtual_address && rva < virtual_address + span {
            let into = rva - virtual_address;
            if into >= raw_size {
                return None; // uninitialised data: nothing to run
            }
            return Some(raw_pointer + into);
        }
    }
    None
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
    let mut out = Vec::with_capacity(cases.len());

    let (mapped, image_size) = map_image(&image);

    let mut at = 0usize;
    let mut ran = 0u32;
    while at + 8 + 16 * 8 + 256 <= cases.len() {
        let rva = u32::from_le_bytes(cases[at..at + 4].try_into().unwrap()) as usize;
        let len = u32::from_le_bytes(cases[at + 4..at + 8].try_into().unwrap()) as usize;
        at += 8;
        let mut regs = [0u64; 16];
        for (i, slot) in regs.iter_mut().enumerate() {
            *slot = u64::from_le_bytes(cases[at + i * 8..at + i * 8 + 8].try_into().unwrap());
        }
        at += 128;
        let stack = &cases[at..at + 256];
        at += 256;

        if len == 0 || rva + len > image_size {
            out.extend_from_slice(&regs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            out.extend_from_slice(stack);
            continue;
        }

        unsafe {
            // Bytes go in at the page's start; RIP-relative operands would
            // resolve against the wrong base, which is why the caller filters
            // them out before sending a case here.
            let mut ours = regs;
            let mut stk = Aligned([0u8; 256]);
            stk.0.copy_from_slice(stack);
            REGS = ours.as_mut_ptr();
            STK = stk.0.as_mut_ptr();
            CODE = (mapped as usize + rva) as *const u8;
            call_guest();
            let mut back = [0u64; 16];
            back.copy_from_slice(&OUT);
            // `rsp` comes back as an address into our buffer: report it the way
            // the record asked for it, as an offset.
            let base = stk.0.as_ptr() as u64;
            back[4] = back[4].wrapping_sub(base);
            out.extend_from_slice(&back.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            out.extend_from_slice(&stk.0);
            ran += 1;
        }
    }
    unsafe { VirtualFree(mapped as *mut c_void, 0, 0x8000) };
    std::fs::File::create(&args[3])
        .expect("out")
        .write_all(&out)
        .expect("out");
    eprintln!("oracle: {ran} functions ran");
}
