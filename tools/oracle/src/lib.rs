//! The guest call stub, as both a library function and a C symbol.
//!
//! Two users, one piece of assembly. The judge (`main.rs`) uses `run_case` to
//! run a batch of functions and report what they did. The generated C uses
//! `guest_bridge` to call a function that has not been lifted yet, so that a
//! lifted function still calls *something* — the original bytes — where the
//! port is incomplete. That is what a recompiler does while it is being
//! written, and it is also what makes the differential test possible at scale:
//! without it only the 0.5% of functions that call nothing could be checked.
//!
//! Both paths run the guest on the arena, not on the host stack, so the two
//! sides of the comparison see the same memory. They also carry the vector
//! register file in and out, for the same reason: xmm6-xmm15 are non-volatile,
//! so almost every function in this image spills them to its frame in the first
//! six instructions. If the guest's spills are not the harness's spills, the
//! comparison sees sixteen bytes of noise in the arena and calls it a
//! disagreement.

use std::arch::asm;

static mut REGS: *mut u64 = std::ptr::null_mut();
static mut VEC: *mut u64 = std::ptr::null_mut();
static mut CODE: *const u8 = std::ptr::null();

static mut HOST_RSP: u64 = 0;
static mut SAVE_RBX: u64 = 0;
static mut SAVE_RBP: u64 = 0;
static mut GUEST_RSP: u64 = 0;

static mut OUT: [u64; 16] = [0; 16];
static mut VEC_OUT: [u64; 64] = [0; 64];

/// How far above `r[4]` to start the guest, so that the `call` below lands it
/// exactly on `r[4]`. Eight when entering a function from a harness (the guest
/// should see the record's `rsp`, not that minus the return address it did not
/// push); zero when bridging out of lifted C (the guest is in the middle of a
/// frame that already has its return address, and a `call` from there must put
/// the callee at `r[4] - 8`, which is what the hardware would do).
static mut PRECHARGE: u64 = 0;

/// Calls `CODE` with the sixteen registers in `REGS` and the sixteen vector
/// registers in `VEC`, then writes what they hold afterwards back into `OUT`
/// and `VEC_OUT`.
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
        // The vector file, before any GPR is loaded: rax is still scratch here,
        // and `vmovdqu` is the instruction that touches no general register.
        "mov rax, [rip + {vec}]",
        "vmovdqu ymm0,  [rax + 0]",
        "vmovdqu ymm1,  [rax + 32]",
        "vmovdqu ymm2,  [rax + 64]",
        "vmovdqu ymm3,  [rax + 96]",
        "vmovdqu ymm4,  [rax + 128]",
        "vmovdqu ymm5,  [rax + 160]",
        "vmovdqu ymm6,  [rax + 192]",
        "vmovdqu ymm7,  [rax + 224]",
        "vmovdqu ymm8,  [rax + 256]",
        "vmovdqu ymm9,  [rax + 288]",
        "vmovdqu ymm10, [rax + 320]",
        "vmovdqu ymm11, [rax + 352]",
        "vmovdqu ymm12, [rax + 384]",
        "vmovdqu ymm13, [rax + 416]",
        "vmovdqu ymm14, [rax + 448]",
        "vmovdqu ymm15, [rax + 480]",
        // Guest stack: the record's `rsp`, pre-charged by the return address
        // the `call` below is about to push, so that inside the guest `rsp` is
        // exactly what the lifted C computes.
        "mov rax, [rip + {regs}]",
        "mov rax, [rax + 32]",
        "add rax, [rip + {pre}]",
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
        // And the vector file back out, now that no GPR holds guest state.
        // `lea`, not a load: `VEC_OUT` is an array, so `[rip + VEC_OUT]` would
        // be its first element and not its address.
        "lea rax, [rip + {vout}]",
        "vmovdqu [rax + 0], ymm0",
        "vmovdqu [rax + 32], ymm1",
        "vmovdqu [rax + 64], ymm2",
        "vmovdqu [rax + 96], ymm3",
        "vmovdqu [rax + 128], ymm4",
        "vmovdqu [rax + 160], ymm5",
        "vmovdqu [rax + 192], ymm6",
        "vmovdqu [rax + 224], ymm7",
        "vmovdqu [rax + 256], ymm8",
        "vmovdqu [rax + 288], ymm9",
        "vmovdqu [rax + 320], ymm10",
        "vmovdqu [rax + 352], ymm11",
        "vmovdqu [rax + 384], ymm12",
        "vmovdqu [rax + 416], ymm13",
        "vmovdqu [rax + 448], ymm14",
        "vmovdqu [rax + 480], ymm15",
        "mov rbx, [rip + {srbx}]",
        "mov rbp, [rip + {srbp}]",
        "mov rsp, [rip + {host}]",
        host = sym HOST_RSP,
        srbx = sym SAVE_RBX,
        srbp = sym SAVE_RBP,
        guest = sym GUEST_RSP,
        pre = sym PRECHARGE,
        regs = sym REGS,
        vec = sym VEC,
        code = sym CODE,
        out = sym OUT,
        vout = sym VEC_OUT,
        // The guest owns all sixteen general registers; the Win64 clobber list
        // covers the volatile half, and these cover the rest. Without them the
        // compiler is free to keep a local in rbx/rbp and find it gone on
        // return.
        out("rsi") _,
        out("rdi") _,
        out("r12") _,
        out("r13") _,
        out("r14") _,
        out("r15") _,
        // And all sixteen vector registers, which `clobber_abi("win64")` covers
        // only up to xmm5.
        out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
        out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
        out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _,
        clobber_abi("win64"),
    );
}

/// Runs `code` with the registers in `regs` and the vector file in `vec`, and
/// returns the general registers afterwards. `vec` is updated in place: the
/// judge's one primitive, and the reason a spill of `xmm6` is checked rather
/// than excused.
pub fn run_case(regs: &mut [u64; 16], vec: &mut [u64; 64], code: *const u8) -> [u64; 16] {
    unsafe {
        REGS = regs.as_mut_ptr();
        VEC = vec.as_mut_ptr();
        CODE = code;
        PRECHARGE = 8;
        call_guest();
        std::ptr::copy_nonoverlapping(
            std::ptr::addr_of!(VEC_OUT) as *const u64,
            vec.as_mut_ptr(),
            64,
        );
        std::ptr::read(std::ptr::addr_of!(OUT))
    }
}

/// `void guest_bridge(uint64_t *r, uint64_t *vec, const void *target)` for the
/// generated C.
///
/// Call it where the port is not: the guest state is `r` and `vec`, the callee
/// is the original bytes, and it returns with both updated. One slot of memory
/// differs between the bridge and a guest `call` — the return address at
/// `r[4] - 8`, a host address here and an image address there — and the harness
/// excludes exactly that slot, which is why the generated C reports it.
#[no_mangle]
pub extern "C" fn guest_bridge(r: *mut u64, vec: *mut u64, target: *const u8) {
    unsafe {
        if std::env::var_os("LIFT_TRACE").is_some() {
            eprintln!("bridge target={:p} rsp={:#x}", target, *r.add(4));
        }
        REGS = r;
        VEC = vec;
        CODE = target;
        PRECHARGE = 0;
        call_guest();
        std::ptr::copy_nonoverlapping(
            std::ptr::addr_of!(OUT) as *const u64,
            r,
            16,
        );
        std::ptr::copy_nonoverlapping(
            std::ptr::addr_of!(VEC_OUT) as *const u64,
            vec,
            64,
        );
    }
}
