//! the guest's whole address space as one stretch of the host's, each
//! guest address at its own offset from the start, so recompiled code reaches
//! memory with one load or store instead of a page table walk. pages the
//! guest has not mapped fault, and the handler puts a page of zeros there,
//! which is what reading unmapped memory gives anyway. only for 64-bit
//! hosts: the view is 4 GiB and a page, more than a 32-bit host has, whose
//! recompiled code goes through the page tables instead.

use std::sync::atomic::{AtomicUsize, Ordering};

/// 4 GiB and a page, an access at the very top may run past the end.
const SPAN: usize = (1 << 32) + 0x1000;
/// views at once, tests make many.
const SLOTS: usize = 64;

static VIEWS: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
/// how many unmapped pages the code touched.
pub static FAULTS: AtomicUsize = AtomicUsize::new(0);

/// the fault handler there was before ours, for faults that are not ours.
static mut PREVIOUS: Option<libc::sigaction> = None;

pub struct View {
    base: *mut u8,
    slot: usize,
}

impl View {
    /// a view with nothing mapped, none when the host cannot have one.
    pub fn new() -> Option<View> {
        install();
        // SAFETY: a reservation of address space only, checked
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                SPAN,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return None;
        }
        let base = base as *mut u8;
        let Some(slot) = VIEWS.iter().position(|view| view.compare_exchange(0, base as usize, Ordering::AcqRel, Ordering::Acquire).is_ok()) else {
            // SAFETY: the reservation just made
            unsafe { libc::munmap(base as *mut libc::c_void, SPAN) };
            return None;
        };
        Some(View { base, slot })
    }

    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// size bytes of the shared file from offset at vaddr.
    pub fn map(&self, vaddr: u32, size: u32, fd: i32, offset: u64) -> bool {
        // SAFETY: inside the reservation, which only this view uses
        let mapped = unsafe {
            libc::mmap(
                self.base.add(vaddr as usize) as *mut libc::c_void,
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                offset as libc::off_t,
            )
        };
        mapped != libc::MAP_FAILED
    }

    /// leaves size bytes at vaddr unmapped again.
    pub fn unmap(&self, vaddr: u32, size: u32) {
        // SAFETY: as in map
        unsafe {
            libc::mmap(
                self.base.add(vaddr as usize) as *mut libc::c_void,
                size as usize,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
                -1,
                0,
            );
        }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        VIEWS[self.slot].store(0, Ordering::Release);
        // SAFETY: the reservation View::new made
        unsafe { libc::munmap(self.base as *mut libc::c_void, SPAN) };
    }
}

fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: installing a handler, the old one kept for the rest
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_fault as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_NODEFER;
            libc::sigemptyset(&mut action.sa_mask);
            let mut old: libc::sigaction = std::mem::zeroed();
            if libc::sigaction(libc::SIGSEGV, &action, &mut old) == 0 {
                PREVIOUS = Some(old);
            }
        }
    });
}

extern "C" fn on_fault(signal: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: the kernel hands a valid siginfo, and only async-signal-safe
    // calls follow
    unsafe {
        let address = (*info).si_addr() as usize;
        for view in &VIEWS {
            let base = view.load(Ordering::Acquire);
            if base != 0 && (base..base + SPAN).contains(&address) {
                let page = address & !0xFFF;
                libc::mmap(
                    page as *mut libc::c_void,
                    0x1000,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                    -1,
                    0,
                );
                FAULTS.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // not ours, the handler before decides
        let previous = std::ptr::addr_of!(PREVIOUS).read();
        match previous {
            Some(old) if old.sa_flags & libc::SA_SIGINFO != 0 && old.sa_sigaction > 1 => {
                let handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) = std::mem::transmute(old.sa_sigaction);
                handler(signal, info, context);
            }
            Some(old) if old.sa_sigaction > 1 => {
                let handler: extern "C" fn(libc::c_int) = std::mem::transmute(old.sa_sigaction);
                handler(signal);
            }
            _ => {
                // the default, the fault comes again and ends the process
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = libc::SIG_DFL;
                libc::sigaction(signal, &action, std::ptr::null_mut());
            }
        }
    }
}
