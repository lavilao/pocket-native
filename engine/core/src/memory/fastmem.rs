//! the guest's whole address space as one stretch of the host's, each
//! guest address at its own offset from the start, so recompiled code reaches
//! memory with one load or store instead of a page table walk. pages the
//! guest has not mapped fault, and the handler puts a page of zeros there,
//! which is what reading unmapped memory gives anyway. a 64-bit host holds
//! the guest's whole 4 GiB, every address going in plain; a 32-bit host has
//! no room for that, so its view is 1 GiB and the code masks every address
//! into it, which the guest never notices, everything it may touch lying
//! below 0x40000000.

use std::sync::atomic::{AtomicUsize, Ordering};

/// the view's span, and a page over it, since an access at the very top may
/// run past the end: the guest's 4 GiB on a 64-bit host, and on a 32-bit one
/// 1 GiB, about all its 3 GiB of user space that can be spared, the code
/// masking addresses into the low GiB, where the whole guest lives anyway.
#[cfg(target_pointer_width = "64")]
const SPAN: usize = (1 << 32) + 0x1000;
#[cfg(target_pointer_width = "32")]
const SPAN: usize = (1 << 30) + 0x1000;
/// views at once, tests make many.
const SLOTS: usize = 64;

static VIEWS: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
/// how many unmapped pages the code touched.
pub static FAULTS: AtomicUsize = AtomicUsize::new(0);

/// the fault handler there was before ours, for faults that are not ours.
static mut PREVIOUS: Option<libc::sigaction> = None;

/// a view reserved early, in android_main before EGL/ART fragment the
/// 32-bit address space, that View::new adopts instead of reserving.
/// (base, slot) once set; 0 means no early reservation.
static EARLY_BASE: AtomicUsize = AtomicUsize::new(0);
static EARLY_SLOT: AtomicUsize = AtomicUsize::new(usize::MAX);

pub struct View {
    base: *mut u8,
    slot: usize,
}

/// reserve the view's address space now, before the process fragments
/// its 32-bit address space. View::new adopts it instead of reserving
/// again; if no early reservation was made this does nothing.
pub fn reserve_early() {
    if EARLY_BASE.load(Ordering::SeqCst) != 0 {
        return;
    }
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
        log::warn!("fastmem: early reservation of {} MiB failed: {}", SPAN >> 20, std::io::Error::last_os_error());
        return;
    }
    let base = base as usize;
    let Some(slot) = VIEWS.iter().position(|view| view.compare_exchange(0, base, Ordering::AcqRel, Ordering::Acquire).is_ok()) else {
        // SAFETY: the reservation just made
        unsafe { libc::munmap(base as *mut libc::c_void, SPAN) };
        return;
    };
    EARLY_SLOT.store(slot, Ordering::SeqCst);
    EARLY_BASE.store(base, Ordering::SeqCst);
    log::info!("fastmem: early 1 GiB view reserved at 0x{base:08X}");
}

impl View {
    /// a view with nothing mapped, none when the host cannot have one.
    pub fn new() -> Option<View> {
        // adopt the early reservation from android_main, if there is one:
        // it got the biggest hole before ART/EGL fragmented the space
        let early = EARLY_BASE.swap(0, Ordering::SeqCst);
        if early != 0 {
            let slot = EARLY_SLOT.load(Ordering::SeqCst);
            return Some(View { base: early as *mut u8, slot });
        }
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
            log::warn!("fastmem: could not reserve {} MiB of address space: {}", SPAN >> 20, std::io::Error::last_os_error());
            return None;
        }
        let base = base as *mut u8;
        let Some(slot) = VIEWS.iter().position(|view| view.compare_exchange(0, base as usize, Ordering::AcqRel, Ordering::Acquire).is_ok()) else {
            // SAFETY: the reservation just made
            unsafe { libc::munmap(base as *mut libc::c_void, SPAN) };
            log::warn!("fastmem: all {} views are taken", SLOTS);
            return None;
        };
        Some(View { base, slot })
    }

    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// size bytes of the shared file from offset at vaddr.
    pub fn map(&self, vaddr: u32, size: u32, fd: i32, offset: u64) -> bool {
        // only what the view holds, whatever its width: the kernel hands the
        // guest nothing above the low GiB that a 32-bit host's view covers.
        if (vaddr as u64) + (size as u64) > SPAN as u64 {
            return false;
        }
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
        if (vaddr as u64) + (size as u64) > SPAN as u64 {
            return;
        }
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
            // subtracting, rather than adding to base, which on a 32-bit host
            // would wrap the view's end around the address space
            if base != 0 && address >= base && address - base < SPAN {
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
