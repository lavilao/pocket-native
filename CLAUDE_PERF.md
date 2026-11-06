# Pocket Native on Redmi 9 (Armv7, PowerVR) — performance diagnosis request

You are helping debug a 3DS recompiler/emulator ("Pocket Native", fork of fearkov/zakuro)
running as an Android app on a **Redmi 9** (dandelion, Android 11/API 30, **32-bit only**,
3.9 GiB RAM, GPU PowerVR Rogue GE8320, OpenGL ES 3.2, MTK driver 1.13).

Game: Rune Factory 4. Native code built once by recompiler, loaded as `code/<title id>.so`
in the app's internal storage. Runs at **~4–5 fps**. Suspected root cause: the CPU *fastmem*
"view" reservation fails, so the loader rejects the recompiled library and the game falls
back to the interpreter (~5 fps). This report is about the performance bug only. A second,
unrelated bug (PowerVR UniFlex segfault in `PVRUniFlexCompileToHw`) is diagnosed elsewhere.

## Device and symptom

- `adb shell mount` shows `/storage/9E69-2AB0` (SD card, fuse/sdcardfs), `/data/media` sdcardfs.
- Game launches, renders, `BufferQueueProducer: queueBuffer: fps=4.42...5.31` (interpreted).
- Previous logcat showed no app log lines — only system/crash buffer. The app logs to
  `POCKET_NATIVE_LOG=<game>/data/log.txt` (see `android/src/lib.rs:135-137` `set_up`,
  `frontend/src/lib.rs:104-109`). That file has **not been readable from here**:
  `/storage/9E69-2AB0/Android/data/io.github.pocketnative.test/files/` contains only `games/`.
  The game folder comes from the launcher intent extra `game`; the data folder that exists
  but never shows `log.txt` suggests the file sits under the game's own folder on shared
  storage, or the process died before `set_up` could create it. Any of the warn lines
  below would pinpoint the failing half — get `cat <game>/data/log.txt` or `adb shell
  run-as` on the internal data path.

## How the fast path is supposed to work

1. `recompiler` emits functions that read/write guest memory through `ctx->fastmem`
   (`recompiler/src/codegen/mod.rs:290,309`), and exports `recomp_fastmem = 1`
   when built with `RECOMP_FASTMEM`. The emitted code does one load/store at
   `view + guest_vaddr` instead of a page-table walk.
2. The host gives it a "view": a single `mmap(PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE)`
   reservation of 1 GiB on 32-bit (`engine/core/src/memory/fastmem.rs:34`) or 4 GiB on 64-bit.
   Slices of guest-addressable memory are then mapped into the view with
   `mmap(MAP_SHARED|MAP_FIXED, fd, offset)` so both the interpreter's physical store
   and the recompiled code alias the same pages.
3. `Memory::fastmem()` returns `view.base()` only if **both** a view exists **and**
   `phys.backing(FCRAM_PADDR, PAGE_SIZE).is_some()` — i.e. the shared memfd backing exists.
   Two independent failure points, identical symptoms, different warn texts
   (`fastmem: could not reserve…` vs `physical memory: … failed: …`).
4. The loader rejects the library when `needs_fastmem() && fastmem().is_null()`
   (`engine/core/src/loader.rs:99`) and warns `"could not load {}, {error}, interpreting
   everything"`.

## Code

### 1. `engine/core/src/memory/fastmem.rs` — View (full, 223 lines, abridged to relevant parts)

```rust
//! the guest's whole address space as one stretch of the host's, each
//! guest address at its own offset from the start, so recompiled code reaches
//! memory with one load or store instead of a page table walk. ...
//! a 64-bit host holds the guest's whole 4 GiB, every address going in plain;
//! a 32-bit host has no room for that, so its view is 1 GiB and the code masks
//! every address into it, which the guest never notices, everything it may
//! touch lying below 0x40000000.

use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_pointer_width = "64")]
const SPAN: usize = (1 << 32) + 0x1000;
#[cfg(target_pointer_width = "32")]
const SPAN: usize = (1 << 30) + 0x1000;  // 1 GiB + page
const SLOTS: usize = 64;

static VIEWS: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
static EARLY_BASE: AtomicUsize = AtomicUsize::new(0);
static EARLY_SLOT: AtomicUsize = AtomicUsize::new(usize::MAX);

pub struct View { base: *mut u8, slot: usize }

// called from android_main (android/src/lib.rs:46) BEFORE the window/EGL/audio
// are created, to grab the biggest contiguous hole in the 32-bit address space.
pub fn reserve_early() {
    if EARLY_BASE.load(Ordering::SeqCst) != 0 { return; }
    install();
    let base = unsafe {
        libc::mmap(std::ptr::null_mut(), SPAN, libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0)
    };
    if base == libc::MAP_FAILED {
        log::warn!("fastmem: early reservation of {} MiB failed: {}",
            SPAN >> 20, std::io::Error::last_os_error());
        return;
    }
    let base = base as usize;
    let Some(slot) = VIEWS.iter().position(|view|
        view.compare_exchange(0, base, Ordering::AcqRel, Ordering::Acquire).is_ok()
    ) else {
        unsafe { libc::munmap(base as *mut libc::c_void, SPAN) };
        return;
    };
    EARLY_SLOT.store(slot, Ordering::SeqCst);
    EARLY_BASE.store(base, Ordering::SeqCst);
    log::info!("fastmem: early 1 GiB view reserved at 0x{base:08X}");
}

impl View {
    pub fn new() -> Option<View> {
        // adopt the early reservation from android_main, if there is one
        let early = EARLY_BASE.swap(0, Ordering::SeqCst);
        if early != 0 {
            let slot = EARLY_SLOT.load(Ordering::SeqCst);
            return Some(View { base: early as *mut u8, slot });
        }
        install();
        let base = unsafe {
            libc::mmap(std::ptr::null_mut(), SPAN, libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0)
        };
        if base == libc::MAP_FAILED {
            log::warn!("fastmem: could not reserve {} MiB of address space: {}",
                SPAN >> 20, std::io::Error::last_os_error());
            return None;
        }
        let base = base as *mut u8;
        let Some(slot) = VIEWS.iter().position(|view|
            view.compare_exchange(0, base as usize, Ordering::AcqRel, Ordering::Acquire).is_ok()
        ) else {
            unsafe { libc::munmap(base as *mut libc::c_void, SPAN) };
            log::warn!("fastmem: all {} views are taken", SLOTS);
            return None;
        };
        Some(View { base, slot })
    }

    /// size bytes of the shared file from offset at vaddr.
    pub fn map(&self, vaddr: u32, size: u32, fd: i32, offset: u64) -> bool {
        if (vaddr as u64) + (size as u64) > SPAN as u64 { return false; }
        let mapped = unsafe {
            libc::mmap(self.base.add(vaddr as usize) as *mut libc::c_void,
                size as usize, libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED, fd, offset as libc::off_t)
        };
        mapped != libc::MAP_FAILED
    }
    // unmap() does the inverse (MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED|MAP_NORESERVE)
}
// SIGSEGV handler `on_fault`: for any address inside a View, mmap a zero page
// (MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED) and resume. For foreign faults it
// chains to PREVIOUS handler or resets to SIG_DFL.
```

### 2. `engine/core/src/memory/physical.rs` — Shared (≈300 MiB file)

```rust
struct Shared { fd: i32, base: *mut u8, len: usize }

#[cfg(any(target_os = "linux", target_os = "android"))]
fn new(len: usize) -> Option<Shared> {
    unsafe {
        let fd = libc::memfd_create(c"zakuro-physical".as_ptr(), libc::MFD_CLOEXEC);
        if fd >= 0 {
            if libc::ftruncate(fd, len as libc::off_t) == 0 {
                let base = libc::mmap(null_mut(), len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
                if base != MAP_FAILED {
                    return Some(Shared { fd, base: base as *mut u8, len });
                }
                log::warn!("physical memory: mmap({len} bytes, shared) failed: {}", …);
            } else {
                log::warn!("physical memory: ftruncate({len} bytes) failed: {}", …);
            }
            libc::close(fd);
        } else {
            log::warn!("physical memory: memfd_create failed: {}", …);
        }
        // ANONYMOUS FALLBACK (added in ce332f2): if memfd/ftruncate/shared-mmap
        // fails, fall back to plain anonymous memory instead of giving up.
        // `backing()` returns None for fd < 0, so fastmem cleanly degrades
        // to the page-table path rather than crashing on mmap(fd=-1).
        let base = libc::mmap(null_mut(), len, PROT_READ|PROT_WRITE,
            MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
        if base == MAP_FAILED {
            log::warn!("physical memory: mmap({len} bytes, anonymous) failed: {}", …);
            return None;
        }
        Some(Shared { fd: -1, base: base as *mut u8, len })
    }
}
```

### 3. `engine/core/src/memory/physical.rs` — `PhysicalMemory::new` backing store

```rust
pub fn new(new3ds: bool, app_bytes: u32) -> PhysicalMemory {
    let total = fcram_size(new3ds);
    // SYSTEM capped at 32 MiB, BASE gets the rest; new3ds extra 128 MiB to Application.
    let sizes = [total as usize, VRAM_SIZE as usize, DSP_RAM_SIZE as usize, AXI_WRAM_SIZE as usize];
    let shared = Shared::new(sizes.iter().sum());   // one file, four offsets
    let mut own = Vec::new();
    let mut offset = 0;
    let [fcram, vram, dsp_ram, axi_wram] = sizes.map(|len| {
        let store = match &shared {
            Some(shared) => Store { ptr: unsafe { shared.base.add(offset) }, len, offset: Some(offset as u64) },
            None => {
                let mut memory = vec![0u8; len].into_boxed_slice();
                let ptr = memory.as_mut_ptr();
                own.push(memory);
                Store { ptr, len, offset: None }
            }
        };
        offset += len;
        store
    });
    PhysicalMemory { fcram, vram, dsp_ram, axi_wram, shared, _own: own, … }
}

/// The shared-file part of backing — THE fastmem gate:
pub fn backing(&self, paddr: u32, size: u32) -> Option<(i32, u64)> {
    let shared = self.shared.as_ref()?;
    // ANONYMOUS FALLBACK HAS NO FILE: the view cannot map pages from it,
    // so fastmem stays off and the page tables take over.
    if shared.fd < 0 { return None; }
    let (base, store) = match paddr { … };
    let offset = (paddr - base) as u64;
    let start = store.offset?;
    (offset + size as u64 <= store.len as u64).then_some((shared.fd, start + offset))
}
```

### 4. `engine/core/src/memory/mod.rs` — Memory::new + fastmem()

```rust
impl Memory {
    pub fn new(new3ds: bool, app_bytes: u32) -> Memory {
        // the view first, while the process still has its biggest clear
        // stretch of address space ... runs before EGL and the game's threads
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let view = fastmem::View::new();            // 1 GiB reservation
        Memory {
            phys: PhysicalMemory::new(new3ds, app_bytes),  // ~270-300 MiB memfd AFTER the view
            read_table: vec![null_mut(); PAGE_TABLE_ENTRIES],  // 4 GiB / 4 KiB  = 1 M entries ≈ 8 MiB (32-bit ptrs)
            write_table: vec![null_mut(); PAGE_TABLE_ENTRIES], // another ≈ 8 MiB
            mappings: BTreeMap::new(), faults: BTreeMap::new(), new_fault: None,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            view,
            linear_base: 0,
        }
    }

    pub fn fastmem(&self) -> *mut u8 {
        // requires BOTH view.is_some() AND phys.backing(FCRAM_PADDR, PAGE_SIZE).is_some()
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if let (Some(view), true) = (&self.view, self.phys.backing(FCRAM_PADDR, PAGE_SIZE).is_some()) {
            return view.base();
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        log::warn!("fastmem: no view for recompiled code (view: {}, shared backing: {})",
            self.view.is_some(), self.phys.backing(FCRAM_PADDR, PAGE_SIZE).is_some());
        std::ptr::null_mut()
    }
}

// Memory::map(): when view exists, each guest mapping is also mapped INTO the view:
//   if let Some(view) = &self.view {
//       match self.phys.backing(paddr, size) {
//           Some((fd, offset)) => { view.map(vaddr, size, fd, offset) or log error }
//           None => view.unmap(vaddr, size)
//       }
//   }
```

### 5. `engine/core/src/loader.rs` — the rejection site (lines 90-121)

```rust
// loader.rs
let opened = crate::recompiled::Library::open(&path).and_then(|library| {
    if library.needs_fastmem() && system.memory.fastmem().is_null() {
        Err("it was built for fastmem, which this host has no view for".to_owned())
    } else { Ok(library) }
});
match opened {
    Ok(library) => { /* run recompiled code with fastmem */
        log::info!("running recompiled code from {}, {}", path.display(), library.describe());
        system.recompiled = Some(library);
    }
    Err(error) => log::warn!("could not load {}, {error}, interpreting everything", path.display()),
}
```

`needs_fastmem()` is `recompiler/abi/src/lib.rs:201-202`:
`self.fastmem` is true iff the library exports symbol `recomp_fastmem` (set at load via
`symbol(b"recomp_fastmem").is_ok()`, `recompiler/abi/src/lib.rs:193`). Whether the
installed `.so` was built with `RECOMP_FASTMEM` decides if rejection can happen at all.

### 6. `android/src/lib.rs` — `android_main` (reserves view early)

```rust
#[no_mangle]
fn android_main(app: AndroidApp) {
    // the 1 GiB fastmem reservation first, before the window, EGL and the
    // game thread fragment the 32-bit address space: on a 32-bit host this
    // is the hardest mapping to place and the difference between fastmem
    // working and the game running interpreted
    #[cfg(target_os = "android")]
    zakuro_core::reserve_early();
    let own = app.internal_data_path().unwrap_or_else(|| PathBuf::from("/data/local/tmp/pocket-native"));
    let folder = game_extra(&app)?;            // game's folder on shared storage
    let data = folder.join("data");
    let _ = std::fs::create_dir_all(&data);
    let _ = std::fs::create_dir_all(own.join("config"));
    set_up(&own, &folder, &data);              // sets POCKET_NATIVE_LOG=data/log.txt,
    …                                        //   ZAKURO_FAST=1, ZAKURO_RECOMPILED=own/code/<id>.so
    // …window, EGL, input, game thread…
    let game = std::thread::Builder::new().name("game".into())
        .spawn(move || pocket_native::run(vec![rom]))…
}
```

Note: `reserve_early()` is in `engine/core/src/lib.rs:9` (`pub use memory::reserve_early`).
`set_up` at `android/src/lib.rs:135` also sets `XDG_CACHE_HOME=own/cache` (shader cache)
and `TMPDIR=data`; reading `data/env.txt` for user-tunable switches.

### 7. Commit state

```
86fd68c add the zakuro-core dependency to pocket-native-android so android_main can call reserve_early()
ce332f2 apply Claude's diagnosis: Shared::new falls back to anonymous mmap …,
            fastmem view reserved in android_main …,
            tables[] sized, shade() values[7] unrolled, DYNAMIC stand-in trivial shader
ff10871 three fixes from the Redmi 9 run: sample all textures before any branch …,
            port upstream zakuro c772fcf for the garbled shared font …,
            and create the fastmem view before the shared physical memory …
```

The two most recent commits are the current best attempt at the performance bug
(early reservation + anonymous fallback). The game still runs at ~5 fps after them,
and no log was obtainable to confirm which of the two warn lines fires.

## What we know, what we don't

**Known**
- Still ~4–5 fps after `ce332f2`/`86fd68c` (BufferQueueProducer fps=4.95–5.31 on 11-06 07:57).
- FPS ~5 is the interpreter cost (software CPU emulation of ARM, page-table walk per
  instruction/fetch). Recompiled with fastmem should reach display rate (60 fps or device
  limit); compiled without fastmem (page-table callbacks) should be noticeably faster than
  interpreted but slower than fastmem.
- The log is not in logcat: `frontend/src/lib.rs:104-109` routes logs to
  `POCKET_NATIVE_LOG` file instead of stderr, and there is no `android_log` integration.
  System/crash buffer only (tombstone came from there).
- `grep -a` on the session file shows no readable `fastmem:` log line at all — the app's
  own log was never pulled.

**Don't know** (which one of the three fires — needs one log line, or a code change to
emit it to logcat)
1. `fastmem: early reservation of 1024 MiB failed: …` (`View::new` path never taken because
   `EARLY_BASE` adoption code is bypassed if `reserve_early` returned early — but
   early reservation seems to have succeeded if there is no warning? unknown).
2. `fastmem: could not reserve 1024 MiB of address space: …` (`View::new` fallback).
3. `physical memory: memfd_create failed: …` / `ftruncate failed` / `mmap(… bytes) failed`.
4. `fastmem: no view for recompiled code (view: true/false, shared backing: true/false)` —
   the pair that distinguishes View vs Shared failure directly.
5. `could not load …, interpreting everything`.

## Hypotheses ranked (please argue with these)

1. **View reservation succeeds (early), Shared memfd succeeds, but the .so was built
   with fastmem and something else rejects it.** Unlikely — rejection requires
   `fastmem().is_null()`. If both succeeded, fastmem returns base and the library is
   accepted. Check: confirm the installed `.so` was built with `RECOMP_FASTMEM`.
2. **Shared `mmap(MAP_SHARED)` on ~270–300 MiB fails due to address-space fragmentation.**
   By `Memory::new()` time, ART's heap, EGL, the game/ROM file handling, and the
   just-created 1 GiB reservation have fragmented 3 GiB user space. `memfd_create` +
   `ftruncate` may succeed but the final `mmap(..., len, MAP_SHARED, fd, 0)` needs a
   contiguous ~270 MiB hole — the most likely failure after the view took the biggest hole.
   The anonymous fallback then returns `Shared{fd:-1}`, `backing()` returns `None`,
   `fastmem()` returns null. **Same symptom as view failure.**
3. **View reservation fails** (either early or late) because the 1 GiB hole is gone.
   `reserve_early()` runs before the window, but after ART started the process and after
   `android_main` set up the `game_extra` folder and created data dirs — maybe already
   too late. Also `reserve_early()` might never be called if the `android` crate builds
   without the zakuro-core dependency (fixed at `86fd68c`, but was the APK rebuilt after?).
4. **memfd_create silently killed by seccomp (SIGSYS).** Earlier evidence: `ioprio_set`
   (syscall 314) killed the process with SIGSYS and had to be guarded
   `cfg(not(target_os = "android"))`. If `memfd_create` were blocked the same way, the
   process would die in `Shared::new` before any log. **But the game runs** at 5 fps,
   so it wasn't killed. `memfd_create` is widely used by the framework on Android 11.
5. **The library was built WITHOUT fastmem** — then it wouldn't be rejected, would run
   via page-table callbacks, and still be slowish (~10–30 fps?). But the observed ~5 fps
   suggests interpreter. Check `nm -D` / `readelf --dyn-syms title.so` for `recomp_fastmem`:
   one line decides.
6. **Perf is slow even WITH fastmem** (game is shader-bound, not CPU-bound). Then Oracle
   would be higher fps with fastmem but still low. Plausible second-order: the game may
   be shader-bound (PowerVR), so fixing fastmem might only lift 5→15 fps. The shader
   bug's separate diagnosis mentions slow shader compiles and a big fragment shader.

## Questions

1. **Order of failure**: view (1 GiB `PROT_NONE`) vs shared (~270 MiB `MAP_SHARED`)?
   Can we prove it without a log — e.g. does the fastmem fallback path ever leave a
   tombstone or secondary trace? What is the smallest change that makes one warn line
   reachable on device: mirror `log::warn!` to `__android_log_print` (Java/`android_log`
   tag), or write `data/log.txt` to a launcher-readable location?
2. **Fix for fragmentation**:
   (a) Reserve the **shared memfd early too**, alongside the view, in `android_main` —
       static `OnceLock<Shared>` / `(fd, base)` adopted by `PhysicalMemory::new`.
       The memfd also does `ftruncate(len)` — that's a RAM commitment (tmpfs pages are
       copy-on-write sparse, so ftruncate of 270 MiB is cheap until written, but you
       commit the VA). Which is the right VA reservation strategy?
   (b) Make the view retry with successively smaller spans if 1 GiB fails
       (512 MiB, 256 MiB). What does the compiled code assume about span? It masks
       `(addr & 0x3FFFFFFF)`-ish into the low GiB; a 256 MiB view would need 0xFFFFFFF
       masking — does it still cover everything the guest touches? Check what the
       guest's highest real address is (FCRAM base + size). FCRAM top ~0x30000000
       (New3DS FCRAM 256 MiB?), maybe 256 MiB suffices. But `View::map` guard
       `vaddr+size > SPAN` would silently unmap higher mappings.
   (c) Reserve view with `MAP_FIXED` at a known hint address (e.g. high user space)
       instead of letting the kernel choose.
   (d) Reserve at **library load time (`.init_array`)** even earlier than `android_main`
       — is there a cargo-correct way to run before ART allocates its bulk heap?
3. **Fallback when memfd/mmap-shared fails**: the anonymous fallback currently disables
   fastmem for the whole session. Is there a shared-memory fallback that preserves
   aliasing without a fd? Options: `memfd_create` → `ashmem_create_region`
   (deprecated but present on API 30) → anonymous `MAP_SHARED` (no alias guarantee).
   Could `View` be changed to map from `Store` pointers directly instead of via `fd+offset`
   when the backing is anonymous? That would keep fastmem on anonymous backing — but
   then two views must not alias the same RAM twice; `View::map` would need a different
   implementation (copy-on-write via the existing Store pointer?).
4. **Interpreter cost check**: measure Memory interpreter vs recompiled throughput on this
   exact device. Is `ZAKURO_FAST=1` or `ZAKURO_HIST` relevant? `ftruncate` of 270 MiB
   maps tmpfs files — does that commit OOM on a 3.9 GiB RAM device? `Shared::new`'s
   `ftruncate` is sparse (no allocation until pages are faulted in), but `mmap(MAP_SHARED)`
   of the full file commits VA. Could we map `ftruncate` lazily or in chunks to reduce
   VA pressure?
5. **Scheduling / threads**: `Memory::new` runs on the "game" thread after window/EGL/audio.
   If we move `reserve_early()` (and a new shared reservation) to process start (native
   `JNI_OnLoad`? `android_main` is already first native code, so same thing), will
   1 GiB + 270 MiB holes be findable? Alternatively, can we shrink the 1 GiB reservation:
   the guest never touches above ~0x40000000 — the view only needs to cover actually
   mapped guest memory (FCRAM+Vram+etc.). How small could the span be while keeping
   `write8(view+vaddr)` correct for every guest address the game uses?
6. **Built library check**: how does `Build a game.bat` / `scripts/` build the `.so`? Is
   `RECOMP_FASTMEM` on by default? If the only path is "build without fastmem and accept
   page-table callbacks", which one should ship? What fps would a non-fastmem library get
   on this SoC (measure with a desktop Armv7 emulator vs We can't measure on device)?
7. **Profiling without a log**: suggest `ZAKURO_INTERP_STATS=1` / fallback counter
   (`Library::fallbacks`, `recompiled.rs:173`) surfaced to logcat, or if that is always
   zero with the interpreter path, the fallback counter is a proxy for the interpreter's
   share — how to attribute slow fps to interpreter vs shader stalls?
8. **Storage**: the current `log.txt` lands in the game's `data/` folder, which we could
   not read when we expected. Confirm whether the game "folder" is on internal or
   external storage (SD card) for this build — the `games` folder found at
   `/storage/9E69-2AB0/Android/data/io.github.pocketnative.test/files/games/` was empty
   except `games/`. Where is the "game folder" that the launcher passes via intent extra?
   Should the log go to `app.internal_data_path()/log.txt` (`run-as` readable) instead?
9. **Risk**: does the current `reserve_early()` need `MAP_NORESERVE`? That's a request on
   Linux; Android 11 respects it for anonymous. Is there risk of `reserve_early()`
   succeeding (1 GiB reserved) but `View` then failing at `SYSTEM` region due to
   fragmentation — and now the view sits unused (VA wasted)? Monitor with `/proc/<pid>/maps`?
10. **Minimum acceptance test**: no device + no ability to read the log. What can we test
    on the host (x86/arm64 Linux) — is there a 32-bit ARM emulator + 1 GiB virtual
    constraint test? Or a test that the reservation path runs before EGL? What's the
    smallest build + 1-device test loop to validate the fix?

## Ask

Ranked root cause + smallest general fix that doesn't regress desktop/64-bit Linux
(from both fastmem and non-fastmem paths), with files/symbols cited. Diagnosis, not a
full implementation. Also: is the anonymous fallback actually a perf regression
(it turns a hard failure into silent fastmem-off)?

## What I know is standard

- `memfd_create` is allowed on Android 11 (framework uses it), but seccomp **can** block
  it per-app for certain configurations; `ioprio_set` (314) was blocked and killed the
  process with SIGSYS. Guarded now.
- PowerVR GE8320 UniFlex crashes are a separate bug (handled elsewhere). Shader compiles
  also hurt fps a lot on this driver.
- `View::new()` adopting the early reservation: `EARLY_BASE.swap(0)` — if called twice
  (e.g. two `Memory::new()` in tests), the second gets nothing. `SLOTS=64` covers tests.

## Evidence to fetch next (adb)

```bash
# app's log — find the game folder first
adb shell "ls /storage/9E69-2AB0/Android/data/io.github.pocketnative.test/files/games/"
adb shell "ls -la /storage/9E69-2AB0/Android/data/io.github.pocketnative.test/files/games/*/data/ 2>/dev/null"
adb shell "cat /storage/9E69-2AB0/Android/data/io.github.pocketnative.test/files/games/*/data/log.txt"
# internal data once-app-run (debuggable?)
adb shell "run-as io.github.pocketnative.test cat files/data/log.txt 2>/dev/null || echo no-run-as"

# check the library was built with fastmem
adb shell "ls -la /data/data/io.github.pocketnative.test/files/code/ 2>/dev/null"
adb shell "run-as io.github.pocketnative.test readelf --dyn-syms files/code/*.so 2>/dev/null | grep -i fastmem"
# fallback: read the .so off the SD card staging folder

# fps / crash sanity
adb shell "logcat -d 2>/dev/null | grep -E 'queueBuffer.*Game|fps=' | tail -5"
```

## History

- Commit `ff10871` (Redmi 9 run): view created before shared physical memory, log added.
- Commit `ce332f2`: Claude's first diagnosis applied — anonymous fallback in `Shared::new`,
  early `reserve_early()` in `android_main`, shader hardening (unrelated to perf).
- Commit `86fd68c`: android crate gets `zakuro-core` dependency so `reserve_early()` compiles.
