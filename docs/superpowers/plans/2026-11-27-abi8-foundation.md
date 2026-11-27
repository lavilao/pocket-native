# ABI 8 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine able to run ABI 8 recompiled libraries while ABI 5–7 libraries keep working, then switch the recompiler to emit ABI 8.

**Architecture:** Two `Context` layouts share the prefix through `user`. The engine loads a library, reads its `recomp_abi`, and picks `Modern` (ABI 8) or `Legacy` (ABI 5–7). The fork's 1 GiB fastmem view is appended after `trail` on modern and stays after `user` on legacy. The recompiler is ported from 3dsrecomp `d143f2c` file by file, reapplying the fork's fastmem, a64, precache, update/Mii/GBA-VC and compile-pipeline work.

**Tech Stack:** Rust 2021, C11 generated code, `recomp-abi` crate, `zakuro-core` engine, GitHub Actions (ubuntu-latest, NDK `armv7a-linux-androideabi29-clang`).

**Spec:** `docs/superpowers/specs/2026-11-26-abi8-foundation-design.md`

**Pinned upstream:** `/tmp/opencode/up/3dsrecomp-d143f2c` (3dsrecomp `d143f2c`) and `/tmp/opencode/up/zakuro-14e5786` (zakuro `14e5786`). Re-download with the commands in Task 0 if `/tmp/opencode` was wiped.

## Global Constraints

- **No local builds or tests.** Every verification runs in GitHub Actions. A task is done when its push is green (or when the failure is a known, documented next-task dependency).
- Primary target is 32-bit Android (`armeabi-v7a`). aarch64 Android and x86_64 desktop are regression targets only.
- ABI 5/6/7 libraries, including 32-bit fastmem libraries, keep running unchanged.
- The fork's `RECOMP_FASTMEM` 1 GiB view stays. On modern it is the last `Context` field, after `trail`; on legacy it stays after `user`.
- Do not touch the `3dsrecomp-upstream` branch.
- New builds eventually emit ABI 8 only; the engine keeps a legacy loader for old libraries.
- `OLDEST_ABI = 5`, `GENERATION = 6`.
- `stack_limit` is `usize` (4 bytes on armv7); tick arithmetic is `u64`; only the budget is `i32`.
- Pin upstream to `d143f2c` / `14e5786`. Rebase later, not during this slice.

## Review Focus

- A 32-bit host must not assume 64-bit `usize` or spare address space: `stack_limit`, `RECOMP_VIEW_OFFSET` masking, and `Piece`/`Origin` offsets stay `u32`.
- A legacy library must never see modern fields: `fastmem` after `user` for ABI 5–7, after `trail` for ABI 8. A wrong offset corrupts guest memory silently.
- An ABI 8 library built by upstream (no trailing `fastmem`) must still run; never read past `trail` for it.
- Ticks and the lent mutex must reconcile with the kernel when code hands back: a wrong `holding` count deadlocks the title.
- `natives` must have an escape hatch (`ZAKURO_NATIVES=0`) and must not change results on armv7 before it is measured.

---

## Task 0: Refresh the pinned upstream trees

**Files:**
- Create: `/tmp/opencode/up/3dsrecomp-d143f2c/` (reference only)
- Create: `/tmp/opencode/up/zakuro-14e5786/` (reference only)

- [ ] **Step 1: Download and extract both trees**

```bash
mkdir -p /tmp/opencode/up && cd /tmp/opencode/up
curl -sL -o 3ds.tar.gz https://codeload.github.com/fearkov/3dsrecomp/tar.gz/d143f2c
curl -sL -o zak.tar.gz https://codeload.github.com/fearkov/zakuro/tar.gz/14e5786
tar xzf 3ds.tar.gz && tar xzf zak.tar.gz
test -f 3dsrecomp-d143f2c/abi/recomp.h && test -f zakuro-14e5786/crates/zakuro-core/src/recompiled.rs
```

Expected: both `test` calls succeed.

- [ ] **Step 2: Confirm the pinned files**

```bash
grep -m1 '^#define RECOMP_ABI' /tmp/opencode/up/3dsrecomp-d143f2c/abi/recomp.h
wc -l /tmp/opencode/up/3dsrecomp-d143f2c/src/{discover,natives,verify,build}.rs
```

Expected: `#define RECOMP_ABI 8`, and the four files exist.

---

## Task 1: Add ABI 8 types beside the ABI 7 line

This task is additive: `ABI` stays `7` and `HEADER` stays the ABI 7 header, so every existing test and library keeps working. The new types are exercised by a layout test only.

**Files:**
- Create: `recompiler/abi/recomp8.h`
- Create: `recompiler/abi/src/abi8.rs`
- Modify: `recompiler/abi/src/lib.rs`
- Test: `recompiler/abi/src/abi8.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Produces: `recomp_abi::abi8::{Context8, Host8, Code8, Entry8, Module8, Piece, Run, Origin, Origins, Library8, HEADER8, ABI8, TRAIL, NO_TICKS, NOTHING_HELD, STOPPED, STACK, stack_limit}`
- Produces: `recomp_abi::{Loaded, open, linked}` (dispatchers; `Library` remains the legacy ABI 7 type until Task 6)

- [ ] **Step 1: Create `recomp8.h` from upstream plus the fork field**

```bash
cp /tmp/opencode/up/3dsrecomp-d143f2c/abi/recomp.h recompiler/abi/recomp8.h
```

Then apply these two edits.

Edit 1 — add the trailing field after `trail`:

```c
    uint32_t done;
    uint32_t trail[RECOMP_TRAIL];
    /* fork extension: the host's 1 GiB view of guest memory, or null.
       upstream ABI 8 code never reads this field, so upstream libraries
       also run against this header. code built with RECOMP_FASTMEM reads
       and writes through it. */
    uint8_t *fastmem;
};
```

Edit 2 — insert the fork fastmem block immediately before `#define C_EQ (F_Z)`:

```c
#ifdef RECOMP_FASTMEM
/* one access at the address's own offset in the host's view of the guest's
   memory, which has every page mapped. a 64-bit host holds the guest's whole
   4 GiB, so the address goes in whole; a 32-bit host can only hold 1 GiB of
   it, and the address is masked into that, which changes nothing the guest
   can notice, everything it may touch being below 0x40000000. */
#undef mem_read8
#undef mem_read16
#undef mem_read32
#undef mem_write8
#undef mem_write16
#undef mem_write32
/* the view, which generated code holds in a local: a store through it
   could be to the context as far as the compiler knows, which would have it
   load ctx->fastmem again after every one */
#define RECOMP_VIEW(ctx) ((ctx)->fastmem)
#if UINTPTR_MAX == 0xFFFFFFFFu
#define RECOMP_VIEW_OFFSET(address) ((uint32_t)(address) & 0x3FFFFFFFu)
#else
#define RECOMP_VIEW_OFFSET(address) ((uint32_t)(address))
#endif
#define mem_read8(ctx, address) (*(const uint8_t *)(RECOMP_VIEW(ctx) + RECOMP_VIEW_OFFSET(address)))
#define mem_read16(ctx, address) ({ uint16_t __v; __builtin_memcpy(&__v, RECOMP_VIEW(ctx) + RECOMP_VIEW_OFFSET(address), 2); __v; })
#define mem_read32(ctx, address) ({ uint32_t __v; __builtin_memcpy(&__v, RECOMP_VIEW(ctx) + RECOMP_VIEW_OFFSET(address), 4); __v; })
#define mem_write8(ctx, address, value) do { *(RECOMP_VIEW(ctx) + RECOMP_VIEW_OFFSET(address)) = (uint8_t)(value); } while (0)
#define mem_write16(ctx, address, value) do { uint16_t __v = (value); __builtin_memcpy(RECOMP_VIEW(ctx) + RECOMP_VIEW_OFFSET(address), &__v, 2); } while (0)
#define mem_write32(ctx, address, value) do { uint32_t __v = (value); __builtin_memcpy(RECOMP_VIEW(ctx) + RECOMP_VIEW_OFFSET(address), &__v, 4); } while (0)
#endif
```

- [ ] **Step 2: Write `abi8.rs` from upstream's `abi/src/lib.rs`**

Take `/tmp/opencode/up/3dsrecomp-d143f2c/abi/src/lib.rs` and rename, in the copied file only:

| upstream | abi8.rs |
|---|---|
| `HEADER` | `HEADER8` (`include_str!("../recomp8.h")`) |
| `ABI` | `ABI8` = `8` |
| `Code` | `Code8` |
| `Context` | `Context8` |
| `Host` | `Host8` |
| `Entry` | `Entry8` |
| `Module` | `Module8` |
| `Library` | `Library8` |
| `open` | `open8` |
| `linked` | `linked8` |

Then make these structural changes:

1. Add the trailing field to `Context8`:

```rust
    pub done: u32,
    pub trail: [u32; TRAIL],
    /// fork extension: the 1 GiB view, after everything upstream reads.
    pub fastmem: *mut u8,
```

2. Add `fastmem: bool` to `Library8`, set by `open8` from the `recomp_fastmem` symbol, and expose:

```rust
    pub fn needs_fastmem(&self) -> bool {
        self.fastmem
    }
```

3. `linked8` accepts `OLDEST_ABI..=ABI8` so the engine can hand it any modern `Linked`; the legacy dispatcher handles 5–7.

4. Keep upstream's `Origins`, `Piece`, `Run`, `Origin`, `CodeHash`, `check`, `interpret`, `interpret_at`, `set_natives`, `origins`, `place`, `lookup`, `describe`, `entries`, `modules`.

5. Keep the layout test, but add `AT(Context, fastmem)` and `SIZE`/`AT` lines for the fork field, and print `RECOMP_ABI 8`.

- [ ] **Step 3: Wire the module and dispatchers in `lib.rs`**

Add at the top of `recompiler/abi/src/lib.rs`:

```rust
pub mod abi8;
pub use abi8::{
    Context8, Host8, Code8, Entry8, Module8, Origin, Origins, Piece, Run, Library8,
    HEADER8, ABI8, TRAIL, NO_TICKS, NOTHING_HELD, STOPPED, STACK, stack_limit,
};
```

Add the dispatcher below the existing `Library`:

```rust
/// a library the engine can run, picked by the version it was built for.
pub enum Loaded {
    Modern(abi8::Library8),
    Legacy(Library),
}

impl Loaded {
    pub fn abi(&self) -> u32 {
        match self {
            Loaded::Modern(code) => code.abi(),
            Loaded::Legacy(code) => ABI,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Loaded::Modern(code) => code.describe(),
            Loaded::Legacy(code) => code.describe(),
        }
    }

    pub fn needs_fastmem(&self) -> bool {
        match self {
            Loaded::Modern(code) => code.needs_fastmem(),
            Loaded::Legacy(code) => code.needs_fastmem(),
        }
    }
}

#[cfg(feature = "load")]
pub fn open(path: &Path) -> Result<Loaded, String> {
    // read recomp_abi first, then hand the path to the right loader
    let abi = unsafe {
        let library = libloading::Library::new(path).map_err(|e| e.to_string())?;
        let symbol = library.get::<*const u32>(b"recomp_abi").map_err(|e| e.to_string())?;
        **symbol
    };
    match abi {
        8 => abi8::open8(path).map(Loaded::Modern),
        5..=7 => Library::open(path).map(Loaded::Legacy),
        other => Err(format!("it was built for version {other} of the interface, this is {ABI8}")),
    }
}

pub fn linked(linked: &Linked) -> Result<Loaded, String> {
    match linked.abi {
        8 => abi8::linked8(linked).map(Loaded::Modern),
        5..=7 => Library::linked(linked).map(Loaded::Legacy),
        other => Err(format!("it was built for version {other} of the interface, this is {ABI8}")),
    }
}
```

`Linked.abi` is private today; add `pub fn abi(&self) -> u32 { self.abi }` to the existing `Linked` in `lib.rs`.

- [ ] **Step 4: Commit**

```bash
git add recompiler/abi/recomp8.h recompiler/abi/src/abi8.rs recompiler/abi/src/lib.rs
git commit -m "recomp-abi: the ABI 8 types beside the ABI 7 line"
git push origin abi8-foundation
```

Expected in CI: `cargo test -p recomp-abi` runs the new layout test; every existing test stays green.

---

## Task 2: Engine loads and runs both layouts

**Files:**
- Modify: `engine/core/src/recompiled.rs`
- Modify: `engine/core/src/lib.rs:1711-1790` (the polling fixture)
- Test: `engine/core/src/recompiled.rs` (`#[cfg(test)]`)

**Interfaces:**
- Consumes: `recomp_abi::{Loaded, open, linked, Context8, Host8, Code8, LegacyContext, LegacyHost, Code7}`
- Produces: `recompiled::FoundCode::{Modern(Code8), Legacy(Code7)}`
- Produces: `recompiled::Library { code: Loaded, ... }` with `lookup`, `run`, `place`, `needs_fastmem`, `has_code`

- [ ] **Step 1: Add the legacy types to `recomp-abi`**

In `recompiler/abi/src/lib.rs`, add beside the existing `Context`/`Host`:

```rust
pub type Code7 = unsafe extern "C" fn(*mut LegacyContext);

#[repr(C)]
pub struct LegacyContext {
    pub r: [u32; 16],
    pub n: u8, pub z: u8, pub c: u8, pub v: u8, pub q: u8, pub thumb: u8, pub ge: u8,
    pub exclusive: u8,
    pub budget: i32,
    pub exit: u32,
    pub svc: u32,
    pub depth: u32,
    pub exclusive_address: u32,
    pub tls: u32,
    pub read_pages: *const *mut u8,
    pub write_pages: *const *mut u8,
    pub vfp: *mut u32,
    pub fpscr: *mut u32,
    pub host: *const LegacyHost,
    pub user: *mut c_void,
    pub fastmem: *mut u8,
}

#[repr(C)]
pub struct LegacyHost {
    pub read8: unsafe extern "C" fn(*mut LegacyContext, u32) -> u8,
    pub read16: unsafe extern "C" fn(*mut LegacyContext, u32) -> u16,
    pub read32: unsafe extern "C" fn(*mut LegacyContext, u32) -> u32,
    pub write8: unsafe extern "C" fn(*mut LegacyContext, u32, u8),
    pub write16: unsafe extern "C" fn(*mut LegacyContext, u32, u16),
    pub write32: unsafe extern "C" fn(*mut LegacyContext, u32, u32),
    pub interpret: unsafe extern "C" fn(*mut LegacyContext, u32, u32),
    pub lookup: unsafe extern "C" fn(*mut LegacyContext, u32) -> Option<Code7>,
    pub svc: unsafe extern "C" fn(*mut LegacyContext, u32),
}
```

`LegacyContext`'s field order must match the current ABI 7 `Context` exactly, including `fastmem` after `user`. Add a layout test that compiles the current `recomp.h` (the ABI 7 one) and compares offsets, same pattern as upstream's test.

- [ ] **Step 2: Make the engine `Library` hold `Loaded`**

In `engine/core/src/recompiled.rs`:

```rust
use recomp_abi::{Code7, Code8, Context8, Host8, LegacyContext, LegacyHost, Loaded, EXIT_BUDGET, EXIT_SVC, EXIT_UNWIND};

/// code found for an address, in whichever layout the library uses.
pub enum FoundCode {
    Modern(Code8),
    Legacy(Code7),
}
```

Change `Library::open`/`linked` to `recomp_abi::open(path)` / `recomp_abi::linked(linked)`, and `Library::new` to take `Loaded`.

`lookup` returns `Option<FoundCode>`:

```rust
fn lookup(&self, address: u32) -> Option<FoundCode> {
    match &self.code {
        Loaded::Modern(code) => code.lookup(address).map(FoundCode::Modern),
        Loaded::Legacy(code) => code.lookup(address).map(FoundCode::Legacy),
    }
}
```

`place` and `describe` dispatch on `Loaded`. `needs_fastmem` already dispatches in `recomp-abi`.

- [ ] **Step 3: Build the right context and call the right code**

Split the current `run` body into `run_modern` and `run_legacy`, sharing `Machine`, `load`, `store`, `float_mode`, and the memory callbacks.

Modern context construction:

```rust
let mut ctx = Context8 {
    // registers/flags/pages/vfp/fpscr/host/user as today
    stack_limit: recomp_abi::stack_limit(),
    tick_end: 0,
    tick_last: 0,
    tick_floor: i32::MAX,
    tick_cost: 0,
    tick_at: recomp_abi::NO_TICKS,
    ticks: 0,
    wait_at: recomp_abi::NO_TICKS,
    release_at: recomp_abi::NO_TICKS,
    held: 0,
    holding: 0,
    held_floor: recomp_abi::NOTHING_HELD,
    done: 0,
    trail: [0; recomp_abi::TRAIL],
    fastmem,
};
```

Legacy context construction sets `fastmem` after `user`, exactly as the current code does.

Call sites:

```rust
match code {
    FoundCode::Modern(code) => {
        ctx8.exit = 0;
        ctx8.depth = 0;
        unsafe { code(&mut ctx8) };
    }
    FoundCode::Legacy(code) => {
        ctx7.exit = 0;
        ctx7.depth = 0;
        unsafe { code(&mut ctx7) };
    }
}
```

The loop must keep one context alive across calls; use an enum:

```rust
enum Running {
    Modern(Context8),
    Legacy(LegacyContext),
}
```

with `budget()`, `set_budget()`, `exit()`, `set_exit()`, `pc()`, `set_pc()`, `thumb()` helpers, so the stop logic stays in one place.

- [ ] **Step 4: Add the two host vtables**

`HOST8` and `HOST7` have the same callbacks, differing only in pointer type and `lookup` return. Write one macro or two small modules:

```rust
macro_rules! host {
    ($name:ident, $ctx:ty, $code:ty) => {
        static $name: ... = ...;
    };
}
```

The memory callbacks cast `(*ctx).user` to `*mut Machine`; `user` is at the same offset in both layouts (the prefix through `user` is identical). `interpret`, `svc` and `lookup` call the same `Machine` helpers; only `lookup`'s return type differs.

- [ ] **Step 5: Update the polling fixture in `engine/core/src/lib.rs`**

`polling` becomes ABI 8-shaped:

```rust
unsafe extern "C" fn polling<const IN_PLACE: bool>(ctx: *mut recomp_abi::Context8) -> u32 {
    // body unchanged, but every `return;` becomes `return (*ctx).r[15] | (*ctx).thumb as u32;`
    // and the final path returns the same
}
```

Build `Entry8 { address, code }` and `Linked::new(..., recomp_abi::ABI8, ...)` for the modern fixture; keep an `Entry`/`Code7` path for the legacy fixture.

- [ ] **Step 6: Commit**

```bash
git add engine/core/src/recompiled.rs engine/core/src/lib.rs recompiler/abi/src/lib.rs
git commit -m "engine: run ABI 8 and ABI 5-7 libraries side by side"
git push origin abi8-foundation
```

Expected in CI: workspace tests green; the polling fixture runs both layouts.

---

## Task 3: Ticks, lent mutex and trail in the engine

**Files:**
- Create: `engine/core/src/spin.rs` (from `/tmp/opencode/up/zakuro-14e5786/crates/zakuro-core/src/spin.rs`)
- Modify: `engine/core/src/lib.rs` (`System::{tick_in_place, tick_due, answered_in_place, ticks_in_place, svc_in_place}`)
- Modify: `engine/core/src/kernel/` (`lendable`, `held_as`, `held`, mutex state)
- Modify: `engine/core/src/recompiled.rs` (`Machine::{take_back, held_from, ticks_from, lend}`, `run` budget/limit)

**Interfaces:**
- Consumes: `recomp_abi::{NOTHING_HELD, NO_TICKS, TRAIL}`, `kernel::svc::SYSTEM_TICK_COST`
- Produces: `Machine { limit, due, held, lends, tick_wanted }`, `caught_up`, `lend`, `ticks_from`, `held_from`, `take_back`
- Produces: `System::tick_in_place(cycles, at, event, deadline, limit, library) -> Option<...>`, `tick_due(event, deadline) -> Option<u64>`, `answered_in_place(&trail, done, tick_last, after)`, `ticks_in_place(at, ticks, tick_last)`

- [ ] **Step 1: Port `spin.rs`**

Copy upstream's `spin.rs` into `engine/core/src/spin.rs`, then rename `crate::kernel::svc::SYSTEM_TICK_COST` only if the fork's path differs (it does not: `engine/core/src/kernel/svc.rs:1016`). Add `mod spin;` to `engine/core/src/lib.rs`.

- [ ] **Step 2: Port the kernel hooks**

Compare upstream `crates/zakuro-core/src/kernel/` with `engine/core/src/kernel/`. Port the mutex helpers (`lendable`, `held_as`, `held`) and the tick helpers, keeping the fork's 32-bit and service changes. The upstream signatures are the contract; the fork's `System` is the receiver.

- [ ] **Step 3: Port the `Machine` methods**

From upstream `recompiled.rs:100-195`, port `take_back`, `held_from`, `ticks_from`, `lend` and the `Machine` fields `limit`, `due`, `held`, `lends`, `tick_wanted`. Keep the fork's `cycles`, `budget`, `ran`, `deadline`, `event`, `left`, `went_on`.

- [ ] **Step 4: Wire `run`**

In the modern loop, before each call:

```rust
machine.ticks_from(&mut ctx);
```

After the call, when the code answered svcs itself:

```rust
machine.take_back(&mut ctx, &mut system);
```

Set `ctx.tick_cost = crate::kernel::svc::SYSTEM_TICK_COST as i32`, `ctx.tick_floor`, `ctx.tick_at`, and the wait/release fields from `machine.held`. Mirror upstream's `run` (lines 277–390) but keep the fork's `Stop::{Svc, Left, Declined, Exit}` mapping.

- [ ] **Step 5: Commit**

```bash
git add engine/core/src/spin.rs engine/core/src/lib.rs engine/core/src/kernel engine/core/src/recompiled.rs
git commit -m "engine: the code answers ticks and the lent mutex itself"
git push origin abi8-foundation
```

Expected in CI: new tests for `ticks_from`/`held_from` with synthetic contexts; existing tests green.

---

## Task 4: Natives, origins, stale and interpret in the engine

**Files:**
- Modify: `engine/core/src/recompiled.rs`
- Modify: `engine/core/src/lib.rs` (`ZAKURO_NATIVES=0` logging)

**Interfaces:**
- Produces: `Library::set_natives(bool) -> bool`, `Library::check(&mut self, memory: &mut Memory) -> usize`, `Library::interpret(unit, start) -> bool`, `Library::interpret_at(unit, address) -> usize`, `Library::generation() -> u32`, `Library::outdated() -> bool`
- Consumes: `recomp_abi::abi8::{Origins, check, interpret, interpret_at}`

- [ ] **Step 1: Add the modern-only methods**

```rust
pub fn set_natives(&mut self, on: bool) -> bool {
    match &mut self.code {
        Loaded::Modern(code) => code.set_natives(on),
        Loaded::Legacy(_) => false,
    }
}
```

`check`/`interpret`/`interpret_at` return `0`/`false` for legacy. `check` reads guest memory through `Memory::read_bytes`.

- [ ] **Step 2: Call `check` when a module is placed and at load**

In `place`, after `code.place(index, base)`, call `self.check(memory)` for modern. In `Library::open`, log the stale count and warn once. Keep the fork's `found` cache invalidation.

- [ ] **Step 3: Honour `ZAKURO_NATIVES=0`**

At load, if the variable is `0` and `set_natives(false)` returns true, log `"the library's natives are off, the title's own code runs for them"`.

- [ ] **Step 4: Commit**

```bash
git add engine/core/src/recompiled.rs engine/core/src/lib.rs
git commit -m "engine: natives, origins and interpret for ABI 8"
git push origin abi8-foundation
```

Expected in CI: a synthetic `Origins` table marks a function stale, `lookup` skips it, and `check` clears it.

---

## Task 5: Fixtures and CI gates for both layouts

**Files:**
- Create: `recompiler/abi/tests/fixture7.c`
- Create: `recompiler/abi/tests/fixture8.c`
- Create: `engine/core/tests/legacy.rs`
- Create: `engine/core/tests/modern.rs`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/ci-32bit.yml`

- [ ] **Step 1: Build a legacy fixture in CI**

`fixture7.c` includes the current ABI 7 `recomp.h`, exports `recomp_abi = 7`, `recomp_fastmem = 1`, and one function that writes through `ctx->fastmem` and returns `void`. Build it with `cc -shared -fPIC -o fixture7.so fixture7.c` on the CI runner.

- [ ] **Step 2: Build a modern fixture in CI**

`fixture8.c` includes `recomp8.h`, exports `recomp_abi = 8`, `recomp_generation = 6`, `recomp_fastmem = 1`, a one-function `recomp_entries`, and an `Origins` table with one piece and one function. Build it the same way.

- [ ] **Step 3: Engine tests**

`engine/core/tests/legacy.rs` opens `fixture7.so`, sets `fastmem` to a 1 GiB-masked buffer, runs the function, and asserts the write landed at the masked offset. `modern.rs` does the same through `Context8`, plus a `check`/`interpret` assertion.

- [ ] **Step 4: armv7 generated-C gate**

In `ci-32bit.yml`, add a step that runs the recompiler on a synthetic program and compiles the generated C with `armv7a-linux-androideabi29-clang -c`. This is the gate that catches 32-bit pointer/size assumptions. It can land with a stub program now and use a real game later.

- [ ] **Step 5: Commit**

```bash
git add recompiler/abi/tests engine/core/tests .github/workflows/ci.yml .github/workflows/ci-32bit.yml
git commit -m "ci: fixtures for ABI 7 and ABI 8, and the armv7 generated-C gate"
git push origin abi8-foundation
```

Expected in CI: both fixture tests pass on x86_64 and aarch64; the armv7 job compiles the generated C.

---

## Task 6: Switch the header and the version constants to ABI 8

**Files:**
- Modify: `recompiler/abi/recomp.h` (replace with `recomp8.h`)
- Create: `recompiler/abi/recomp7.h` (the old header, fixtures only)
- Modify: `recompiler/abi/src/lib.rs`
- Modify: `recompiler/src/codegen/mod.rs` (`HEADER` import, emitted `RECOMP_ABI`)

- [ ] **Step 1: Move the headers**

```bash
git mv recompiler/abi/recomp.h recompiler/abi/recomp7.h
git mv recompiler/abi/recomp8.h recompiler/abi/recomp.h
```

Update `fixture7.c` to include `recomp7.h`; the legacy layout test compiles `recomp7.h`.

- [ ] **Step 2: Flip the constants**

In `recompiler/abi/src/lib.rs`:

```rust
pub const ABI: u32 = 8;
pub const OLDEST_ABI: u32 = 5;
pub const GENERATION: u32 = 6;
```

`HEADER` now includes the ABI 8 header. `Library` becomes `Loaded`'s modern variant: rename the old `Library` to `Legacy` and make `Library` the enum from Task 1. Keep the `open`/`linked` dispatchers as the public entry points.

- [ ] **Step 3: Keep the legacy loader reachable**

`Loaded::Legacy` uses `LegacyContext`/`Code7`; the engine's `FoundCode::Legacy` already does. The legacy header is only for fixtures and tests; the engine never compiles legacy code.

- [ ] **Step 4: Commit**

```bash
git add recompiler/abi recompiler/src/codegen/mod.rs
git commit -m "recomp-abi: interface 8 is the one builds are made for"
git push origin abi8-foundation
```

Expected in CI: the layout test compiles the ABI 8 header; legacy fixture tests still pass.

---

## Task 7: Vendor the ABI 8 codegen

**Files:**
- Replace: `recompiler/src/codegen/{mod,arm,thumb,vfp}.rs`
- Add: `recompiler/src/codegen/{locals,hoist}.rs`
- Keep: `recompiler/src/codegen/a64.rs`, `a64_harness.c`

- [ ] **Step 1: Copy upstream's codegen**

```bash
cp /tmp/opencode/up/3dsrecomp-d143f2c/src/codegen/{mod,arm,thumb,vfp,locals,hoist}.rs recompiler/src/codegen/
```

- [ ] **Step 2: Reapply the fork's fastmem emission**

In `codegen/mod.rs`, restore the fork's per-file view local and `RECOMP_VIEW` redefinition, and the `RECOMP_FASTMEM`/`recomp_fastmem` symbol emission. Search the fork's pre-vendor file at `git show HEAD~1:recompiler/src/codegen/mod.rs` for `RECOMP_VIEW` and `recomp_fastmem`; keep those hunks.

- [ ] **Step 3: Reapply the `a64` dispatch**

Upstream's `codegen/mod.rs` has no `a64` flag. Add the fork's dispatch back without disturbing `locals`/`hoist`: `Scope { accesses, plan }` stays, and the `a64` path is chosen before the C path. The a64 assembly must return `u32` and `RECOMP_STOPPED` on host takeover.

- [ ] **Step 4: Adapt `compile.rs` to the new `Program`/`Analysis`/`Piece`**

Keep the fork's jobs/memory-budget pipeline. Upstream's `compile.rs` is 347 lines; the fork's is 210. Merge by keeping the fork's scheduler and taking upstream's `Piece`/`Origin` plumbing.

- [ ] **Step 5: Commit**

```bash
git add recompiler/src/codegen recompiler/src/compile.rs
git commit -m "recompiler: upstream ABI 8 codegen with the fork's fastmem and a64"
git push origin abi8-foundation
```

Expected in CI: codegen unit tests pass; the armv7 generated-C gate compiles the fastmem path.

---

## Task 8: Vendor discovery and hints

**Files:**
- Replace: `recompiler/src/discover.rs`
- Add: `recompiler/hints.txt`
- Modify: `recompiler/src/build.rs`, `recompiler/src/main.rs`

- [ ] **Step 1: Copy upstream's discovery and hints**

```bash
cp /tmp/opencode/up/3dsrecomp-d143f2c/src/discover.rs recompiler/src/discover.rs
cp /tmp/opencode/up/3dsrecomp-d143f2c/hints.txt recompiler/hints.txt
```

- [ ] **Step 2: Keep the fork's `.hints` lookup**

Upstream bundles `hints.txt` via `include_str!`. The fork also reads a per-library `.hints`. Merge both: the bundled hints are the default, the per-library file overrides.

- [ ] **Step 3: Adapt `build.rs`/`main.rs` to `Analysis`/`split_huge`/`OWN_FUNCTION`**

Upstream's `Analysis` feeds `compile.rs` and `build.rs`. Keep the fork's precache/shaders/Mii/update handling; change only the discovery call sites.

- [ ] **Step 4: Commit**

```bash
git add recompiler/src/discover.rs recompiler/src/build.rs recompiler/src/main.rs recompiler/hints.txt
git commit -m "recompiler: upstream discovery, split_huge and bundled hints"
git push origin abi8-foundation
```

Expected in CI: discovery tests pass; a synthetic program produces pieces and `OWN_FUNCTION` subfunctions.

---

## Task 9: Vendor natives and ABI 8 verification

**Files:**
- Add: `recompiler/src/natives.rs`
- Replace: `recompiler/src/verify.rs`
- Modify: `recompiler/src/build.rs`, `recompiler/src/lib.rs`

- [ ] **Step 1: Copy upstream's natives and verify**

```bash
cp /tmp/opencode/up/3dsrecomp-d143f2c/src/natives.rs recompiler/src/natives.rs
cp /tmp/opencode/up/3dsrecomp-d143f2c/src/verify.rs recompiler/src/verify.rs
```

- [ ] **Step 2: Integrate natives with `overrides.rs`**

A handwritten override wins over a native. `build.rs` calls `natives::find`/`natives::source`, writes `natives.c`, and extends `replaced`. Keep the fork's `precache` and `shaders` emission.

- [ ] **Step 3: Keep the fork's verify checks**

Upstream's `verify.rs` tests ABI 8 run-through behaviour. Keep the fork's checks for encrypted dumps, update titles and Mii parts; they live in `rom.rs`/`build.rs` if `verify.rs` no longer owns them.

- [ ] **Step 4: Commit**

```bash
git add recompiler/src/natives.rs recompiler/src/verify.rs recompiler/src/build.rs recompiler/src/lib.rs
git commit -m "recompiler: host natives and ABI 8 verification"
git push origin abi8-foundation
```

Expected in CI: natives tests pass; `ZAKURO_NATIVES=0` is covered by an engine test.

---

## Task 10: Merge ROM, build, main and the fork's extras

**Files:**
- Modify: `recompiler/src/rom.rs`, `build.rs`, `main.rs`, `lib.rs`
- Keep: `recompiler/src/cro.rs`, `image.rs`, `overrides.rs`, `port.rs`, `precache/`, `shaders.rs`

- [ ] **Step 1: Merge upstream's ABI 8 `rom.rs`/`build.rs`/`main.rs`**

Upstream's versions are at `/tmp/opencode/up/3dsrecomp-d143f2c/src/`. Keep the fork's update title id, Mii parts, GBA-VC detection and encrypted-dump handling. Keep the `Mods` plumbing already on `recompiler-update`.

- [ ] **Step 2: Keep the fork's compile pipeline**

`compile.rs` keeps the fork's jobs/memory budget. Adapt the new `Program`/`Analysis`/`Piece` types.

- [ ] **Step 3: Export the new modules**

`lib.rs` exports `natives`; keeps `precache`, `shaders`, `a64`, `Mods`.

- [ ] **Step 4: Commit**

```bash
git add recompiler/src
git commit -m "recompiler: the fork's ROM, update and mods work on the ABI 8 APIs"
git push origin abi8-foundation
```

Expected in CI: the recompiler builds and the synthetic program round-trips.

---

## Task 11: Docs and the final gate

**Files:**
- Modify: `recompiler/docs/interface.md`
- Modify: `recompiler/docs/upstream-port.md`
- Modify: `docs/PLAN.md`

- [ ] **Step 1: Document interface 8**

Take upstream's `docs/interface.md` (591 lines) and add a fork section: the trailing `fastmem`, the 1 GiB mask on 32-bit, the legacy window 5–7, and the fact that upstream ABI 8 libraries run without the trailing field.

- [ ] **Step 2: Mark the deferred work done**

`docs/upstream-port.md` lists the ABI 8 work as waiting. Move each item to "landed" with the commit that did it.

- [ ] **Step 3: Final CI**

Push and confirm `ci.yml` and `ci-32bit.yml` are green, the armv7 APK is uploaded, and the fixture tests pass on both architectures.

- [ ] **Step 4: Commit**

```bash
git add recompiler/docs docs/PLAN.md
git commit -m "docs: interface 8 and the end of the deferred ABI 8 list"
git push origin abi8-foundation
```

---

## Self-review notes

- Spec coverage: header/ABI crate (Tasks 1, 6), codegen (7), discovery/hints (8), natives/verify (9), ROM/build/main/compile (10), engine host dual loading (2), ticks/mutex/trail (3), natives/origins/interpret (4), CI fixtures and armv7 gate (5), docs (11).
- The spec's "new builds emit ABI 8 only" lands at Task 6; until then the recompiler emits ABI 7 and the engine can already run ABI 8 fixtures.
- The spec's "engine does not compile legacy libraries" holds: `recomp7.h` is a fixture header only.
- The spec's `Library` enum is `recomp_abi::Loaded` in this plan; the name is an implementation detail, the two-variant dispatch is the contract.
