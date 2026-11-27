# ABI 8 foundation — design

**Date:** 2026-11-26
**Branch:** `abi8-foundation`, based on `origin/recompiler-update` (`ee1e9e0`)
**Status:** design approved in conversation; spec awaiting review
**Boundary:** the `3dsrecomp-upstream` branch belongs to someone else and is not
touched by this work.

## Why

`recompiler-update` ported upstream 3dsrecomp's mod-era changes onto the fork's ABI 7
line, but deliberately deferred ABI 8. That leaves the fork without the newer
codegen, discovery, natives, and the recompiled fast paths (self-answered ticks,
lent mutex, clock-poll loop skipping). Several later features depend on ABI 8.

This slice finishes the ABI 8 foundation end-to-end: the recompiler emits ABI 8,
the engine runs it, and existing ABI 5/6/7 libraries keep working.

**Primary target: 32-bit Android (`armeabi-v7a`).** aarch64 Android, x86_64
desktop, and handheld Linux stay buildable and tested, but they are regression
targets only. The design must not assume 64-bit pointers, 64-bit `usize`, or
spare address space.

## Goals

- Adopt upstream 3dsrecomp's ABI 8 recompiler line and reapply the fork's work.
- Keep the fork's `fastmem` 1 GiB view on ABI 8, alongside upstream's page
  tables and spans.
- Keep ABI 5/6/7 libraries running unchanged, including legacy 32-bit fastmem
  libraries.
- Port the engine-side ABI 8 host: ticks, lent mutex, trail, `stack_limit`,
  interpret, natives, origins/stale.
- Validate the 32-bit path in CI without local builds.

## Non-goals (follow-on slices)

- Runtime mods loader (the compile-side Mods API is already here; ABI 8
  origins/stale plumbing lands, the mods folder loader does not).
- Vulkan renderer.
- Save states, cheats/60 FPS, cameras, microphone, Download Play, 3D modes,
  widescreen, idle/spin skipping, launch options.
- Desktop GUI or packaging work.
- Performance tuning beyond keeping the ABI 7 baseline and validating on device.

## Decisions

| Decision | Choice |
|---|---|
| Port strategy | Vendor upstream ABI 8, then reapply fork extensions |
| Slice scope | Recompiler and engine host together, end-to-end |
| Old libraries | ABI 5/6/7 keep running on a legacy Context layout |
| Fastmem on ABI 8 | Keep both: fastmem for `RECOMP_FASTMEM`, page tables/spans otherwise |
| New builds | Emit ABI 8 only |
| Upstream source | 3dsrecomp `main` at `d143f2c` (Nov 24) |
| Verification | GitHub Actions only; no local builds |

## Architecture

### ABI version window

- `RECOMP_ABI = 8`
- `OLDEST_ABI = 5` (the fork never supported 4)
- `GENERATION = 6` (adopt upstream's generation counter)
- `recomp_abi::open`/`linked` accept `5..=8` and return a `Loaded` enum.

### Two Context layouts, one shared prefix

`Context8` is upstream's ABI 8 Context through `trail`, plus a fork field at the
end:

```c
/* upstream ABI 8 fields ... */
uint32_t done;
uint32_t trail[RECOMP_TRAIL];
/* fork extension */
uint8_t *fastmem;
```

`LegacyContext` is the fork's current ABI 7 layout through `user`, plus
`fastmem` where it is today:

```c
/* registers, flags, budget/exit/svc, page tables, VFP, host, user */
uint8_t *fastmem;
```

Everything through `user` is identical in both layouts. That common prefix is
what Host callbacks need. The layouts diverge only after `user`:

| Field/feature | Legacy 5–7 | Modern 8 |
|---|---|---|
| `fastmem` | after `user` | after `trail` |
| `stack_limit` | absent | before every call |
| `tick_end`, `tick_last`, `tick_floor`, `tick_cost`, `tick_at`, `ticks` | absent | host maintains |
| `wait_at`, `release_at`, `held`, `holding`, `held_floor` | absent | host maintains |
| `done`, `trail` | absent | host maintains |
| `Origins`/`stale` | absent | loaded and checked |
| `RECOMP_NATIVE` | absent | supported |
| `Code` return | `void` | `u32` |

Upstream-built ABI 8 libraries also run: they never read past `trail`, so the
trailing fork `fastmem` is invisible to them.

### Code and Host types

```rust
pub type Code8 = unsafe extern "C" fn(*mut Context8) -> u32;
pub type Code7 = unsafe extern "C" fn(*mut LegacyContext);
```

Two Host vtables, `HOST8` and `HOST7`, have the same field order but different
`lookup`/`Code` types. Callbacks share helpers that cast to the common prefix;
only ABI-specific work branches on the loaded library.

The engine's `recompiled::Library` becomes an enum over `Modern` and `Legacy`.
The legacy wrapper ignores the missing return value and keeps using `r[15]`.

### Fastmem

- Modern code compiled with `RECOMP_FASTMEM` reads `ctx->fastmem` from the
  trailing field.
- Legacy code reads the old offset after `user`.
- The engine sets the correct field for the loaded ABI.
- Non-fastmem modern code uses upstream's page tables and spans.
- The 1 GiB view remains the fork's 32-bit performance path.

### Headers and docs

- `recompiler/abi/recomp.h` becomes the ABI 8 header with the trailing
  `fastmem` field and the existing `RECOMP_FASTMEM` macros.
- A legacy header (ABI 7) is kept for fixtures/tests only; the engine does not
  compile legacy libraries.
- `recompiler/docs/interface.md` becomes the ABI 8 document plus a fork section
  covering fastmem and the legacy compatibility window.

## Recompiler port

### File dispositions

| Upstream | Fork destination | Action |
|---|---|---|
| `abi/recomp.h` | `recompiler/abi/recomp.h` | Import ABI 8, append `fastmem`, keep `RECOMP_FASTMEM` |
| `abi/src/lib.rs` | `recompiler/abi/src/lib.rs` | Import ABI 8; add `LegacyContext`, `Host7`, `Code7`, `Loaded` |
| `src/codegen/{mod,arm,thumb,vfp,locals,hoist}.rs` | same | Replace with ABI 8; re-add `a64` dispatch and fastmem emission |
| `src/discover.rs` | same | Replace with upstream guesses/trials/`split_huge` |
| `src/natives.rs` | same | Add; integrate with `overrides.rs` (handwritten override wins) |
| `src/verify.rs` | same | Replace with ABI 8 checks; keep fork checks |
| `src/arm.rs`, `src/thumb.rs` | same | Merge upstream ABI 8 changes |
| `src/rom.rs`, `src/build.rs`, `src/main.rs` | same | Merge upstream ABI 8 with fork update/Mii/GBA-VC/Mods/hints |
| `src/compile.rs` | same | Keep fork jobs/memory-budget pipeline; adapt to new `Program`/`Analysis`/`Piece` |
| `src/lib.rs` | same | Export `natives`; keep `precache`, `shaders`, `a64`, `Mods` |
| `src/codegen/a64.rs`, `a64_harness.c` | same | Keep fork-only; update to ABI 8 Context/return convention |
| `src/precache/*`, `src/shaders.rs` | same | Keep; adapt only where APIs moved |
| `hints.txt` | `recompiler/hints.txt` | Add upstream bundled hints; merge with fork `.hints` lookup |
| `docs/interface.md` | `recompiler/docs/interface.md` | Replace with ABI 8; add fork fastmem/legacy section |

### What the fork keeps

- `fastmem` in the ABI 8 Context and generated code.
- `RECOMP_A64=1` assembly path and harness. It is a 64-bit path; it is not part
  of 32-bit validation.
- Shader precache (`precache/`, `shaders.rs`).
- Update title id, Mii parts, GBA-VC detection, encrypted-dump handling.
- The fork's `compile.rs` job/memory-budget compiler.
- Mods plumbing already on `recompiler-update`.

### Integration points

- `codegen/mod.rs` currently takes an `a64` flag; upstream's ABI 8 version does
  not. Re-add the dispatch without disturbing `locals`/`hoist`.
- `discover.rs`'s new `Analysis` shape feeds `compile.rs` and `build.rs`; adapt
  both.
- `natives.rs` needs `image::Segment`/`Text` APIs from the ported `image.rs`;
  verify the mod-era `image.rs` still exposes them.
- `verify.rs` ABI 8 expects `Origins`/`stale`; `port.rs` and `overrides.rs` must
  agree on ownership.
- `hints.txt` is bundled via `include_str!`; the fork's per-library `.hints`
  lookup stays.

### Commit sequence

1. ABI crate + header: ABI 8, legacy layout, fastmem, version constants.
2. Codegen: upstream ABI 8 + fastmem + a64 dispatch.
3. Discovery + hints: upstream analysis, `split_huge`, bundled hints.
4. Natives + verify: host-native divisions, ABI 8 verification.
5. ROM/build/main/compile merge: fork features on the new APIs.
6. Docs.

Each commit adapts its call sites so CI stays green. No local builds.

## Engine host

`engine/core/src/recompiled.rs`:

- **Dual library loading:** `Library::{Modern, Legacy}` selected from
  `recomp_abi`. Modern uses `Context8`/`Code8`; legacy uses
  `LegacyContext`/`Code7`.
- **Fastmem:** modern field after `trail`; legacy field after `user`; both point
  at the 1 GiB view on 32-bit. Page-table and fastmem paths coexist.
- **`stack_limit`:** `usize` (4 bytes on armv7), set before every modern call to
  `stack_here - STACK`. Generated code unwinds past it.
- **Ticks:** set `tick_end`, `tick_floor`, `tick_cost`, `tick_at`; read back
  `ticks`, `tick_last`, and `trail`. Tick arithmetic stays `u64`; only the
  budget is `i32`.
- **Lent mutex:** set `wait_at`, `release_at`, `held`, `holding`, `held_floor`;
  reconcile `holding` with the kernel's mutex state when the code hands back.
- **Natives:** expose `recomp_natives`, add `set_natives(false)` and the
  `ZAKURO_NATIVES=0` path. The cost model is tested on armv7 before it is
  trusted by default.
- **Origins/stale:** load `Origins`, compare code hashes, mark stale functions
  for overrides (and later mods).
- **Interpret/lookup:** modern `lookup` returns `Code8`; legacy returns `Code7`.
  `interpret` is shared. `Library::interpret`/`interpret_at` is exposed for the
  engine to run chosen functions.
- **Step loop:** handle the ABI 8 return value and
  `Stop::{Svc, Left, Declined, Exit}`; sync tick/mutex/trail state before and
  after each call.
- **Fork memory model stays:** 1 GiB fastmem view, page tables, 32-bit
  address-space handling.

## Validation and CI

No local builds. Every stage is pushed and must pass CI before the next.

### `ci.yml`

- Existing workspace build/test with `zakuro-gpu/gles`.
- New ABI 8 tests:
  - C/Rust layout sync for `Context8` and `LegacyContext`.
  - Codegen: `locals`, `hoist`, spans, `RECOMP_FASTMEM`.
  - Discovery: guesses/trials, `split_huge`, `OWN_FUNCTION`.
  - Natives: pattern finding, override generation, cost model.
  - Verify: ABI 8 checks, origins/stale.
  - Engine host: synthetic contexts for ticks, lent mutex, trail, interpret,
    natives, origins/stale.

### `ci-32bit.yml` (gate)

- armv7 Android build of `pocket-native-android`.
- New step: generate C from a synthetic program with the recompiler and compile
  it with the NDK's `armv7a-linux-androideabi29-clang`. This catches 32-bit
  pointer/size assumptions in generated code.
- Upload the armv7 APK for device testing.

### Legacy fixture

- A small ABI 7 library built in CI from the legacy header.
- An engine test loads it and runs a function, proving the legacy `fastmem`
  offset and `void` return path.
- The fixture is also compiled for armv7 to catch layout drift.

### 64-bit and x86

- aarch64 Android and x86_64 desktop builds/tests stay green.
- They are not performance gates for this slice.

### Device check

- The armv7 APK is installed on the target device.
- Smoke test: a recompiled game boots, runs recompiled code, uses fastmem, and
  does not fall back to the interpreter beyond the expected gaps.
- Compare against the ABI 7 baseline; record any regression.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| 32-bit address space: fastmem view + guest memory + C stack | Verify the memory map on armv7; keep `STACK` at 1 MiB and measure |
| Upstream codegen assumes 64-bit sizes | NDK armv7 compile of generated C in CI |
| `Origins` hash checks and natives cost too much on low-end ARM | Measure on device; keep `ZAKURO_NATIVES=0` escape hatch |
| Self-answered ticks/mutex change timing | Device smoke test on R36S-class hardware |
| Legacy ABI 7 fastmem context receives ABI 8 fields | Separate layouts + legacy fixture test |
| Large vendor diff obscures fork changes | Staged commits; fork extensions listed above |
| Upstream moves during the port | Pin to `d143f2c`; rebase later |

## Success criteria

- New builds emit ABI 8 and run on armv7 Android.
- Existing ABI 5/6/7 libraries, including 32-bit fastmem libraries, still run.
- ABI 8 fastmem works alongside page tables/spans.
- Ticks, lent mutex, trail, `stack_limit`, interpret, natives, and origins/stale
  are exercised by tests and CI.
- `ci.yml` and `ci-32bit.yml` are green; the armv7 APK is produced.
- Device smoke test passes with no regression against the ABI 7 baseline.

## References

- Fork branch: `origin/recompiler-update`, `ee1e9e0`
- Deferred ABI 8 map: `recompiler/docs/upstream-port.md`
- Upstream 3dsrecomp: `fearkov/3dsrecomp` `main` at `d143f2c`
- Upstream Zakuro ABI 8 host: `fearkov/zakuro` `main` at `14e5786`
- ABI 8 interface: upstream `docs/interface.md`
