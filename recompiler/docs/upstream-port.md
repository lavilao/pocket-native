# upstream port, this branch

3dsrecomp upstream (fearkov) moved on since this fork's recompiler was taken from it
(~Sep 28). Its newer line went to interface version 8; this fork keeps its own
ABI 7 line with the 32-bit work. This branch ports what can cross over without
touching that interface, and maps what has to wait for it.

## what landed here

taken from upstream whole, nothing of ours in them:

- src/cro.rs, src/image.rs — mods' code and modules recompile in place of the
  title's, a mod's exheader says where its code segments sit
- src/overrides.rs, src/port.rs — guess judging before counting, port checks the
  code linked into it against memory
- src/thumb.rs — the decoder's newer fixes

merged (both sides kept):

- src/lib.rs — upstream's Mods plumbing (what a mod swaps in: code, exheader,
  RomFS files), programs/module_files/static_module take it now; this fork's
  pub modules (precache, shaders, a64) stay
- src/rom.rs — this fork's update title id, Mii parts and GBA Virtual Console
  detection kept; upstream's encrypted-dump telling (one decrypted without
  fixing flags still loads, a real encryption says so) and its test fixtures
  added; both test sets run
- src/arm.rs — both sides' changes
- Cargo.toml / abi/Cargo.toml

adapted to the new signatures:

- src/main.rs, src/build.rs call sites pass Mods::default()

## kept as this fork has them, on purpose

- abi/recomp.h, abi/src/lib.rs — the ABI 7 line, untouched
- src/codegen/* — upstream's newer codegen (locals/hoist optimizer, pieces and
  origins, self-answering SVC_TICK/WAIT/RELEASE, uint32_t functions) is written
  against ABI 8's recomp.h contract; emitting it against this fork's header
  would change generated code's shape under the same ABI number
- src/discover.rs — upstream rewrote discovery around its guesses/trials/split
  machinery that feeds the new codegen; this fork's gap-scan line stands on its
  own and its games build with it
- src/compile.rs — this fork's own jobs/memory-budget compiler
- src/verify.rs — upstream's now tests ABI 8 run-through behavior
- docs/interface.md — describes this fork's interface 7; docs/overrides.md and
  docs/port.md stay fork-side too, upstream's newer overrides doc describes
  its ABI 8 override returns

## waiting for the ABI 8 line

- natives.rs — the C library's divisions run on the host (needs ABI 8's
  register-pair reads, budget handling and RECOMP_NATIVE in the header)
- src/codegen/locals.rs, src/codegen/hoist.rs — registers/flags/vfp in locals,
  shared page-table look plans
- split_huge discovery + OWN_FUNCTION subfunctions
- the code answering GetSystemTick and the lent mutex itself
- Library::interpret/interpret_at plumbing

verified here: cargo test --workspace --features zakuro-gpu/gles, everything
green, and cargo check --locked matches CI's build.
