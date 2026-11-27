# the upstream

zakuro (fearkov/zakuro) is where the engine comes from, and this fork's one
job is to keep it running on a 32-bit android phone.

## what this fork is on, and what it is not

the engine here is zakuro's `engine/` line, merged up to v0.6.1. zakuro's
master has since moved to a `crates/` line, and it is on that line — not
this one — that these live:

- the Vulkan renderer, `crates/zakuro-gpu/src/raster/hardware.rs`: it draws
  the 3D on Vulkan itself and asks 1.3 of a driver (`VK_KHR_push_descriptor`,
  dynamic rendering, synchronization2)
- the Vulkan presenter, `crates/zakuro-gpu/src/backend/vulkan.rs`, and the
  `backend`, `device` and `instance` split under it
- what came with them: frame interpolation, texture copies and depth maps on
  the GOU, and the `Pass`/twins machinery in `raster.rs`

so none of that is in this tree, whatever a commit message claims. the
Vulkan work here is the fork's own, in `engine/gpu/src/vulkan/`: a 1.1
device, and a presenter that puts the software rasterizer's screens and the
menu over them on the window through a 1.1 surface and swapchain. the menu's
`Show screens with` setting picks it, and then GLES does not start at all: the
3D is drawn on the CPU and Vulkan shows the pictures. upstream's own renderer,
the one that draws the 3D on the GPU, is still the gap above.

## how the two lines meet

- this fork's master takes upstream's `engine/` line whole when zakuro tags
  one; merged so far: v0.6.0, v0.6.1
- two files in recompiler/ stay fork-side on purpose, everything else
  in there is zakuro's as it stands:
  - abi/recomp.h — the fastmem view on a 32-bit host can only hold
    1 GiB of the guest's 4, so the address is masked into it; a 64-bit
    host holds the whole 4 GiB as it is
  - src/rom.rs — a .cia of an update names its build after the
    update's title id, which the running system looks it up by
- the app is built 32-bit only on a push, in its own workflow,
  ci-32bit.yml, and its artifact is the APK
  (pocket-native-32bit-apk); ci.yml, which builds the workspace, tests it
  and cross-builds a 64-bit library, runs for a pull request or by hand

## why the recompiler does not follow 3dsrecomp anymore

fearkov's 3dsrecomp went on to interface 8 after the recompiler in
here was taken from it, and taking its newer line meant porting the
engine by hand onto this fork's ABI 7 while keeping the 32-bit work.
zakuro's own line carries the recompiler that runs the games zakuro
runs — its line has been taken whole here instead, and the interface
8 work waits until zakuro itself takes it, so it comes in from the
upstream the same way everything else does.

## when a game runs on zakuro but not here

the app on the phone recompiles each game itself, and it keeps the
build it made: a game that misbehaves after an update may be running
code the old app recompiled. delete the game's .so in files/code/
(or clear the app's data) and let the new one recompile it before
blaming the engine; the version in the log line "Pocket Native x.y.z"
tells which line the app on the phone was built from.
