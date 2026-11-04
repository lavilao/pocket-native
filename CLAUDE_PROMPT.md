# Pocket Native on Redmi 9 (Armv7, PowerVR) — two bugs to diagnose

You are helping debug a 3DS recompiler/emulator ("Pocket Native", fork of fearkov/zakuro) running as an Android app on a **Redmi 9** (dandelion, Android 11/API 30, **32-bit only**, 3.9 GiB RAM, GPU **PowerVR Rogue GE8320**, OpenGL ES 3.2, MTK driver 1.13).

## What happens

1. **The game runs at ~4 fps** because the recompiled native library is rejected ("interpreted everything" path). Root cause: the CPU *fastmem* "view" (a large mmap reservation used by the recompiled code for memory access) fails to be created on this 32-bit device, so the loader falls back to the interpreter.
2. **The process then crashes** (SIGSEGV) inside the **PowerVR shader compiler**: backtrace is `libusc.so` → `PVRUniFlexCompileToHw` (the driver's hardware compiler) → `libglslcompiler.so` (`GLSLCompileToUniflex`) → `libGLESv2_mtk.so`, thread name `gles`. Fault address `0x3f80025c` (a float value being dereferenced as a pointer - looks like a UniFlex compiler bug). Previously (before the latest shader change) the same shader **failed to compile cleanly** with "Calls to any function that may require a gradient calculation inside a conditional block may return undefined results" and the driver returned an error (game showed black 3D); now the compiler **segfaults** instead.

A previous change (moving all texture() calls out of conditionals, so implicit derivatives stay in uniform flow) fixed the "compile error" case but now the UniFlex compiler crashes. The shader is a single huge 758-line fragment shader.

No NDK source, no ability to read the app's log file from here - what I have is the adb crash buffer and the code.

## Bug 1 - the fastmem view fails on 32-bit Android

Relevant code:

### engine/core/src/memory/fastmem.rs (head + View::new)
```rust
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
```

### engine/core/src/memory/physical.rs (Shared::new)
```rust
    fn new(len: usize) -> Option<Shared> {
        // SAFETY: plain system calls, every result checked
        unsafe {
            let fd = libc::memfd_create(c"zakuro-physical".as_ptr(), libc::MFD_CLOEXEC);
            if fd < 0 {
                log::warn!("physical memory: memfd_create failed: {}", std::io::Error::last_os_error());
                return None;
            }
            if libc::ftruncate(fd, len as libc::off_t) != 0 {
                log::warn!("physical memory: ftruncate({} bytes) failed: {}", len, std::io::Error::last_os_error());
                libc::close(fd);
                return None;
            }
            let base = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0);
            if base == libc::MAP_FAILED {
                log::warn!("physical memory: mmap({} bytes, shared) failed: {}", len, std::io::Error::last_os_error());
                libc::close(fd);
                return None;
            }
            Some(Shared { fd, base: base as *mut u8, len })
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn new(_len: usize) -> Option<Shared> {
        None
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        // SAFETY: the mapping and descriptor Shared::new made
        unsafe {
            libc::munmap(self.base as *mut libc::c_void, self.len);
            libc::close(self.fd);
```

### engine/core/src/memory/mod.rs (Memory::new + fastmem())
```rust
impl Memory {
    pub fn new(new3ds: bool, app_bytes: u32) -> Memory {
        // the view first, while the process still has its biggest clear
        // stretch of address space: on a 32-bit host the 1 GiB reservation
        // is the hardest mapping to place, harder than the shared file's
        // few hundred MiB, and this runs before EGL and the game's threads
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let view = fastmem::View::new();
        Memory {
            phys: PhysicalMemory::new(new3ds, app_bytes),
            read_table: vec![std::ptr::null_mut(); PAGE_TABLE_ENTRIES],
            write_table: vec![std::ptr::null_mut(); PAGE_TABLE_ENTRIES],
            mappings: BTreeMap::new(),
            faults: BTreeMap::new(),
            new_fault: None,
            #[cfg(any(target_os = "linux", target_os = "android"))]
            view,
            linear_base: 0,
        }
    }

    pub fn with_linear_base(mut self, linear_base: VAddr) -> Memory {
        self.linear_base = linear_base;
        self
    }

    /// whether the GPU can read an address, the linear heap and VRAM are
    /// all it is given.
    fn gpu_reads(&self, vaddr: VAddr) -> bool {
        vaddr.wrapping_sub(self.linear_base) < FCRAM_SIZE_NEW3DS || (VRAM_VADDR..VRAM_VADDR + VRAM_SIZE).contains(&vaddr)
    }

    /// tells the GPU's journal physical memory was written, a service or
    /// the kernel writing what the GPU may read.
    fn note_physical(&self, paddr: u32, len: u32) {
        if paddr >= FCRAM_PADDR {
            zakuro_gpu::journal::note(self.linear_base.wrapping_add(paddr - FCRAM_PADDR), len);
        } else if (VRAM_PADDR..VRAM_PADDR + VRAM_SIZE).contains(&paddr) {
            zakuro_gpu::journal::note(VRAM_VADDR + (paddr - VRAM_PADDR), len);
        }
    }

    /// where the guest's address 0 is in the host's view of all of it, null
    /// without one.
    pub fn fastmem(&self) -> *mut u8 {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if let (Some(view), true) = (&self.view, self.phys.backing(FCRAM_PADDR, PAGE_SIZE).is_some()) {
            return view.base();
```

### engine/core/src/loader.rs (rejection)
```rust
    if let Some(linked) = &system.config.linked {
        if linked.program_id() != title.program_id() {
            log::warn!(
                "the code linked in is for title {:016X}, not {:016X}, interpreting everything",
                linked.program_id(),
                title.program_id()
            );
        } else {
            match crate::recompiled::Library::linked(linked) {
                Ok(library) => {
                    log::info!("running the recompiled code linked in, {}", library.describe());
                    system.recompiled = Some(library);
                }
                Err(error) => log::warn!("could not use the code linked in, {error}, interpreting everything"),
            }
        }
    } else if let Some(path) = system.config.recompiled.clone().or_else(|| installed(&system, &title)) {
        let path = if path.is_dir() { path.join(recomp_abi::library_name(title.program_id())) } else { path };
        let opened = crate::recompiled::Library::open(&path).and_then(|library| {
            if library.needs_fastmem() && system.memory.fastmem().is_null() {
                Err("it was built for fastmem, which this host has no view for".to_owned())
            } else {
                Ok(library)
            }
        });
        match opened {
            Ok(library) => {
                log::info!("running recompiled code from {}, {}", path.display(), library.describe());
                system.recompiled = Some(library);
                let text = exheader_text(&title);
                system.hints = Some(crate::hints::Hints::new(&path, text));
            }
            Err(error) => log::warn!("could not load {}, {error}, interpreting everything", path.display()),
        }
    }

    if system.config.hardware_renderer && system.config.software_draws {
        match system.gpu.enable_panel() {
            Ok(()) => log::info!("switch: drawing in software, the screens shown through OpenGL ES"),
            Err(error) => log::warn!("{error}, drawing in software with no panel"),
        }
```

`Memory::fastmem()` returns `view.base()` only if **both** `view` is `Some` **and** `phys.backing(FCRAM_PADDR, PAGE_SIZE).is_some()` (i.e. the shared memfd backing exists). Two independent points of failure. Hypotheses to evaluate:
- Does Android 11's app seccomp block `memfd_create` with SIGSYS (silent death, nothing logs) — an earlier similar case: `ioprio_set` (syscall 314) killed the process; guarded now with `cfg(not(target_os = "android"))`.
- Is the 1 GiB `MAP_NORESERVE` reservation failing because of fragmentation: by the time `Memory::new()` runs in the game thread, ART's heap, EGL, and the app's other mappings have fragmented the 3 GiB user space so no contiguous 1 GiB hole exists?
- Is `Shared::new()` failing: ftruncate of ~300+ MiB on a device with 3.9 GiB RAM but possibly a quota/vm limit, or `mmap(MAP_SHARED)` failing for the same fragmentation reason?
- Ordering: `View::new()` runs before `PhysicalMemory::new()` — so the view wins the biggest hole. Maybe the shared file should be created first, or the view should retry with `MAP_NORESERVE` + smaller span after failure, or the view should reserve at process start (in `android_main` before EGL/ART bulk allocations).

What's the smallest, most general fix that doesn't regress 64-bit Linux/desktop? e.g. (a) memfd -> ashmem fallback, (b) reserve view+shared in android_main before window/EGL/audio, (c) retry with smaller span, (d) make the shared file pageable lazily?

## Bug 2 - PowerVR UniFlex segfaults compiling the fragment shader

### engine/gpu/src/raster/gles.rs - fragment_source
```rust
fn fragment_source(constants: &[u32; FRAGMENT_CONSTANTS], writes_depth: bool, lighting: &[u32]) -> String {
    let mut out = String::from("#version 310 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n");
    if writes_depth {
        out.push_str("#define WRITES_DEPTH\n");
    }
    let mut skipping = false;
    for line in FRAGMENT_SOURCE.lines().skip(1) {
        if !lighting.is_empty() && line.starts_with("void shade(") {
            out += &lighting_source(lighting);
            skipping = true;
            continue;
        }
        if skipping {
            if line.starts_with("precise vec3 color_operand") {
                skipping = false;
            } else {
                continue;
            }
        }
        if let Some(rest) = line.strip_prefix("layout(constant_id = ") {
            let (id, rest) = rest.split_once(')').unwrap_or_default();
            let name = rest.trim().trim_start_matches("const uint ").split(' ').next().unwrap_or_default();
            out += &format!("const uint {name} = {}u;\n", constants[id.parse::<usize>().unwrap_or(0)]);
        } else if line.contains("noperspective in float in_depth") {
            out += "layout(location = 3) in float in_depth_w;\nlayout(location = 6) in float in_w;\n#define in_depth (in_depth_w / in_w)\n";
        } else {
            out += &line.replace("set = 0, ", "").replace("precise ", "");
            out.push('\n');
        }
    }
    if crate::fast_with(crate::speedup::HALF) {
        // in fast mode the colors at half precision, which the GPU works
        // twice as fast as full, what places texels, depth and lights kept
        // full
        for (from, to) in [
            ("precision highp float;\nprecision highp int;\nprecision highp sampler2D;", "precision mediump float;\nprecision highp int;\nprecision mediump sampler2D;"),
            ("in vec4 in_texcoords01", "in highp vec4 in_texcoords01"),
            ("in vec2 in_texcoord2", "in highp vec2 in_texcoord2"),
            ("in vec3 in_view", "in highp vec3 in_view"),
            ("in float in_depth_w", "in highp float in_depth_w"),
            ("in float in_w", "in highp float in_w"),
            ("sample_unit(uint unit, vec2 uv, vec4", "sample_unit(uint unit, highp vec2 uv, vec4"),
            ("    vec2 st = ", "    highp vec2 st = "),
            ("    float depth = in_depth", "    highp float depth = in_depth"),
            ("    vec3 view = in_view", "    highp vec3 view = in_view"),
            ("vec3 offset = -view", "highp vec3 offset = -view"),
        ] {
            out = out.replace(from, to);
        }
    }
    if ADRENO.load(std::sync::atomic::Ordering::Relaxed) {
        out = out.replace(SWITCH_COMPARE, ADRENO_COMPARE);
    }
    out
}


```

### engine/gpu/shaders/raster.frag (full, 758 lines, #version 450 desktop / #version 310 es on GLES)
```glsl
#version 450

// the PICA200's fragment stages for one draw, the texture units, fragment
// lighting and the six combiners, then the alpha test. they follow the
// software rasterizer, which follows the hardware.

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec4 in_texcoords01;
layout(location = 2) in vec2 in_texcoord2;
layout(location = 3) noperspective in float in_depth;
layout(location = 4) in vec4 in_quaternion;
layout(location = 5) in vec3 in_view;

layout(location = 0) out vec4 out_color;

// what decides the shape of the fragment stages comes in as specialization
// constants, a pipeline for each combination, so the driver compiles away
// the combiners' switches and the stages a draw does not use. per combiner
// stage the source, operand, combiner and scale registers, then the buffer
// update, the texture units' configuration, whether lighting is on, the
// alpha test's switch and function, and the depth mode
layout(constant_id = 0) const uint SOURCE0 = 0u;
layout(constant_id = 1) const uint SOURCE1 = 0u;
layout(constant_id = 2) const uint SOURCE2 = 0u;
layout(constant_id = 3) const uint SOURCE3 = 0u;
layout(constant_id = 4) const uint SOURCE4 = 0u;
layout(constant_id = 5) const uint SOURCE5 = 0u;
layout(constant_id = 6) const uint OPERAND0 = 0u;
layout(constant_id = 7) const uint OPERAND1 = 0u;
layout(constant_id = 8) const uint OPERAND2 = 0u;
layout(constant_id = 9) const uint OPERAND3 = 0u;
layout(constant_id = 10) const uint OPERAND4 = 0u;
layout(constant_id = 11) const uint OPERAND5 = 0u;
layout(constant_id = 12) const uint COMBINER0 = 0u;
layout(constant_id = 13) const uint COMBINER1 = 0u;
layout(constant_id = 14) const uint COMBINER2 = 0u;
layout(constant_id = 15) const uint COMBINER3 = 0u;
layout(constant_id = 16) const uint COMBINER4 = 0u;
layout(constant_id = 17) const uint COMBINER5 = 0u;
layout(constant_id = 18) const uint SCALE0 = 0u;
layout(constant_id = 19) const uint SCALE1 = 0u;
layout(constant_id = 20) const uint SCALE2 = 0u;
layout(constant_id = 21) const uint SCALE3 = 0u;
layout(constant_id = 22) const uint SCALE4 = 0u;
layout(constant_id = 23) const uint SCALE5 = 0u;
layout(constant_id = 24) const uint UPDATE = 0u;
layout(constant_id = 25) const uint TEXTURE_CONFIG = 0u;
layout(constant_id = 26) const uint LIGHTING = 0u;
layout(constant_id = 27) const uint ALPHA_TEST = 0u;
layout(constant_id = 28) const uint DEPTH_MODE = 0u;
// set, a generic pipeline that reads the above from the draw's registers
// instead, to draw with while the one for its combination compiles. it
// keeps lighting and the procedural texture's switch, a lot of code it goes
// without when they are off. it masks the rest as hardware.rs masks the
// constants, so the two draw the same
layout(constant_id = 29) const uint DYNAMIC = 0u;

layout(set = 0, binding = 0) uniform sampler2D texture0;
layout(set = 0, binding = 1) uniform sampler2D texture1;
layout(set = 0, binding = 2) uniform sampler2D texture2;
// ETC1A4's alpha, which the GPU's ETC1 leaves out
layout(set = 0, binding = 3) uniform sampler2D alpha0;
layout(set = 0, binding = 4) uniform sampler2D alpha1;
layout(set = 0, binding = 5) uniform sampler2D alpha2;

struct Light {
    vec4 specular0;
    vec4 specular1;
    vec4 diffuse;
    vec4 ambient;
    vec4 position;
    vec4 direction;
    vec4 spot;
    // bias and scale into the distance table
    vec4 distance;
    // x, 1 directional, 2 two sided, 4 and 8 geometric factors, 16 distance
    // attenuation, 32 spotlight, 64 shadowed, y, which light it is
    uvec4 flags;
};

layout(std140, set = 0, binding = 3) uniform Draw {
    // per stage the source, operand, combiner, constant and scale registers
    uvec4 tev[12];
    // the buffer update and buffer color registers, the alpha test and the
    // texture unit configuration
    uvec4 misc;
    // per unit the configuration word and the border color
    uvec4 units[3];
    // x, 1 a w-buffer, 2 depth from the fragment's z, y, lighting is on,
    // z and w, the depth map's scale and offset
    uvec4 flags;
    // x, the configuration, y, how many lights, z, bump mapping, w, shadow
    uvec4 light_config;
    // x, fresnel into the primary alpha, y, into the secondary, z, clamp
    // highlights, w, 1 when the half vector is read, 2 the view vector
    uvec4 light_flags;
    // distribution 0 and 1, fresnel, reflection red, green and blue and the
    // spotlight, each whether it is read, its input, abs and scale
    uvec4 lookups[7];
    vec4 global_ambient;
    Light lights[8];
    // the procedural texture's registers, the configuration, the noise's
    // u and v and its frequencies, then its table's configuration and
    // offset
    uvec4 proctex[2];
    // the six combiner stages' constant colors and the buffer color, then in
    // x the alpha test's reference, as floats
    vec4 colors[7];
    vec4 reference;
    // x, the fog's color
    uvec4 fog;
};

// a combiner stage's source, operand, combiner and scale registers, the
// pipeline's constants or, generic, the draw's
uvec4 stage_registers(uint stage, uvec4 constants) {
    if (DYNAMIC == 0u) {
        return constants;
    }
    uvec4 registers = tev[stage * 2u];
    return uvec4(registers.x & 0x0FFF0FFFu, registers.y & 0x00777FFFu, registers.z & 0x000F000Fu, tev[stage * 2u + 1u].x & 0x00030003u);
}

layout(std430, set = 0, binding = 4) readonly buffer Tables {
    // 24 tables of 256 entries, a value and the step to the next, then the
    // procedural texture's noise, color map and alpha map, 128 entries
    // each, and its 256 colors and their steps, two pairs an entry, then
    // the fog's 128, a value and a step
    vec2 tables[];
};

const uint PROCTEX_MAPS = 24u * 256u;
const uint PROCTEX_COLORS = PROCTEX_MAPS + 3u * 128u;
const uint PROCTEX_STEPS = PROCTEX_COLORS + 2u * 256u;
const uint FOG_TABLE = PROCTEX_STEPS + 2u * 256u;

const uint DISTRIBUTION0 = 0u;
const uint DISTRIBUTION1 = 1u;
const uint FRESNEL = 3u;
const uint REFLECT_BLUE = 4u;
const uint REFLECT_GREEN = 5u;
const uint REFLECT_RED = 6u;
const uint SPOTLIGHT = 8u;
const uint DISTANCE = 16u;

vec4 unpack_color(uint value) {
    return unpackUnorm4x8(value);
}

// a texture unit's sample, with v = 0 at the bottom row as the PICA has it
// and the border color outside the texture where the unit clamps to it. a
// unit reading a drawn buffer in place has its rows the other way up. the
// texture configuration constant tells, bit 16 on for each unit in place
// and bit 20 for each that never clamps to the border, the generic pipeline
// reads both from the unit
vec2 sample_st(uint unit, vec2 uv) {
    uint config = units[unit].x;
    bool in_place = DYNAMIC != 0u ? units[unit].z != 0u : (TEXTURE_CONFIG & (0x10000u << unit)) != 0u;
    // precise as the combiners are. the coordinate itself the driver may
    // still interpolate a hair apart from one shader to another
    vec2 st = vec2(uv.x, in_place ? uv.y : 1.0 - uv.y);
    return st;
}

// a unit's sample from the values main already fetched. selection only, no
// texture call: texture() needs implicit derivatives, which are undefined
// outside uniform flow, and some drivers (PowerVR) refuse to compile it
// under a branch at all, so every fetch happens unconditionally in main.
vec4 sample_unit(uint unit, vec2 uv, vec4 t0, vec4 t1, vec4 t2, vec4 a0, vec4 a1, vec4 a2) {
    uint config = units[unit].x;
    uint wrap_t = (config >> 8) & 7u;
    uint wrap_s = (config >> 12) & 7u;
    vec2 st = sample_st(unit, uv);
    if (DYNAMIC != 0u || (TEXTURE_CONFIG & (0x100000u << unit)) == 0u) {
        bool border_s = (wrap_s & 3u) == 1u && (st.x < 0.0 || st.x >= 1.0);
        bool border_t = (wrap_t & 3u) == 1u && (st.y < 0.0 || st.y >= 1.0);
        if (border_s || border_t) {
            return unpack_color(units[unit].y);
        }
    }
    // bit 24 on for each unit with its alpha apart
    if (DYNAMIC != 0u ? units[unit].w != 0u : (TEXTURE_CONFIG & (0x1000000u << unit)) != 0u) {
        return unit == 0u ? a0 : unit == 1u ? a1 : a2;
    }
    return unit == 0u ? t0 : unit == 1u ? t1 : t2;
}

precise float lookup(uint table, uint entry, float delta) {
    vec2 value = tables[table * 256u + entry];
    return value.x + value.y * delta;
}

// a procedural texture map read at a coordinate from 0 to 1
precise float proctex_lookup(uint map, float coordinate) {
    float at = coordinate * 128.0;
    uint entry = min(uint(max(at, 0.0)), 127u);
    vec2 value = tables[PROCTEX_MAPS + map * 128u + entry];
    return value.x + (at - float(entry)) * value.y;
}

vec4 proctex_color(uint entry, uint base) {
    return vec4(tables[base + entry * 2u], tables[base + entry * 2u + 1u]);
}

// a pseudo-random value from -1 to 1 for a point of the noise's grid
float proctex_random(uint x, uint y) {
    const uint rows[16] = uint[16](0u, 4u, 10u, 8u, 4u, 9u, 7u, 12u, 5u, 15u, 13u, 14u, 11u, 15u, 2u, 11u);
    const uint mixes[16] = uint[16](10u, 2u, 15u, 8u, 0u, 7u, 4u, 5u, 5u, 13u, 2u, 6u, 13u, 9u, 3u, 14u);
    uint u = (((x % 9u + 2u) * 3u) & 0xFu) ^ rows[(x / 9u) & 0xFu];
    uint v = (((y % 9u + 2u) * 3u) & 0xFu) ^ rows[(y / 9u) & 0xFu];
    if ((u & 3u) == 1u) {
        v += 4u;
    }
    v ^= (u & 1u) * 6u;
    v += 10u + u;
    v &= 0xFu;
    v ^= mixes[u];
    return -1.0 + float(v) * 2.0 / 15.0;
}

float proctex_shift(float other, uint mode, uint clamping) {
    float amount = clamping == 3u ? 1.0 : 0.5;
    int o = int(other);
    if (mode == 1u) {
        return amount * float((o / 2) % 2);
    } else if (mode == 2u) {
        return amount * float(((o + 1) / 2) % 2);
    }
    return 0.0;
}

precise float proctex_clamp(float c, uint mode) {
    switch (mode) {
        case 0u: return c > 1.0 ? 0.0 : c;
        case 1u: return min(c, 1.0);
        case 2u: return c - floor(c);
        case 3u: {
            int whole = int(c);
            float part = c - float(whole);
            return (whole % 2) == 0 ? part : 1.0 - part;
        }
        case 4u: return c <= 0.5 ? 0.0 : 1.0;
        default: return clamp(c, 0.0, 1.0);
    }
}

precise float proctex_combine(float u, float v, uint function) {
    float len = sqrt(u * u + v * v);
    switch (function) {
        case 0u: return u;
        case 1u: return u * u;
        case 2u: return v;
        case 3u: return v * v;
        case 4u: return (u + v) * 0.5;
        case 5u: return (u * u + v * v) * 0.5;
        case 6u: return min(len, 1.0);
        case 7u: return min(u, v);
        case 8u: return max(u, v);
        case 9u: return min(((u + v) * 0.5 + len) * 0.5, 1.0);
        default: return 0.0;
    }
}

// texture 3, made up from its coordinates, as the software rasterizer has
// it. precise as the combiners are, as is lighting
precise vec4 procedural(vec2 uv) {
    uint config = proctex[0].x;
    uvec2 clamps = uvec2(config & 7u, (config >> 3) & 7u);
    float u = abs(uv.x);
    float v = abs(uv.y);
    vec2 shifts = vec2(proctex_shift(v, (config >> 16) & 3u, clamps.x), proctex_shift(u, (config >> 18) & 3u, clamps.y));
    if ((config & (1u << 15)) != 0u) {
        uint noise_u = proctex[0].y;
        uint noise_v = proctex[0].z;
        vec2 frequency = unpackHalf2x16(proctex[0].w);
        float x = 9.0 * frequency.x * abs(u + float(noise_u >> 16) / 4096.0);
        float y = 9.0 * frequency.y * abs(v + float(noise_v >> 16) / 4096.0);
        uint xi = uint(x);
        uint yi = uint(y);
        float xf = x - float(xi);
        float yf = y - float(yi);
        float g0 = proctex_random(xi, yi) * (xf + yf);
        float g1 = proctex_random(xi + 1u, yi) * (xf + yf - 1.0);
        float g2 = proctex_random(xi, yi + 1u) * (xf + yf - 1.0);
        float g3 = proctex_random(xi + 1u, yi + 1u) * (xf + yf - 2.0);
        float s = proctex_lookup(0u, xf);
        float t = proctex_lookup(0u, yf);
        float noise = mix(mix(g0, g1, s), mix(g2, g3, s), t);
        u = abs(u + noise * float(int(noise_u << 16) >> 16) / 4095.0);
        v = abs(v + noise * float(int(noise_v << 16) >> 16) / 4095.0);
    }
    u = proctex_clamp(u + shifts.x, clamps.x);
    v = proctex_clamp(v + shifts.y, clamps.y);

    float coordinate = proctex_lookup(1u, proctex_combine(u, v, (config >> 6) & 0xFu));
    uint lut = proctex[1].x;
    float width = float((lut >> 11) & 0xFFu);
    float at = float(proctex[1].y & 0xFFu) + coordinate * max(width - 1.0, 0.0);
    uint filtering = lut & 7u;
    vec4 color;
    if (filtering == 1u || filtering == 3u || filtering == 5u) {
        uint entry = min(uint(max(at, 0.0)), 255u);
        color = proctex_color(entry, PROCTEX_COLORS) + (at - float(entry)) * proctex_color(entry, PROCTEX_STEPS);
    } else {
        color = proctex_color(min(uint(max(round(at), 0.0)), 255u), PROCTEX_COLORS);
    }
    color = clamp(color / 255.0, 0.0, 1.0);
    // alpha of its own skips the color table, the map gives it
    if ((config & (1u << 14)) != 0u) {
        color.a = clamp(proctex_lookup(2u, proctex_combine(u, v, (config >> 10) & 0xFu)), 0.0, 1.0);
    }
    return color;
}

precise vec3 rotate(vec4 q, vec3 v) {
    vec3 inner = cross(q.xyz, v);
    return v + 2.0 * cross(q.xyz, inner + v * q.w);
}

precise float quantize(float c) {
    return floor(clamp(c, 0.0, 1.0) * 255.0) / 255.0;
}

// what fragment lighting leaves in the primary and secondary colors
void shade(vec4 textures[4], out vec4 diffuse_out, out vec4 specular_out) {
    uint config = light_config.x;
    float length_q = length(in_quaternion);
    vec4 q = length_q > 0.0 ? in_quaternion / length_q : vec4(0.0, 0.0, 0.0, 1.0);

    uint shadow_word = light_config.w;
    vec4 shadow = vec4(1.0);
    if ((shadow_word & 1u) != 0u) {
        shadow = textures[(shadow_word >> 4) & 3u];
        if ((shadow_word & 0x100u) != 0u) {
            shadow = vec4(1.0) - shadow;
        }
    }

    vec3 surface_normal = vec3(0.0, 0.0, 1.0);
    vec3 surface_tangent = vec3(1.0, 0.0, 0.0);
    uint bump = light_config.z;
    uint bump_mode = bump & 3u;
    if (bump_mode == 1u) {
        surface_normal = textures[(bump >> 4) & 3u].xyz * 2.0 - 1.0;
        if ((bump & 0x100u) != 0u) {
            surface_normal.z = sqrt(max(1.0 - dot(surface_normal.xy, surface_normal.xy), 0.0));
        }
    } else if (bump_mode == 2u) {
        surface_tangent = textures[(bump >> 4) & 3u].xyz * 2.0 - 1.0;
    }
    vec3 normal = rotate(q, surface_normal);
    vec3 tangent = config == 8u ? rotate(q, surface_tangent) : surface_tangent;
    bool needs_half = (light_flags.w & 1u) != 0u;
    bool needs_view = (light_flags.w & 2u) != 0u;
    vec3 view = in_view;
    vec3 norm_view = needs_view ? normalize(view) : vec3(0.0);

    precise vec4 diffuse_sum = vec4(0.0, 0.0, 0.0, 1.0);
    precise vec4 specular_sum = vec4(0.0, 0.0, 0.0, 1.0);
    uint count = light_config.y;
    for (uint slot = 0u; slot < count; slot++) {
        Light light = lights[slot];
        uint flags = light.flags.x;
        uint number = light.flags.y;
        bool two_sided = (flags & 2u) != 0u;
        vec3 light_vector = (flags & 1u) != 0u ? light.direction.xyz : normalize(light.position.xyz + view);
        vec3 half_vector = vec3(0.0);
        vec3 half_unit = vec3(0.0);
        if (needs_half) {
            half_vector = norm_view + light_vector;
            half_unit = normalize(half_vector);
        }

        float distance = 1.0;
        if ((flags & 16u) != 0u) {
            vec3 offset = -view - light.position.xyz;
            float place = clamp(light.distance.y * length(offset) + light.distance.x, 0.0, 1.0);
            float entry = clamp(floor(place * 256.0), 0.0, 255.0);
            distance = lookup(DISTANCE + number, uint(entry), place * 256.0 - entry);
        }

        float values[7];
        for (uint i = 0u; i < 7u; i++) {
            values[i] = 1.0;
            uvec4 l = lookups[i];
            if (l.x == 0u || (i == 6u && (flags & 32u) == 0u)) {
                continue;
            }
            float result = 0.0;
            switch (l.y) {
                case 0u: result = dot(normal, half_unit); break;
                case 1u: result = dot(norm_view, half_unit); break;
                case 2u: result = dot(normal, norm_view); break;
                case 3u: result = dot(light_vector, normal); break;
                case 4u: result = dot(light_vector, light.spot.xyz); break;
                case 5u:
                    if (config == 8u) {
                        float along = dot(normal, half_unit);
                        result = dot(half_unit - normal * along, tangent);
                    }
                    break;
                default: break;
            }
            uint entry;
            float delta;
            if (l.z != 0u) {
                result = two_sided ? abs(result) : max(result, 0.0);
                float e = clamp(floor(result * 256.0), 0.0, 255.0);
                entry = uint(e);
                delta = result * 256.0 - e;
            } else {
                // the signed index wraps into the upper half of the table
                float e = clamp(floor(result * 128.0), -128.0, 127.0);
                entry = uint(int(e)) & 0xFFu;
                delta = result * 128.0 - e;
            }
            uint table = i == 0u ? DISTRIBUTION0
                : i == 1u ? DISTRIBUTION1
                : i == 2u ? FRESNEL
                : i == 3u ? REFLECT_RED
                : i == 4u ? REFLECT_GREEN
                : i == 5u ? REFLECT_BLUE
                : SPOTLIGHT + number;
            values[i] = uintBitsToFloat(l.w) * lookup(table, entry, delta);
        }

        float spot = values[6];
        float red = values[3];
        vec3 reflect_color = vec3(red, lookups[4].x != 0u ? values[4] : red, lookups[5].x != 0u ? values[5] : red);
        vec3 specular0 = light.specular0.rgb * values[0];
        vec3 specular1 = values[1] * reflect_color * light.specular1.rgb;

        // only the last light applies fresnel
        if (slot == count - 1u && lookups[2].x != 0u) {
            if (light_flags.x != 0u) {
                diffuse_sum.a = values[2];
            }
            if (light_flags.y != 0u) {
                specular_sum.a = values[2];
            }
        }

        float facing = dot(light_vector, normal);
        facing = two_sided ? abs(facing) : max(facing, 0.0);
        float highlights = light_flags.z != 0u && facing == 0.0 ? 0.0 : 1.0;
        if ((flags & 12u) != 0u) {
            float length2 = dot(half_vector, half_vector);
            float factor = length2 == 0.0 ? 0.0 : min(facing / length2, 1.0);
            if ((flags & 4u) != 0u) {
                specular0 *= factor;
            }
            if ((flags & 8u) != 0u) {
                specular1 *= factor;
            }
        }

        bool shadowed = (flags & 64u) != 0u && (shadow_word & 1u) != 0u;
        vec3 shadow_primary = shadowed && (shadow_word & 0x200u) != 0u ? shadow.rgb : vec3(1.0);
        vec3 shadow_secondary = shadowed && (shadow_word & 0x400u) != 0u ? shadow.rgb : vec3(1.0);
        vec3 diffuse = (light.diffuse.rgb * facing + light.ambient.rgb) * distance * spot;
        vec3 specular = (specular0 + specular1) * highlights * distance * spot;
        diffuse_sum.rgb += diffuse * shadow_primary;
        specular_sum.rgb += specular * shadow_secondary;
    }

    if ((shadow_word & 0x800u) != 0u) {
        if (light_flags.x != 0u) {
            diffuse_sum.a *= shadow.a;
        }
        if (light_flags.y != 0u) {
            specular_sum.a *= shadow.a;
        }
    }
    diffuse_sum.rgb += global_ambient.rgb;
    diffuse_out = vec4(quantize(diffuse_sum.r), quantize(diffuse_sum.g), quantize(diffuse_sum.b), quantize(diffuse_sum.a));
    specular_out = vec4(quantize(specular_sum.r), quantize(specular_sum.g), quantize(specular_sum.b), quantize(specular_sum.a));
}

precise vec3 color_operand(vec4 s, uint operand) {
    switch (operand) {
        case 0x1u: return vec3(1.0) - s.rgb;
        case 0x2u: return vec3(s.a);
        case 0x3u: return vec3(1.0 - s.a);
        case 0x4u: return vec3(s.r);
        case 0x5u: return vec3(1.0 - s.r);
        case 0x8u: return vec3(s.g);
        case 0x9u: return vec3(1.0 - s.g);
        case 0xCu: return vec3(s.b);
        case 0xDu: return vec3(1.0 - s.b);
        default: return s.rgb;
    }
}

precise float alpha_operand(vec4 s, uint operand) {
    switch (operand) {
        case 0x0u: return s.a;
        case 0x1u: return 1.0 - s.a;
        case 0x2u: return s.r;
        case 0x3u: return 1.0 - s.r;
        case 0x4u: return s.g;
        case 0x5u: return 1.0 - s.g;
        case 0x6u: return s.b;
        default: return 1.0 - s.b;
    }
}

// the operations' results are precise, so the driver fuses no multiply
// into an add, which it would do one way where it knows the operation and
// another where the generic shader does not, rounding differently
vec3 combine_rgb(uint op, vec3 a, vec3 b, vec3 c) {
    precise vec3 result;
    switch (op) {
        case 0u: result = a; break;
        case 1u: result = a * b; break;
        case 2u: result = min(a + b, vec3(1.0)); break;
        case 3u: result = clamp(a + b - 0.5, 0.0, 1.0); break;
        case 4u: result = a * c + b * (vec3(1.0) - c); break;
        case 5u: result = max(a - b, vec3(0.0)); break;
        case 6u:
        case 7u: {
            // both inputs are signed values packed into 0..1
            float d = clamp(dot(a * 2.0 - 1.0, b * 2.0 - 1.0), 0.0, 1.0);
            result = vec3(d);
            break;
        }
        case 8u: result = min(a * b + c, vec3(1.0)); break;
        default: result = min(a + b, vec3(1.0)) * c; break;
    }
    return result;
}

float combine_alpha(uint op, float a, float b, float c) {
    precise float result;
    switch (op) {
        case 0u: result = a; break;
        case 1u: result = a * b; break;
        case 2u: result = min(a + b, 1.0); break;
        case 3u: result = clamp(a + b - 0.5, 0.0, 1.0); break;
        case 4u: result = a * c + b * (1.0 - c); break;
        case 5u: result = max(a - b, 0.0); break;
        case 6u:
        case 7u: result = a; break;
        case 8u: result = min(a * b + c, 1.0); break;
        default: result = min(a + b, 1.0) * c; break;
    }
    return result;
}

uint operation(uint raw) {
    uint op = raw & 0xFu;
    return op > 9u ? 9u : op;
}

float scale_factor(uint raw) {
    uint s = raw & 3u;
    return s == 1u ? 2.0 : s == 2u ? 4.0 : 1.0;
}

bool compare(uint function, float value, float reference) {
    switch (function) {
        case 0u: return false;
        case 1u: return true;
        case 2u: return value == reference;
        case 3u: return value != reference;
        case 4u: return value < reference;
        case 5u: return value <= reference;
        case 6u: return value > reference;
        default: return value >= reference;
    }
}

// a source's value for a combiner stage
vec4 source_value(uint s, vec4 primary, vec4 fragment_primary, vec4 fragment_secondary, vec4 textures[4], vec4 previous, vec4 held, vec4 constant) {
    switch (s) {
        case 0x0u: return primary;
        case 0x1u: return fragment_primary;
        case 0x2u: return fragment_secondary;
        case 0x3u: return textures[0];
        case 0x4u: return textures[1];
        case 0x5u: return textures[2];
        case 0x6u: return textures[3];
        case 0xDu: return held;
        case 0xEu: return constant;
        default: return previous;
    }
}

// one combiner stage, its registers and the buffer update register
// constant in each pipeline but the generic one
void combine_stage(
    uint stage, uvec4 registers, uint update,
    vec4 primary, vec4 fragment_primary, vec4 fragment_secondary, vec4 textures[4],
    inout vec4 previous, inout vec4 held, inout vec4 next_buffer
) {
    uint source = registers.x;
    uint operand = registers.y;
    uint combiner = registers.z;
    uint scale = registers.w;
    // a stage handing the previous color on as it is changes nothing, as
    // the color is between 0 and 1 already. titles leave most stages so,
    // and the generic shader, which cannot drop them as it compiles, skips
    // them as it runs
    bool passes = (source & 0x000F000Fu) == 0x000F000Fu && (operand & 0x700Fu) == 0u
        && (combiner & 0x000F000Fu) == 0u && scale == 0u;
    if (!passes) {
        vec4 constant = colors[stage];
        vec4 rgb_in[3];
        vec4 alpha_in[3];
        for (uint i = 0u; i < 3u; i++) {
            rgb_in[i] = source_value((source >> (i * 4u)) & 0xFu, primary, fragment_primary, fragment_secondary, textures, previous, held, constant);
            alpha_in[i] = source_value((source >> (16u + i * 4u)) & 0xFu, primary, fragment_primary, fragment_secondary, textures, previous, held, constant);
        }
        uint color_op = operation(combiner);
        uint alpha_op = operation(combiner >> 16);
        vec3 rgb = combine_rgb(
            color_op,
            color_operand(rgb_in[0], operand & 0xFu),
            color_operand(rgb_in[1], (operand >> 4) & 0xFu),
            color_operand(rgb_in[2], (operand >> 8) & 0xFu)
        );
        float alpha = color_op == 7u
            ? rgb.r
            : combine_alpha(
                alpha_op,
                alpha_operand(alpha_in[0], (operand >> 12) & 0x7u),
                alpha_operand(alpha_in[1], (operand >> 16) & 0x7u),
                alpha_operand(alpha_in[2], (operand >> 20) & 0x7u)
            );
        precise vec4 scaled = clamp(vec4(rgb * scale_factor(scale), alpha * scale_factor(scale >> 16)), 0.0, 1.0);
        previous = scaled;
    }

    held = next_buffer;
    if (stage < 4u) {
        if ((update & (0x100u << stage)) != 0u) {
            next_buffer.rgb = previous.rgb;
        }
        if ((update & (0x1000u << stage)) != 0u) {
            next_buffer.a = previous.a;
        }
    }
}

#ifdef WRITES_DEPTH
// what the GPU shaded maps z/w itself, which the clipper got exactly.
// precise as the combiners are. built so only for a w-buffer and for a
// depth map the viewport can't hold, the rest leave depth to the
// rasterizer, which lets the GPU skip what is hidden before shading it
float fragment_depth() {
    precise float depth = in_depth;
    uint depth_mode = DYNAMIC != 0u ? flags.x : DEPTH_MODE;
    if ((depth_mode & 2u) != 0u) {
        depth = -gl_FragCoord.z * uintBitsToFloat(flags.z) + uintBitsToFloat(flags.w);
    }
    if ((depth_mode & 1u) != 0u) {
        depth /= gl_FragCoord.w;
    }
    return clamp(depth, 0.0, 1.0);
}
#else
// the viewport mapped the depth as the PICA does
float fragment_depth() {
    return gl_FragCoord.z;
}
#endif

void main() {
    vec4 primary = clamp(in_color, 0.0, 1.0);

    // the units the configuration turns on, unit 2 can read coordinate
    // set 1 instead of its own
    uint texture_config = DYNAMIC != 0u ? (misc.w & 0x2307u) | (TEXTURE_CONFIG & 0x400u) : TEXTURE_CONFIG;
    vec4 textures[4] = vec4[4](vec4(0.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0), vec4(0.0, 0.0, 0.0, 1.0));
    // every fetch unconditionally, before any branch, so texture()'s
    // implicit derivatives stay in uniform flow (see sample_unit); units
    // a draw does not use are sampled from the blank texture and dropped
    vec2 st0 = sample_st(0u, in_texcoords01.xy);
    vec2 st1 = sample_st(1u, in_texcoords01.zw);
    vec2 st2 = sample_st(2u, (texture_config & (1u << 13)) != 0u ? in_texcoords01.zw : in_texcoord2);
    vec4 t0 = texture(texture0, st0);
    vec4 t1 = texture(texture1, st1);
    vec4 t2 = texture(texture2, st2);
    vec4 a0 = vec4(t0.rgb, texture(alpha0, st0).r);
    vec4 a1 = vec4(t1.rgb, texture(alpha1, st1).r);
    vec4 a2 = vec4(t2.rgb, texture(alpha2, st2).r);
    if ((texture_config & 1u) != 0u) {
        textures[0] = sample_unit(0u, in_texcoords01.xy, t0, t1, t2, a0, a1, a2);
    }
    if ((texture_config & 2u) != 0u) {
        textures[1] = sample_unit(1u, in_texcoords01.zw, t0, t1, t2, a0, a1, a2);
    }
    if ((texture_config & 4u) != 0u) {
        textures[2] = sample_unit(2u, (texture_config & (1u << 13)) != 0u ? in_texcoords01.zw : in_texcoord2, t0, t1, t2, a0, a1, a2);
    }
    if ((texture_config & (1u << 10)) != 0u) {
        uint set = min((texture_config >> 8) & 3u, 2u);
        textures[3] = procedural(set == 0u ? in_texcoords01.xy : (set == 1u ? in_texcoords01.zw : in_texcoord2));
    }

    // without fragment lighting the primary fragment color is the vertex
    // color and there is no specular term
    vec4 fragment_primary = primary;
    vec4 fragment_secondary = vec4(0.0, 0.0, 0.0, 1.0);
    if (LIGHTING != 0u) {
        shade(textures, fragment_primary, fragment_secondary);
    }

    vec4 previous = primary;
    // the buffer lags a stage behind, the first stage reads zero, the second
    // the configured buffer color
    vec4 held = vec4(0.0);
    vec4 next_buffer = colors[6];
    uint update = (DYNAMIC != 0u ? misc.x : UPDATE) & 0xFF00u;
    combine_stage(0u, stage_registers(0u, uvec4(SOURCE0, OPERAND0, COMBINER0, SCALE0)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(1u, stage_registers(1u, uvec4(SOURCE1, OPERAND1, COMBINER1, SCALE1)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(2u, stage_registers(2u, uvec4(SOURCE2, OPERAND2, COMBINER2, SCALE2)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(3u, stage_registers(3u, uvec4(SOURCE3, OPERAND3, COMBINER3, SCALE3)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(4u, stage_registers(4u, uvec4(SOURCE4, OPERAND4, COMBINER4, SCALE4)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);
    combine_stage(5u, stage_registers(5u, uvec4(SOURCE5, OPERAND5, COMBINER5, SCALE5)), update, primary, fragment_primary, fragment_secondary, textures, previous, held, next_buffer);

    // the color leaves as whole bytes, the way the software path truncates.
    // precise, as everything from the operands on is, or where the driver
    // knows the operands it works 1 - a times 255 out as 255 - 255a, which
    // rounds to the other side of a byte
    precise vec4 color = floor(previous * 255.0);
    uint alpha_test = DYNAMIC != 0u ? misc.z & 0x71u : ALPHA_TEST;
    if ((alpha_test & 1u) != 0u && !compare((alpha_test >> 4) & 7u, color.a, reference.x)) {
        discard;
    }
    // the fog, by the fragment's depth through its table, the table's factor
    // of the color and the rest of the fog's, which the alpha test came
    // before. 5 turns it on, 7 is gas
    // known when built, as the rest of the update register, so programs
    // without it leave it and its table out
    uint fog_mode = DYNAMIC != 0u ? misc.x : UPDATE;
    if ((fog_mode & 7u) == 5u) {
        float depth = fragment_depth();
        precise float index = ((fog_mode & 0x10000u) != 0u ? 1.0 - depth : depth) * 128.0;
        precise float entry = clamp(floor(index), 0.0, 127.0);
        vec2 lookup = tables[FOG_TABLE + uint(entry)];
        precise float factor = clamp(lookup.x + lookup.y * (index - entry), 0.0, 1.0);
        vec3 fog_color = vec3(fog.x & 0xFFu, (fog.x >> 8) & 0xFFu, (fog.x >> 16) & 0xFFu);
        color.rgb = floor(factor * color.rgb + (1.0 - factor) * fog_color);
    }
    precise vec4 written = color / 255.0;
    // what nothing reads, as the alpha test constant tells, leaves as zero
    if ((ALPHA_TEST & 0x100u) != 0u) {
        written.rgb = vec3(0.0);
    }
    if ((ALPHA_TEST & 0x200u) != 0u) {
        written.a = 0.0;
    }
    out_color = written;

#ifdef WRITES_DEPTH
    gl_FragDepth = fragment_depth();
#endif
}
```

### Notes
- Single mega-shader for the PICA200 fragment stages: 3 texture units (+3 separate alpha samplers), procedural textures, lighting, 6 combiners, alpha test.
- Contains `vec2 tables[]` (unsized uniform array, dynamically indexed: `tables[table * 256u + entry]`, `tables[FOG_TABLE + uint(entry)]`). Unreal results: UniFlex crashes.
- The shader used to have `texture()` inside `if (unit == 0u)` branches: PowerVR emitted the "gradient in conditional block" warnings and returned a compile error (recoverable, black 3D).
- The "fix" — 6 unconditional `texture()` fetches at the top of `main()`, then `sample_unit()` just *selects* between already-fetched vec4s via ternary — eliminated the error but now UniFlex **segfaults** (fault addr is a float: `0x3f80025c`). Known UniFlex triggers: too many live texture fetches, dynamic indexing into unsized uniform arrays, oversized shaders.
- Generic-shader mode (`ZAKURO_GENERIC_SHADERS`) exists that uses a single program variant (fewer compiles) but the same shader body. `ZAKURO_SYNC_SHADERS` compiles shaders on the render thread instead of a background thread (driver thread-safety concerns).

### Questions
1. Most likely cause of the segfault: (a) 6 unconditional texture() fetches, (b) the dynamically-indexed `tables[]` uniform array, (c) something else in the 758-line shader? What's the minimal shader change to avoid UniFlex crashing?
2. Should we offer a "simple raster" path for GLES that compiles away unused stages at a higher level, or split the mega-shader?
3. Should Android/PowerVR default to `ZAKURO_SYNC_SHADERS=1`, or is background-thread shader compilation itself the trigger (the crash thread is named "gles")?

## Ask

For each bug: most likely root cause, and the smallest, most general fix that doesn't regress desktop/64-bit Linux. Cite file/symbol names. This is diagnosis, not a full implementation.
