//! drawing PICA200 draws on the host's GPU through Vulkan 1.1.
//!
//! the title's vertices are shaded on the CPU, as the rasterizer works them
//! out in screen space already, and the GPU does what costs, the pixels: the
//! texture units, fragment lighting, the six combiners, the alpha, stencil
//! and depth tests and blending, from the same fragment program the OpenGL
//! ES renderer builds. that program is GLSL, which naga turns into SPIR-V
//! when a draw first needs it, and the resulting pipelines, samplers and
//! texture images are kept for the draws after.
//!
//! buffers live in guest memory: a color or depth buffer is brought up to an
//! image the first time a draw needs it and goes back down when something
//! else reads it, so the rest of the emulator sees no difference. display
//! transfers, fills and copies from one texture to another are left to the
//! software path, which the flush before them makes correct.
//!
//! the screens are shown by the OpenGL ES panel, not here: this renderer
//! draws, and its buffers come back to guest memory for that panel to show.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;

use super::gles::sources;
use super::host::{Draw, Geometry};
use super::{color_words, BoundTexture, DepthMap, Screen, Wrap, COLOR_WORDS, TEXTURE_UNIT_BASES};
use crate::format::{morton_offset, ColorFormat};
use crate::lighting::Tables;
use crate::registers::*;
use crate::{proctex, GpuMemory};

/// the words of a draw's uniform block, as the OpenGL ES renderer packs the
/// same block: the combiners, the texture units and the flags, then the
/// lighting, the procedural texture's registers, the colors and the fog.
const PROCTEX_WORDS: usize = 8;
const FOG_WORDS: usize = 4;
const UNIFORM_WORDS: usize = 12 * 4 + 4 + 3 * 4 + 4 + 328 + PROCTEX_WORDS + COLOR_WORDS + FOG_WORDS;
/// the lighting tables, then the procedural texture's and the fog's.
const TABLES_SIZE: usize = (24 * 256 + 3 * proctex::MAP_ENTRIES + 4 * proctex::COLOR_ENTRIES + crate::fog::ENTRIES) * 8;
const FRAGMENT_CONSTANTS: usize = 30;

const STAGE_REGISTERS: [usize; 6] = [0x0C0, 0x0C8, 0x0D0, 0x0D8, 0x0F0, 0x0F8];
const REG_UPDATE_BUFFER: usize = 0x0E0;
const REG_BUFFER_COLOR: usize = 0x0FD;
const LIGHTING_BASE: usize = 68;
/// GPUREG_BLEND_COLOR, the constant the blend factors scale by.
const REG_BLEND_COLOR: usize = 0x103;

/// a vertex is six vectors: position, color, texture coordinates 0 and 1,
/// coordinate 2 with the mapped depth, the quaternion and the view vector.
const VERTEX_SIZE: u64 = 24 * 4;

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const DEPTH_FORMAT: vk::Format = vk::Format::D24_UNORM_S8_UINT;

/// how many bytes a draw may write into the ring before it is submitted.
const RING_FLUSH: vk::DeviceSize = 1 << 20;

/// the PICA's compare functions in its own order, as Vulkan names them.
const COMPARES: [vk::CompareOp; 8] = [
    vk::CompareOp::NEVER,
    vk::CompareOp::ALWAYS,
    vk::CompareOp::EQUAL,
    vk::CompareOp::NOT_EQUAL,
    vk::CompareOp::LESS,
    vk::CompareOp::LESS_OR_EQUAL,
    vk::CompareOp::GREATER,
    vk::CompareOp::GREATER_OR_EQUAL,
];

/// the PICA's stencil operations in its own order.
const STENCIL_OPS: [vk::StencilOp; 8] = [
    vk::StencilOp::KEEP,
    vk::StencilOp::ZERO,
    vk::StencilOp::REPLACE,
    vk::StencilOp::INCREMENT_AND_CLAMP,
    vk::StencilOp::DECREMENT_AND_CLAMP,
    vk::StencilOp::INVERT,
    vk::StencilOp::INCREMENT_AND_WRAP,
    vk::StencilOp::DECREMENT_AND_WRAP,
];

/// the PICA's blend factors in its own order.
const FACTORS: [vk::BlendFactor; 15] = [
    vk::BlendFactor::ZERO,
    vk::BlendFactor::ONE,
    vk::BlendFactor::SRC_COLOR,
    vk::BlendFactor::ONE_MINUS_SRC_COLOR,
    vk::BlendFactor::DST_COLOR,
    vk::BlendFactor::ONE_MINUS_DST_COLOR,
    vk::BlendFactor::SRC_ALPHA,
    vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
    vk::BlendFactor::DST_ALPHA,
    vk::BlendFactor::ONE_MINUS_DST_ALPHA,
    vk::BlendFactor::CONSTANT_COLOR,
    vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
    vk::BlendFactor::CONSTANT_ALPHA,
    vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
    vk::BlendFactor::SRC_ALPHA_SATURATE,
];

/// the PICA's blend equations in its own order.
const EQUATIONS: [vk::BlendOp; 5] = [
    vk::BlendOp::ADD,
    vk::BlendOp::SUBTRACT,
    vk::BlendOp::REVERSE_SUBTRACT,
    vk::BlendOp::MIN,
    vk::BlendOp::MAX,
];

/// a buffer the renderer draws into or from, by address and size.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TargetKey {
    addr: u32,
    width: u32,
    height: u32,
    /// none for a color buffer, else the bytes each depth sample takes.
    depth: Option<u8>,
}

/// whether a buffer at a target's address, its whole extent, touches a range.
fn overlaps(key: &TargetKey, addr: u32, len: u32) -> bool {
    key.addr < addr.saturating_add(len) && addr < key.addr.saturating_add(key.width * key.height * 4)
}

struct Target {
    id: u64,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    layout: vk::ImageLayout,
    /// draws went into it since it was last brought down to memory.
    dirty: bool,
    /// the image holds the buffer's bytes, draws not yet included.
    loaded: bool,
    key: TargetKey,
    format: ColorFormat,
}

struct Texture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    layout: vk::ImageLayout,
    /// holds the texels it was made from alive, whose address keys it.
    _texels: Option<Arc<[[u8; 4]]>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SamplerKey {
    linear: bool,
    wrap_s: u32,
    wrap_t: u32,
    lod: (i16, u8, u8, bool, bool),
}

/// a fragment program, by the constants that shape it and the lighting
/// written into it.
#[derive(PartialEq, Eq, Hash)]
struct ShaderKey {
    constants: [u32; FRAGMENT_CONSTANTS],
    writes_depth: bool,
    lighting: Vec<u32>,
}

/// a pipeline, by everything a draw's command buffer cannot set for itself.
#[derive(PartialEq, Eq, Hash)]
struct PipelineKey {
    shader: [u32; FRAGMENT_CONSTANTS],
    writes_depth: bool,
    lighting: Vec<u32>,
    with_depth: bool,
    blend: Option<(u32, u32)>,
    depth: Option<u32>,
    depth_write: bool,
    stencil: Option<(u32, u32, bool)>,
    color_mask: u32,
}

pub(crate) struct Vulkan {
    /// kept alive: the instance is made through it, and goes before it.
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    non_coherent: bool,

    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    recording: bool,

    ring: vk::Buffer,
    ring_memory: vk::DeviceMemory,
    ring_ptr: *mut u8,
    ring_size: vk::DeviceSize,
    ring_at: vk::DeviceSize,
    uniform_alignment: vk::DeviceSize,
    storage_alignment: vk::DeviceSize,

    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,

    render_pass_plain: vk::RenderPass,
    render_pass_depth: vk::RenderPass,

    vertex_module: vk::ShaderModule,
    fragment_modules: HashMap<ShaderKey, vk::ShaderModule>,
    pipelines: HashMap<PipelineKey, vk::Pipeline>,
    samplers: HashMap<SamplerKey, vk::Sampler>,
    textures: HashMap<(usize, usize), Texture>,
    replaced: HashMap<u64, Texture>,
    targets: HashMap<TargetKey, Target>,
    framebuffers: HashMap<(u64, u64), vk::Framebuffer>,
    next_id: u64,

    dummy: Texture,
    dummy_sampler: vk::Sampler,
    /// the buffers a draw staged its vertices, uniforms and tables in, to
    /// free them once the GPU is done with the batch.
    staging: Vec<(vk::Buffer, vk::DeviceMemory)>,
    scale: u32,
}

impl Vulkan {
    pub fn new() -> Result<Vulkan, String> {
        // SAFETY: the loader is only opened, nothing is asked of it yet
        let entry = unsafe { ash::Entry::load().map_err(|error| format!("no Vulkan loader here: {error}"))? };
        let application = vk::ApplicationInfo::default()
            .application_name(c"pocket-native")
            .api_version(vk::make_api_version(0, 1, 1, 0));
        let instance_info = vk::InstanceCreateInfo::default().application_info(&application);
        // SAFETY: a plain instance with no layers or extensions
        let instance = unsafe { entry.create_instance(&instance_info, None).map_err(|error| format!("no Vulkan instance: {error}"))? };

        let (physical, family) = pick_device(&instance)?;
        // SAFETY: the device is asked for a graphics queue alone, on Vulkan 1.1
        let device = unsafe {
            let priorities = [1.0f32];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
            let features = vk::PhysicalDeviceFeatures::default();
            let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_features(&features);
            instance.create_device(physical, &info, None).map_err(|error| format!("could not open the Vulkan device: {error}"))?
        };
        // SAFETY: the queue family was checked to have a graphics queue
        let queue = unsafe { device.get_device_queue(family, 0) };
        // SAFETY: a device just made
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let non_coherent = !coherent_host_memory(&memory_properties);

        let command_pool = create_command_pool(&device, family)?;
        // SAFETY: a pool of that family
        let (command_buffer, fence) = unsafe {
            let buffer = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
                .map_err(|error| format!("no command buffer: {error}"))?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|error| format!("no fence: {error}"))?;
            (buffer, fence)
        };

        let (ring, ring_memory, ring_ptr, ring_size) = create_ring(&instance, &device, physical, 32 << 20)?;
        let limits = unsafe { instance.get_physical_device_properties(physical) }.limits;
        let uniform_alignment = limits.min_uniform_buffer_offset_alignment.max(4) as vk::DeviceSize;
        let storage_alignment = limits.min_storage_buffer_offset_alignment.max(4) as vk::DeviceSize;

        let (descriptor_layout, pipeline_layout) = create_descriptor_layout(&device)?;
        // SAFETY: a device just made
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(65536)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(6 * 65536),
                            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLER).descriptor_count(6 * 65536),
                            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(65536),
                            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(65536),
                        ])
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                    None,
                )
                .map_err(|error| format!("no descriptor pool: {error}"))?
        };

        let render_pass_plain = create_render_pass(&device, false)?;
        let render_pass_depth = create_render_pass(&device, true)?;

        let vertex_source = sources::VERTEX_SOURCE.replace("#version 310 es\n", "#version 450\n");
        let vertex_module = make_module(&device, &vertex_source, naga::ShaderStage::Vertex)?;

        let mut renderer = Vulkan {
            _entry: entry,
            instance,
            physical,
            device,
            queue,
            non_coherent,
            command_pool,
            command_buffer,
            fence,
            recording: false,
            ring,
            ring_memory,
            ring_ptr,
            ring_size,
            ring_at: 0,
            uniform_alignment,
            storage_alignment,
            descriptor_layout,
            pipeline_layout,
            descriptor_pool,
            render_pass_plain,
            render_pass_depth,
            vertex_module,
            fragment_modules: HashMap::new(),
            pipelines: HashMap::new(),
            samplers: HashMap::new(),
            textures: HashMap::new(),
            replaced: HashMap::new(),
            targets: HashMap::new(),
            framebuffers: HashMap::new(),
            next_id: 1,
            dummy: Texture { image: vk::Image::null(), memory: vk::DeviceMemory::null(), view: vk::ImageView::null(), layout: vk::ImageLayout::UNDEFINED, _texels: None },
            dummy_sampler: vk::Sampler::null(),
            staging: Vec::new(),
            scale: 1,
        };
        renderer.dummy = renderer.make_texture(1, 1, 1, &[0, 0, 0, 0])?;
        renderer.make_dummy_sampler()?;
        log::info!("vulkan: drawing through Vulkan 1.1 on the host's GPU");
        Ok(renderer)
    }

    fn make_dummy_sampler(&mut self) -> Result<(), String> {
        // SAFETY: a device just made
        self.dummy_sampler = unsafe {
            self.device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::NEAREST)
                        .min_filter(vk::Filter::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .map_err(|error| format!("no sampler: {error}"))?
        };
        Ok(())
    }

    pub fn name(&self) -> &str {
        "vulkan"
    }

    /// the vertices are shaded on the CPU, the GPU draws their pixels.
    pub fn shades(&self) -> bool {
        false
    }

    pub fn fetches(&self) -> bool {
        false
    }

    /// the textures come to it decoded, not in the PICA's own formats.
    pub fn packs(&self) -> bool {
        false
    }

    pub fn scale(&self) -> u32 {
        self.scale
    }

    /// how many times the console's resolution to draw at. the images here
    /// are at the console's size, so a scale over one is not taken.
    pub fn set_scale(&mut self, scale: f32) -> f32 {
        let _ = scale;
        self.scale = 1;
        1.0
    }

    /// whether the screens can be shown by this renderer, which they cannot:
    /// the panel draws them with OpenGL ES.
    pub fn presents(&self) -> bool {
        false
    }

    /// nothing reads a buffer the renderer drew back as a texture yet, so a
    /// drawn texture is decoded from guest memory by the rasterizer.
    pub fn holds(&self, _texture: &super::DrawnTexture) -> bool {
        false
    }

    pub fn flush_draws(&mut self) {}

    /// guest memory was written by something else over a range: a buffer
    /// that overlaps it comes up from memory again before its next draw.
    pub fn overwritten(&mut self, addr: u32, len: u32) {
        for (key, target) in self.targets.iter_mut() {
            if overlaps(key, addr, len) {
                target.loaded = false;
            }
        }
    }

    /// a fill wrote guest memory itself, so what the renderer draws from a
    /// buffer it held has to come up again before the next draw.
    pub fn filled(&mut self, addr: u32, bytes: &[u8]) -> Result<(), String> {
        self.overwritten(addr, bytes.len() as u32);
        Ok(())
    }

    pub fn fill_only(&mut self, _addr: u32, _len: u32, _pattern: &[u8]) -> Result<bool, String> {
        Ok(false)
    }

    pub fn before_fill<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32, _width: u32) -> Result<(), String> {
        self.flush_overlapping(memory, addr, len)
    }

    pub fn before_overwrite<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        self.flush_overlapping(memory, addr, len)
    }

    pub fn sync<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        self.flush_overlapping(memory, addr, len)
    }

    pub fn prepare_read<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32, _texture: Option<&super::DrawnTexture>) -> Result<(), String> {
        self.flush_overlapping(memory, addr, len)
    }

    /// the displays' transfers run in software, guest memory being right
    /// after the flush.
    pub fn display_transfer<M: GpuMemory>(&mut self, memory: &mut M, _transfer: &super::host::Transfer) -> Result<bool, String> {
        self.flush(memory)?;
        Ok(false)
    }

    pub fn texture_copy<M: GpuMemory>(&mut self, memory: &mut M, _input: u32, _output: u32, _size: u32, _input_lines: (u32, u32), _output_lines: (u32, u32)) -> Result<bool, String> {
        self.flush(memory)?;
        Ok(false)
    }

    /// no buffer is offered to the panel as an image of its own.
    pub fn screen(&mut self, _addr: u32, _size: (u32, u32), _stride: u32, _format: ColorFormat, _guest: &[u8]) -> Option<crate::ScreenRef> {
        None
    }

    pub fn picture(&mut self, _screen: crate::ScreenRef) -> Result<Option<crate::Picture>, String> {
        Ok(None)
    }

    pub fn present(&mut self, _shown: [Option<(crate::PanelImage, [i32; 4])>; 2], _cursor: Option<(i32, i32)>, _overlays: Vec<Arc<crate::Overlay>>, _smooth: bool, _interpolate: bool, _frames: u32) {}

    pub fn can_mirror(&self, _first: [(u32, u32); 2], _other: [(u32, u32); 2]) -> bool {
        false
    }

    pub fn mirror(&mut self, _first: [(u32, u32); 2], _other: [(u32, u32); 2]) -> Result<(), String> {
        Ok(())
    }

    /// ends the GPU's work and frees what the renderer holds, for the
    /// process to end.
    pub fn stop(&mut self) {
        // SAFETY: everything made here, and nothing the GPU runs still reads
        unsafe {
            let _ = self.device.device_wait_idle();
            for (buffer, memory) in self.staging.drain(..) {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            let textures: Vec<Texture> = std::mem::take(&mut self.textures).into_values().collect();
            for texture in textures {
                self.destroy_texture(texture);
            }
            let replaced: Vec<Texture> = std::mem::take(&mut self.replaced).into_values().collect();
            for texture in replaced {
                self.destroy_texture(texture);
            }
            if self.dummy.view != vk::ImageView::null() {
                let dummy = std::mem::replace(
                    &mut self.dummy,
                    Texture { image: vk::Image::null(), memory: vk::DeviceMemory::null(), view: vk::ImageView::null(), layout: vk::ImageLayout::UNDEFINED, _texels: None },
                );
                self.destroy_texture(dummy);
            }
            if self.dummy_sampler != vk::Sampler::null() {
                self.device.destroy_sampler(self.dummy_sampler, None);
                self.dummy_sampler = vk::Sampler::null();
            }
            let targets: Vec<Target> = std::mem::take(&mut self.targets).into_values().collect();
            for target in targets {
                self.device.destroy_image_view(target.view, None);
                self.device.destroy_image(target.image, None);
                self.device.free_memory(target.memory, None);
            }
            let framebuffers: Vec<vk::Framebuffer> = std::mem::take(&mut self.framebuffers).into_values().collect();
            for framebuffer in framebuffers {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            let pipelines: Vec<vk::Pipeline> = std::mem::take(&mut self.pipelines).into_values().collect();
            for pipeline in pipelines {
                self.device.destroy_pipeline(pipeline, None);
            }
            let modules: Vec<vk::ShaderModule> = std::mem::take(&mut self.fragment_modules).into_values().collect();
            for module in modules {
                self.device.destroy_shader_module(module, None);
            }
            let samplers: Vec<vk::Sampler> = std::mem::take(&mut self.samplers).into_values().collect();
            for sampler in samplers {
                self.device.destroy_sampler(sampler, None);
            }
            self.device.destroy_shader_module(self.vertex_module, None);
            self.device.destroy_render_pass(self.render_pass_plain, None);
            self.device.destroy_render_pass(self.render_pass_depth, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_buffer(self.ring, None);
            self.device.free_memory(self.ring_memory, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }

    // -- the drawing -------------------------------------------------------

    pub fn draw<M: GpuMemory>(&mut self, memory: &mut M, draw: &Draw<'_>) -> Result<(), String> {
        let triangles = match draw.geometry {
            Geometry::Triangles(triangles) => triangles,
            // the vertex shader would have to run on the GPU for these,
            // which this renderer leaves to the software path
            Geometry::Shaded(_) => return Err("the Vulkan renderer shades vertices on the CPU".to_owned()),
        };
        let [left, bottom, right, top] = draw.scissor;
        let (left, bottom) = (left.max(0) as u32, bottom.max(0) as u32);
        let (right, top) = (right.clamp(0, draw.width as i32) as u32, top.clamp(0, draw.height as i32) as u32);
        if triangles.is_empty() || right <= left || top <= bottom || draw.width == 0 || draw.height == 0 {
            return Ok(());
        }

        let r = draw.registers;
        let color_key = TargetKey { addr: draw.target, width: draw.width, height: draw.height, depth: None };
        self.ensure_target(memory, color_key, draw.format)?;
        let depth_key = draw.depth.map(|(addr, bytes)| TargetKey { addr, width: draw.width, height: draw.height, depth: Some(bytes as u8) });
        if let Some(key) = depth_key {
            self.ensure_target(memory, key, draw.format)?;
        }
        if self.ring_at > self.ring_size - RING_FLUSH {
            self.flush(memory)?;
        }
        self.begin()?;

        // the textures the units read, uploaded from their decoded texels
        let mut images = [self.dummy.view; 6];
        let mut samplers = [self.dummy_sampler; 6];
        let mut enabled = 0u32;
        for (unit, texture) in draw.textures.iter().enumerate() {
            let Some(texture) = texture else { continue };
            images[unit] = match &texture.replacement {
                Some(material) => match self.replacement(material)? {
                    Some(view) => view,
                    None => self.texture(texture)?,
                },
                None => self.texture(texture)?,
            };
            samplers[unit] = self.sampler(texture)?;
            enabled |= 1 << unit;
        }

        let depth_map = draw.depth_map;
        let (writes_depth, depth_flags) = match depth_map.w_buffer {
            true => (true, 1),
            false => (false, 0),
        };
        let sources = [false; 3];
        let mut words = Vec::with_capacity(UNIFORM_WORDS);
        for base in STAGE_REGISTERS {
            words.extend([r[base], r[base + 1], r[base + 2], r[base + 3], r[base + 4], 0, 0, 0]);
        }
        let texture_config = (r[REG_TEXTURE_CONFIG] & !7) | enabled;
        words.extend([r[REG_UPDATE_BUFFER], r[REG_BUFFER_COLOR], r[REG_ALPHA_TEST], texture_config]);
        for (unit, base) in TEXTURE_UNIT_BASES.into_iter().enumerate() {
            // the bias in the top half of the in place flag's word
            let bias = ((r[base + 3] << 19) as i32 >> 19) as u16 as u32;
            words.extend([r[base + 2], r[base], bias << 16, 0u32]);
            let _ = unit;
        }
        words.extend([depth_flags, draw.lighting.is_some() as u32, depth_map.scale.to_bits(), depth_map.offset.to_bits()]);
        match draw.lighting {
            Some(lighting) => lighting.pack(&mut words),
            None => words.resize(UNIFORM_WORDS - PROCTEX_WORDS - COLOR_WORDS - FOG_WORDS, 0),
        }
        let base = proctex::REG_CONFIG;
        words.extend([r[base], r[base + 1], r[base + 2], r[base + 3], r[base + 4], r[base + 5], 0, 0]);
        words.extend(color_words(r));
        words.extend([r[crate::fog::REG_COLOR], 0, 0, 0]);
        if words.len() != UNIFORM_WORDS {
            return Err(format!("the draw's uniform block came to {} words, not {UNIFORM_WORDS}", words.len()));
        }

        let (constants, lighting) = sources::fragment_constants(r, enabled, 0, sources, depth_flags, writes_depth, draw.lighting.map(|_| &words[LIGHTING_BASE..]));

        // the state the pipeline holds
        let color_mask = if r[REG_COLOR_BUFFER_WRITE] != 0 { (r[REG_DEPTH_COLOR_MASK] >> 8) & 0xF } else { 0 };
        let mask = r[REG_DEPTH_COLOR_MASK];
        let writable = r[REG_DEPTH_STENCIL_WRITE] != 0;
        let depth_test = mask & 1 != 0;
        let depth_write = writable && mask & (1 << 12) != 0;
        let stencil = depth_key.filter(|key| key.depth == Some(4)).and_then(|_| super::Stencil::read(r, writable));
        let stencil_compare = match r[REG_STENCIL_TEST] & 1 {
            0 => r[REG_STENCIL_TEST] & !0x70 | 0x10,
            _ => r[REG_STENCIL_TEST],
        };
        let blend = (r[REG_COLOR_OPERATION] & 0x100 != 0).then_some(r[REG_BLEND_FUNC]);
        let color_mask = match crate::blend::LogicOp::read(r) {
            Some(crate::blend::LogicOp::Noop) => 0,
            _ => color_mask,
        };
        if color_mask == 0 && blend.is_none() {
            // nothing would be written, but the depth and stencil may still be
        }

        let pipeline_key = PipelineKey {
            shader: constants,
            writes_depth,
            lighting: lighting.clone(),
            with_depth: depth_key.is_some(),
            blend: blend.map(|config| (config, r[REG_BLEND_COLOR])),
            depth: depth_key.filter(|_| depth_test || depth_write).map(|_| {
                if depth_test {
                    COMPARES[((mask >> 4) & 7) as usize].as_raw() as u32
                } else {
                    vk::CompareOp::ALWAYS.as_raw() as u32
                }
            }),
            depth_write: depth_key.is_some() && depth_write,
            stencil: stencil.map(|_| (stencil_compare, r[REG_STENCIL_OP] & 0x777, writable)),
            color_mask,
        };
        let pipeline = self.pipeline(&pipeline_key, &constants, writes_depth, &lighting)?;

        // what the draw reads and writes, in the ring
        let uniform_bytes = memory_bytes(&words);
        let uniforms = self.push(&uniform_bytes, self.uniform_alignment)?;
        let tables = self.tables(draw.tables, draw.proctex_tables, draw.fog_table)?;
        let vertices = self.push(&vertex_bytes(triangles, (draw.width as f32, draw.height as f32), depth_map), 4)?;

        let set = self.descriptor_set(images, samplers, uniforms, tables)?;

        // the framebuffer, the color buffer and its depth
        let color = self.targets.get(&color_key).expect("a target just made").clone_ref();
        let depth = depth_key.and_then(|key| self.targets.get(&key).map(TargetRef::of));
        let framebuffer = self.framebuffer(color_key, color.id, color.view, depth.as_ref())?;

        let extent = vk::Extent2D { width: draw.width, height: draw.height };
        let render_pass = if depth_key.is_some() { self.render_pass_depth } else { self.render_pass_plain };
        // SAFETY: the command buffer is recording, every handle was made on
        // this device and the barrier names the layouts the images are in
        unsafe {
            let device = &self.device;
            let cb = self.command_buffer;
            let attachments = [
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
            ];
            let clear = attachments[..if depth_key.is_some() { 2 } else { 1 }].to_vec();
            device.cmd_begin_render_pass(
                cb,
                &vk::RenderPassBeginInfo::default().render_pass(render_pass).framebuffer(framebuffer).render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent }).clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            let viewport = vk::Viewport { x: 0.0, y: 0.0, width: draw.width as f32, height: draw.height as f32, min_depth: 0.0, max_depth: 1.0 };
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: left as i32, y: bottom as i32 }, extent: vk::Extent2D { width: right - left, height: top - bottom } }]);
            if let Some((config, color)) = pipeline_key.blend {
                let constant = color.to_le_bytes().map(|c| c as f32 / 255.0);
                device.cmd_set_blend_constants(cb, &constant);
                let _ = config;
            }
            if let Some((compare, _, writable)) = pipeline_key.stencil {
                device.cmd_set_stencil_reference(cb, vk::StencilFaceFlags::FRONT_AND_BACK, ((compare >> 16) & 0xFF) as u32);
                device.cmd_set_stencil_compare_mask(cb, vk::StencilFaceFlags::FRONT_AND_BACK, (compare >> 24) & 0xFF);
                device.cmd_set_stencil_write_mask(cb, vk::StencilFaceFlags::FRONT_AND_BACK, if writable { (compare >> 8) & 0xFF } else { 0 });
            }
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[set], &[]);
            device.cmd_bind_vertex_buffers(cb, 0, &[self.ring], &[vertices]);
            device.cmd_draw(cb, (triangles.len() * 3) as u32, 1, 0, 0);
            device.cmd_end_render_pass(cb);
        }

        for key in [Some(color_key), depth_key].into_iter().flatten() {
            if let Some(target) = self.targets.get_mut(&key) {
                target.dirty = true;
            }
        }
        Ok(())
    }

    /// submits the draws recorded, brings every buffer they drew into back
    /// to guest memory and starts afresh.
    pub fn flush<M: GpuMemory>(&mut self, memory: &mut M) -> Result<(), String> {
        self.submit()?;
        let dirty: Vec<TargetKey> = self.targets.iter().filter(|(_, target)| target.dirty).map(|(key, _)| *key).collect();
        for key in dirty {
            self.download(memory, key)?;
            if let Some(target) = self.targets.get_mut(&key) {
                target.dirty = false;
            }
        }
        Ok(())
    }

    /// submits the draws recorded and brings back the buffers a range
    /// touches, the others staying on the device until something reads one.
    fn flush_overlapping<M: GpuMemory>(&mut self, memory: &mut M, addr: u32, len: u32) -> Result<(), String> {
        self.submit()?;
        let dirty: Vec<TargetKey> = self.targets.iter().filter(|(key, target)| target.dirty && overlaps(key, addr, len)).map(|(key, _)| *key).collect();
        for key in dirty {
            self.download(memory, key)?;
            if let Some(target) = self.targets.get_mut(&key) {
                target.dirty = false;
            }
        }
        Ok(())
    }

    // -- command buffers ---------------------------------------------------

    fn begin(&mut self) -> Result<(), String> {
        if self.recording {
            return Ok(());
        }
        // SAFETY: nothing is recording, the fence was waited on
        unsafe {
            let device = &self.device;
            device.reset_fences(&[self.fence]).map_err(|error| error.to_string())?;
            device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty()).map_err(|error| error.to_string())?;
            device
                .begin_command_buffer(self.command_buffer, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(|error| error.to_string())?;
            // the ring's contents of the batch before are done with
            device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty()).map_err(|error| error.to_string())?;
        }
        for (buffer, memory) in self.staging.drain(..) {
            // SAFETY: the buffer was made here and the GPU is idle
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        }
        self.ring_at = 0;
        self.recording = true;
        Ok(())
    }

    /// submits what was recorded and waits for it, so guest memory can be
    /// read.
    fn submit(&mut self) -> Result<(), String> {
        if !self.recording {
            return Ok(());
        }
        self.recording = false;
        // SAFETY: the command buffer is recording on this device
        unsafe {
            let device = &self.device;
            device.end_command_buffer(self.command_buffer).map_err(|error| error.to_string())?;
            let buffers = [self.command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&buffers);
            device.queue_submit(self.queue, &[submit], self.fence).map_err(|error| format!("the GPU would not take the draws: {error}"))?;
            device.wait_for_fences(&[self.fence], true, u64::MAX).map_err(|error| format!("the GPU would not finish: {error}"))?;
        }
        Ok(())
    }

    // -- buffers -----------------------------------------------------------

    fn push(&mut self, bytes: &[u8], align: vk::DeviceSize) -> Result<vk::DeviceSize, String> {
        let at = self.ring_at.next_multiple_of(align);
        let end = at + bytes.len() as vk::DeviceSize;
        if end > self.ring_size {
            return Err("the Vulkan upload ring is full".to_owned());
        }
        // SAFETY: the ring was mapped for the process's life, and the range
        // is inside it
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ring_ptr.add(at as usize), bytes.len());
        }
        self.ring_at = end;
        Ok(at)
    }

    /// the lighting, procedural texture and fog tables, as the fragment
    /// program reads them.
    fn tables(&mut self, lighting: &Tables, procedural: &proctex::Tables, fog: &crate::fog::Table) -> Result<vk::DeviceSize, String> {
        let (colors, steps) = procedural.colors();
        let parts: [&[f32]; 5] = [
            lighting.entries().as_flattened().as_flattened(),
            procedural.maps().as_flattened().as_flattened(),
            colors.as_flattened(),
            steps.as_flattened(),
            fog.entries().as_flattened(),
        ];
        let mut bytes = Vec::with_capacity(TABLES_SIZE);
        for part in parts {
            for value in part {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes.resize(TABLES_SIZE, 0);
        let alignment = self.storage_alignment;
        self.push(&bytes, alignment)
    }

    fn ensure_target<M: GpuMemory>(&mut self, memory: &mut M, key: TargetKey, format: ColorFormat) -> Result<(), String> {
        if self.targets.get(&key).is_some_and(|target| target.loaded) {
            return Ok(());
        }
        if self.targets.contains_key(&key) {
            // something else wrote the buffer since it was drawn: what it
            // holds now goes up over the drawing
            self.begin()?;
            let staged = self.stage(memory, &key, format)?;
            let depth = key.depth.is_some();
            let target = self.targets.get_mut(&key).expect("a target just checked");
            target.dirty = false;
            let (image, layout) = (target.image, target.layout);
            let mut fresh = Target { id: target.id, image, memory: target.memory, view: target.view, layout, dirty: false, loaded: true, key, format };
            self.transition(&mut fresh, if depth { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }, &staged, false)?;
            self.targets.insert(key, fresh);
            return Ok(());
        }
        let depth = key.depth.is_some();
        // a 16-bit depth buffer goes in the same 24-bit image as the rest,
        // its samples scaled, so one render pass takes them all
        let vk_format = if depth { DEPTH_FORMAT } else { COLOR_FORMAT };
        let usage = if depth {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED
        };
        let (image, memory_handle, view) = create_image(&self.instance, &self.device, self.physical, key.width, key.height, 1, vk_format, usage)?;
        let id = self.next_id;
        self.next_id += 1;
        let mut target = Target { id, image, memory: memory_handle, view, layout: vk::ImageLayout::UNDEFINED, dirty: false, loaded: true, key, format };
        // what guest memory holds goes up first, so the draw adds to it
        self.begin()?;
        let staged = self.stage(memory, &key, format)?;
        self.transition(&mut target, if depth { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }, &staged, false)?;
        self.targets.insert(key, target);
        Ok(())
    }

    /// brings a buffer from guest memory into a staging buffer, as the
    /// image wants it.
    fn stage<M: GpuMemory>(&mut self, memory: &mut M, key: &TargetKey, format: ColorFormat) -> Result<Staged, String> {
        let (width, height) = (key.width, key.height);
        let (buffer, allocation, pointer) = self.staging_buffer((width * height * 4) as vk::DeviceSize)?;
        let mut bytes = vec![0u8; (width * height * 4) as usize];
        match (key.depth, key.depth) {
            (None, _) => {
                let bpp = format.bytes_per_pixel() as u32;
                let mut pixels = vec![0u8; (width * height * bpp) as usize];
                memory.read(key.addr, &mut pixels);
                for y in 0..height {
                    for x in 0..width {
                        let at = morton_offset(x, y, width, bpp) as usize;
                        let pixel = format.decode(&pixels[at..at + bpp as usize]);
                        let out = ((y * width + x) * 4) as usize;
                        bytes[out..out + 4].copy_from_slice(&pixel);
                    }
                }
            }
            (Some(sample), _) => {
                let mut pixels = vec![0u8; (width * height * sample as u32) as usize];
                memory.read(key.addr, &mut pixels);
                let sample = sample as u32;
                for y in 0..height {
                    for x in 0..width {
                        let at = morton_offset(x, y, width, sample) as usize;
                        let (depth, stencil) = match sample {
                            2 => {
                                let d16 = u16::from_le_bytes([pixels[at], pixels[at + 1]]) as u64;
                                (((d16 * 0xFF_FFFF + 0x7FFF) / 0xFFFF) as u32, 0)
                            }
                            3 => (u32::from_le_bytes([pixels[at], pixels[at + 1], pixels[at + 2], 0]), 0),
                            _ => {
                                let word = u32::from_le_bytes([pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]);
                                (word & 0xFF_FFFF, word >> 24)
                            }
                        };
                        let out = ((y * width + x) * 4) as usize;
                        let _ = sample;
                        bytes[out..out + 4].copy_from_slice(&(depth << 8 | stencil).to_ne_bytes());
                    }
                }
            }
        }
        // SAFETY: the staging buffer is mapped for its life and covers the rows
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len());
        }
        self.flush_staging(allocation)?;
        let _ = allocation;
        Ok(Staged { buffer })
    }

    /// moves an image to a layout, uploading the staged bytes when it is
    /// their new content.
    fn transition(&mut self, target: &mut Target, to: vk::ImageLayout, staged: &Staged, download: bool) -> Result<(), String> {
        let depth = target.key.depth.is_some();
        let aspect = if depth { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::COLOR };
        // SAFETY: the command buffer is recording and the image is the target's
        unsafe {
            let device = &self.device;
            let cb = self.command_buffer;
            let barrier = |layout: vk::ImageLayout| vk::ImageMemoryBarrier::default()
                .old_layout(target.layout)
                .new_layout(layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(target.image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            let stage = if depth { vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS } else { vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT };
            if !download {
                let transfer = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                if target.layout != transfer {
                    let barriers = [barrier(transfer)];
                    device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &barriers);
                }
                let (offset, extent) = (vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Extent3D { width: target.key.width, height: target.key.height, depth: 1 });
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                    .image_offset(offset)
                    .image_extent(extent);
                device.cmd_copy_buffer_to_image(cb, staged.buffer, target.image, transfer, &[region]);
            }
            let barriers = [barrier(to)];
            device.cmd_pipeline_barrier(
                cb,
                if !download { vk::PipelineStageFlags::TRANSFER } else { stage },
                stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
        target.layout = to;
        Ok(())
    }

    /// brings a buffer's drawing back down to guest memory.
    fn download<M: GpuMemory>(&mut self, memory: &mut M, key: TargetKey) -> Result<(), String> {
        if !self.targets.contains_key(&key) {
            return Ok(());
        }
        self.begin()?;
        let (width, height) = (key.width, key.height);
        let depth = key.depth.is_some();
        let (image, layout) = {
            let target = self.targets.get(&key).expect("a target just checked");
            (target.image, target.layout)
        };
        let aspect = if depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
        let bytes = (width * height * 4) as vk::DeviceSize;
        let (buffer, allocation, pointer) = self.staging_buffer(bytes)?;
        // SAFETY: the command buffer is recording and the image is the target's
        unsafe {
            let device = &self.device;
            let cb = self.command_buffer;
            let to_transfer = vk::ImageMemoryBarrier::default()
                .old_layout(layout)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_transfer]);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_extent(vk::Extent3D { width, height, depth: 1 });
            device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buffer, &[region]);
            let back = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(if depth { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[back]);
        }
        if let Some(target) = self.targets.get_mut(&key) {
            target.layout = if depth { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL };
        }
        self.submit()?;
        self.invalidate_staging(allocation)?;
        let mut pixels = vec![0u8; bytes as usize];
        // SAFETY: the staging buffer is mapped and the GPU finished with it
        unsafe {
            std::ptr::copy_nonoverlapping(pointer, pixels.as_mut_ptr(), pixels.len());
        }
        let format = self.targets.get(&key).map_or(ColorFormat::Rgba8, |target| target.format);
        let mut out = vec![0u8; (width * height * if depth { key.depth.unwrap_or(4) as u32 } else { format.bytes_per_pixel() as u32 }) as usize];
        match key.depth {
            None => {
                let bpp = format.bytes_per_pixel();
                for y in 0..height {
                    for x in 0..width {
                        let at = ((y * width + x) * 4) as usize;
                        let pixel = [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]];
                        let out_at = morton_offset(x, y, width, bpp as u32) as usize;
                        format.encode(pixel, &mut out[out_at..out_at + bpp]);
                    }
                }
            }
            Some(sample) => {
                let sample = sample as u32;
                for y in 0..height {
                    for x in 0..width {
                        let at = ((y * width + x) * 4) as usize;
                        let out_at = morton_offset(x, y, width, sample) as usize;
                        match sample {
                            2 => {
                                let word = u32::from_ne_bytes([pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]);
                                let d24 = word >> 8;
                                let d16 = ((d24 as u64 * 0xFFFF + 0x7F_FFFF) / 0xFF_FFFF) as u16;
                                out[out_at..out_at + 2].copy_from_slice(&d16.to_le_bytes());
                            }
                            3 => {
                                let word = u32::from_ne_bytes([pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]);
                                out[out_at..out_at + 3].copy_from_slice(&(word >> 8).to_le_bytes()[..3]);
                            }
                            _ => {
                                let word = u32::from_ne_bytes([pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]);
                                let depth24 = word >> 8;
                                out[out_at..out_at + 4].copy_from_slice(&(depth24 | (word & 0xFF) << 24).to_ne_bytes());
                            }
                        }
                    }
                }
            }
        }
        memory.write(key.addr, &out);
        self.release_staging(buffer, allocation);
        Ok(())
    }

    // -- images ------------------------------------------------------------

    fn make_texture(&mut self, width: u32, height: u32, levels: u32, texels: &[u8]) -> Result<Texture, String> {
        let (image, memory, view) = create_image(
            &self.instance,
            &self.device,
            self.physical,
            width,
            height,
            levels.max(1),
            COLOR_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let mut texture = Texture { image, memory, view, layout: vk::ImageLayout::UNDEFINED, _texels: None };
        self.begin()?;
        // SAFETY: the command buffer is recording
        unsafe {
            let device = &self.device;
            let cb = self.command_buffer;
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: levels, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
        }
        let mut offset = 0usize;
        for level in 0..levels {
            let (w, h) = ((width >> level).max(1), (height >> level).max(1));
            let size = (w * h * 4) as usize;
            let end = (offset + size).min(texels.len());
            let bytes = &texels[offset..end];
            if bytes.len() == size {
                let (buffer, allocation, pointer) = self.staging_buffer(size as vk::DeviceSize)?;
                // SAFETY: the staging buffer is mapped and covers the bytes
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, size);
                }
                self.flush_staging(allocation)?;
                // SAFETY: the command buffer is recording, the staging buffer holds the level
                unsafe {
                    let region = vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: level, base_array_layer: 0, layer_count: 1 })
                        .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
                    self.device.cmd_copy_buffer_to_image(self.command_buffer, buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
                }
                self.release_staging(buffer, allocation);
            }
            offset += size;
        }
        // SAFETY: the command buffer is recording
        unsafe {
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: levels, base_array_layer: 0, layer_count: 1 });
            self.device.cmd_pipeline_barrier(self.command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
        }
        texture.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        Ok(texture)
    }

    /// the image of a texture, made from its decoded texels the first time
    /// it is drawn with.
    fn texture(&mut self, bound: &BoundTexture) -> Result<vk::ImageView, String> {
        let key = (Arc::as_ptr(&bound.texels) as *const u8 as usize, bound.texels.len());
        if let Some(texture) = self.textures.get(&key) {
            return Ok(texture.view);
        }
        let bytes: Vec<u8> = bound.texels.iter().flatten().copied().collect();
        let mut texture = self.make_texture(bound.width, bound.height, bound.levels.max(1), &bytes)?;
        texture._texels = Some(bound.texels.clone());
        let view = texture.view;
        self.textures.insert(key, texture);
        Ok(view)
    }

    /// the picture a texture pack drew in place of a texture, read in when
    /// the draw first needs it.
    fn replacement(&mut self, material: &Arc<crate::pack::Material>) -> Result<Option<vk::ImageView>, String> {
        if let Some(texture) = self.replaced.get(&material.hash()) {
            return Ok(Some(texture.view));
        }
        let Some(picture) = material.picture(4096) else { return Ok(None) };
        let mut texels = Vec::with_capacity(picture.texels.len());
        for level in &picture.levels {
            let at = level.offset as usize;
            let end = at + (level.width * level.height * 4) as usize;
            if end <= picture.texels.len() {
                texels.extend_from_slice(&picture.texels[at..end]);
            }
        }
        let texture = self.make_texture(picture.width, picture.height, picture.levels.len().max(1) as u32, &texels)?;
        material.release();
        let view = texture.view;
        self.replaced.insert(material.hash(), texture);
        Ok(Some(view))
    }

    fn sampler(&mut self, bound: &BoundTexture) -> Result<vk::Sampler, String> {
        let key = SamplerKey {
            linear: bound.linear || bound.lod.linear,
            wrap_s: wrap_number(bound.wrap_s),
            wrap_t: wrap_number(bound.wrap_t),
            lod: (bound.lod.bias, bound.lod.min, bound.lod.max, bound.lod.linear, bound.lod.between),
        };
        if let Some(sampler) = self.samplers.get(&key) {
            return Ok(*sampler);
        }
        let wrap = |value: u32| match value {
            0 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            1 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            2 => vk::SamplerAddressMode::REPEAT,
            _ => vk::SamplerAddressMode::MIRRORED_REPEAT,
        };
        let filter = if key.linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
        // SAFETY: a device just made
        let sampler = unsafe {
            self.device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(filter)
                        .min_filter(filter)
                        .mipmap_mode(if key.lod.3 { vk::SamplerMipmapMode::LINEAR } else { vk::SamplerMipmapMode::NEAREST })
                        .address_mode_u(wrap(key.wrap_s))
                        .address_mode_v(wrap(key.wrap_t))
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .min_lod(0.0)
                        .max_lod(15.0)
                        .mip_lod_bias(0.0)
                        .max_anisotropy(1.0),
                    None,
                )
                .map_err(|error| format!("no sampler: {error}"))?
        };
        self.samplers.insert(key, sampler);
        Ok(sampler)
    }

    // -- pipelines ---------------------------------------------------------

    fn pipeline(&mut self, key: &PipelineKey, constants: &[u32; FRAGMENT_CONSTANTS], writes_depth: bool, lighting: &[u32]) -> Result<vk::Pipeline, String> {
        if let Some(pipeline) = self.pipelines.get(key) {
            return Ok(*pipeline);
        }
        let shader_key = ShaderKey { constants: *constants, writes_depth, lighting: lighting.to_vec() };
        let fragment = match self.fragment_modules.get(&shader_key) {
            Some(module) => *module,
            None => {
                let source = sources::fragment_source(constants, writes_depth, lighting);
                let module = make_module(&self.device, &source, naga::ShaderStage::Fragment)?;
                self.fragment_modules.insert(ShaderKey { constants: *constants, writes_depth, lighting: lighting.to_vec() }, module);
                module
            }
        };
        let depth = key.with_depth;
        let stencil = key.stencil;
        let blend = match key.blend {
            Some((config, _)) => {
                let factor = |raw: u32| FACTORS.get((raw & 0xF) as usize).copied().unwrap_or(vk::BlendFactor::ONE);
                let equation = |raw: u32| EQUATIONS.get((raw & 7) as usize).copied().unwrap_or(vk::BlendOp::ADD);
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(color_mask_flags(key.color_mask))
                    .blend_enable(true)
                    .src_color_blend_factor(factor(config >> 16))
                    .dst_color_blend_factor(factor(config >> 20))
                    .color_blend_op(equation(config))
                    .src_alpha_blend_factor(factor(config >> 24))
                    .dst_alpha_blend_factor(factor(config >> 28))
                    .alpha_blend_op(equation(config >> 8))
            }
            None => vk::PipelineColorBlendAttachmentState::default().color_write_mask(color_mask_flags(key.color_mask)).blend_enable(false),
        };
        let dynamic = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
        ];
        let binding = [vk::VertexInputBindingDescription::default().binding(0).stride(VERTEX_SIZE as u32).input_rate(vk::VertexInputRate::VERTEX)];
        let attributes: Vec<vk::VertexInputAttributeDescription> = (0..6)
            .map(|location| {
                vk::VertexInputAttributeDescription::default()
                    .location(location)
                    .binding(0)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(location * 16)
            })
            .collect();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(self.vertex_module).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(fragment).name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&binding).vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(key.depth.is_some())
            .depth_write_enable(key.depth_write)
            .depth_compare_op(vk::CompareOp::from_raw(key.depth.unwrap_or(vk::CompareOp::ALWAYS.as_raw() as u32) as i32))
            .depth_bounds_test_enable(false)
            .stencil_test_enable(stencil.is_some())
            .front(match stencil {
                Some((compare, ops, _)) => vk::StencilOpState::default()
                    .fail_op(STENCIL_OPS[(ops & 7) as usize])
                    .pass_op(STENCIL_OPS[((ops >> 8) & 7) as usize])
                    .depth_fail_op(STENCIL_OPS[((ops >> 4) & 7) as usize])
                    .compare_op(COMPARES[((compare >> 4) & 7) as usize]),
                None => vk::StencilOpState::default(),
            })
            .back(match stencil {
                Some((compare, ops, _)) => vk::StencilOpState::default()
                    .fail_op(STENCIL_OPS[(ops & 7) as usize])
                    .pass_op(STENCIL_OPS[((ops >> 8) & 7) as usize])
                    .depth_fail_op(STENCIL_OPS[((ops >> 4) & 7) as usize])
                    .compare_op(COMPARES[((compare >> 4) & 7) as usize]),
                None => vk::StencilOpState::default(),
            });
        let blend_attachments = [blend];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.pipeline_layout)
            .render_pass(if depth { self.render_pass_depth } else { self.render_pass_plain })
            .subpass(0);
        // SAFETY: every stage was made here and the render pass matches
        let pipelines = unsafe { self.device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None) };
        let pipeline = match pipelines {
            Ok(pipelines) => pipelines[0],
            Err((_, error)) => return Err(format!("the Vulkan pipeline would not build: {error}")),
        };
        self.pipelines.insert(
            PipelineKey {
                shader: key.shader,
                writes_depth: key.writes_depth,
                lighting: key.lighting.clone(),
                with_depth: key.with_depth,
                blend: key.blend,
                depth: key.depth,
                depth_write: key.depth_write,
                stencil: key.stencil,
                color_mask: key.color_mask,
            },
            pipeline,
        );
        Ok(pipeline)
    }

    fn framebuffer(
        &mut self,
        color_key: TargetKey,
        color_id: u64,
        color_view: vk::ImageView,
        depth: Option<&TargetRef>,
    ) -> Result<vk::Framebuffer, String> {
        let key = (color_id, depth.map_or(0, |depth| depth.id));
        if let Some(framebuffer) = self.framebuffers.get(&key) {
            return Ok(*framebuffer);
        }
        let attachments = match depth {
            Some(depth) => vec![color_view, depth.view],
            None => vec![color_view],
        };
        // SAFETY: the views are the buffers' and the render pass expects them
        let framebuffer = unsafe {
            self.device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(if depth.is_some() { self.render_pass_depth } else { self.render_pass_plain })
                        .attachments(&attachments)
                        .width(color_key.width)
                        .height(color_key.height)
                        .layers(1),
                    None,
                )
                .map_err(|error| format!("no framebuffer: {error}"))?
        };
        self.framebuffers.insert(key, framebuffer);
        Ok(framebuffer)
    }

    fn descriptor_set(&mut self, images: [vk::ImageView; 6], samplers: [vk::Sampler; 6], uniforms: vk::DeviceSize, tables: vk::DeviceSize) -> Result<vk::DescriptorSet, String> {
        // SAFETY: a pool with room for the sets of a batch
        let set = unsafe {
            self.device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.descriptor_pool).set_layouts(&[self.descriptor_layout]))
                .map_err(|error| format!("no descriptor set: {error}"))?[0]
        };
        let image_infos: Vec<vk::DescriptorImageInfo> = images.iter().map(|view| vk::DescriptorImageInfo::default().image_view(*view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)).collect();
        let sampler_infos: Vec<vk::DescriptorImageInfo> = samplers.iter().map(|sampler| vk::DescriptorImageInfo::default().sampler(*sampler)).collect();
        let uniform = [vk::DescriptorBufferInfo::default().buffer(self.ring).offset(uniforms).range((UNIFORM_WORDS * 4) as vk::DeviceSize)];
        let table = [vk::DescriptorBufferInfo::default().buffer(self.ring).offset(tables).range(TABLES_SIZE as vk::DeviceSize)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&image_infos[..3]),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&image_infos[3..]),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(6).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uniform),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(7).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&table),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(16).descriptor_type(vk::DescriptorType::SAMPLER).image_info(&sampler_infos[..3]),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(19).descriptor_type(vk::DescriptorType::SAMPLER).image_info(&sampler_infos[3..]),
        ];
        // SAFETY: the set was made from this layout and the buffers and views
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }

    // -- staging -----------------------------------------------------------

    fn staging_buffer(&mut self, size: vk::DeviceSize) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        let (buffer, memory, pointer) = create_buffer(&self.instance, &self.device, self.physical, size, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER, true)?;
        self.staging.push((buffer, memory));
        Ok((buffer, memory, pointer))
    }

    fn flush_staging(&mut self, memory: vk::DeviceMemory) -> Result<(), String> {
        if self.non_coherent {
            // SAFETY: a range inside the buffer the memory backs
            unsafe {
                self.device
                    .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default().memory(memory).offset(0).size(vk::WHOLE_SIZE)])
                    .map_err(|error| format!("could not flush staging memory: {error}"))?;
            }
        }
        Ok(())
    }

    fn invalidate_staging(&mut self, memory: vk::DeviceMemory) -> Result<(), String> {
        if self.non_coherent {
            // SAFETY: as above
            unsafe {
                self.device
                    .invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default().memory(memory).offset(0).size(vk::WHOLE_SIZE)])
                    .map_err(|error| format!("could not read staging memory: {error}"))?;
            }
        }
        Ok(())
    }

    fn release_staging(&mut self, buffer: vk::Buffer, _memory: vk::DeviceMemory) {
        // the buffer stays until the batch is submitted, which is when the
        // GPU stops reading it
        let _ = buffer;
    }

    fn destroy_texture(&mut self, texture: Texture) {
        // SAFETY: the texture's own handles, and the GPU is idle
        unsafe {
            self.device.destroy_image_view(texture.view, None);
            self.device.destroy_image(texture.image, None);
            self.device.free_memory(texture.memory, None);
        }
    }
}

/// a target's handles, for the framebuffer and the copy that goes with it.
#[derive(Clone, Copy)]
struct TargetRef {
    id: u64,
    view: vk::ImageView,
}

impl TargetRef {
    fn of(target: &Target) -> TargetRef {
        TargetRef { id: target.id, view: target.view }
    }
}

impl Target {
    fn clone_ref(&self) -> TargetRef {
        TargetRef { id: self.id, view: self.view }
    }
}

/// a staging buffer holding an upload, and what frees it.
struct Staged {
    buffer: vk::Buffer,
}

fn color_mask_flags(mask: u32) -> vk::ColorComponentFlags {
    let mut flags = vk::ColorComponentFlags::empty();
    if mask & 1 != 0 {
        flags |= vk::ColorComponentFlags::R;
    }
    if mask & 2 != 0 {
        flags |= vk::ColorComponentFlags::G;
    }
    if mask & 4 != 0 {
        flags |= vk::ColorComponentFlags::B;
    }
    if mask & 8 != 0 {
        flags |= vk::ColorComponentFlags::A;
    }
    flags
}

fn wrap_number(wrap: Wrap) -> u32 {
    match wrap {
        Wrap::ClampToEdge => 0,
        Wrap::ClampToBorder => 1,
        Wrap::Repeat => 2,
        Wrap::MirroredRepeat => 3,
    }
}

/// the draw's uniform words as the bytes a uniform buffer wants.
fn memory_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// the draw's triangles as the vertices the vertex program reads, y turned
/// over for Vulkan's window, whose rows run down from the top.
fn vertex_bytes(triangles: &[[Screen; 3]], (width, height): (f32, f32), depth_map: DepthMap) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(triangles.len() * 3 * VERTEX_SIZE as usize);
    for vertex in triangles.iter().flatten() {
        let w = 1.0 / vertex.inv_w;
        let x = vertex.x / width * 2.0 - 1.0;
        let y = 1.0 - vertex.y / height * 2.0;
        let depth = vertex.z * depth_map.scale + depth_map.offset;
        let (c, t, q, view) = (vertex.color_over_w, vertex.texcoords_over_w, vertex.quaternion_over_w, vertex.view_over_w);
        let values: [f32; 24] = [
            x * w,
            y * w,
            0.0,
            w,
            c[0] * w,
            c[1] * w,
            c[2] * w,
            c[3] * w,
            t[0][0] * w,
            t[0][1] * w,
            t[1][0] * w,
            t[1][1] * w,
            t[2][0] * w,
            t[2][1] * w,
            depth,
            0.0,
            q[0] * w,
            q[1] * w,
            q[2] * w,
            q[3] * w,
            view[0] * w,
            view[1] * w,
            view[2] * w,
            view[3] * w,
        ];
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// GLSL the OpenGL ES renderer builds, as Vulkan takes it: desktop GLSL
/// rather than GLSL ES, the sampler uniforms split into the images and the
/// samplers Vulkan binds apart, and the half unpacking spelled out so no
/// device needs its 16-bit floats, which Vulkan 1.1 does not have.
fn vulkan_glsl(source: &str) -> String {
    let mut out = source
        .replace("#version 310 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n", "#version 450\n")
        .replace("#version 310 es\nprecision highp float;\n", "#version 450\n");
    for (unit, name) in ["texture0", "texture1", "texture2", "alpha0", "alpha1", "alpha2"].into_iter().enumerate() {
        let declaration = format!("layout(binding = {unit}) uniform sampler2D {name};");
        let replacement = format!("layout(binding = {unit}) uniform texture2D {name}_tex;\nlayout(binding = {}) uniform sampler {name}_smp;\n#define {name} sampler2D({name}_tex, {name}_smp)", 16 + unit);
        out = out.replace(&declaration, &replacement);
    }
    out = out
        .replace("layout(std140, binding = 3) uniform Draw {", "layout(std140, binding = 6) uniform Draw {")
        .replace("layout(std430, binding = 4) readonly buffer Tables {", "layout(std430, binding = 7) readonly buffer Tables {");
    if let Some(at) = out.find('\n') {
        let helper = "\nvec2 zk_unpack_half2(uint v) {\n    vec2 out_values;\n    for (int i = 0; i < 2; ++i) {\n        uint h = (v >> (uint(i) * 16u)) & 0xFFFFu;\n        uint sign = (h >> 15) & 1u;\n        uint exponent = (h >> 10) & 0x1Fu;\n        uint mantissa = h & 0x3FFu;\n        uint bits;\n        if (exponent == 0u) {\n            bits = (mantissa == 0u) ? (sign << 31) : (sign << 31) | 0x3F800000u;\n        } else if (exponent == 0x1Fu) {\n            bits = (sign << 31) | 0x7F800000u | (mantissa << 13);\n        } else {\n            bits = (sign << 31) | ((exponent + 112u) << 23) | (mantissa << 13);\n        }\n        out_values[i] = uintBitsToFloat(bits);\n    }\n    return out_values;\n}\n#define unpackHalf2x16 zk_unpack_half2\n";
        out.insert_str(at + 1, helper);
    }
    out
}

/// GLSL to SPIR-V, or why naga would not.
fn compile(source: &str, stage: naga::ShaderStage) -> Result<Vec<u32>, String> {
    use naga::back::spv;
    use naga::front::glsl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let source = vulkan_glsl(source);
    let run = || -> Result<Vec<u32>, String> {
        let module = glsl::Frontend::default()
            .parse(&glsl::Options::from(stage), &source)
            .map_err(|errors| errors.emit_to_string_with_path(&source, "a Vulkan shader"))?;
        let info = Validator::new(ValidationFlags::all(), Capabilities::empty()).validate(&module).map_err(|error| error.emit_to_string_with_path(&source, "a Vulkan shader"))?;
        let options = spv::Options { lang_version: (1, 0), flags: spv::WriterFlags::empty(), force_loop_bounding: false, ..spv::Options::default() };
        let pipeline = spv::PipelineOptions { shader_stage: stage, entry_point: "main".into() };
        spv::write_vec(&module, &info, &options, Some(&pipeline)).map_err(|error| error.to_string())
    };
    std::panic::catch_unwind(run).unwrap_or_else(|_| Err("naga panicked".to_owned()))
}

fn make_module(device: &ash::Device, source: &str, stage: naga::ShaderStage) -> Result<vk::ShaderModule, String> {
    let words = compile(source, stage)?;
    // SAFETY: the words are SPIR-V naga wrote and validated
    unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }.map_err(|error| format!("no shader module: {error}"))
}

fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32), String> {
    // SAFETY: a live instance, asked for its devices
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|error| format!("no Vulkan devices: {error}"))?;
    for device in devices {
        // SAFETY: a device of this instance
        let properties = unsafe { instance.get_physical_device_properties(device) };
        if properties.api_version < vk::make_api_version(0, 1, 1, 0) {
            continue;
        }
        // SAFETY: as above
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in families.iter().enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                let format = unsafe { instance.get_physical_device_format_properties(device, DEPTH_FORMAT) };
                let depth = format.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
                if depth {
                    log::info!(
                        "vulkan: {} (api {}.{}.{}), queue family {index}",
                        properties.device_name_as_c_str().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default(),
                        properties.api_version >> 22,
                        (properties.api_version >> 12) & 0x3FF,
                        properties.api_version & 0xFFF,
                    );
                    return Ok((device, index as u32));
                }
            }
        }
    }
    Err("no Vulkan device draws into a depth buffer".to_owned())
}

fn coherent_host_memory(properties: &vk::PhysicalDeviceMemoryProperties) -> bool {
    (0..properties.memory_type_count as usize).any(|index| {
        let flags = properties.memory_types[index].property_flags;
        flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
    })
}

fn find_memory_type(instance: &ash::Instance, device: vk::PhysicalDevice, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32, String> {
    // SAFETY: a live device
    let properties = unsafe { instance.get_physical_device_memory_properties(device) };
    for index in 0..properties.memory_type_count {
        if bits & (1 << index) != 0 && properties.memory_types[index as usize].property_flags.contains(flags) {
            return Ok(index);
        }
    }
    Err("no memory of the kind the renderer needs".to_owned())
}

fn create_command_pool(device: &ash::Device, family: u32) -> Result<vk::CommandPool, String> {
    // SAFETY: the family has a graphics queue
    unsafe {
        device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
            .map_err(|error| format!("no command pool: {error}"))
    }
}

fn create_ring(instance: &ash::Instance, device: &ash::Device, physical: vk::PhysicalDevice, size: vk::DeviceSize) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8, vk::DeviceSize), String> {
    let (buffer, memory, pointer) = create_buffer(instance, device, physical, size, vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, true)?;
    Ok((buffer, memory, pointer, size))
}

fn create_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mapped: bool,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
    // SAFETY: a buffer of this device, and memory of a kind it has
    unsafe {
        let buffer = device
            .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)
            .map_err(|error| format!("no buffer: {error}"))?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = find_memory_type(instance, physical, requirements.memory_type_bits, flags)?;
        let memory = device
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index), None)
            .map_err(|error| format!("no buffer memory: {error}"))?;
        device.bind_buffer_memory(buffer, memory, 0).map_err(|error| format!("could not bind buffer memory: {error}"))?;
        let pointer = if mapped { device.map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty()).map_err(|error| format!("could not map a buffer: {error}"))? as *mut u8 } else { std::ptr::null_mut() };
        Ok((buffer, memory, pointer))
    }
}

fn create_image(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    width: u32,
    height: u32,
    levels: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), String> {
    // SAFETY: an image of this device, and memory of a kind it has
    unsafe {
        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(levels.max(1))
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .map_err(|error| format!("no image: {error}"))?;
        let requirements = device.get_image_memory_requirements(image);
        let index = find_memory_type(instance, physical, requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let memory = device
            .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index), None)
            .map_err(|error| format!("no image memory: {error}"))?;
        device.bind_image_memory(image, memory, 0).map_err(|error| format!("could not bind image memory: {error}"))?;
        let aspect = match format {
            vk::Format::D24_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            vk::Format::D16_UNORM => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        };
        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: if aspect == vk::ImageAspectFlags::COLOR { levels.max(1) } else { 1 }, base_array_layer: 0, layer_count: 1 }),
                None,
            )
            .map_err(|error| format!("no image view: {error}"))?;
        Ok((image, memory, view))
    }
}

fn create_descriptor_layout(device: &ash::Device) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout), String> {
    let mut bindings = Vec::new();
    for binding in 0..6u32 {
        bindings.push(vk::DescriptorSetLayoutBinding::default().binding(binding).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT));
    }
    for binding in 16..22u32 {
        bindings.push(vk::DescriptorSetLayoutBinding::default().binding(binding).descriptor_type(vk::DescriptorType::SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT));
    }
    bindings.push(vk::DescriptorSetLayoutBinding::default().binding(6).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT));
    bindings.push(vk::DescriptorSetLayoutBinding::default().binding(7).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT));
    // SAFETY: a device just made
    unsafe {
        let layout = device
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
            .map_err(|error| format!("no descriptor set layout: {error}"))?;
        let pipeline = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[layout]), None)
            .map_err(|error| format!("no pipeline layout: {error}"))?;
        Ok((layout, pipeline))
    }
}

fn create_render_pass(device: &ash::Device, depth: bool) -> Result<vk::RenderPass, String> {
    let color = vk::AttachmentDescription::default()
        .format(COLOR_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let mut attachments = vec![color];
    let mut references = vec![vk::AttachmentReference::default().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    if depth {
        attachments.push(
            vk::AttachmentDescription::default()
                .format(DEPTH_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::LOAD)
                .stencil_store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
        references.push(vk::AttachmentReference::default().attachment(1).layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL));
    }
    let mut subpass = vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS).color_attachments(&references[..1]);
    if depth {
        subpass = subpass.depth_stencil_attachment(&references[1]);
    }
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
    // SAFETY: a device just made
    unsafe {
        device
            .create_render_pass(&vk::RenderPassCreateInfo::default().attachments(&attachments).subpasses(&[subpass]).dependencies(&[dependency]), None)
            .map_err(|error| format!("no render pass: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the fragment programs the renderer builds are what Vulkan takes:
    /// each one naga would turn into SPIR-V, checked without a GPU.
    #[test]
    fn the_fragment_programs_translate() {
        let mut failures = Vec::new();
        for lighting in [0u32, 1] {
            for writes_depth in [false, true] {
                for first in [0u32, 0x0012_3456] {
                    let mut constants = [0u32; FRAGMENT_CONSTANTS];
                    constants[26] = lighting;
                    constants[0] = first;
                    let source = sources::fragment_source(&constants, writes_depth, &[]);
                    if let Err(error) = compile(&source, naga::ShaderStage::Fragment) {
                        failures.push(format!("lighting {lighting}, depth {writes_depth}, first {first:x}: {error}"));
                    }
                }
            }
        }
        assert!(failures.is_empty(), "fragment programs Vulkan would not take: {failures:#?}");
    }

    #[test]
    fn the_vertex_program_translates() {
        let source = sources::VERTEX_SOURCE.replace("#version 310 es\n", "#version 450\n");
        if let Err(error) = compile(&source, naga::ShaderStage::Vertex) {
            panic!("the vertex program Vulkan would not take: {error}");
        }
    }
}
