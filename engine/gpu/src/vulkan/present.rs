//! the screens and the menu on a Vulkan 1.1 swapchain.
//!
//! the engine draws the 3D in software and hands this the pictures it made —
//! the same `PanelImage::Pixels` an OpenGL ES presenter gets — and this puts
//! them on the window: instance, surface, device, swapchain, one render pass
//! with a colour attachment, a pipeline, a descriptor set a screen, classic
//! barriers, semaphores and fences, present. the menu and the boxes over the
//! screens come as the `Overlay` meshes GLES draws, and go in the same pass
//! through a second pipeline that blends them over the screens.
//!
//! nothing here is Vulkan 1.3: the uniform block and the sampled picture are
//! ordinary descriptor set bindings (no push descriptors), a real render pass
//! does the colour attachment's layout transitions (no dynamic rendering),
//! and the barriers are vkCmdPipelineBarrier, not its 2 form.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

use ash::vk;

use crate::{Overlay, OverlayTexture, PanelImage};

/// frames the CPU may get ahead of the GPU, each with its own commands, its
/// fence and its acquire semaphore.
const FRAMES: usize = 2;
/// the colour formats taken when the surface offers them, best first.
const FORMATS: [vk::Format; 3] = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM, vk::Format::A2B10G10R10_UNORM_PACK32];
/// the format the engine's screen pixels are in, straight RGBA8.
const SCREEN_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// a uniform block is two vec4: the quad's place and the picture's uv.
const UNIFORM_BYTES: u64 = 32;
/// an overlay corner: its place in clip space, its uv and its colour.
const CORNER_STRIDE: u64 = 20;
/// the sets the menu's first pool is made for: a menu's few, with room to
/// spare. every one taken makes another pool twice the size, so no number of
/// textures runs out.
const OVERLAY_SETS: u32 = 16;
/// the id the touch cursor's one white pixel is kept under, which no overlay
/// the frontend sends uses.
const CURSOR_TEXTURE: u64 = u64::MAX;

/// the shaders, WGSL turned into SPIR-V by naga when the presenter is made.
const SHADERS: &str = r#"
struct Uniforms {
    // left, top, width and height in clip space, y down
    rect: vec4<f32>,
    // where in the picture the quad starts and how much of it it takes
    uv: vec4<f32>,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var picture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct FsIn {
    @location(0) uv: vec2<f32>,
};

// one triangle over the quad's place, its corners reaching past it
@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let across = corners[index] * 0.5 + vec2<f32>(0.5, 0.5);
    var out: VsOut;
    out.position = vec4<f32>(uniforms.rect.xy + across * uniforms.rect.zw, 0.0, 1.0);
    out.uv = uniforms.uv.xy + across * uniforms.uv.zw;
    return out;
}

@fragment
fn fs(in: FsIn) -> @location(0) vec4<f32> {
    return textureSample(picture, samp, in.uv);
}
"#;

/// the overlay's shaders: the corners come in as clip space already, so the
/// vertex stage passes them and the picture's uv along, and the fragment one
/// blends the picture's premultiplied colour with the corner's.
const OVERLAY_SHADERS: &str = r#"
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var picture: texture_2d<f32>;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(@location(0) position: vec2<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec4<f32>) -> Corner {
    var out: Corner;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

struct FsIn {
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@fragment
fn fs(in: FsIn) -> @location(0) vec4<f32> {
    return textureSample(picture, samp, in.uv) * in.color;
}
"#;

/// a corner of an overlay mesh as the frame's vertex buffer takes it: clip
/// space, its uv and its premultiplied colour, which is what
/// `crate::OverlayVertex` turns into once it is placed.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Corner {
    position: [f32; 2],
    uv: [f32; 2],
    color: [u8; 4],
}

/// a screen's picture and where it goes.
struct Screen {
    staging: Option<Staging>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    size: (u32, u32),
    /// the image has held a picture before, so a barrier takes it from what it
    /// was read as rather than from nothing.
    initialized: bool,
    /// this frame has a picture for the screen at all.
    shown: bool,
    /// left, top, width and height, in the layout's pixels.
    area: [i32; 4],
}

impl Screen {
    fn empty() -> Screen {
        Screen {
            staging: None,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            size: (0, 0),
            initialized: false,
            shown: false,
            area: [0, 0, 0, 0],
        }
    }
}

/// a host-visible buffer the picture's bytes go to, kept mapped.
struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: usize,
}

impl Staging {
    fn empty() -> Staging {
        Staging { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), mapped: std::ptr::null_mut(), capacity: 0 }
    }
}

/// a uniform block's buffer, kept mapped.
struct Uniform {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
}

/// one of the overlay's textures on the device, kept by its id, with the set
/// its meshes bind it through.
struct Picture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
    set: vk::DescriptorSet,
    /// the pool the set came from, whose room it gives back when the texture
    /// goes.
    pool: usize,
    size: [u32; 2],
}

/// the sets the menu's textures take: a pool made for so many, and one more
/// pool, twice the size, when they are all taken. a set a dropped texture
/// gives back is taken up again, so a session's textures never run out and the
/// pools stop growing once the churn settles.
struct SetBudget {
    /// the sets the first pool is made for.
    first: u32,
    /// each pool's sets and how many of them are taken.
    pools: Vec<(u32, u32)>,
}

impl SetBudget {
    fn new(first: u32) -> SetBudget {
        SetBudget { first: first.max(1), pools: Vec::new() }
    }

    /// the pool the next set is taken from, making one when every pool is
    /// full.
    fn take(&mut self) -> usize {
        match self.pools.iter().position(|(capacity, taken)| taken < capacity) {
            Some(pool) => {
                self.pools[pool].1 += 1;
                pool
            }
            None => {
                let capacity = match self.pools.last() {
                    Some((capacity, _)) => capacity.saturating_mul(2).max(self.first),
                    None => self.first,
                };
                self.pools.push((capacity, 1));
                self.pools.len() - 1
            }
        }
    }

    /// a set of a pool given back, for the next take to use.
    fn give(&mut self, pool: usize) {
        if let Some((capacity, taken)) = self.pools.get_mut(pool) {
            *taken = taken.saturating_sub(1).min(*capacity);
        }
    }

    /// what a pool was made for.
    fn capacity(&self, pool: usize) -> Option<u32> {
        self.pools.get(pool).map(|(capacity, _)| *capacity)
    }

    /// how many of a pool's sets are taken.
    fn taken(&self, pool: usize) -> Option<u32> {
        self.pools.get(pool).map(|(_, taken)| *taken)
    }

    fn pools(&self) -> usize {
        self.pools.len()
    }
}

/// one mesh of an overlay as this frame draws it: where its indices start in
/// the frame's index buffer, how many, which vertex they start at, the set its
/// texture is bound through, and the part of the window it shows in.
struct OverlayDraw {
    first: u32,
    count: u32,
    vertex_offset: i32,
    set: vk::DescriptorSet,
    scissor: vk::Rect2D,
}

/// the screens on a Vulkan 1.1 device's swapchain.
pub struct Presenter {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_fn: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    swapchain_fn: ash::khr::swapchain::Device,
    queue: vk::Queue,
    family: u32,

    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    /// the layout the areas are in, and how it fits the window.
    layout: (i32, i32),
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    /// whether each swapchain image was drawn into since it was made, which
    /// picks the barrier's old layout.
    drawn: Vec<bool>,
    framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// the menu's pipeline: the same pass and layout, blended over the screens.
    overlay_pipeline: vk::Pipeline,
    /// the pools the menu's textures take their sets from, made as they fill.
    overlay_pools: Vec<vk::DescriptorPool>,
    overlay_budget: SetBudget,
    overlay_textures: HashMap<u64, Picture>,
    /// the set the touch cursor's white pixel is bound through.
    cursor_set: vk::DescriptorSet,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    // a frame's own set a screen, so a frame being recorded never rewrites the
    // descriptors of the frame still on the GPU
    sets: [[vk::DescriptorSet; 2]; FRAMES],
    nearest: vk::Sampler,
    linear: vk::Sampler,

    // a frame's own images, staging buffers and uniform blocks a screen: the
    // GPU may still be reading the other frame's
    screens: [[Screen; 2]; FRAMES],
    uniforms: [[Uniform; 2]; FRAMES],
    /// a frame's own corners, indices and meshes of the overlay: the GPU may
    /// still be reading the other frame's
    overlay_vertices: [Staging; FRAMES],
    overlay_indices: [Staging; FRAMES],
    overlay_draws: [Vec<OverlayDraw>; FRAMES],
    command_pool: vk::CommandPool,
    commands: Vec<vk::CommandBuffer>,
    /// waited on when a frame starts, one a frame.
    available: Vec<vk::Semaphore>,
    /// signalled when the frame drawing into an image finished, one an image.
    finished: Vec<vk::Semaphore>,
    fences: Vec<vk::Fence>,
    frame: usize,
}

impl Presenter {
    /// a presenter on the window `set_window` gave last, or why there is none:
    /// no library, no surface, no device whose queue draws and shows, no
    /// swapchain.
    pub fn new() -> Result<Presenter, String> {
        let window = super::window();
        if window.is_null() {
            return Err("no window to show the screens in".into());
        }
        let entry = super::entry()?;
        let instance = super::instance(&entry, &surface_extensions())?;
        let surface_fn = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = match unsafe { make_surface(&entry, &instance, window) } {
            Ok(surface) => surface,
            Err(error) => {
                // SAFETY: nothing made from the instance outlives this
                unsafe { instance.destroy_instance(None) };
                return Err(error);
            }
        };
        Presenter::build(entry, instance, surface_fn, surface)
    }

    fn build(
        entry: ash::Entry,
        instance: ash::Instance,
        surface_fn: ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<Presenter, String> {
        let (physical, family) = match super::pick_presenting(&instance, &surface_fn, surface) {
            Ok(found) => found,
            Err(error) => {
                // SAFETY: tearing down what this call made
                unsafe {
                    surface_fn.destroy_surface(surface, None);
                    instance.destroy_instance(None);
                }
                return Err(error);
            }
        };
        let priority = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priority)];
        let extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_extension_names(&extensions);
        // SAFETY: a device on the GPU the surface takes, with a family the
        // instance listed for it and an extension the GPU offers
        let device = match unsafe { instance.create_device(physical, &info, None) } {
            Ok(device) => device,
            Err(error) => {
                // SAFETY: tearing down what this call made
                unsafe {
                    surface_fn.destroy_surface(surface, None);
                    instance.destroy_instance(None);
                }
                return Err(format!("no Vulkan device: {error}"));
            }
        };
        let swapchain_fn = ash::khr::swapchain::Device::new(&instance, &device);
        // SAFETY: a queue of the family the device was made with
        let queue = unsafe { device.get_device_queue(family, 0) };
        let mut presenter = Presenter {
            entry,
            instance,
            surface_fn,
            surface,
            physical,
            device,
            swapchain_fn,
            queue,
            family,
            swapchain: vk::SwapchainKHR::null(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D { width: 0, height: 0 },
            layout: (1, 1),
            images: Vec::new(),
            views: Vec::new(),
            drawn: Vec::new(),
            framebuffers: Vec::new(),
            render_pass: vk::RenderPass::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            overlay_pipeline: vk::Pipeline::null(),
            overlay_pools: Vec::new(),
            overlay_budget: SetBudget::new(OVERLAY_SETS),
            overlay_textures: HashMap::new(),
            cursor_set: vk::DescriptorSet::null(),
            overlay_vertices: std::array::from_fn(|_| Staging::empty()),
            overlay_indices: std::array::from_fn(|_| Staging::empty()),
            overlay_draws: std::array::from_fn(|_| Vec::new()),
            descriptor_pool: vk::DescriptorPool::null(),
            sets: [[vk::DescriptorSet::null(); 2]; FRAMES],
            nearest: vk::Sampler::null(),
            linear: vk::Sampler::null(),
            screens: std::array::from_fn(|_| [Screen::empty(), Screen::empty()]),
            uniforms: std::array::from_fn(|_| [Uniform::empty(), Uniform::empty()]),
            command_pool: vk::CommandPool::null(),
            commands: Vec::new(),
            available: Vec::new(),
            finished: Vec::new(),
            fences: Vec::new(),
            frame: 0,
        };
        if let Err(error) = presenter.prepare() {
            presenter.destroy_objects();
            // it let go of everything already, its drop running again would
            // hand the device and instance over twice
            std::mem::forget(presenter);
            return Err(error);
        }
        Ok(presenter)
    }

    /// the swapchain, then everything drawn into its images: the render pass
    /// needs its format, the framebuffers need both.
    fn prepare(&mut self) -> Result<(), String> {
        self.make_swapchain()?;
        let device = self.device.clone();
        self.render_pass = unsafe { create_render_pass(&device, self.format) }?;
        self.descriptor_layout = unsafe { create_descriptor_layout(&device) }?;
        self.pipeline_layout = unsafe { create_pipeline_layout(&device, self.descriptor_layout) }?;
        self.pipeline = unsafe { create_pipeline(&device, self.pipeline_layout, self.render_pass) }?;
        self.overlay_pipeline = unsafe { create_overlay_pipeline(&device, self.pipeline_layout, self.render_pass) }?;
        self.nearest = unsafe { create_sampler(&device, false) }?;
        self.linear = unsafe { create_sampler(&device, true) }?;
        let (pool, sets) = unsafe { create_descriptors(&device, self.descriptor_layout) }?;
        self.descriptor_pool = pool;
        self.sets = sets;
        for frame in 0..FRAMES {
            for screen in 0..2 {
                self.uniforms[frame][screen] = unsafe { create_uniform(&device, &self.instance, self.physical) }?;
            }
        }
        // SAFETY: a pool for commands on the family the device was made with
        self.command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|error| format!("no Vulkan command pool: {error}"))?;
        // SAFETY: buffers of a pool this presenter owns
        self.commands = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(FRAMES as u32),
            )
        }
        .map_err(|error| format!("no Vulkan command buffers: {error}"))?;
        for _ in 0..FRAMES {
            // SAFETY: plain semaphores and fences of this device
            let available = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(|error| format!("no Vulkan semaphore: {error}"))?;
            self.available.push(available);
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None) }
                .map_err(|error| format!("no Vulkan fence: {error}"))?;
            self.fences.push(fence);
        }
        // one white pixel, for the touch cursor's squares
        // SAFETY: an image, a view and a sampler of this device, a set of the
        // pool this call makes, and a copy into the image before any frame
        // samples it
        unsafe {
            let (image, memory) = create_screen_image(&device, &self.instance, self.physical, 1, 1)?;
            let view = create_view(&device, image, SCREEN_FORMAT)?;
            let sampler = create_sampler(&device, false)?;
            let (set, pool) = self.overlay_set(view, sampler)?;
            upload_overlay(
                &device,
                &self.instance,
                self.physical,
                self.command_pool,
                self.queue,
                image,
                true,
                [1, 1],
                [0, 0],
                &[255, 255, 255, 255],
            )?;
            self.cursor_set = set;
            self.overlay_textures.insert(CURSOR_TEXTURE, Picture { image, memory, view, sampler, set, pool, size: [1, 1] });
        }
        self.make_framebuffers()
    }

    /// a swapchain for the surface as it is now, its images and their views,
    /// and a semaphore an image.
    fn make_swapchain(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        // SAFETY: queries of the surface this presenter made
        let capabilities = unsafe { self.surface_fn.get_physical_device_surface_capabilities(self.physical, self.surface) }
            .map_err(|error| format!("no Vulkan surface capabilities: {error}"))?;
        let formats = unsafe { self.surface_fn.get_physical_device_surface_formats(self.physical, self.surface) }
            .map_err(|error| format!("no Vulkan surface formats: {error}"))?;
        let format = pick_format(&formats).ok_or("the surface offers no colour format this takes")?;
        let modes = unsafe { self.surface_fn.get_physical_device_surface_present_modes(self.physical, self.surface) }
            .map_err(|error| format!("no Vulkan present modes: {error}"))?;
        // fifo is the one every device has; a handheld wants a steady rate
        let mode = match modes.contains(&vk::PresentModeKHR::MAILBOX) {
            true => vk::PresentModeKHR::MAILBOX,
            false => vk::PresentModeKHR::FIFO,
        };
        let extent = current_extent(&capabilities);
        if extent.width == 0 || extent.height == 0 {
            return Err("the window has no size to show the screens in".into());
        }
        let mut count = capabilities.min_image_count + 1;
        if capabilities.max_image_count != 0 {
            count = count.min(capabilities.max_image_count);
        }
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(count)
            .image_format(format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(mode)
            .clipped(true)
            .old_swapchain(self.swapchain);
        // SAFETY: a swapchain on the surface, the old one handed over to be
        // replaced
        let swapchain = unsafe { self.swapchain_fn.create_swapchain(&info, None) }
            .map_err(|error| format!("no Vulkan swapchain: {error}"))?;
        if self.swapchain != vk::SwapchainKHR::null() {
            // SAFETY: the swapchain the new one replaced
            unsafe { self.swapchain_fn.destroy_swapchain(self.swapchain, None) };
        }
        self.swapchain = swapchain;
        self.format = format;
        self.extent = extent;
        // SAFETY: images of a swapchain this presenter owns
        self.images = unsafe { self.swapchain_fn.get_swapchain_images(swapchain) }
            .map_err(|error| format!("no Vulkan swapchain images: {error}"))?;
        self.drawn = vec![false; self.images.len()];
        for view in self.views.drain(..) {
            unsafe { device.destroy_image_view(view, None) };
        }
        for &image in &self.images {
            self.views.push(unsafe { create_view(&device, image, format) }?);
        }
        // one a swapchain image, so waiting to show an image never waits on a
        // frame that still has to signal it
        for semaphore in self.finished.drain(..) {
            unsafe { device.destroy_semaphore(semaphore, None) };
        }
        for _ in 0..self.images.len() {
            let semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(|error| format!("no Vulkan semaphore: {error}"))?;
            self.finished.push(semaphore);
        }
        Ok(())
    }

    /// a framebuffer an image, for the render pass the presenter has.
    fn make_framebuffers(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        for framebuffer in self.framebuffers.drain(..) {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }
        for &view in &self.views {
            let framebuffer = unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachments(&[view])
                        .width(self.extent.width)
                        .height(self.extent.height)
                        .layers(1),
                    None,
                )
            }
            .map_err(|error| format!("no Vulkan framebuffer: {error}"))?;
            self.framebuffers.push(framebuffer);
        }
        Ok(())
    }

    /// the device's name, for the log.
    pub fn name(&self) -> String {
        // SAFETY: a query on the device this presenter picked
        let properties = unsafe { self.instance.get_physical_device_properties(self.physical) };
        // SAFETY: a device's name is a NUL-terminated string of its properties
        unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy().into_owned()
    }

    /// puts a frame's screens, and the menu over them, on the window. a frame
    /// the window will not take yet is dropped; the next one draws into a
    /// swapchain made for its size.
    pub fn present(&mut self, shown: [Option<(PanelImage, [i32; 4])>; 2], smooth: bool, overlays: Vec<Arc<Overlay>>, cursor: Option<(i32, i32)>) {
        if let Err(error) = self.draw(shown, smooth, overlays, cursor) {
            log::error!("Vulkan could not show a frame, {error}");
        }
    }

    fn draw(&mut self, shown: [Option<(PanelImage, [i32; 4])>; 2], smooth: bool, overlays: Vec<Arc<Overlay>>, cursor: Option<(i32, i32)>) -> Result<(), String> {
        let device = self.device.clone();
        let frame = self.frame;
        // nothing this frame touches may still be in the GPU's hands: the
        // fence of the frame two presents ago went, this one's is waited on
        // before its uniforms and staging buffers are written
        // SAFETY: a fence this presenter made
        unsafe { device.wait_for_fences(&[self.fences[frame]], true, u64::MAX) }
            .map_err(|error| format!("no Vulkan fence: {error}"))?;
        if self.swapchain == vk::SwapchainKHR::null() {
            self.make_swapchain()?;
            self.make_framebuffers()?;
        }
        self.stage(shown, frame)?;
        self.update_uniforms(smooth, frame);
        self.stage_overlays(&overlays, frame, cursor);

        // SAFETY: an acquire on the swapchain and the frame's semaphore
        let acquired = unsafe { self.swapchain_fn.acquire_next_image(self.swapchain, u64::MAX, self.available[frame], vk::Fence::null()) };
        let index = match acquired {
            Ok((index, _)) => index as usize,
            // the window changed size or went: the next frame gets a swapchain
            // made for it
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate()?;
                return Ok(());
            }
            Err(error) => return Err(format!("no Vulkan image to draw into: {error}")),
        };
        // SAFETY: a fence waited on above, a buffer of the frame's own
        unsafe {
            device.reset_fences(&[self.fences[frame]]).map_err(|error| format!("no Vulkan fence: {error}"))?;
            device
                .reset_command_buffer(self.commands[frame], vk::CommandBufferResetFlags::empty())
                .map_err(|error| format!("no Vulkan command buffer: {error}"))?;
        }
        let command = self.commands[frame];
        // SAFETY: a command buffer this presenter records into
        unsafe {
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(|error| format!("no Vulkan command buffer: {error}"))?;
            for index in 0..self.screens[frame].len() {
                if self.screens[frame][index].shown {
                    copy_screen(&device, command, &mut self.screens[frame][index]);
                }
            }
            let old = match self.drawn[index] {
                true => vk::ImageLayout::PRESENT_SRC_KHR,
                false => vk::ImageLayout::UNDEFINED,
            };
            transition(&device, command, self.images[index], old, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
        self.drawn[index] = true;

        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: self.extent };
        let clears = [vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }];
        let pass = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[index])
            .render_area(area)
            .clear_values(&clears);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        // SAFETY: recording into the command buffer this frame owns, on the
        // framebuffer of the image it acquired
        unsafe {
            device.cmd_begin_render_pass(command, &pass, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_viewport(command, 0, &viewports);
            device.cmd_set_scissor(command, 0, &[area]);
        }
        for screen in 0..self.screens[frame].len() {
            if !self.screens[frame][screen].shown {
                continue;
            }
            // SAFETY: a set this presenter filled this frame for this screen
            unsafe {
                device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.sets[frame][screen]], &[]);
                device.cmd_draw(command, 3, 1, 0, 0);
            }
        }
        if !self.overlay_draws[frame].is_empty() {
            // SAFETY: the frame's own corners and indices, and the sets and
            // textures the overlay came with
            unsafe {
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.overlay_pipeline);
                device.cmd_bind_vertex_buffers(command, 0, &[self.overlay_vertices[frame].buffer], &[0]);
                device.cmd_bind_index_buffer(command, self.overlay_indices[frame].buffer, 0, vk::IndexType::UINT32);
            }
            for draw in &self.overlay_draws[frame] {
                if draw.count == 0 || draw.scissor.extent.width == 0 || draw.scissor.extent.height == 0 {
                    continue;
                }
                // SAFETY: a mesh of this frame's overlay, its scissor inside the
                // window, and the set its texture was written into
                unsafe {
                    device.cmd_set_scissor(command, 0, &[draw.scissor]);
                    device.cmd_bind_descriptor_sets(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[draw.set], &[]);
                    device.cmd_draw_indexed(command, draw.count, 1, draw.first, draw.vertex_offset, 0);
                }
            }
        }
        // SAFETY: ending the pass and the buffer this frame recorded
        unsafe {
            device.cmd_end_render_pass(command);
            device.end_command_buffer(command).map_err(|error| format!("no Vulkan command buffer: {error}"))?;
        }

        let waits = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        // the submit and the present hold on to these past this statement, so
        // they are not made up in place in the builder
        let waiting = [self.available[frame]];
        let buffers = [command];
        let signalled = [self.finished[index]];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&waiting)
            .wait_dst_stage_mask(&waits)
            .command_buffers(&buffers)
            .signal_semaphores(&signalled);
        // SAFETY: a submit of the frame's own command buffer, fence and
        // semaphores
        unsafe { device.queue_submit(self.queue, &[submit], self.fences[frame]) }
            .map_err(|error| format!("no Vulkan frame submitted: {error}"))?;
        let drawn = [self.finished[index]];
        let swapchains = [self.swapchain];
        let indices = [index as u32];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&drawn)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: a present of the image just drawn into
        let shown = unsafe { self.swapchain_fn.queue_present(self.queue, &present) };
        self.frame = (frame + 1) % FRAMES;
        match shown {
            Ok(false) => Ok(()),
            // the window changed size or went: the next frame gets a swapchain
            // made for it
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => self.recreate(),
            Err(error) => Err(format!("Vulkan could not show the frame: {error}")),
        }
    }

    /// the swapchain and what was made from its images, for the window as it
    /// is now. the device waits: a present in flight is touching those images.
    fn recreate(&mut self) -> Result<(), String> {
        let device = self.device.clone();
        // SAFETY: everything in flight finishes before its images go
        unsafe { device.device_wait_idle() }.map_err(|error| format!("no Vulkan device: {error}"))?;
        self.drop_swapchain();
        self.make_swapchain()?;
        self.make_framebuffers()
    }

    /// the swapchain and what was made from its images, not the device.
    fn drop_swapchain(&mut self) {
        let device = self.device.clone();
        for framebuffer in self.framebuffers.drain(..) {
            // SAFETY: framebuffers of this presenter, nothing in flight
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }
        for view in self.views.drain(..) {
            unsafe { device.destroy_image_view(view, None) };
        }
        if self.swapchain != vk::SwapchainKHR::null() {
            unsafe { self.swapchain_fn.destroy_swapchain(self.swapchain, None) };
            self.swapchain = vk::SwapchainKHR::null();
        }
        self.images.clear();
        self.drawn.clear();
    }

    /// this frame's pictures into the screens' images, a new size making a new
    /// image, and the areas they go to.
    fn stage(&mut self, shown: [Option<(PanelImage, [i32; 4])>; 2], frame: usize) -> Result<(), String> {
        let device = self.device.clone();
        let mut areas = [None, None];
        for (index, screen) in shown.into_iter().enumerate() {
            let Some((picture, area)) = screen else {
                self.screens[frame][index].shown = false;
                continue;
            };
            let (pixels, width, height) = match picture {
                PanelImage::Pixels(pixels, width, height) => (pixels, width, height),
                // a picture the host's GPU drew is an OpenGL texture, which
                // this cannot sample. with the Vulkan backend chosen the 3D is
                // drawn in software, so it does not come up
                PanelImage::Gpu(_) => {
                    self.screens[frame][index].shown = false;
                    log::warn!("a screen the host GPU drew cannot go to Vulkan, it is left out");
                    continue;
                }
            };
            if width == 0 || height == 0 || pixels.len() < (width as usize * height as usize * 4) {
                self.screens[frame][index].shown = false;
                continue;
            }
            self.screens[frame][index].area = area;
            self.screens[frame][index].shown = true;
            areas[index] = Some(area);
            if self.screens[frame][index].size != (width, height) {
                // SAFETY: the device waits, so nothing made from the old image
                // is in flight when it goes
                unsafe { device.device_wait_idle() }.map_err(|error| format!("no Vulkan device: {error}"))?;
                // a new image, and the set that names it is written again below
                unsafe { destroy_screen(&device, &mut self.screens[frame][index]) };
                let (image, memory) = unsafe { create_screen_image(&device, &self.instance, self.physical, width, height) }?;
                let view = unsafe { create_view(&device, image, SCREEN_FORMAT) }?;
                self.screens[frame][index].image = image;
                self.screens[frame][index].memory = memory;
                self.screens[frame][index].view = view;
                self.screens[frame][index].size = (width, height);
                self.screens[frame][index].initialized = false;
            }
            let needs = pixels.len();
            if self.screens[frame][index].staging.as_ref().is_none_or(|staging| staging.capacity < needs) {
                if let Some(staging) = self.screens[frame][index].staging.take() {
                    // SAFETY: a buffer of this presenter, nothing in flight
                    unsafe { destroy_staging(&device, staging) };
                }
                self.screens[frame][index].staging = Some(unsafe {
                    create_staging(&device, &self.instance, self.physical, needs, vk::BufferUsageFlags::TRANSFER_SRC)
                }?);
            }
            let staging = self.screens[frame][index].staging.as_ref().expect("a staging buffer was just made");
            // SAFETY: a mapped buffer of exactly this many bytes
            unsafe { std::ptr::copy_nonoverlapping(pixels.as_ptr(), staging.mapped, needs) };
        }
        self.layout = layout_of(&areas);
        Ok(())
    }

    /// where each screen goes in clip space, and its picture's uv, into the
    /// uniform blocks; then the sampler and picture into the sets.
    fn update_uniforms(&mut self, smooth: bool, frame: usize) {
        let (width, height) = (self.extent.width as f32, self.extent.height as f32);
        let (layout_width, layout_height) = (self.layout.0 as f32, self.layout.1 as f32);
        let scale = (width / layout_width).min(height / layout_height);
        let left = (width - layout_width * scale) / 2.0;
        let top = (height - layout_height * scale) / 2.0;
        for index in 0..self.screens[frame].len() {
            let [x, y, w, h] = self.screens[frame][index].area;
            let (x, y) = (left + x as f32 * scale, top + y as f32 * scale);
            let (w, h) = (w as f32 * scale, h as f32 * scale);
            // clip space, y down, from the window's top left corner
            let words: [f32; 8] = [2.0 * x / width - 1.0, 2.0 * y / height - 1.0, 2.0 * w / width, 2.0 * h / height, 0.0, 0.0, 1.0, 1.0];
            // SAFETY: a mapped uniform block of UNIFORM_BYTES
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), self.uniforms[frame][index].mapped, UNIFORM_BYTES as usize) };
        }
        let sampler = match smooth {
            true => self.linear,
            false => self.nearest,
        };
        let device = self.device.clone();
        for index in 0..self.screens[frame].len() {
            let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
            let picture_info = [vk::DescriptorImageInfo::default()
                .image_view(self.screens[frame][index].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let block = [vk::DescriptorBufferInfo::default().buffer(self.uniforms[frame][index].buffer).offset(0).range(UNIFORM_BYTES)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[frame][index])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[frame][index])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&picture_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[frame][index])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&block),
            ];
            // SAFETY: sets from this presenter's pool, handles it owns, and
            // no command buffer in flight reads them (the frame's fence was
            // waited on)
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    /// the menu's textures onto the device, and its meshes into this frame's
    /// corners and indices, placed where the layout puts them.
    fn stage_overlays(&mut self, overlays: &[Arc<Overlay>], frame: usize, cursor: Option<(i32, i32)>) {
        self.overlay_draws[frame].clear();
        if overlays.is_empty() {
            return;
        }
        let device = self.device.clone();
        let (width, height) = (self.extent.width as f32, self.extent.height as f32);
        let (layout_width, layout_height) = (self.layout.0.max(1) as f32, self.layout.1.max(1) as f32);
        let scale = (width / layout_width).min(height / layout_height);
        let left = (width - layout_width * scale) / 2.0;
        let top = (height - layout_height * scale) / 2.0;
        let bottom = self.screens[frame][1].shown.then_some(self.screens[frame][1].area);
        // a texture the menu changes or lets go of may be one the other frame
        // still reads
        if overlays.iter().any(|overlay| !overlay.textures.is_empty() || !overlay.free.is_empty()) {
            // SAFETY: nothing in flight may read what is about to be written or
            // destroyed
            unsafe {
                let _ = device.device_wait_idle();
            }
        }
        let mut corners: Vec<Corner> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut draws: Vec<OverlayDraw> = Vec::new();
        // the touch cursor, where GLES puts it and as big: a darker square with
        // a lighter one inside, the two marks cursor_marks gives
        if let Some((x, y)) = cursor {
            if self.cursor_set != vk::DescriptorSet::null() {
                let to = |value: i32| (left + value as f32 * scale).round() as i32;
                for (size, shade) in crate::cursor_marks(scale) {
                    let [x0, y0, wide, tall] = cursor_rect(to(x), to(y), size);
                    let Some(scissor) = scissor_of([0, 0, self.extent.width, self.extent.height], [x0, y0, wide, tall], 1.0, 1.0, self.extent) else { continue };
                    let light = (shade * 255.0) as u8;
                    let colour = [light, light, light, 255];
                    let vertex_offset = corners.len() as i32;
                    let (x1, y1) = (x0 + wide, y0 + tall);
                    for (px, py) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
                        corners.push(Corner { position: [2.0 * px / width - 1.0, 2.0 * py / height - 1.0], uv: [0.5, 0.5], color: colour });
                    }
                    let first = indices.len() as u32;
                    indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
                    draws.push(OverlayDraw { first, count: 6, vertex_offset, set: self.cursor_set, scissor });
                }
            }
        }
        for overlay in overlays {
            let mut sets: HashMap<u64, vk::DescriptorSet> = HashMap::new();
            for texture in &overlay.textures {
                match self.overlay_picture(texture) {
                    Ok(set) => {
                        sets.insert(texture.id, set);
                    }
                    Err(error) => log::warn!("a texture of the menu did not go to Vulkan, {error}"),
                }
            }
            for id in &overlay.free {
                if let Some(picture) = self.overlay_textures.remove(id) {
                    let pool = picture.pool;
                    // SAFETY: the device was let finish above, so no frame reads
                    // the set, and a pool outlives every set it made
                    unsafe {
                        if let Some(handle) = self.overlay_pools.get(pool) {
                            let _ = device.free_descriptor_sets(*handle, &[picture.set]);
                        }
                        destroy_picture(&device, picture);
                    }
                    self.overlay_budget.give(pool);
                }
            }
            // where the overlay's space sits in the window: over the bottom
            // screen, over the whole layout, or the menu's 640x480 in the
            // middle, as GLES places them
            let (rect, space) = if overlay.bottom && bottom.is_some() {
                let (x, y, fit) = crate::fitted(bottom.unwrap_or_default());
                ([left + x * scale, top + y * scale, 640.0 * fit * scale, 480.0 * fit * scale], (640.0, 480.0))
            } else if overlay.whole {
                ([left, top, layout_width * scale, layout_height * scale], (layout_width, layout_height))
            } else {
                let down = if layout_height > layout_width { 0.0 } else { (layout_height - 480.0) / 2.0 };
                ([left + (layout_width - 640.0) / 2.0 * scale, top + down * scale, 640.0 * scale, 480.0 * scale], (640.0, 480.0))
            };
            let (step_x, step_y) = (rect[2] / space.0, rect[3] / space.1);
            for mesh in &overlay.meshes {
                let Some(&set) = sets.get(&mesh.texture) else { continue };
                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    continue;
                }
                let Some(scissor) = scissor_of(mesh.clip, rect, step_x, step_y, self.extent) else { continue };
                let vertex_offset = corners.len() as i32;
                for vertex in &mesh.vertices {
                    let x = rect[0] + vertex.position[0] * step_x;
                    let y = rect[1] + vertex.position[1] * step_y;
                    corners.push(Corner {
                        position: [2.0 * x / width - 1.0, 2.0 * y / height - 1.0],
                        uv: vertex.uv,
                        color: vertex.color,
                    });
                }
                let first = indices.len() as u32;
                indices.extend_from_slice(&mesh.indices);
                draws.push(OverlayDraw { first, count: mesh.indices.len() as u32, vertex_offset, set, scissor });
            }
        }
        if draws.is_empty() {
            return;
        }
        // SAFETY: the corners and indices just built, as bytes
        let vertex_bytes = unsafe { std::slice::from_raw_parts(corners.as_ptr().cast::<u8>(), std::mem::size_of_val(corners.as_slice())) };
        if let Err(error) = unsafe {
            fill_stream(&device, &self.instance, self.physical, &mut self.overlay_vertices[frame], vertex_bytes, vk::BufferUsageFlags::VERTEX_BUFFER)
        } {
            log::warn!("the menu's corners did not go to Vulkan, {error}");
            return;
        }
        // SAFETY: the indices just built, as bytes
        let index_bytes = unsafe { std::slice::from_raw_parts(indices.as_ptr().cast::<u8>(), std::mem::size_of_val(indices.as_slice())) };
        if let Err(error) = unsafe {
            fill_stream(&device, &self.instance, self.physical, &mut self.overlay_indices[frame], index_bytes, vk::BufferUsageFlags::INDEX_BUFFER)
        } {
            log::warn!("the menu's indices did not go to Vulkan, {error}");
            return;
        }
        self.overlay_draws[frame] = draws;
    }

    /// a texture the menu draws with, made or changed, and the set its meshes
    /// bind it through. the device is let finish before a texture is written,
    /// as stage_overlays did for the frame.
    fn overlay_picture(&mut self, texture: &OverlayTexture) -> Result<vk::DescriptorSet, String> {
        let device = self.device.clone();
        let known = self.overlay_textures.get(&texture.id).map(|picture| (picture.set, picture.size));
        let (set, first) = match known {
            // a part of a texture already there, or the same size again
            Some((set, size)) if texture.offset.is_some() || size == texture.size => (set, false),
            _ => {
                if let Some(old) = self.overlay_textures.remove(&texture.id) {
                    let pool = old.pool;
                    // SAFETY: the device was let finish before this frame's
                    // overlays were staged, so no frame reads the set
                    unsafe {
                        if let Some(handle) = self.overlay_pools.get(pool) {
                            let _ = device.free_descriptor_sets(*handle, &[old.set]);
                        }
                        destroy_picture(&device, old);
                    }
                    self.overlay_budget.give(pool);
                }
                let [width, height] = texture.size;
                // SAFETY: an image and a view of this device, a sampler, and a
                // set of this presenter's pool
                let (image, memory) = unsafe { create_screen_image(&device, &self.instance, self.physical, width.max(1), height.max(1)) }?;
                let view = unsafe { create_view(&device, image, SCREEN_FORMAT) }?;
                let sampler = unsafe { create_sampler(&device, texture.linear) }?;
                let (set, pool) = unsafe { self.overlay_set(view, sampler) }?;
                self.overlay_textures.insert(texture.id, Picture { image, memory, view, sampler, set, pool, size: texture.size });
                (set, true)
            }
        };
        let rows = (texture.pixels.len() / 4) as u32 / texture.size[0].max(1);
        if rows == 0 || texture.size[0] == 0 {
            return Ok(set);
        }
        let image = self.overlay_textures.get(&texture.id).map(|picture| picture.image).ok_or("a menu texture went away")?;
        // SAFETY: an image of this presenter's, a staging buffer of the call's
        // own, and a one-shot command buffer of its pool
        unsafe {
            upload_overlay(
                &device,
                &self.instance,
                self.physical,
                self.command_pool,
                self.queue,
                image,
                first,
                [texture.size[0], rows],
                texture.offset.unwrap_or([0, 0]),
                &texture.pixels,
            )
        }?;
        Ok(set)
    }

    /// a new set with a texture and its sampler in it, and the pool it came
    /// from: one of those made, or one more when they are all taken.
    unsafe fn overlay_set(&mut self, view: vk::ImageView, sampler: vk::Sampler) -> Result<(vk::DescriptorSet, usize), String> {
        let device = self.device.clone();
        let pool = self.overlay_budget.take();
        if pool == self.overlay_pools.len() {
            let capacity = self.overlay_budget.capacity(pool).unwrap_or(OVERLAY_SETS);
            // SAFETY: another pool of this device, for the sets the menu's
            // textures take
            let made = unsafe { create_overlay_pool(&device, capacity) }?;
            self.overlay_pools.push(made);
        }
        let handle = self.overlay_pools[pool];
        let layouts = [self.descriptor_layout];
        // SAFETY: a set of a pool this presenter made, of a layout it takes
        let set = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(handle).set_layouts(&layouts))
        }
        .map_err(|error| format!("no Vulkan descriptor set for a menu texture: {error}"))?;
        let set = set[0];
        let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler)];
        let picture_info = [vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&sampler_info),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&picture_info),
        ];
        // SAFETY: a set of this presenter's, before any frame binds it
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok((set, pool))
    }

    /// lets everything go: the frames in flight first, then the objects in the
    /// order they were made.
    fn destroy_objects(&mut self) {
        let device = self.device.clone();
        if device.handle() != vk::Device::null() {
            // SAFETY: nothing in flight may touch what is about to go
            unsafe {
                let _ = device.device_wait_idle();
            }
        }
        self.drop_swapchain();
        for semaphore in self.finished.drain(..) {
            // SAFETY: semaphores of this presenter, nothing in flight
            unsafe { device.destroy_semaphore(semaphore, None) };
        }
        for semaphore in self.available.drain(..) {
            unsafe { device.destroy_semaphore(semaphore, None) };
        }
        for fence in self.fences.drain(..) {
            unsafe { device.destroy_fence(fence, None) };
        }
        if self.command_pool != vk::CommandPool::null() {
            unsafe { device.destroy_command_pool(self.command_pool, None) };
        }
        for frame in &mut self.screens {
            for screen in frame {
                unsafe { destroy_screen(&device, screen) };
            }
        }
        for frame in &mut self.uniforms {
            for uniform in frame {
                if uniform.buffer != vk::Buffer::null() {
                    unsafe {
                        device.unmap_memory(uniform.memory);
                        device.destroy_buffer(uniform.buffer, None);
                        device.free_memory(uniform.memory, None);
                    }
                }
            }
        }
        for stream in self.overlay_vertices.iter_mut().chain(self.overlay_indices.iter_mut()) {
            if stream.buffer != vk::Buffer::null() {
                unsafe { destroy_staging(&device, std::mem::replace(stream, Staging::empty())) };
            }
        }
        for (_, picture) in self.overlay_textures.drain() {
            unsafe { destroy_picture(&device, picture) };
        }
        for pool in self.overlay_pools.drain(..) {
            unsafe { device.destroy_descriptor_pool(pool, None) };
        }
        if self.overlay_pipeline != vk::Pipeline::null() {
            unsafe { device.destroy_pipeline(self.overlay_pipeline, None) };
        }
        if self.descriptor_pool != vk::DescriptorPool::null() {
            unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };
        }
        if self.descriptor_layout != vk::DescriptorSetLayout::null() {
            unsafe { device.destroy_descriptor_set_layout(self.descriptor_layout, None) };
        }
        if self.pipeline != vk::Pipeline::null() {
            unsafe { device.destroy_pipeline(self.pipeline, None) };
        }
        if self.pipeline_layout != vk::PipelineLayout::null() {
            unsafe { device.destroy_pipeline_layout(self.pipeline_layout, None) };
        }
        if self.render_pass != vk::RenderPass::null() {
            unsafe { device.destroy_render_pass(self.render_pass, None) };
        }
        for sampler in [self.nearest, self.linear] {
            if sampler != vk::Sampler::null() {
                unsafe { device.destroy_sampler(sampler, None) };
            }
        }
        if device.handle() != vk::Device::null() {
            unsafe { device.destroy_device(None) };
        }
        if self.surface != vk::SurfaceKHR::null() {
            unsafe { self.surface_fn.destroy_surface(self.surface, None) };
            self.surface = vk::SurfaceKHR::null();
        }
        unsafe { self.instance.destroy_instance(None) };
        let _ = &self.entry;
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        self.destroy_objects();
    }
}

impl Uniform {
    fn empty() -> Uniform {
        Uniform { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), mapped: std::ptr::null_mut() }
    }
}

/// where a cursor mark of a size sits over a spot, its centre on the spot and
/// its left and top whole pixels, as GLES cuts it.
fn cursor_rect(x: i32, y: i32, size: i32) -> [f32; 4] {
    [(x - size / 2) as f32, (y - size / 2) as f32, size as f32, size as f32]
}

/// where a mesh's clip sits in the window, cut to it, or nothing if none of
/// it shows.
fn scissor_of(clip: [u32; 4], rect: [f32; 4], step_x: f32, step_y: f32, extent: vk::Extent2D) -> Option<vk::Rect2D> {
    let x = (rect[0] + clip[0] as f32 * step_x).floor().max(0.0);
    let y = (rect[1] + clip[1] as f32 * step_y).floor().max(0.0);
    let right = (rect[0] + clip[2] as f32 * step_x).ceil().min(extent.width as f32);
    let bottom = (rect[1] + clip[3] as f32 * step_y).ceil().min(extent.height as f32);
    if right <= x || bottom <= y {
        return None;
    }
    Some(vk::Rect2D {
        offset: vk::Offset2D { x: x as i32, y: y as i32 },
        extent: vk::Extent2D { width: (right - x) as u32, height: (bottom - y) as u32 },
    })
}

/// a stream buffer a frame fills with its corners or indices, made bigger when
/// what goes in it outgrows it.
unsafe fn fill_stream(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    stream: &mut Staging,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<(), String> {
    if stream.buffer == vk::Buffer::null() || stream.capacity < bytes.len() {
        if stream.buffer != vk::Buffer::null() {
            // the other frame may still be drawing with it, and this grows
            // only when a menu outgrows it
            // SAFETY: nothing in flight may read what is about to go
            unsafe {
                let _ = device.device_wait_idle();
                destroy_staging(device, std::mem::replace(stream, Staging::empty()));
            }
        }
        // SAFETY: a buffer of this device for bytes the CPU writes
        *stream = unsafe { create_staging(device, instance, physical, bytes.len().max(64), usage) }?;
    }
    // SAFETY: a mapped buffer of at least this many bytes
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), stream.mapped, bytes.len()) };
    Ok(())
}

/// a menu texture and what binds it, let go of.
unsafe fn destroy_picture(device: &ash::Device, picture: Picture) {
    // SAFETY: handles of this device, off its last frame
    unsafe {
        if picture.sampler != vk::Sampler::null() {
            device.destroy_sampler(picture.sampler, None);
        }
        if picture.view != vk::ImageView::null() {
            device.destroy_image_view(picture.view, None);
        }
        if picture.image != vk::Image::null() {
            device.destroy_image(picture.image, None);
        }
        if picture.memory != vk::DeviceMemory::null() {
            device.free_memory(picture.memory, None);
        }
    }
}

/// the pool the menu's textures get their sets from, of the layout the screens
/// use: a sampler and a sampled picture.
unsafe fn create_overlay_pool(device: &ash::Device, size: u32) -> Result<vk::DescriptorPool, String> {
    let sizes = [
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLER).descriptor_count(size),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(size),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(size),
    ];
    // SAFETY: a pool of this device for the sets the menu's textures take
    unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(size).pool_sizes(&sizes), None) }
        .map_err(|error| format!("no Vulkan descriptor pool for the menu: {error}"))
}

/// copies a menu texture's pixels into its image and leaves it ready to
/// sample. first says the image has never held a picture.
unsafe fn upload_overlay(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    image: vk::Image,
    first: bool,
    size: [u32; 2],
    at: [u32; 2],
    pixels: &[u8],
) -> Result<(), String> {
    // SAFETY: a staging buffer of this device, for these bytes
    let staging = unsafe { create_staging(device, instance, physical, pixels.len().max(1), vk::BufferUsageFlags::TRANSFER_SRC) }?;
    // SAFETY: a mapped buffer of exactly this many bytes
    unsafe { std::ptr::copy_nonoverlapping(pixels.as_ptr(), staging.mapped, pixels.len()) };
    // SAFETY: a command buffer of this presenter's pool, and a fence of its
    // device
    let command = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    }
    .map_err(|error| format!("no Vulkan command buffer for a menu texture: {error}"))?[0];
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .map_err(|error| format!("no Vulkan fence for a menu texture: {error}"))?;
    let run = || -> Result<(), String> {
        // SAFETY: a one-shot command buffer of this call's own
        unsafe {
            device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                .map_err(|error| format!("no Vulkan command buffer for a menu texture: {error}"))?;
        }
        let old = match first {
            true => vk::ImageLayout::UNDEFINED,
            false => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        // SAFETY: the image, and a buffer this call made to copy into it
        unsafe {
            transition(device, command, image, old, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: at[0] as i32, y: at[1] as i32, z: 0 },
                image_extent: vk::Extent3D { width: size[0].max(1), height: size[1].max(1), depth: 1 },
            };
            device.cmd_copy_buffer_to_image(command, staging.buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            transition(device, command, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            device
                .end_command_buffer(command)
                .map_err(|error| format!("no Vulkan command buffer for a menu texture: {error}"))?;
        }
        let buffers = [command];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        // SAFETY: the queue this presenter made its device with, and the fence
        // this call made to wait the copy out
        unsafe {
            device
                .queue_submit(queue, &[submit], fence)
                .map_err(|error| format!("a menu texture was not copied: {error}"))?;
            device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|error| format!("a menu texture was not copied: {error}"))?;
        }
        Ok(())
    };
    let done = run();
    // SAFETY: the fence and command buffer of this call, and its staging
    // buffer, the copy done
    unsafe {
        device.destroy_fence(fence, None);
        device.free_command_buffers(command_pool, &[command]);
        destroy_staging(device, staging);
    }
    done
}

/// the menu's pipeline: the same pass and pipeline layout, corners from a
/// vertex buffer, and its premultiplied colour blended over what is there.
unsafe fn create_overlay_pipeline(device: &ash::Device, layout: vk::PipelineLayout, render_pass: vk::RenderPass) -> Result<vk::Pipeline, String> {
    let module = unsafe { create_shader_module(device, OVERLAY_SHADERS, "menu's") }?;
    let vertex = CString::new("vs").expect("a name without a NUL");
    let fragment = CString::new("fs").expect("a name without a NUL");
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(vertex.as_c_str()),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(fragment.as_c_str()),
    ];
    let bindings = [vk::VertexInputBindingDescription { binding: 0, stride: CORNER_STRIDE as u32, input_rate: vk::VertexInputRate::VERTEX }];
    let attributes = [
        vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 0 },
        vk::VertexInputAttributeDescription { location: 1, binding: 0, format: vk::Format::R32G32_SFLOAT, offset: 8 },
        vk::VertexInputAttributeDescription { location: 2, binding: 0, format: vk::Format::R8G8B8A8_UNORM, offset: 16 },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blending)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    // SAFETY: a pipeline of this device, with the module and layouts above
    let made = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    // SAFETY: the module was only for making the pipeline
    unsafe { device.destroy_shader_module(module, None) };
    match made {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((_, error)) => Err(format!("no Vulkan pipeline for the menu: {error}")),
    }
}

/// the layout the areas are in: as far right and down as they reach.
fn layout_of(areas: &[Option<[i32; 4]>; 2]) -> (i32, i32) {
    let mut right = 1;
    let mut bottom = 1;
    for area in areas.iter().flatten() {
        right = right.max(area[0] + area[2]);
        bottom = bottom.max(area[1] + area[3]);
    }
    (right.max(1), bottom.max(1))
}

/// the window's size, as the surface reports it, or a size within its bounds.
fn current_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }
    let width = capabilities.min_image_extent.width.max(capabilities.max_image_extent.width.min(1280));
    let height = capabilities.min_image_extent.height.max(capabilities.max_image_extent.height.min(720));
    vk::Extent2D { width, height }
}

/// a format of the surface's that this draws in, the first of FORMATS it
/// offers, or whatever it offers first.
fn pick_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::Format> {
    formats
        .iter()
        .find(|format| FORMATS.contains(&format.format) && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .or_else(|| formats.iter().find(|format| FORMATS.contains(&format.format)))
        .or_else(|| formats.first())
        .map(|format| format.format)
}

/// the surface's extensions: the surface itself, and on android the window it
/// is made from.
#[cfg(target_os = "android")]
fn surface_extensions() -> Vec<*const c_char> {
    vec![ash::khr::surface::NAME.as_ptr(), ash::khr::android_surface::NAME.as_ptr()]
}

#[cfg(not(target_os = "android"))]
fn surface_extensions() -> Vec<*const c_char> {
    vec![ash::khr::surface::NAME.as_ptr()]
}

/// a surface on the window set_window gave: an android window, which is the
/// window a handheld app is given. a handheld's panel is KMS otherwise, which
/// Vulkan shows to through a display extension most drivers lack.
#[cfg(target_os = "android")]
unsafe fn make_surface(entry: &ash::Entry, instance: &ash::Instance, window: *mut c_void) -> Result<vk::SurfaceKHR, String> {
    // SAFETY: an android window the activity handed the renderer
    let info = vk::AndroidSurfaceCreateInfoKHR::default().window(window.cast());
    let android = ash::khr::android_surface::Instance::new(entry, instance);
    unsafe { android.create_android_surface(&info, None) }.map_err(|error| format!("no Vulkan android surface: {error}"))
}

#[cfg(not(target_os = "android"))]
unsafe fn make_surface(_entry: &ash::Entry, _instance: &ash::Instance, _window: *mut c_void) -> Result<vk::SurfaceKHR, String> {
    Err("the screens go to a panel KMS drives, which Vulkan cannot show to here".into())
}

/// the memory type a requirement takes with the properties asked for.
unsafe fn memory_type(instance: &ash::Instance, physical: vk::PhysicalDevice, bits: u32, wanted: vk::MemoryPropertyFlags) -> Result<u32, String> {
    // SAFETY: a query of the device this presenter picked
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    (0..memory.memory_type_count)
        .find(|index| bits & (1 << index) != 0 && memory.memory_types[*index as usize].property_flags.contains(wanted))
        .ok_or_else(|| "no Vulkan memory of the kind a picture needs".to_owned())
}

/// an image a screen's picture is sampled from, device local.
unsafe fn create_screen_image(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory), String> {
    // SAFETY: an image of this device, in the format the shader samples
    let image = unsafe {
        device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(SCREEN_FORMAT)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
    }
    .map_err(|error| format!("no Vulkan image for a screen: {error}"))?;
    // SAFETY: the image's requirements and a memory type for them
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let kind = unsafe { memory_type(instance, physical, requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) }?;
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(kind),
            None,
        )
    }
    .map_err(|error| format!("no Vulkan memory for a screen: {error}"))?;
    // SAFETY: memory the image's requirements asked for
    if let Err(error) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.destroy_image(image, None);
            device.free_memory(memory, None);
        }
        return Err(format!("no Vulkan memory for a screen: {error}"));
    }
    Ok((image, memory))
}

/// an image view of a screen's picture, or of a swapchain image.
unsafe fn create_view(device: &ash::Device, image: vk::Image, format: vk::Format) -> Result<vk::ImageView, String> {
    // SAFETY: a view of an image this presenter made
    unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
    }
    .map_err(|error| format!("no Vulkan image view: {error}"))
}

/// a host-visible buffer a picture's bytes go to, kept mapped.
unsafe fn create_staging(
    device: &ash::Device,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    size: usize,
    usage: vk::BufferUsageFlags,
) -> Result<Staging, String> {
    let size = size as u64;
    // SAFETY: a buffer of this device, for bytes the CPU writes
    let buffer = unsafe {
        device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)
    }
    .map_err(|error| format!("no Vulkan staging buffer: {error}"))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let kind = unsafe {
        memory_type(
            instance,
            physical,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }?;
    let memory = unsafe {
        device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(kind), None)
    }
    .map_err(|error| format!("no Vulkan staging memory: {error}"))?;
    // SAFETY: memory the buffer's requirements asked for
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(format!("no Vulkan staging memory: {error}"));
    }
    // SAFETY: the whole buffer, mapped once and kept
    let mapped = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
        .map_err(|error| format!("no Vulkan staging memory: {error}"))?;
    Ok(Staging { buffer, memory, mapped: mapped.cast(), capacity: size as usize })
}

/// a uniform block's buffer, kept mapped.
unsafe fn create_uniform(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice) -> Result<Uniform, String> {
    // SAFETY: a buffer of this device, for a block the shader reads
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default().size(UNIFORM_BYTES).usage(vk::BufferUsageFlags::UNIFORM_BUFFER).sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|error| format!("no Vulkan uniform buffer: {error}"))?;
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let kind = unsafe {
        memory_type(
            instance,
            physical,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }?;
    let memory = unsafe {
        device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(kind), None)
    }
    .map_err(|error| format!("no Vulkan uniform memory: {error}"))?;
    // SAFETY: memory the buffer's requirements asked for
    if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(format!("no Vulkan uniform memory: {error}"));
    }
    // SAFETY: the whole block, mapped once and kept
    let mapped = unsafe { device.map_memory(memory, 0, UNIFORM_BYTES, vk::MemoryMapFlags::empty()) }
        .map_err(|error| format!("no Vulkan uniform memory: {error}"))?;
    Ok(Uniform { buffer, memory, mapped: mapped.cast() })
}

/// lets a screen's staging buffer, picture and view go.
unsafe fn destroy_screen(device: &ash::Device, screen: &mut Screen) {
    if let Some(staging) = screen.staging.take() {
        // SAFETY: a buffer of this presenter, nothing in flight
        unsafe { destroy_staging(device, staging) };
    }
    // SAFETY: the screen's own image, view and memory
    unsafe {
        if screen.view != vk::ImageView::null() {
            device.destroy_image_view(screen.view, None);
        }
        if screen.image != vk::Image::null() {
            device.destroy_image(screen.image, None);
        }
        if screen.memory != vk::DeviceMemory::null() {
            device.free_memory(screen.memory, None);
        }
    }
    screen.view = vk::ImageView::null();
    screen.image = vk::Image::null();
    screen.memory = vk::DeviceMemory::null();
    screen.size = (0, 0);
    screen.initialized = false;
}

/// lets a staging buffer go.
unsafe fn destroy_staging(device: &ash::Device, staging: Staging) {
    // SAFETY: a buffer of this presenter, nothing in flight
    unsafe {
        device.unmap_memory(staging.memory);
        device.destroy_buffer(staging.buffer, None);
        device.free_memory(staging.memory, None);
    }
}

/// copies a screen's staging buffer into its picture, with the classic
/// barriers around it: nothing before, read as a sampled picture after.
unsafe fn copy_screen(device: &ash::Device, command: vk::CommandBuffer, screen: &mut Screen) {
    let Some(staging) = screen.staging.as_ref() else { return };
    let (width, height) = screen.size;
    let (old, from, access) = match screen.initialized {
        true => (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::AccessFlags::SHADER_READ),
        false => (vk::ImageLayout::UNDEFINED, vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
    };
    let barrier = [vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(screen.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
    let region = [vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D { width, height, depth: 1 },
    }];
    let ready = [vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(screen.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)];
    // SAFETY: recording into the command buffer of the frame, for an image
    // this screen owns
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            from,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        device.cmd_copy_buffer_to_image(command, staging.buffer, screen.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &region);
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &ready,
        );
    }
    screen.initialized = true;
}

/// moves an image between layouts with the classic barrier.
unsafe fn transition(device: &ash::Device, command: vk::CommandBuffer, image: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
    let (from, access) = match old {
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::AccessFlags::empty()),
        _ => (vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags::COLOR_ATTACHMENT_READ),
    };
    let barrier = [vk::ImageMemoryBarrier::default()
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(access)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
    // SAFETY: recording into the frame's command buffer, for its image
    unsafe {
        device.cmd_pipeline_barrier(command, from, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &barrier);
    }
}

/// the render pass: one colour attachment, cleared, then shown.
unsafe fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass, String> {
    let attachments = [vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
    let colors = [vk::AttachmentReference { attachment: 0, layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL }];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors)];
    let dependencies = [vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
    // SAFETY: a render pass of this device, in the swapchain's format
    unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default().attachments(&attachments).subpasses(&subpasses).dependencies(&dependencies),
            None,
        )
    }
    .map_err(|error| format!("no Vulkan render pass: {error}"))
}

/// the set layout: a sampler, a sampled picture and a uniform block, all read
/// by the fragment shader.
unsafe fn create_descriptor_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, String> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
    ];
    // SAFETY: a layout of this device, one set that is all the shaders take
    unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }
        .map_err(|error| format!("no Vulkan descriptor set layout: {error}"))
}

/// a layout with the one set the shader reads, and no push constants.
unsafe fn create_pipeline_layout(device: &ash::Device, set: vk::DescriptorSetLayout) -> Result<vk::PipelineLayout, String> {
    let sets = [set];
    // SAFETY: a layout of this device, with a set it made
    unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets), None) }
        .map_err(|error| format!("no Vulkan pipeline layout: {error}"))
}

/// the screens' shaders: WGSL through naga to SPIR-V, then a module. both entry
/// points come from the one module.
unsafe fn create_shader_module(device: &ash::Device, source: &str, what: &str) -> Result<vk::ShaderModule, String> {
    let module = naga::front::wgsl::parse_str(source).map_err(|error| format!("the {what} shader does not parse: {error}"))?;
    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
    let info = validator.validate(&module).map_err(|error| format!("the {what} shader does not validate: {error}"))?;
    let options = naga::back::spv::Options {
        // 1.0 is what every 1.1 device takes
        lang_version: (1, 0),
        // the default flags turn y around as WebGPU wants it, and this draws
        // in the clip space Vulkan has
        flags: naga::back::spv::WriterFlags::empty(),
        force_loop_bounding: false,
        ..naga::back::spv::Options::default()
    };
    let words = naga::back::spv::write_vec(&module, &info, &options, None)
        .map_err(|error| format!("the {what} shader does not turn into SPIR-V: {error}"))?;
    // SAFETY: words naga wrote for this device's Vulkan
    unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }
        .map_err(|error| format!("no Vulkan shader module: {error}"))
}

/// the pipeline: one triangle a screen, no vertex buffer, no depth, no blend,
/// the viewport and scissor set as the frame is recorded.
unsafe fn create_pipeline(device: &ash::Device, layout: vk::PipelineLayout, render_pass: vk::RenderPass) -> Result<vk::Pipeline, String> {
    let module = unsafe { create_shader_module(device, SHADERS, "screens'") }?;
    let vertex = CString::new("vs").expect("a name without a NUL");
    let fragment = CString::new("fs").expect("a name without a NUL");
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(vertex.as_c_str()),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(fragment.as_c_str()),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blending)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    // SAFETY: a pipeline of this device, with the module and layouts above
    let made = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    // SAFETY: the module was only for making the pipeline
    unsafe { device.destroy_shader_module(module, None) };
    match made {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((_, error)) => Err(format!("no Vulkan pipeline for the screens: {error}")),
    }
}

/// a sampler, smooth or point, clamped to the picture's edge.
unsafe fn create_sampler(device: &ash::Device, smooth: bool) -> Result<vk::Sampler, String> {
    let filter = match smooth {
        true => vk::Filter::LINEAR,
        false => vk::Filter::NEAREST,
    };
    // SAFETY: a sampler of this device
    unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            None,
        )
    }
    .map_err(|error| format!("no Vulkan sampler: {error}"))
}

/// a pool with two sets of the layout, and the two sets.
unsafe fn create_descriptors(
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
) -> Result<(vk::DescriptorPool, [[vk::DescriptorSet; 2]; FRAMES]), String> {
    // a set a screen a frame: the two in flight never share one
    let count = (FRAMES * 2) as u32;
    let sizes = [
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLER).descriptor_count(count),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(count),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(count),
    ];
    // SAFETY: a pool of this device for the handles the sets hold
    let pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(count).pool_sizes(&sizes), None) }
        .map_err(|error| format!("no Vulkan descriptor pool: {error}"))?;
    let layouts = [layout; FRAMES * 2];
    // SAFETY: sets of the pool just made, of a layout it takes
    let sets = unsafe {
        device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))
    }
    .map_err(|error| format!("no Vulkan descriptor sets: {error}"))?;
    let mut made = [[vk::DescriptorSet::null(); 2]; FRAMES];
    for frame in 0..FRAMES {
        made[frame] = [sets[frame * 2], sets[frame * 2 + 1]];
    }
    Ok((pool, made))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the layout is as far right and down as the areas reach, never nothing.
    #[test]
    fn the_layout_holds_every_area() {
        assert_eq!(layout_of(&[Some([0, 0, 400, 240]), Some([0, 240, 320, 240])]), (400, 480));
        assert_eq!(layout_of(&[Some([10, 20, 30, 40]), None]), (40, 60));
        assert_eq!(layout_of(&[None, None]), (1, 1));
    }

    /// the shader the presenter builds naga through is valid and has both
    /// entry points.
    #[test]
    fn the_shaders_compile() {
        for (source, what) in [(SHADERS, "screens"), (OVERLAY_SHADERS, "menu")] {
            let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|error| panic!("the {what} shader does not parse, {error}"));
            let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
            validator.validate(&module).unwrap_or_else(|error| panic!("the {what} shader does not validate, {error}"));
            let names: Vec<&str> = module.entry_points.iter().map(|entry| entry.name.as_str()).collect();
            assert!(names.contains(&"vs") && names.contains(&"fs"), "{what}: {names:?}");
        }
    }

    /// an overlay's clip is cut to the window, and nothing shows outside it.
    #[test]
    fn an_overlays_clip_stays_in_the_window() {
        let extent = vk::Extent2D { width: 640, height: 480 };
        let rect = [0.0, 0.0, 640.0, 480.0];
        let cut = scissor_of([0, 0, 640, 480], rect, 1.0, 1.0, extent).expect("the whole of it shows");
        assert_eq!((cut.offset.x, cut.offset.y, cut.extent.width, cut.extent.height), (0, 0, 640, 480));
        let cut = scissor_of([600, 460, 700, 500], rect, 1.0, 1.0, extent).expect("its top left corner shows");
        assert_eq!((cut.offset.x, cut.offset.y, cut.extent.width, cut.extent.height), (600, 460, 40, 20));
        assert!(scissor_of([700, 500, 800, 600], rect, 1.0, 1.0, extent).is_none());
    }

    /// a cursor's marks sit centred on the finger's spot, as GLES cuts them.
    #[test]
    fn the_cursor_sits_on_its_spot() {
        assert_eq!(cursor_rect(100, 50, 7), [97.0, 47.0, 7.0, 7.0]);
        assert_eq!(cursor_rect(10, 10, 15), [3.0, 3.0, 15.0, 15.0]);
    }

    /// the menu's sets make more pools rather than run out, and a set a
    /// dropped texture gives back is taken up again.
    #[test]
    fn the_sets_grow_and_are_reused() {
        let mut budget = SetBudget::new(4);
        for _ in 0..100 {
            budget.take();
        }
        assert_eq!(budget.taken(0), Some(4), "the first pool filled");
        assert!(budget.pools() > 1, "another pool was made");
        for pool in 0..budget.pools() {
            let capacity = budget.capacity(pool).expect("a pool was made");
            assert!(budget.taken(pool).expect("a pool was made") <= capacity, "a pool gave out more than it holds");
        }

        let mut budget = SetBudget::new(2);
        let first = budget.take();
        budget.take();
        let pools = budget.pools();
        budget.give(first);
        assert_eq!(budget.take(), first, "the pool with room is taken again");
        assert_eq!(budget.pools(), pools, "no pool was added");

        // churn settles: taking and giving never grows past one pool
        let mut budget = SetBudget::new(2);
        for _ in 0..50 {
            let pool = budget.take();
            budget.give(pool);
        }
        assert_eq!(budget.pools(), 1);
    }
}
