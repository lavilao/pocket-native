//! the screens on a Vulkan 1.1 swapchain.
//!
//! the engine draws the 3D in software and hands this the pictures it made —
//! the same `PanelImage::Pixels` an OpenGL ES presenter gets — and this puts
//! them on the window: instance, surface, device, swapchain, one render pass
//! with a colour attachment, a pipeline, a descriptor set a screen, classic
//! barriers, semaphores and fences, present.
//!
//! nothing here is Vulkan 1.3: the uniform block and the sampled picture are
//! ordinary descriptor set bindings (no push descriptors), a real render pass
//! does the colour attachment's layout transitions (no dynamic rendering),
//! and the barriers are vkCmdPipelineBarrier, not its 2 form.

use std::ffi::{c_char, c_void, CStr, CString};

use ash::vk;

use crate::PanelImage;

/// frames the CPU may get ahead of the GPU, each with its own commands, its
/// fence and its acquire semaphore.
const FRAMES: usize = 2;
/// the colour formats taken when the surface offers them, best first.
const FORMATS: [vk::Format; 3] = [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM, vk::Format::A2B10G10R10_UNORM_PACK32];
/// the format the engine's screen pixels are in, straight RGBA8.
const SCREEN_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// a uniform block is two vec4: the quad's place and the picture's uv.
const UNIFORM_BYTES: u64 = 32;

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

/// a uniform block's buffer, kept mapped.
struct Uniform {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
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
        let surface_fn = ash::khr::surface::Instance::load(&entry, &instance);
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
        let swapchain_fn = ash::khr::swapchain::Device::load(&instance, &device);
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

    /// puts a frame's screens on the window. a frame the window will not take
    /// yet is dropped; the next one draws into a swapchain made for its size.
    pub fn present(&mut self, shown: [Option<(PanelImage, [i32; 4])>; 2], smooth: bool) {
        if let Err(error) = self.draw(shown, smooth) {
            log::error!("Vulkan could not show a frame, {error}");
        }
    }

    fn draw(&mut self, shown: [Option<(PanelImage, [i32; 4])>; 2], smooth: bool) -> Result<(), String> {
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
        // SAFETY: ending the pass and the buffer this frame recorded
        unsafe {
            device.cmd_end_render_pass(command);
            device.end_command_buffer(command).map_err(|error| format!("no Vulkan command buffer: {error}"))?;
        }

        let waits = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let buffers = [command];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&[self.available[frame]])
            .wait_destination_stage_mask(&waits)
            .command_buffers(&buffers)
            .signal_semaphores(&[self.finished[index]]);
        // SAFETY: a submit of the frame's own command buffer, fence and
        // semaphores
        unsafe { device.queue_submit(self.queue, &[submit], self.fences[frame]) }
            .map_err(|error| format!("no Vulkan frame submitted: {error}"))?;
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&[self.finished[index]])
            .swapchains(&[self.swapchain])
            .image_indices(&[index as u32]);
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
                self.screens[frame][index].staging = Some(unsafe { create_staging(&device, &self.instance, self.physical, needs) }?);
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
fn surface_extensions() -> Vec<*const c_char> {
    let mut extensions = vec![ash::khr::surface::NAME.as_ptr()];
    #[cfg(target_os = "android")]
    extensions.push(ash::khr::android_surface::NAME.as_ptr());
    extensions
}

/// a surface on the window set_window gave: an android window, which is the
/// window a handheld app is given. a handheld's panel is KMS otherwise, which
/// Vulkan shows to through a display extension most drivers lack.
unsafe fn make_surface(entry: &ash::Entry, instance: &ash::Instance, window: *mut c_void) -> Result<vk::SurfaceKHR, String> {
    let _ = (entry, instance, window);
    #[cfg(target_os = "android")]
    {
        // SAFETY: an android window the activity handed the renderer
        let info = vk::AndroidSurfaceCreateInfoKHR::default().window(window.cast());
        let android = ash::khr::android_surface::Instance::load(entry, instance);
        return unsafe { android.create_android_surface(&info, None) }.map_err(|error| format!("no Vulkan android surface: {error}"));
    }
    #[cfg(not(target_os = "android"))]
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
unsafe fn create_staging(device: &ash::Device, instance: &ash::Instance, physical: vk::PhysicalDevice, size: usize) -> Result<Staging, String> {
    let size = size as u64;
    // SAFETY: a buffer of this device, for bytes the CPU writes
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default().size(size).usage(vk::BufferUsageFlags::TRANSFER_SRC).sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
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
unsafe fn create_shader_module(device: &ash::Device) -> Result<vk::ShaderModule, String> {
    let module = naga::front::wgsl::parse_str(SHADERS).map_err(|error| format!("the screens' shader does not parse: {error}"))?;
    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
    let info = validator.validate(&module).map_err(|error| format!("the screens' shader does not validate: {error}"))?;
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
        .map_err(|error| format!("the screens' shader does not turn into SPIR-V: {error}"))?;
    // SAFETY: words naga wrote for this device's Vulkan
    unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None) }
        .map_err(|error| format!("no Vulkan shader module: {error}"))
}

/// the pipeline: one triangle a screen, no vertex buffer, no depth, no blend,
/// the viewport and scissor set as the frame is recorded.
unsafe fn create_pipeline(device: &ash::Device, layout: vk::PipelineLayout, render_pass: vk::RenderPass) -> Result<vk::Pipeline, String> {
    let module = unsafe { create_shader_module(device) }?;
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
    fn the_screens_shader_compiles() {
        let module = naga::front::wgsl::parse_str(SHADERS).expect("the shader parses");
        let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
        validator.validate(&module).expect("the shader validates");
        let names: Vec<&str> = module.entry_points.iter().map(|entry| entry.name.as_str()).collect();
        assert!(names.contains(&"vs") && names.contains(&"fs"), "{names:?}");
    }
}
