//! the rendering backend interface.

/// which backend a frontend asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RendererKind {
    /// no 3D rasterization: the framebuffers are presented as the fill and
    /// transfer engines leave them.
    #[default]
    Software,
    /// the OpenGL ES renderer, the one that runs on the handhelds.
    OpenGl,
    /// the Vulkan renderer, for a driver that has Vulkan and can spare the
    /// GLES shader compiler.
    Vulkan,
}

impl RendererKind {
    /// the backend a name stands for.
    pub fn parse(name: &str) -> Option<RendererKind> {
        match name.to_ascii_lowercase().as_str() {
            "software" | "none" | "null" => Some(RendererKind::Software),
            "opengl" | "opengl es" | "gl" | "gles" => Some(RendererKind::OpenGl),
            "vulkan" | "vk" => Some(RendererKind::Vulkan),
            _ => None,
        }
    }
}

/// everything a draw needs, handed over without copying the register file.
pub struct DrawCall<'a> {
    pub indexed: bool,
    pub registers: &'a [u32],
}

pub trait Renderer {
    /// name shown in logs and in the window title.
    fn name(&self) -> &'static str;

    /// executes a draw with the current register state.
    fn draw(&mut self, call: DrawCall<'_>);

    /// called once per emulated frame, after both screens have been
    /// transferred, so a deferred backend can flush.
    fn end_frame(&mut self) {}
}

/// the do-nothing backend.
#[derive(Default)]
pub struct SoftwareRenderer {
    pub draws: u64,
}

impl Renderer for SoftwareRenderer {
    fn name(&self) -> &'static str {
        "software"
    }

    fn draw(&mut self, _call: DrawCall<'_>) {
        self.draws += 1;
    }
}
