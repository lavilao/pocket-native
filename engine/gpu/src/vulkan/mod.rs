//! the Vulkan 1.1 device, found and opened, and the screens shown through it.
//!
//! GLES stays the renderer the app draws with by default. with the menu's
//! `Show screens with` setting on Vulkan, the 3D is drawn by the software
//! rasterizer and this puts the pictures, and the menu over them, on the
//! window: `present` is the swapchain end of it, and the source every Vulkan
//! call comes from lives in this module.
//!
//! it is Vulkan 1.1 and nothing newer: the descriptor sets are ordinary
//! ones, a render pass sets up the drawing's attachment, and the barriers are
//! the classic form. a device that has none of 1.3 is still enough here.

use std::ffi::{c_char, c_void, CStr};

use ash::vk;

pub mod present;

/// the native window the screens go to, an ANativeWindow on Android, as
/// `set_window` last took it.
static WINDOW: std::sync::atomic::AtomicPtr<c_void> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// hands the Vulkan presenter the window the activity gave the renderer, or
/// takes it back with null. the GLES renderer gets the same one through its
/// own `set_window`; the app's `zakuro_gpu::set_window` feeds both.
pub fn set_window(window: *mut c_void) {
    WINDOW.store(window, std::sync::atomic::Ordering::Release);
}

/// the window the presenter draws a surface on, null for none.
pub(crate) fn window() -> *mut c_void {
    WINDOW.load(std::sync::atomic::Ordering::Acquire)
}

/// what a device offers a Vulkan renderer.
#[derive(Debug, Clone)]
pub struct Report {
    /// the device as its driver names it.
    pub name: String,
    /// the Vulkan version the driver reports, "1.1.0" and up.
    pub api: String,
    /// the driver's own version number.
    pub driver: u32,
    /// a card of its own rather than a GPU beside the CPU.
    pub discrete: bool,
    /// the queue family draws go on.
    pub queue: u32,
}

impl Report {
    /// the report, one line for the log.
    pub fn line(&self) -> String {
        format!("Vulkan {} on {} (driver {}), queue {}", self.api, self.name, self.driver, self.queue)
    }
}

/// the system's Vulkan library. android's loader is libvulkan.so, while
/// everywhere else the loader answers to libvulkan.so.1.
pub(crate) fn entry() -> Result<ash::Entry, String> {
    #[cfg(target_os = "android")]
    {
        // SAFETY: loading the system's Vulkan library is the documented way to
        // start, and libvulkan.so is the name android gives it
        if let Ok(entry) = unsafe { ash::Entry::load_from("libvulkan.so") } {
            return Ok(entry);
        }
    }
    // SAFETY: ditto, by the loader's own name
    unsafe { ash::Entry::load() }.map_err(|error| format!("no Vulkan library: {error}"))
}

/// a Vulkan 1.1 instance with the extensions given.
pub(crate) fn instance(entry: &ash::Entry, extensions: &[*const c_char]) -> Result<ash::Instance, String> {
    let app = vk::ApplicationInfo::default()
        .application_name(c"pocket-native")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(extensions);
    // SAFETY: an instance of an application info this call owns
    unsafe { entry.create_instance(&info, None) }.map_err(|error| format!("no Vulkan 1.1 instance: {error}"))
}

/// how much a device is the one a renderer wants.
fn rank(kind: vk::PhysicalDeviceType) -> u8 {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
        _ => 0,
    }
}

/// a device with a queue draws can go on, a card of its own over a GPU beside
/// the CPU when there is a choice.
pub(crate) fn pick(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32), String> {
    // SAFETY: a query of the instance
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|error| format!("no Vulkan devices: {error}"))?;
    let mut best: Option<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32)> = None;
    for physical in devices {
        // SAFETY: queries on a device of the instance
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let Some(family) = families.iter().position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS)) else {
            continue;
        };
        let better = best.as_ref().is_none_or(|(_, held, _)| rank(properties.device_type) > rank(held.device_type));
        if better {
            best = Some((physical, properties, family as u32));
        }
    }
    best.ok_or_else(|| "no Vulkan device with a queue for draws".to_owned())
}

/// a device with a queue that draws and can show to the surface, which is what
/// a presenter needs.
pub(crate) fn pick_presenting(
    instance: &ash::Instance,
    surface: &ash::khr::surface::Instance,
    handle: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), String> {
    // SAFETY: a query of the instance
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|error| format!("no Vulkan devices: {error}"))?;
    let mut best: Option<(vk::PhysicalDevice, u32, u8)> = None;
    for physical in devices {
        // SAFETY: queries on a device of the instance and its surface
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = families.iter().enumerate().position(|(index, family)| {
            family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && unsafe { surface.get_physical_device_surface_support(physical, index as u32, handle) }.unwrap_or(false)
        });
        let Some(family) = family else { continue };
        let better = best.as_ref().is_none_or(|(_, _, held)| rank(properties.device_type) > *held);
        if better {
            best = Some((physical, family as u32, rank(properties.device_type)));
        }
    }
    best.map(|(physical, family, _)| (physical, family))
        .ok_or_else(|| "no Vulkan device that both draws and shows the screens".to_owned())
}

/// a version register as "major.minor.patch".
fn version(api: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(api), vk::api_version_minor(api), vk::api_version_patch(api))
}

/// the device's report, or why it gave none.
fn report(instance: &ash::Instance) -> Result<Report, String> {
    let (_, properties, queue) = pick(instance)?;
    // SAFETY: a device's name is a NUL-terminated string of its properties
    let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy().into_owned();
    Ok(Report {
        name,
        api: version(properties.api_version),
        driver: properties.driver_version,
        discrete: properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
        queue,
    })
}

/// what a Vulkan renderer would draw on: the device, its API and its queue.
pub fn describe() -> Result<Report, String> {
    let entry = entry()?;
    let instance = instance(&entry, &[])?;
    let report = report(&instance);
    // SAFETY: nothing made from the instance outlives this call
    unsafe { instance.destroy_instance(None) };
    report
}

/// the report as the log takes it, one line either way.
pub fn line() -> String {
    match describe() {
        Ok(report) => report.line(),
        Err(error) => format!("no Vulkan device: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a version register reads as its three numbers.
    #[test]
    fn a_version_reads_as_its_numbers() {
        assert_eq!(version(vk::make_api_version(0, 1, 1, 0)), "1.1.0");
        assert_eq!(version(vk::make_api_version(0, 1, 3, 281)), "1.3.281");
    }

    /// a card of its own is the one a renderer wants over a GPU beside the
    /// CPU, which beats a virtual one.
    #[test]
    fn a_card_beats_a_gpu_beside_the_cpu() {
        assert!(rank(vk::PhysicalDeviceType::DISCRETE_GPU) > rank(vk::PhysicalDeviceType::INTEGRATED_GPU));
        assert!(rank(vk::PhysicalDeviceType::INTEGRATED_GPU) > rank(vk::PhysicalDeviceType::VIRTUAL_GPU));
        assert!(rank(vk::PhysicalDeviceType::VIRTUAL_GPU) > rank(vk::PhysicalDeviceType::OTHER));
    }
}
