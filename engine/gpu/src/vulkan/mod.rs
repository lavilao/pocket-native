//! a Vulkan 1.1 device, found and opened, and what it offers.
//!
//! the 3D stays on OpenGL ES here. this is the device a Vulkan renderer
//! draws on, and the answer to whether one can run on a given phone at all.
//! upstream zakuro's renderer asks Vulkan 1.3 of a driver — it pushes
//! descriptor sets and draws without render passes under synchronization2 —
//! and this port targets 1.1, so what a driver has of those is reported
//! rather than demanded: a phone that reports 1.1 and push descriptors is a
//! phone the port has to bring the render passes and the older barriers to.

use std::ffi::CStr;

use ash::vk;

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
    /// VK_KHR_push_descriptor, which is how upstream's renderer hands a draw
    /// its textures and uniforms: an extension on 1.1, core from 1.4.
    pub push_descriptors: bool,
    /// dynamic rendering, drawing without render passes: core from 1.3.
    pub dynamic_rendering: bool,
    /// synchronization2, the newer barriers: core from 1.3.
    pub synchronization2: bool,
}

impl Report {
    /// whether a device does all of what upstream's renderer asks for, so
    /// that code would take it as it stands.
    pub fn meets_upstream(&self) -> bool {
        self.push_descriptors && self.dynamic_rendering && self.synchronization2
    }

    /// the report, one line for the log.
    pub fn line(&self) -> String {
        let mut lacks = Vec::new();
        if !self.push_descriptors {
            lacks.push("push descriptors");
        }
        if !self.dynamic_rendering {
            lacks.push("dynamic rendering");
        }
        if !self.synchronization2 {
            lacks.push("synchronization2");
        }
        let lacks = match lacks.is_empty() {
            true => "all upstream's renderer asks for".to_owned(),
            false => format!("no {}", lacks.join(", ")),
        };
        format!("Vulkan {} on {} (driver {}), queue {}, {lacks}", self.api, self.name, self.driver, self.queue)
    }
}

/// the system's Vulkan library. android's loader is libvulkan.so, while
/// everywhere else the loader answers to libvulkan.so.1.
fn entry() -> Result<ash::Entry, String> {
    #[cfg(target_os = "android")]
    {
        // safe: loading the system's Vulkan library is the documented way to
        // start, and libvulkan.so is the name android gives it
        if let Ok(entry) = unsafe { ash::Entry::load_from("libvulkan.so") } {
            return Ok(entry);
        }
    }
    // safe: ditto, by the loader's own name
    unsafe { ash::Entry::load() }.map_err(|error| format!("no Vulkan library: {error}"))
}

/// a Vulkan 1.1 instance, with no surface: draws only, showing the screens
/// stays the panel's job.
fn instance(entry: &ash::Entry) -> Result<ash::Instance, String> {
    let app = vk::ApplicationInfo::default()
        .application_name(c"pocket-native")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    // safe: an instance of an application info this call owns
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
fn pick(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32), String> {
    // safe: a query of the instance
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(|error| format!("no Vulkan devices: {error}"))?;
    let mut best: Option<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32)> = None;
    for physical in devices {
        // safe: queries on a device of the instance
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

/// whether a device lists an extension by name.
fn offers(instance: &ash::Instance, physical: vk::PhysicalDevice, name: &CStr) -> bool {
    // safe: a query on a device of the instance
    unsafe { instance.enumerate_device_extension_properties(physical) }
        .map(|offered| offered.iter().any(|extension| extension.extension_name_as_c_str() == Ok(name)))
        .unwrap_or(false)
}

/// a version register as "major.minor.patch".
fn version(api: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(api), vk::api_version_minor(api), vk::api_version_patch(api))
}

/// the device's report, or why it gave none.
fn report(instance: &ash::Instance) -> Result<Report, String> {
    let (physical, properties, queue) = pick(instance)?;
    // safe: a device's name is a NUL-terminated string of its properties
    let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy().into_owned();
    // from 1.3 these are part of Vulkan itself, before that an extension
    let core_1_3 = properties.api_version >= vk::API_VERSION_1_3;
    Ok(Report {
        name,
        api: version(properties.api_version),
        driver: properties.driver_version,
        discrete: properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
        queue,
        push_descriptors: offers(instance, physical, c"VK_KHR_push_descriptor"),
        dynamic_rendering: core_1_3 || offers(instance, physical, c"VK_KHR_dynamic_rendering"),
        synchronization2: core_1_3 || offers(instance, physical, c"VK_KHR_synchronization2"),
    })
}

/// what a Vulkan renderer would have to draw on: the device, its API, and
/// what it does of what upstream asks.
pub fn describe() -> Result<Report, String> {
    let entry = entry()?;
    let instance = instance(&entry)?;
    let report = report(&instance);
    // safe: nothing made from the instance outlives this call
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

    fn sample(push_descriptors: bool, dynamic_rendering: bool, synchronization2: bool) -> Report {
        Report {
            name: "test".into(),
            api: "1.3.0".into(),
            driver: 0,
            discrete: false,
            queue: 0,
            push_descriptors,
            dynamic_rendering,
            synchronization2,
        }
    }

    /// a device that has everything upstream asks for needs nothing
    /// backported.
    #[test]
    fn a_full_device_meets_upstream() {
        let report = sample(true, true, true);
        assert!(report.meets_upstream());
        assert!(report.line().contains("all upstream's renderer asks for"), "{}", report.line());
    }

    /// a 1.1 phone lacks all three, and the line names them.
    #[test]
    fn a_1_1_device_says_what_it_lacks() {
        let report = sample(false, false, false);
        assert!(!report.meets_upstream());
        let line = report.line();
        assert!(line.contains("no push descriptors, dynamic rendering, synchronization2"), "{line}");
    }

    /// one thing missing is named alone.
    #[test]
    fn one_missing_thing_is_named_alone() {
        let report = sample(true, false, true);
        assert!(!report.meets_upstream());
        assert!(report.line().contains("no dynamic rendering"), "{}", report.line());
    }
}
