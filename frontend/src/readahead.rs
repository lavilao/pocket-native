//! reading the ROM blocks earlier runs read ahead of the title, at idle
//! priority, so that it finds them in memory rather than waiting on the card.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// how many ROM blocks are read ahead, 128 MiB, what memory holds beside
/// the title on a 1 GiB handheld.
const READ_AHEAD_BLOCKS: usize = 2048;

/// reads the blocks of the ROM image, first read first, at idle priority,
/// leaving them in the system's cache.
fn read_ahead(path: &str, blocks: Vec<u32>) {
    let Ok(file) = std::fs::File::open(path) else { return };
    std::thread::spawn(move || {
        use std::os::unix::fs::FileExt;
        // SAFETY: plain calls that lower this thread's own priorities.
        // ioprio_set stays off Android: its seccomp filter kills the
        // process with SIGSYS on a blocked syscall instead of returning
        // an error, so there is no failure to fall back on there. nice
        // goes through setpriority, which Android allows.
        unsafe {
            libc::nice(19);
            #[cfg(not(target_os = "android"))]
            libc::syscall(libc::SYS_ioprio_set, 1, 0, 3 << 13);
        }
        let started = Instant::now();
        let block = zakuro_core::services::fs::ROM_BLOCK;
        let mut buffer = vec![0u8; block];
        let mut read = 0usize;
        for &index in blocks.iter().take(READ_AHEAD_BLOCKS) {
            read += file.read_at(&mut buffer, index as u64 * block as u64).unwrap_or(0);
        }
        log::info!("read {} MiB of the ROM ahead in {:.1} s", read >> 20, started.elapsed().as_secs_f64());
    });
}

/// the blocks earlier runs read, kept in the data directory.
pub struct ReadAhead {
    path: PathBuf,
    earlier: Vec<u32>,
}

impl ReadAhead {
    /// starts reading ahead what earlier runs of this ROM read, when read is
    /// set, and keeps noting what this run reads either way.
    pub fn start(rom: &str, data: &Path, read: bool) -> ReadAhead {
        let path = data.join("rom-blocks");
        let earlier: Vec<u32> = std::fs::read(&path).unwrap_or_default().as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        // POCKET_NO_READAHEAD=1 turns the background reads off entirely.
        let read = read && std::env::var("POCKET_NO_READAHEAD").map(|v| v != "1").unwrap_or(true);
        if read {
            read_ahead(rom, earlier.clone());
        } else {
            log::info!("switch: ROM read-ahead off");
        }
        ReadAhead { path, earlier }
    }

    /// saves the blocks this run read first, then the earlier ones it did not.
    pub fn save(&self) {
        let now = zakuro_core::services::fs::rom_blocks();
        let seen: HashSet<u32> = now.iter().copied().collect();
        let merged: Vec<u8> = now.iter().chain(self.earlier.iter().filter(|block| !seen.contains(block))).take(READ_AHEAD_BLOCKS * 2).flat_map(|block| block.to_le_bytes()).collect();
        if crate::settings::write_safely(&self.path, &merged).is_err() {
            log::warn!("could not save {}", self.path.display());
        }
    }
}
