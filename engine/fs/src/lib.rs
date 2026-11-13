//! ROM container parsing, NCSD cartridge images, CIA archives, NCCH
//! partitions, ExeFS, RomFS and the BLZ compression the CTR SDK applies to
//! .code.

pub mod cia;
pub mod exefs;
pub mod lz77;
pub mod ncch;
pub mod ncsd;
mod reader;
pub mod romfs;
pub mod romfs_build;

use std::path::{Path, PathBuf};
use std::sync::Arc;

pub use cia::{version_name, Cia, Content, TitleKind};
pub use exefs::ExeFs;
pub use ncch::{n3ds_application_memory, CodeSetInfo, ExHeader, MemoryType, NcchHeader, SystemMode};
pub use ncsd::Ncsd;
pub use romfs::RomFs;

#[derive(Debug, thiserror::Error)]
pub enum FsError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("bad magic at 0x{offset:X}: expected {expected:?}, got {got:?}")]
    BadMagic {
        expected: [u8; 4],
        got: [u8; 4],
        offset: usize,
    },

    #[error("read past end of data: offset 0x{offset:X} in a {len}-byte region")]
    Truncated { offset: usize, len: usize },

    #[error("unrecognized ROM format")]
    UnknownFormat,

    #[error("cartridge has no executable partition")]
    NoExecutablePartition,

    #[error("this NCCH is encrypted (crypto method 0x{0:02X}); Zakuro needs a decrypted dump")]
    Encrypted(u8),

    #[error("this .cia is encrypted; Zakuro needs a decrypted one")]
    EncryptedCia,

    #[error("this .cia is {0}, not a game")]
    NotAGame(&'static str),

    #[error("malformed CIA: {0}")]
    BadCia(&'static str),

    #[error("the RomFS can't be read, the dump is still partly encrypted or damaged: {0}")]
    UnreadableRomFs(String),

    #[error("ExeFS has no .code section")]
    NoCode,

    #[error("malformed BLZ stream: {0}")]
    BadLz77(&'static str),

    #[error("malformed RomFS: {0}")]
    BadRomFs(&'static str),

    #[error("path not found in RomFS: {0}")]
    PathNotFound(String),

    #[error("section too large: {what} is {got} bytes, the limit is {max}")]
    TooLarge {
        what: &'static str,
        got: u64,
        max: u64,
    },
}

/// a ROM file read on demand, a block at a time. games run to 4 GiB and a
/// 32-bit host has no spare virtual space for a whole-file mapping (the
/// fastmem view already holds about a GiB of it), so headers, tables and
/// file data come in through pread instead and the kernel's page cache
/// does the keeping. offsets stay u64 all the way, so nothing truncates
/// on a 32-bit host.
#[derive(Debug)]
pub struct RomImage {
    path: PathBuf,
    file: std::fs::File,
    len: u64,
}

impl RomImage {
    pub fn open(path: impl AsRef<Path>) -> Result<RomImage, FsError> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        let len = file.metadata()?.len();
        Ok(RomImage { path, file, len })
    }

    /// a ROM that arrived as an open file, e.g. a file descriptor from a
    /// document picker, rather than as a path.
    pub fn from_file(file: std::fs::File, path: impl AsRef<Path>) -> Result<RomImage, FsError> {
        let len = file.metadata()?.len();
        Ok(RomImage {
            path: path.as_ref().to_path_buf(),
            file,
            len,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// reads exactly buf.len() bytes at offset, without moving the file's
    /// cursor, so concurrent readers need no lock.
    pub fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FsError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset)?;
        }
        #[cfg(not(unix))]
        {
            self.file.read_exact_at(offset, buf)?;
        }
        Ok(())
    }

    /// len bytes at offset, checked against the image before allocating.
    pub fn read_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, FsError> {
        let end = offset.checked_add(len as u64).ok_or(FsError::Truncated {
            offset: offset as usize,
            len: self.len as usize,
        })?;
        if end > self.len {
            return Err(FsError::Truncated {
                offset: offset as usize,
                len: self.len as usize,
            });
        }
        let mut out = vec![0u8; len];
        self.read_exact_at(offset, &mut out)?;
        Ok(out)
    }

    /// read_at with a ceiling, for headers and tables whose stated size
    /// comes from the file and cannot be trusted to be small.
    pub fn read_bounded(&self, offset: u64, len: u64, max: u64, what: &'static str) -> Result<Vec<u8>, FsError> {
        if len > max {
            return Err(FsError::TooLarge { what, got: len, max });
        }
        self.read_at(offset, len as usize)
    }
}

// File::read_exact_at is unix; on Windows it is seek_read.
#[cfg(windows)]
trait ReadExactAt {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()>;
}

#[cfg(windows)]
impl ReadExactAt for std::fs::File {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        use std::os::windows::fs::FileExt;
        self.seek_read(buf, offset)
    }
}

#[cfg(not(any(unix, windows)))]
trait ReadExactAt {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()>;
}

#[cfg(not(any(unix, windows)))]
impl ReadExactAt for std::fs::File {
    fn read_exact_at(&self, _offset: u64, _buf: &mut [u8]) -> std::io::Result<()> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "pread is not available on this platform"))
    }
}

/// a loaded title, the executable NCCH of a cartridge, with its headers parsed
/// and its filesystems located.
pub struct Title {
    /// shared, since a file opened through the fs service reads through it
    /// from another thread while the title is in use.
    image: Arc<RomImage>,
    /// offset of the CXI inside the image.
    ncch_offset: u64,
    pub ncch: NcchHeader,
    pub exheader: ExHeader,
    pub exefs: ExeFs,
    exefs_offset: u64,
    pub romfs: Option<RomFs>,
}

impl Title {
    pub fn load(path: impl AsRef<Path>) -> Result<Title, FsError> {
        let image = Arc::new(RomImage::open(path)?);
        // the format headers live at the start of the file: the first
        // 0x2020 bytes hold a CIA header and every magic (NCSD/NCCH at
        // 0x100), so a bounded head read decides the format.
        let head = image.read_at(0, (image.len().min(0x2020)) as usize)?;

        let ncch_offset = if Ncsd::detect(&head) {
            let ncsd = Ncsd::parse(&image.read_at(0, (image.len().min(0x200)) as usize)?)?;
            log::info!(
                "NCSD cartridge, media id {:016X}, image size {} MiB",
                ncsd.media_id,
                ncsd.image_size / (1024 * 1024)
            );
            ncsd.executable_partition()?.offset
        } else if Cia::detect(&head) {
            let cia = Cia::parse_from(&image)?;
            let (offset, _size) = cia.executable().ok_or(FsError::NoExecutablePartition)?;
            log::info!("CIA archive, title {:016X}", cia.title_id);
            offset
        } else if NcchHeader::detect(&head) {
            0
        } else {
            return Err(FsError::UnknownFormat);
        };

        let ncch = NcchHeader::parse(&image.read_at(ncch_offset, NcchHeader::SIZE)?)?;

        if !ncch.is_decrypted() && ncch.crypto_method() != 0 {
            return Err(FsError::Encrypted(ncch.crypto_method()));
        }

        let exheader_raw = image.read_at(ncch_offset + NcchHeader::SIZE as u64, ExHeader::HASHED_SIZE)?;
        let exheader = ExHeader::parse(&exheader_raw)?;

        let exefs_offset = ncch_offset + ncch.exefs_offset;
        let exefs = ExeFs::parse(&image.read_at(exefs_offset, exefs::HEADER_SIZE as usize)?)?;

        // a game reads its data from the RomFS, without one that reads it
        // stops on its first file
        let romfs = if ncch.has_romfs() {
            let fs = RomFs::parse_from(&image, ncch_offset + ncch.romfs_offset)
                .map_err(|error| FsError::UnreadableRomFs(error.to_string()))?;
            Some(fs)
        } else {
            None
        };

        Ok(Title {
            image,
            ncch_offset,
            ncch,
            exheader,
            exefs,
            exefs_offset,
            romfs,
        })
    }

    pub fn image(&self) -> &RomImage {
        &self.image
    }

    /// the image itself, for a file the fs service reads through while the
    /// system holds the title.
    pub fn shared_image(&self) -> Arc<RomImage> {
        self.image.clone()
    }

    /// whether image is this title's own, whose blocks the frontend reads
    /// ahead; DLC contents are other images, read at their own offsets.
    pub fn is_image(&self, image: &Arc<RomImage>) -> bool {
        Arc::ptr_eq(&self.image, image)
    }

    pub fn ncch_offset(&self) -> u64 {
        self.ncch_offset
    }

    /// raw bytes of an ExeFS entry, read from the image on demand.
    pub fn exefs_file(&self, name: &str) -> Option<Vec<u8>> {
        let entry = self.exefs.find(name)?;
        let start = self.exefs_offset + exefs::HEADER_SIZE + entry.offset;
        self.image.read_at(start, entry.size as usize).ok()
    }

    /// the executable image, decompressed if the exheader says it is BLZ'd.
    pub fn code(&self) -> Result<Vec<u8>, FsError> {
        let raw = self.exefs_file(".code").ok_or(FsError::NoCode)?;
        if self.exheader.compress_code {
            lz77::decompress(&raw)
        } else {
            Ok(raw)
        }
    }

    /// reads len bytes at offset from a RomFS file. this copies, so a
    /// runtime caller should read through the fs service instead, which
    /// already copies into guest memory.
    pub fn read_romfs(&self, file: &romfs::FileEntry, offset: u64, len: usize) -> Option<Vec<u8>> {
        let fs = self.romfs.as_ref()?;
        if offset >= file.data_size {
            return Some(Vec::new());
        }
        let avail = (file.data_size - offset) as usize;
        let len = len.min(avail);
        let start = fs.file_data_offset(file) + offset;
        self.image.read_at(start, len).ok()
    }

    pub fn program_id(&self) -> u64 {
        self.ncch.program_id
    }

    /// human-readable one-liner for the log banner.
    pub fn describe(&self) -> String {
        format!(
            "{} [{}] title={:016X} v{}",
            self.exheader.title, self.ncch.product_code, self.ncch.program_id, self.ncch.version
        )
    }
}

/// downloadable content, from its CIA: contents of data a game opens by
/// their index, each an NCCH with a RomFS, read from the image on demand.
pub struct Dlc {
    image: Arc<RomImage>,
    pub title_id: u64,
    pub version: u16,
    /// every content its TMD lists, those the file holds with a place.
    pub contents: Vec<Content>,
}

impl Dlc {
    pub fn load(path: impl AsRef<Path>) -> Result<Dlc, FsError> {
        let image = Arc::new(RomImage::open(path)?);
        let head = image.read_at(0, (image.len().min(0x2020)) as usize)?;
        if !Cia::detect(&head) {
            return Err(FsError::BadCia("DLC comes as a CIA"));
        }
        let cia = Cia::read_from(&image)?;
        if cia.kind() != TitleKind::Dlc {
            return Err(FsError::BadCia("it is not DLC"));
        }
        // a content the file holds but cannot be read is worse than none: the
        // game would open it and read noise, so the whole archive is refused
        if cia.contents.iter().any(|content| content.offset.is_some() && content.encrypted()) {
            return Err(FsError::EncryptedCia);
        }
        Ok(Dlc { image, title_id: cia.title_id, version: cia.version, contents: cia.contents })
    }

    pub fn image(&self) -> Arc<RomImage> {
        self.image.clone()
    }

    /// the content of an index, none when the TMD does not list it.
    pub fn content(&self, index: u16) -> Option<Content> {
        self.contents.iter().find(|content| content.index == index).copied()
    }

    /// where the RomFS of the content of an index lies in the image, the
    /// window a game reads it through, none when the file leaves the content
    /// out or the NCCH behind it is encrypted or has no RomFS.
    pub fn romfs(&self, index: u16) -> Option<(u64, u64)> {
        let offset = self.contents.iter().find(|content| content.index == index)?.offset?;
        let head = self.image.read_at(offset, NcchHeader::SIZE).ok()?;
        let ncch = NcchHeader::parse(&head).ok()?;
        if !ncch.has_romfs() || (!ncch.is_decrypted() && ncch.crypto_method() != 0) {
            return None;
        }
        let romfs = RomFs::parse_from(&self.image, offset + ncch.romfs_offset).ok()?;
        let end = offset + ncch.romfs_offset + ncch.romfs_size;
        Some((romfs.base, end.saturating_sub(romfs.base)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("zakuro-fs-test-{}-{name}", std::process::id()))
    }

    fn image_with(bytes: &[u8], name: &str) -> (PathBuf, RomImage) {
        let path = scratch(name);
        std::fs::write(&path, bytes).unwrap();
        let image = RomImage::open(&path).unwrap();
        (path, image)
    }

    #[test]
    fn reads_at_zero_in_the_middle_and_at_the_end() {
        let (_path, image) = image_with(b"0123456789abcdef", "reads.3ds");
        assert_eq!(image.len(), 16);
        assert_eq!(image.read_at(0, 4).unwrap(), b"0123");
        assert_eq!(image.read_at(10, 6).unwrap(), b"abcdef");
        assert_eq!(image.read_at(16, 0).unwrap(), Vec::<u8>::new());
        assert!(matches!(image.read_at(16, 1), Err(FsError::Truncated { .. })));
        assert!(matches!(image.read_at(15, 2), Err(FsError::Truncated { .. })));
        assert!(matches!(image.read_at(u64::MAX - 1, 4), Err(FsError::Truncated { .. })));
    }

    #[test]
    fn a_bounded_read_refuses_a_size_past_its_limit() {
        let (_path, image) = image_with(&[0u8; 1024], "bounded.3ds");
        assert!(matches!(
            image.read_bounded(0, 1 << 20, 1024, "a table"),
            Err(FsError::TooLarge { got, max, .. }) if got == 1 << 20 && max == 1024
        ));
        assert_eq!(image.read_bounded(0, 16, 1024, "a table").unwrap().len(), 16);
    }

    /// a sparse image past 4 GiB reads back across the 2^32 boundary,
    /// which is the offset a 32-bit host truncates if it goes through
    /// usize anywhere.
    #[cfg(unix)]
    #[test]
    fn offsets_past_4_gib_are_not_truncated() {
        use std::os::unix::fs::FileExt;
        let path = scratch("big.3ds");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(5 << 30).unwrap();
        file.write_all_at(b"low", 0xFFFF_FFF0).unwrap();
        file.write_all_at(b"high", 0x1_0000_0000).unwrap();
        drop(file);

        let image = RomImage::open(&path).unwrap();
        assert_eq!(image.len(), 5 << 30);
        assert_eq!(image.read_at(0xFFFF_FFF0, 3).unwrap(), b"low");
        assert_eq!(image.read_at(0x1_0000_0000, 4).unwrap(), b"high");
        assert_eq!(&image.read_at(0xFFFF_FFF8, 16).unwrap()[8..12], b"high");
    }

    /// a DLC's CIA of one delivered content, an NCCH whose RomFS holds one
    /// file: the archive, where the content starts and the level-3 image.
    fn dlc_cia(title_id: u64, flags: u16) -> (Vec<u8>, u64, Vec<u8>) {
        use crate::romfs_build::{build, BuildFile};
        // the TMD's own header, past the signature and its padding
        const TMD_HEADER: usize = 0x140;
        // its content records follow 64 content info records
        const RECORDS: usize = 0xC4 + 64 * 0x24;
        let level3 = build(&[BuildFile { path: "addon.bin".into(), data: vec![1, 2, 3, 4] }]);
        // the NCCH: its header, an IVFC header a page long, the level-3 image
        let romfs_offset = 0x200;
        let romfs_size = (0x1000 + level3.len()).next_multiple_of(0x200);
        let mut content = vec![0u8; romfs_offset + romfs_size];
        content[0x100..0x104].copy_from_slice(b"NCCH");
        content[0x1B0..0x1B4].copy_from_slice(&((romfs_offset / 0x200) as u32).to_le_bytes());
        content[0x1B4..0x1B8].copy_from_slice(&((romfs_size / 0x200) as u32).to_le_bytes());
        // flags[7]: decrypted, and with a RomFS
        content[0x18F] = 4;
        content[romfs_offset..romfs_offset + 4].copy_from_slice(b"IVFC");
        content[romfs_offset + 4..romfs_offset + 8].copy_from_slice(&0x0001_0000u32.to_le_bytes());
        // level 3's blocks are a page, which is where its image then starts
        content[romfs_offset + 0x0C + 2 * 0x18 + 0x10..][..4].copy_from_slice(&12u32.to_le_bytes());
        content[romfs_offset + 0x1000..romfs_offset + 0x1000 + level3.len()].copy_from_slice(&level3);

        let (certificates, ticket) = (0xA00u32, 0x350u32);
        let tmd_size = TMD_HEADER + RECORDS + 0x30;
        let tmd_offset = (0x2020u64 + certificates as u64).next_multiple_of(0x40) + ticket as u64;
        let tmd_offset = tmd_offset.next_multiple_of(0x40);
        let mut data = vec![0u8; 0x2020];
        data[..4].copy_from_slice(&0x2020u32.to_le_bytes());
        data[0x08..0x0C].copy_from_slice(&certificates.to_le_bytes());
        data[0x0C..0x10].copy_from_slice(&ticket.to_le_bytes());
        data[0x10..0x14].copy_from_slice(&(tmd_size as u32).to_le_bytes());
        // content 0 is the one the file holds
        data[0x20] = 0x80;
        data.resize(tmd_offset as usize + tmd_size, 0);
        let tmd = &mut data[tmd_offset as usize..];
        tmd[..4].copy_from_slice(&0x0001_0004u32.to_be_bytes());
        let header = &mut tmd[TMD_HEADER..];
        header[0x4C..0x54].copy_from_slice(&title_id.to_be_bytes());
        header[0x9C..0x9E].copy_from_slice(&0x0400u16.to_be_bytes());
        header[0x9E..0xA0].copy_from_slice(&1u16.to_be_bytes());
        let record = &mut header[RECORDS..];
        record[0x00..0x04].copy_from_slice(&1u32.to_be_bytes());
        record[0x06..0x08].copy_from_slice(&flags.to_be_bytes());
        record[0x08..0x10].copy_from_slice(&(content.len() as u64).to_be_bytes());

        let content_offset = (tmd_offset + tmd_size as u64).next_multiple_of(0x40);
        data.resize(content_offset as usize + content.len(), 0);
        data[content_offset as usize..].copy_from_slice(&content);
        (data, content_offset, level3)
    }

    #[test]
    fn a_dlc_content_reads_through_its_romfs() {
        let (data, content_offset, level3) = dlc_cia(0x0004_008C_0016_4800, 0);
        let path = scratch("addon.cia");
        std::fs::write(&path, &data).unwrap();
        let dlc = Dlc::load(&path).unwrap();
        assert_eq!(dlc.title_id, 0x0004_008C_0016_4800);
        assert_eq!(version_name(dlc.version), "1.0.0");

        // everything past the IVFC header, which is what a game reads
        let (offset, size) = dlc.romfs(0).expect("content 0 has a RomFS");
        assert_eq!(offset, content_offset + 0x200 + 0x1000);
        assert_eq!(size, (0x1000 + level3.len() as u64).next_multiple_of(0x200) - 0x1000);
        assert_eq!(dlc.image().read_at(offset, level3.len()).unwrap(), level3);
        // and an index its TMD does not list is simply not there
        assert_eq!(dlc.romfs(1), None);

        // a game is not DLC, and a content too encrypted to read is refused
        std::fs::write(&path, dlc_cia(0x0004_0000_0016_4800, 0).0).unwrap();
        assert!(matches!(Dlc::load(&path), Err(FsError::BadCia(_))));
        std::fs::write(&path, dlc_cia(0x0004_008C_0016_4800, 1).0).unwrap();
        assert!(matches!(Dlc::load(&path), Err(FsError::EncryptedCia)));
    }
}
