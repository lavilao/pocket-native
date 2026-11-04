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

pub use cia::Cia;
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
    image: RomImage,
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
        let image = RomImage::open(path)?;
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
            log::info!("CIA archive, title {:016X}", cia.title_id);
            cia.executable.0
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
}
