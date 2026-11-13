//! CIA (.cia) installable archives, the way titles come from the eShop. a
//! header, a certificate chain, a ticket, the title's metadata (TMD) and the
//! contents, each section starting on a 64-byte boundary. the content with
//! index 0 is the executable NCCH, the same as a cartridge's first partition.

use crate::reader::Reader;
use crate::FsError;

/// the header's size, which is also its first field.
const HEADER_SIZE: u32 = 0x2020;
const ALIGN: u64 = 0x40;
/// where the TMD's content records start past its signature, after the
/// header and the 64 content info records.
const CONTENT_RECORDS: usize = 0xC4 + 64 * 0x24;
const CONTENT_RECORD_SIZE: usize = 0x30;
/// a content's type flag for being encrypted with the title key.
const CONTENT_ENCRYPTED: u16 = 0x0001;

/// what a title is, by the high half of its id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TitleKind {
    /// a game, or anything else the console runs.
    Game,
    /// an update, the code and the data that replace a game's.
    Update,
    /// downloadable content, data a game reads besides its own.
    Dlc,
    /// a game's Download Play child, the program it sends others.
    DlpChild,
}

/// a title version as the console shows it, major.minor.micro.
pub fn version_name(version: u16) -> String {
    format!("{}.{}.{}", version >> 10, version >> 4 & 0x3F, version & 0xF)
}

impl TitleKind {
    pub fn of(title_id: u64) -> TitleKind {
        match title_id >> 32 {
            0x0004_000E => TitleKind::Update,
            0x0004_008C => TitleKind::Dlc,
            0x0004_0001 => TitleKind::DlpChild,
            _ => TitleKind::Game,
        }
    }
}

/// a CIA's title, its version and its contents.
#[derive(Debug, Clone)]
pub struct Cia {
    pub title_id: u64,
    /// the title's version, as the TMD has it.
    pub version: u16,
    /// every content the TMD lists, in its order, the file holding some.
    pub contents: Vec<Content>,
}

/// one content of a title, as its TMD record describes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Content {
    pub index: u16,
    pub id: u32,
    /// the record's type flags, encrypted, optional and the like.
    pub flags: u16,
    pub size: u64,
    /// where its bytes start in the file, none when the file leaves it out.
    pub offset: Option<u64>,
}

impl Content {
    pub fn encrypted(&self) -> bool {
        self.flags & CONTENT_ENCRYPTED != 0
    }
}

/// the most a title metadata can be: a signature, the header and 64
/// content records. anything larger is not a TMD.
const MAX_TMD: u64 = 1 << 20;

impl Cia {
    /// returns true if data starts with a CIA header.
    pub fn detect(data: &[u8]) -> bool {
        data.len() >= HEADER_SIZE as usize && data[..4] == HEADER_SIZE.to_le_bytes()
    }

    /// any title's CIA, a game's, an update's or a DLC's, read out of a ROM
    /// image: only its header and its title metadata are read, not the
    /// whole archive.
    pub fn read_from(image: &crate::RomImage) -> Result<Cia, FsError> {
        let header = image.read_at(0, HEADER_SIZE as usize)?;
        let r = Reader::new(&header);
        let align = |n: u64| n.next_multiple_of(ALIGN);
        let certificates = r.u32(0x08)? as u64;
        let ticket = r.u32(0x0C)? as u64;
        let tmd_size = r.u32(0x10)? as u64;
        let tmd_offset = align(align(align(HEADER_SIZE as u64) + certificates) + ticket);
        let tmd_bytes = image.read_bounded(tmd_offset, tmd_size, MAX_TMD, "a CIA's title metadata")?;
        Self::from_tmd(&r, &Reader::new(&tmd_bytes), align(tmd_offset + tmd_size))
    }

    /// any title's CIA out of bytes already in memory.
    pub fn read(data: &[u8]) -> Result<Cia, FsError> {
        let r = Reader::new(data);
        let align = |n: u64| n.next_multiple_of(ALIGN);
        let certificates = r.u32(0x08)? as u64;
        let ticket = r.u32(0x0C)? as u64;
        let tmd_size = r.u32(0x10)? as u64;
        let tmd_offset = align(align(align(HEADER_SIZE as u64) + certificates) + ticket);
        let tmd = r.at(tmd_offset as usize)?;
        Self::from_tmd(&r, &tmd, align(tmd_offset + tmd_size))
    }

    /// the title and contents of the TMD in tmd, whose bytes start at
    /// contents_offset in the archive the header r came from.
    fn from_tmd(r: &Reader, tmd: &Reader, contents_offset: u64) -> Result<Cia, FsError> {
        // the signature's size goes by its type, padded to 64 bytes
        let signature = match tmd.u32_be(0)? & 0xFFFF {
            0x0000 | 0x0003 => 4 + 0x200 + 0x3C,
            0x0001 | 0x0004 => 4 + 0x100 + 0x3C,
            0x0002 | 0x0005 => 4 + 0x3C + 0x40,
            _ => return Err(FsError::BadCia("unknown TMD signature type")),
        };
        let tmd = tmd.at(signature)?;
        let title_id = tmd.u64_be(0x4C)?;
        let version = tmd.u16_be(0x9C)?;

        // the contents follow in the order of their records, those the
        // header's index bitmap says are in the file
        let present = |index: u16| {
            let byte = r.u8(0x20 + index as usize / 8).unwrap_or(0);
            byte & (0x80 >> (index % 8)) != 0
        };
        let align = |n: u64| n.next_multiple_of(ALIGN);
        let mut offset = contents_offset;
        let mut contents = Vec::new();
        for i in 0..tmd.u16_be(0x9E)? as usize {
            let record = tmd.at(CONTENT_RECORDS + i * CONTENT_RECORD_SIZE)?;
            let index = record.u16_be(0x04)?;
            let size = record.u64_be(0x08)?;
            let here = present(index).then_some(offset);
            if here.is_some() {
                offset = align(offset + size);
            }
            contents.push(Content { index, id: record.u32_be(0x00)?, flags: record.u16_be(0x06)?, size, offset: here });
        }
        Ok(Cia { title_id, version, contents })
    }

    /// a game's CIA, which has its executable as content 0, decrypted.
    pub fn parse_from(image: &crate::RomImage) -> Result<Cia, FsError> {
        Self::read_from(image)?.as_game()
    }

    pub fn parse(data: &[u8]) -> Result<Cia, FsError> {
        Self::read(data)?.as_game()
    }

    /// the same CIA, once it is known to be a game's, one whose executable
    /// content is there and decrypted.
    fn as_game(self) -> Result<Cia, FsError> {
        match self.kind() {
            TitleKind::Update => return Err(FsError::NotAGame("an update")),
            TitleKind::Dlc => return Err(FsError::NotAGame("downloadable content")),
            TitleKind::Game | TitleKind::DlpChild => {}
        }
        let executable = self.content(0).filter(|content| content.offset.is_some()).ok_or(FsError::NoExecutablePartition)?;
        if executable.encrypted() {
            return Err(FsError::EncryptedCia);
        }
        Ok(self)
    }

    pub fn kind(&self) -> TitleKind {
        TitleKind::of(self.title_id)
    }

    /// the content of an index, none when the TMD does not list it.
    pub fn content(&self, index: u16) -> Option<Content> {
        self.contents.iter().find(|content| content.index == index).copied()
    }

    /// where the executable content, index 0, is in the file, and its size.
    pub fn executable(&self) -> Option<(u64, u64)> {
        let content = self.content(0)?;
        Some((content.offset?, content.size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a CIA of two contents, the executable and a manual, with sections of
    /// sizes that need aligning.
    fn cia(title_id: u64, content_type: u16, present: u8) -> Vec<u8> {
        let (certificates, ticket) = (0xA00u32, 0x350u32);
        let tmd_size = 0x140 + CONTENT_RECORDS + 2 * CONTENT_RECORD_SIZE;
        let mut data = vec![0u8; 0x2020];
        data[..4].copy_from_slice(&HEADER_SIZE.to_le_bytes());
        data[0x08..0x0C].copy_from_slice(&certificates.to_le_bytes());
        data[0x0C..0x10].copy_from_slice(&ticket.to_le_bytes());
        data[0x10..0x14].copy_from_slice(&(tmd_size as u32).to_le_bytes());
        data[0x20] = present;

        let tmd_offset = 0x2040 + 0xA00 + 0x380;
        data.resize(tmd_offset + tmd_size, 0);
        let tmd = &mut data[tmd_offset..];
        tmd[..4].copy_from_slice(&0x0001_0004u32.to_be_bytes());
        let header = &mut tmd[0x140..];
        header[0x4C..0x54].copy_from_slice(&title_id.to_be_bytes());
        header[0x9E..0xA0].copy_from_slice(&2u16.to_be_bytes());
        for (i, (index, size)) in [(0u16, 0x600u64), (1, 0x200)].into_iter().enumerate() {
            let record = &mut header[CONTENT_RECORDS + i * CONTENT_RECORD_SIZE..];
            record[0x04..0x06].copy_from_slice(&index.to_be_bytes());
            record[0x06..0x08].copy_from_slice(&if index == 0 { content_type } else { 0 }.to_be_bytes());
            record[0x08..0x10].copy_from_slice(&size.to_be_bytes());
        }
        data
    }

    #[test]
    fn the_executable_is_the_first_content() {
        let data = cia(0x0004_0000_0016_4800, 0, 0xC0);
        assert!(Cia::detect(&data));
        let cia = Cia::parse(&data).unwrap();
        assert_eq!(cia.title_id, 0x0004_0000_0016_4800);
        assert_eq!(cia.kind(), TitleKind::Game);
        let tmd_end = 0x2040 + 0xA00 + 0x380 + 0x140 + CONTENT_RECORDS + 2 * CONTENT_RECORD_SIZE;
        assert_eq!(cia.executable(), Some(((tmd_end as u64).next_multiple_of(0x40), 0x600)));
    }

    #[test]
    fn encrypted_updates_and_missing_games_are_refused() {
        assert!(matches!(Cia::parse(&cia(0x0004_0000_0016_4800, 1, 0xC0)), Err(FsError::EncryptedCia)));
        assert!(matches!(Cia::parse(&cia(0x0004_000E_0016_4800, 0, 0xC0)), Err(FsError::NotAGame(_))));
        assert!(matches!(Cia::parse(&cia(0x0004_0000_0016_4800, 0, 0x40)), Err(FsError::NoExecutablePartition)));
    }

    /// a DLC's CIA read as DLC: the index bitmap says content 0 is in the
    /// file and content 1 is not, and both are still listed with their size.
    #[test]
    fn a_dlc_lists_every_content_and_places_the_ones_it_holds() {
        let mut data = cia(0x0004_008C_0016_4800, 0, 0xA0);
        let tmd_offset = 0x2040 + 0xA00 + 0x380;
        let tmd_size = 0x140 + CONTENT_RECORDS + 2 * CONTENT_RECORD_SIZE;
        let cia = Cia::read(&data).unwrap();
        assert_eq!(cia.kind(), TitleKind::Dlc);
        assert_eq!(cia.contents.len(), 2);
        let held = (tmd_offset + tmd_size).next_multiple_of(0x40) as u64;
        assert_eq!(cia.content(0), Some(Content { index: 0, id: 0, flags: 0, size: 0x600, offset: Some(held) }));
        assert_eq!(cia.content(1), Some(Content { index: 1, id: 0, flags: 0, size: 0x200, offset: None }));
        // a content the TMD does not list reads as nothing
        assert_eq!(cia.content(2), None);
        // a game's parse refuses it
        assert!(matches!(Cia::parse(&data), Err(FsError::NotAGame(_))));
        // and so does one whose held content is encrypted
        data[tmd_offset + 0x140 + CONTENT_RECORDS + 0x06..tmd_offset + 0x140 + CONTENT_RECORDS + 0x08]
            .copy_from_slice(&1u16.to_be_bytes());
        assert!(Cia::read(&data).unwrap().content(0).unwrap().encrypted());
    }

    #[test]
    fn versions_read_as_the_console_shows_them() {
        assert_eq!(version_name(0x0400), "1.0.0");
        assert_eq!(version_name(0x0401), "1.0.1");
        assert_eq!(version_name(0x0802), "2.0.2");
    }
}
