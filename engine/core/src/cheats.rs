//! the running title's cheats: `cheats/<TitleID>.txt` in the data folder, in
//! the files Citra and Azahar read, applied once a frame while they are on.
//!
//! a file is a list of `[name]` headers, each followed by its code lines of
//! two words of eight hex digits. `0`/`1`/`2` write 32, 16 and 8 bits, `3` to
//! `A` compare a value before letting the next line run, `B` loads the offset,
//! `D3`/`DC` set and add to it, `C`…`D1` loop, `DD` fires only while buttons
//! are held, and `D0`/`D2` end a block. a line this engine does not know is
//! logged and skipped, and a code that was never named is left out.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use zakuro_cpu::Bus;

/// how a title's cheats are found: `cheats/<TitleID>.txt` in the data
/// folder, upper or lower case, as the games' cheat lists are named.
pub fn path(data_dir: &Path, program_id: u64) -> Option<PathBuf> {
    let dir = data_dir.join("cheats");
    [format!("{program_id:016X}"), format!("{program_id:016x}")]
        .into_iter()
        .map(|name| dir.join(name).with_extension("txt"))
        .find(|path| path.is_file())
}

/// how a comparison answers, the codes `3` to `A`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Test {
    Greater,
    Less,
    Equal,
    NotEqual,
}

/// one line of a cheat, as the interpreter runs it.
#[derive(Debug, Clone)]
enum Line {
    /// a write of 4, 2 or 1 bytes at address + offset.
    Write { width: u8, address: u32, value: u32 },
    /// a comparison of the word, or the masked halfword, at address +
    /// offset: the line after it runs only when it passes.
    Compare {
        width: u8,
        mask: u32,
        test: Test,
        address: u32,
        value: u32,
    },
    /// `DD`: the block runs only while the buttons it names are held.
    Joker { buttons: u32 },
    /// `D3`: the offset writes go through.
    SetOffset(u32),
    /// `DC`: adds to that offset.
    AddOffset(u32),
    /// `B`: the offset read from memory at address + offset.
    LoadOffset(u32),
    /// `C`: the lines after it, up to its `D1`, run count times.
    Loop { count: u32, end: usize },
    /// `D1`: where a loop comes back to.
    LoopEnd,
    /// `D0`: the end of a conditional block.
    Terminator,
    /// `D2`: the rest of the cheat is left alone this frame.
    FullTerminator,
}

/// one `[name]` of a cheat file and its codes.
#[derive(Debug, Clone)]
pub struct Cheat {
    pub name: String,
    pub enabled: bool,
    lines: Vec<Line>,
}

impl Cheat {
    /// runs its lines over memory.
    fn run<M: Bus>(&self, memory: &mut M, buttons: u32) {
        let mut offset = 0u32;
        self.run_range(memory, 0, self.lines.len(), &mut offset, buttons, 0);
    }

    /// runs lines from..to, false when a full terminator stopped them.
    fn run_range<M: Bus>(&self, memory: &mut M, from: usize, to: usize, offset: &mut u32, buttons: u32, depth: u32) -> bool {
        let mut index = from;
        // a comparison that failed holds back the one line after it.
        let mut skip_next = false;
        while index < to {
            if skip_next {
                skip_next = false;
                index += 1;
                continue;
            }
            match &self.lines[index] {
                Line::Write { width, address, value } => {
                    let at = address.wrapping_add(*offset);
                    match width {
                        4 => memory.write32(at, *value),
                        2 => memory.write16(at, *value as u16),
                        _ => memory.write8(at, *value as u8),
                    }
                }
                Line::Compare { width, mask, test, address, value } => {
                    let at = address.wrapping_add(*offset);
                    let read = if *width == 4 { memory.read32(at) } else { u32::from(memory.read16(at)) };
                    let held = read & mask;
                    let passes = match test {
                        Test::Greater => held > *value,
                        Test::Less => held < *value,
                        Test::Equal => held == *value,
                        Test::NotEqual => held != *value,
                    };
                    if !passes {
                        skip_next = true;
                    }
                }
                Line::Joker { buttons: wanted } => {
                    if buttons & wanted != *wanted {
                        // the block is left out, up to its terminator.
                        index = self.skip_block(index + 1, to);
                        continue;
                    }
                }
                Line::SetOffset(value) => *offset = *value,
                Line::AddOffset(value) => *offset = offset.wrapping_add(*value),
                Line::LoadOffset(address) => *offset = memory.read32(address.wrapping_add(*offset)),
                Line::Loop { count, end } => {
                    let end = (*end).min(to);
                    // a runaway count would hang the frame; a code needs more
                    // than sixty thousand turns of its body to be broken.
                    if depth < 8 {
                        for _ in 0..(*count).min(0x1_0000) {
                            if !self.run_range(memory, index + 1, end, offset, buttons, depth + 1) {
                                return false;
                            }
                        }
                    }
                    index = end;
                    continue;
                }
                Line::LoopEnd | Line::Terminator => {}
                Line::FullTerminator => return false,
            }
            index += 1;
        }
        true
    }

    /// the end of a block a joker code did not run: past its terminator.
    fn skip_block(&self, from: usize, to: usize) -> usize {
        (from..to)
            .find(|&index| matches!(self.lines[index], Line::Terminator | Line::FullTerminator))
            .map_or(to, |index| index + 1)
    }

    /// pairs each loop with its end, nested ones included, so running a code
    /// never has to look for it.
    fn resolve_loops(&mut self) {
        let mut lines = std::mem::take(&mut self.lines);
        let mut open: Vec<usize> = Vec::new();
        for index in 0..lines.len() {
            match lines[index] {
                Line::Loop { .. } => open.push(index),
                Line::LoopEnd => {
                    if let Some(start) = open.pop() {
                        if let Line::Loop { end, .. } = &mut lines[start] {
                            *end = index;
                        }
                    }
                }
                _ => {}
            }
        }
        // a loop without its end runs to the end of the cheat.
        let end_of_cheat = lines.len();
        for start in open {
            if let Line::Loop { end, .. } = &mut lines[start] {
                *end = end_of_cheat;
            }
        }
        self.lines = lines;
    }
}

/// every cheat of the running title.
#[derive(Debug, Default, Clone)]
pub struct Cheats {
    pub cheats: Vec<Cheat>,
    /// the buttons held, which the joker codes ask about.
    pub buttons: u32,
}

impl Cheats {
    /// reads the title's cheat file, if it has one.
    pub fn load(data_dir: Option<&Path>, program_id: u64) -> Cheats {
        let Some(data_dir) = data_dir else { return Cheats::default() };
        let Some(path) = path(data_dir, program_id) else { return Cheats::default() };
        let Ok(text) = std::fs::read_to_string(&path) else {
            log::warn!("cheats: {} could not be read", path.display());
            return Cheats::default();
        };
        let cheats = Cheats::from_text(&text);
        log::info!("cheats: {} codes in {}", cheats.cheats.len(), path.display());
        cheats
    }

    /// the codes a cheat file holds, none of them on.
    pub fn from_text(text: &str) -> Cheats {
        Cheats { cheats: parse(text), buttons: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.cheats.is_empty()
    }

    /// the codes' names, in the order the file lists them.
    pub fn names(&self) -> Vec<String> {
        self.cheats.iter().map(|cheat| cheat.name.clone()).collect()
    }

    /// turns on the codes named and leaves the rest off, as the menu's
    /// settings say.
    pub fn set(&mut self, enabled: &BTreeMap<String, bool>) {
        for cheat in &mut self.cheats {
            cheat.enabled = enabled.get(&cheat.name).copied().unwrap_or(false);
        }
    }

    /// writes every enabled code over memory, once a frame.
    pub fn apply<M: Bus>(&self, memory: &mut M) {
        for cheat in self.cheats.iter().filter(|cheat| cheat.enabled) {
            cheat.run(memory, self.buttons);
        }
    }
}

/// reads a cheat file's text.
pub fn parse(text: &str) -> Vec<Cheat> {
    let mut cheats: Vec<Cheat> = Vec::new();
    let mut current: Option<Cheat> = None;
    for (number, raw) in text.lines().enumerate() {
        let line = raw.trim();
        // blank lines and `*` lines, which cheat lists keep notes on.
        if line.is_empty() || line.starts_with('*') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')) {
            if let Some(cheat) = current.take() {
                cheats.push(cheat);
            }
            current = Some(Cheat { name: name.trim().to_owned(), enabled: false, lines: Vec::new() });
            continue;
        }
        let Some(cheat) = current.as_mut() else {
            log::warn!("cheats: line {} is a code before any [name], it is skipped", number + 1);
            continue;
        };
        match parse_line(line) {
            Ok(Some(parsed)) => cheat.lines.push(parsed),
            Ok(None) => {}
            Err(why) => log::warn!("cheats: {} line {}: {why}, it is skipped", cheat.name, number + 1),
        }
    }
    if let Some(cheat) = current.take() {
        cheats.push(cheat);
    }
    for cheat in &mut cheats {
        cheat.resolve_loops();
    }
    cheats
}

/// reads one code line, None for one that is left out on purpose.
fn parse_line(line: &str) -> Result<Option<Line>, String> {
    let mut words = line.split_whitespace();
    let (Some(first), Some(second)) = (words.next(), words.next()) else {
        return Err("a code is two words of eight hex digits".into());
    };
    if words.next().is_some() {
        return Err("a code is two words, not more".into());
    }
    let first = u32::from_str_radix(first, 16).map_err(|_| format!("{first} is not hex"))?;
    let second = u32::from_str_radix(second, 16).map_err(|_| format!("{second} is not hex"))?;
    let address = first & 0x0FFF_FFFF;
    Ok(Some(match first >> 28 {
        0x0 => Line::Write { width: 4, address, value: second },
        0x1 => Line::Write { width: 2, address, value: second & 0xFFFF },
        0x2 => Line::Write { width: 1, address, value: second & 0xFF },
        0x3 => Line::Compare { width: 4, mask: u32::MAX, test: Test::Greater, address, value: second },
        0x4 => Line::Compare { width: 4, mask: u32::MAX, test: Test::Less, address, value: second },
        0x5 => Line::Compare { width: 4, mask: u32::MAX, test: Test::Equal, address, value: second },
        0x6 => Line::Compare { width: 4, mask: u32::MAX, test: Test::NotEqual, address, value: second },
        // the 16-bit comparisons carry their mask above their value.
        0x7 | 0x8 | 0x9 | 0xA => Line::Compare {
            width: 2,
            mask: second >> 16,
            test: match first >> 28 {
                0x7 => Test::Greater,
                0x8 => Test::Less,
                0x9 => Test::Equal,
                _ => Test::NotEqual,
            },
            address,
            value: second & 0xFFFF,
        },
        0xB => Line::LoadOffset(address),
        0xC => Line::Loop { count: second, end: 0 },
        // the D codes carry which one they are in the nibble below the
        // opcode, so D3, DC and DD are told apart by their first word.
        0xD => match (first >> 24) & 0xF {
            0x0 => Line::Terminator,
            0x1 => Line::LoopEnd,
            0x2 => Line::FullTerminator,
            0x3 => Line::SetOffset(second),
            0xC => Line::AddOffset(second),
            0xD => Line::Joker { buttons: second },
            other => return Err(format!("the D{other:X} code is not supported")),
        },
        0xE => return Err("the E (patch) code is not supported".into()),
        other => return Err(format!("opcode {other:X} is not one of the file's")),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a page of memory a code can be run over.
    struct Page(Vec<u8>);

    impl Page {
        fn new() -> Page {
            Page(vec![0; 0x400])
        }

        fn word(&self, at: u32) -> u32 {
            let at = at as usize;
            u32::from_le_bytes([self.0[at], self.0[at + 1], self.0[at + 2], self.0[at + 3]])
        }
    }

    impl Bus for Page {
        fn read8(&mut self, addr: u32) -> u8 {
            self.0.get(addr as usize).copied().unwrap_or(0)
        }
        fn read16(&mut self, addr: u32) -> u16 {
            let at = addr as usize;
            u16::from_le_bytes([self.0.get(at).copied().unwrap_or(0), self.0.get(at + 1).copied().unwrap_or(0)])
        }
        fn read32(&mut self, addr: u32) -> u32 {
            self.word(addr)
        }
        fn write8(&mut self, addr: u32, value: u8) {
            if let Some(byte) = self.0.get_mut(addr as usize) {
                *byte = value;
            }
        }
        fn write16(&mut self, addr: u32, value: u16) {
            for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
                self.write8(addr + i as u32, byte);
            }
        }
        fn write32(&mut self, addr: u32, value: u32) {
            for (i, byte) in value.to_le_bytes().into_iter().enumerate() {
                self.write8(addr + i as u32, byte);
            }
        }
    }

    fn enabled(text: &str) -> Cheats {
        let mut cheats = Cheats::from_text(text);
        for cheat in &mut cheats.cheats {
            cheat.enabled = true;
        }
        cheats
    }

    /// the three widths write where the code says, and nowhere else.
    #[test]
    fn the_writes_land_where_the_code_says() {
        let cheats = enabled("[writes]\n00000010 11223344\n10000010 00005566\n2000001C 00000077\n");
        let mut memory = Page::new();
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x10), 0x1122_3344);
        assert_eq!(memory.read16(0x10), 0x5566);
        assert_eq!(memory.read8(0x1C), 0x77);
        assert_eq!(memory.word(0x14), 0, "a write touched a word beside it");
    }

    /// the offset codes carry later writes to where they point.
    #[test]
    fn the_offset_carries_the_writes() {
        let cheats = enabled("[offset]\nD3000000 00000100\n00000010 00000001\nDC000000 00000004\n00000010 00000002\n");
        let mut memory = Page::new();
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x110), 1);
        assert_eq!(memory.word(0x114), 2);
    }

    /// a comparison that fails holds back the line after it, and one that
    /// passes lets it through.
    #[test]
    fn a_comparison_holds_back_the_line_after_it() {
        let cheats = enabled("[if]\n50000020 00000001\n00000024 0000000A\n");
        let mut memory = Page::new();
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x24), 0, "a write ran though its condition failed");

        memory.write32(0x20, 1);
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x24), 0x0A);
    }

    /// a loop runs its body as many times as it says, each turn after the
    /// offset moved on.
    #[test]
    fn a_loop_runs_its_body_again_and_again() {
        let cheats = enabled("[loop]\nC0000000 00000003\n00000010 000000FF\nDC000000 00000004\nD1000000 00000000\n");
        let mut memory = Page::new();
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x10), 0xFF);
        assert_eq!(memory.word(0x14), 0xFF);
        assert_eq!(memory.word(0x18), 0xFF);
        assert_eq!(memory.word(0x1C), 0, "the loop ran a turn too many");
    }

    /// a joker code runs its block only while its buttons are held.
    #[test]
    fn a_joker_code_waits_for_its_buttons() {
        let text = "[joker]\nDD000000 00000100\n00000010 00000001\nD0000000 00000000\n";
        let mut cheats = enabled(text);
        let mut memory = Page::new();
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x10), 0, "a joker code ran unheld");

        cheats.buttons = 0x100;
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x10), 1);
    }

    /// a full terminator leaves the rest of the cheat alone.
    #[test]
    fn a_full_terminator_stops_the_cheat() {
        let cheats = enabled("[stop]\n00000010 00000001\nD2000000 00000000\n00000014 00000002\n");
        let mut memory = Page::new();
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x10), 1);
        assert_eq!(memory.word(0x14), 0);
    }

    /// the names are the file's, and only the switched-on codes run.
    #[test]
    fn only_the_codes_switched_on_run() {
        let mut cheats = Cheats::from_text("[one]\n00000010 00000001\n[two]\n00000014 00000002\n");
        assert_eq!(cheats.names(), vec!["one".to_owned(), "two".to_owned()]);
        cheats.set(&BTreeMap::from([("two".to_owned(), true)]));
        let mut memory = Page::new();
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x10), 0);
        assert_eq!(memory.word(0x14), 2);
    }

    /// a line that is not a code, and one whose opcode is unknown, are
    /// skipped rather than taken for something else.
    #[test]
    fn broken_lines_are_left_out() {
        let cheats = Cheats::from_text("[broken]\nnot a code\nF0000000 00000000\n0000001G 00000001\n00000010 0000002A\n");
        assert_eq!(cheats.cheats.len(), 1);
        let mut cheats = cheats;
        cheats.cheats[0].enabled = true;
        let mut memory = Page::new();
        cheats.apply(&mut memory);
        assert_eq!(memory.word(0x10), 0x2A, "a broken line stopped the ones after it");
    }

    /// a code before any name has no cheat to belong to.
    #[test]
    fn a_code_before_a_name_is_left_out() {
        let cheats = Cheats::from_text("00000010 00000001\n[named]\n00000014 00000002\n");
        assert_eq!(cheats.names(), vec!["named".to_owned()]);
    }
}
