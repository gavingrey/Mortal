use crate::{matches_tu8, t, tu8};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::str::FromStr;
use std::sync::LazyLock;

use ahash::AHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MJAI_PAI_STRINGS_LEN: usize = 3 * 9 + 4 + 3 + 3 + 1;
const MJAI_PAI_STRINGS: [&str; MJAI_PAI_STRINGS_LEN] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", // m
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", // p
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", // s
    "E", "S", "W", "N", "P", "F", "C", // z
    "5mr", "5pr", "5sr", // aka
    "?",   // unknown
];
const DISCARD_PRIORITIES: [u8; 38] = [
    6, 5, 4, 3, 2, 3, 4, 5, 6, // m
    6, 5, 4, 3, 2, 3, 4, 5, 6, // p
    6, 5, 4, 3, 2, 3, 4, 5, 6, // s
    7, 7, 7, 7, 7, 7, 7, // z
    1, 1, 1, // aka
    0, // unknown
];

static MJAI_PAI_STRINGS_MAP: LazyLock<AHashMap<&'static str, Tile>> = LazyLock::new(|| {
    MJAI_PAI_STRINGS
        .iter()
        .enumerate()
        .map(|(id, &s)| (s, Tile::try_from(id).unwrap()))
        .collect()
});

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile(u8);

#[derive(Debug)]
pub enum InvalidTile {
    Number(usize),
    String(String),
}

impl Tile {
    /// # Safety
    /// Calling this method with an out-of-bounds tile ID is undefined behavior.
    #[inline]
    #[must_use]
    pub const fn new_unchecked(id: u8) -> Self {
        Self(id)
    }

    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    #[must_use]
    pub const fn deaka(self) -> Self {
        match self.0 {
            tu8!(5mr) => t!(5m),
            tu8!(5pr) => t!(5p),
            tu8!(5sr) => t!(5s),
            _ => self,
        }
    }

    #[inline]
    #[must_use]
    pub const fn akaize(self) -> Self {
        match self.0 {
            tu8!(5m) => t!(5mr),
            tu8!(5p) => t!(5pr),
            tu8!(5s) => t!(5sr),
            _ => self,
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_aka(self) -> bool {
        matches_tu8!(self.0, 5mr | 5pr | 5sr)
    }

    #[inline]
    #[must_use]
    pub const fn is_jihai(self) -> bool {
        matches_tu8!(self.0, E | S | W | N | P | F | C)
    }

    #[inline]
    #[must_use]
    pub const fn is_yaokyuu(self) -> bool {
        matches_tu8!(
            self.0,
            1m | 9m | 1p | 9p | 1s | 9s | E | S | W | N | P | F | C
        )
    }

    #[inline]
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.0 >= tu8!(?)
    }

    #[inline]
    #[must_use]
    pub const fn next(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let kind = tile.0 / 9;
        let num = tile.0 % 9;

        if kind < 3 {
            Self(kind * 9 + (num + 1) % 9)
        } else if num < 4 {
            Self(3 * 9 + (num + 1) % 4)
        } else {
            Self(3 * 9 + 4 + (num - 4 + 1) % 3)
        }
    }

    #[inline]
    #[must_use]
    pub const fn prev(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let kind = tile.0 / 9;
        let num = tile.0 % 9;
        if kind < 3 {
            Self(kind * 9 + (num + 9 - 1) % 9)
        } else if num < 4 {
            Self(3 * 9 + (num + 4 - 1) % 4)
        } else {
            Self(3 * 9 + 4 + (num - 4 + 3 - 1) % 3)
        }
    }

    /// Permute suits: `perm[original_suit] = target_suit` (0=man, 1=pin, 2=sou).
    /// Honors and unknown tiles unchanged. Aka status preserved.
    #[inline]
    #[must_use]
    pub const fn permute_suit(self, perm: [u8; 3]) -> Self {
        if self.is_unknown() {
            return self;
        }
        let tile = self.deaka();
        let tid = tile.0;
        let kind = tid / 9;
        let ret = if kind < 3 {
            let num = tid % 9;
            Self(perm[kind as usize] * 9 + num)
        } else {
            tile
        };
        if self.is_aka() { ret.akaize() } else { ret }
    }

    #[inline]
    #[must_use]
    pub const fn augment(self) -> Self {
        self.permute_suit([1, 0, 2])
    }

    /// `Ordering::Equal` iff `self == other`
    #[inline]
    #[must_use]
    pub fn cmp_discard_priority(self, other: Self) -> Ordering {
        let l = self.0 as usize;
        let r = other.0 as usize;
        match DISCARD_PRIORITIES[l].cmp(&DISCARD_PRIORITIES[r]) {
            Ordering::Equal => r.cmp(&l),
            o => o,
        }
    }
}

impl Default for Tile {
    fn default() -> Self {
        t!(?)
    }
}

impl TryFrom<u8> for Tile {
    type Error = InvalidTile;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        Self::try_from(v as usize)
    }
}

impl TryFrom<usize> for Tile {
    type Error = InvalidTile;

    fn try_from(v: usize) -> Result<Self, Self::Error> {
        if v >= MJAI_PAI_STRINGS_LEN {
            Err(InvalidTile::Number(v))
        } else {
            Ok(Self(v as u8))
        }
    }
}

impl FromStr for Tile {
    type Err = InvalidTile;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        MJAI_PAI_STRINGS_MAP
            .get(s)
            .copied()
            .ok_or_else(|| InvalidTile::String(s.to_owned()))
    }
}

impl fmt::Debug for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self, f)
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(MJAI_PAI_STRINGS[self.0 as usize])
    }
}

impl<'de> Deserialize<'de> for Tile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tile = String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)?;
        Ok(tile)
    }
}

impl Serialize for Tile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl fmt::Display for InvalidTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("not a valid tile: ")?;
        match self {
            Self::Number(n) => fmt::Display::fmt(n, f),
            Self::String(s) => write!(f, "not a valid tile: \"{s}\""),
        }
    }
}

impl Error for InvalidTile {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn convert() {
        "E".parse::<Tile>().unwrap();
        "5mr".parse::<Tile>().unwrap();
        "?".parse::<Tile>().unwrap();
        Tile::try_from(0_u8).unwrap();
        Tile::try_from(36_u8).unwrap();
        Tile::try_from(37_u8).unwrap();

        "".parse::<Tile>().unwrap_err();
        "0s".parse::<Tile>().unwrap_err();
        "!".parse::<Tile>().unwrap_err();
        Tile::try_from(38_u8).unwrap_err();
        Tile::try_from(u8::MAX).unwrap_err();
    }

    #[test]
    fn next_prev() {
        MJAI_PAI_STRINGS.iter().take(37).for_each(|&s| {
            let tile: Tile = s.parse().unwrap();
            assert_eq!(tile.prev().next(), tile.deaka());
            assert_eq!(tile.next().prev(), tile.deaka());
        });
    }

    #[test]
    fn permute_suit() {
        // Identity permutation
        let id = [0, 1, 2];
        assert_eq!(t!(1m).permute_suit(id), t!(1m));
        assert_eq!(t!(5p).permute_suit(id), t!(5p));
        assert_eq!(t!(9s).permute_suit(id), t!(9s));
        assert_eq!(t!(E).permute_suit(id), t!(E));
        assert_eq!(t!(5mr).permute_suit(id), t!(5mr));
        assert_eq!(t!(?).permute_suit(id), t!(?));

        // man↔pin (same as augment)
        let mp = [1, 0, 2];
        assert_eq!(t!(1m).permute_suit(mp), t!(1p));
        assert_eq!(t!(5p).permute_suit(mp), t!(5m));
        assert_eq!(t!(9s).permute_suit(mp), t!(9s));
        assert_eq!(t!(5mr).permute_suit(mp), t!(5pr));
        assert_eq!(t!(5sr).permute_suit(mp), t!(5sr));
        assert_eq!(t!(C).permute_suit(mp), t!(C));

        // Verify augment() == permute_suit([1,0,2])
        for &s in MJAI_PAI_STRINGS.iter() {
            let tile: Tile = s.parse().unwrap();
            assert_eq!(tile.augment(), tile.permute_suit([1, 0, 2]));
        }

        // pin↔sou
        let ps = [0, 2, 1];
        assert_eq!(t!(1m).permute_suit(ps), t!(1m));
        assert_eq!(t!(5p).permute_suit(ps), t!(5s));
        assert_eq!(t!(9s).permute_suit(ps), t!(9p));
        assert_eq!(t!(5pr).permute_suit(ps), t!(5sr));

        // man→pin→sou→man
        let rot = [1, 2, 0];
        assert_eq!(t!(3m).permute_suit(rot), t!(3p));
        assert_eq!(t!(3p).permute_suit(rot), t!(3s));
        assert_eq!(t!(3s).permute_suit(rot), t!(3m));
        assert_eq!(t!(5mr).permute_suit(rot), t!(5pr));
        assert_eq!(t!(5pr).permute_suit(rot), t!(5sr));
        assert_eq!(t!(5sr).permute_suit(rot), t!(5mr));

        // man→sou→pin→man
        let rot2 = [2, 0, 1];
        assert_eq!(t!(3m).permute_suit(rot2), t!(3s));
        assert_eq!(t!(3p).permute_suit(rot2), t!(3m));
        assert_eq!(t!(3s).permute_suit(rot2), t!(3p));

        // man↔sou
        let ms = [2, 1, 0];
        assert_eq!(t!(1m).permute_suit(ms), t!(1s));
        assert_eq!(t!(5p).permute_suit(ms), t!(5p));
        assert_eq!(t!(9s).permute_suit(ms), t!(9m));
        assert_eq!(t!(5mr).permute_suit(ms), t!(5sr));

        // All 6 perms produce distinct (man, pin, sou) triples
        let perms: [[u8; 3]; 6] = [
            [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0],
        ];
        let results: Vec<(Tile, Tile, Tile)> = perms.iter().map(|&p| {
            (t!(3m).permute_suit(p), t!(3p).permute_suit(p), t!(3s).permute_suit(p))
        }).collect();
        for i in 0..6 {
            for j in (i+1)..6 {
                assert_ne!(results[i], results[j], "perms {i} and {j} collide");
            }
        }

        // Honors unchanged for all perms
        for &p in &perms {
            assert_eq!(t!(E).permute_suit(p), t!(E));
            assert_eq!(t!(P).permute_suit(p), t!(P));
        }
    }
}
