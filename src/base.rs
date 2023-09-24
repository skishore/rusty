// Basics

#[macro_export]
macro_rules! assert_eq_size {
    ($x:ty, $y:expr) => {
        const _: fn() = || {
            let _ = std::mem::transmute::<$x, [u8; $y]>;
        };
    };
}

pub type HashSet<K> = std::collections::HashSet<K>;
pub type HashMap<K, V> = std::collections::HashMap<K, V>;

// Rendering helpers

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Char(pub u16);
assert_eq_size!(Char, 2);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Color(pub u8);
assert_eq_size!(Color, 1);

impl Default for Color {
    fn default() -> Self { Self(0xff) }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Glyph {
    pub ch: Char,
    pub fg: Color,
    pub bg: Color,
}
assert_eq_size!(Glyph, 4);

impl Glyph {
    pub fn wide(ch: char) -> Glyph {
        let ch = Char((ch as u16) + (0xff00 - 0x20));
        Glyph { ch, fg: Color::default(), bg: Color::default() }
    }

    pub fn wdfg(ch: char, fg: i32) -> Glyph {
        let r = (fg >> 8) & 0xf;
        let g = (fg >> 4) & 0xf;
        let b = (fg >> 0) & 0xf;
        let ch = Char((ch as u16) + (0xff00 - 0x20));
        Glyph { ch, fg: Color((16 + b + 6 * g + 36 * r) as u8), bg: Color::default() }
    }
}

// Geometry helpers

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Point(pub i32, pub i32);
assert_eq_size!(Point, 8);

#[derive(Clone, Default)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub size: Point,
    pub default: T,
}

impl<T: Copy> Matrix<T> {
    pub fn new(size: Point, value: T) -> Self {
        let mut data = Vec::new();
        data.resize((size.0 * size.1) as usize, value);
        Self { data, size, default: value }
    }

    pub fn get(&self, point: Point) -> T {
        if !self.contains(point) { return self.default; }
        self.data[self.index(point)]
    }

    pub fn set(&mut self, point: Point, value: T) {
        if !self.contains(point) { return; }
        let index = self.index(point);
        self.data[index] = value;
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    #[inline(always)]
    pub fn contains(&self, point: Point) -> bool {
        let Point(px, py) = point;
        let Point(sx, sy) = self.size;
        return 0 <= px && px < sx && 0 <= py && py < sy;
    }

    #[inline(always)]
    pub fn index(&self, point: Point) -> usize {
        (point.0 + point.1 * self.size.0) as usize
    }
}
