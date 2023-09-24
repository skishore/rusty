// Rendering helpers

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Char(pub u16);

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Color(pub u8);

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Glyph {
    pub ch: Char,
    pub fg: Color,
    pub bg: Color,
}

impl Glyph {
    pub fn wide(ch: char) -> Glyph {
        let ch = Char((ch as u16) + (0xff00 - 0x20));
        Glyph { ch, fg: Color(0), bg: Color(0) }
    }
}

// Geometry helpers

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Point(pub i32, pub i32);

#[derive(Default)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub size: Point,
}

impl<T: Copy> Matrix<T> {
    pub fn new(size: Point, value: T) -> Self {
        let mut data = Vec::new();
        data.resize((size.0 * size.1) as usize, value);
        Self { data, size }
    }

    pub fn get(&self, point: Point) -> T {
        self.data[self.index(point)]
    }

    pub fn set(&mut self, point: Point, value: T) {
        let index = self.index(point);
        self.data[index] = value;
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    #[inline(always)]
    fn index(&self, point: Point) -> usize {
        assert!(0 <= point.0 && point.0 < self.size.0, "{} vs. {}", point.0, self.size.0);
        assert!(0 <= point.1 && point.1 < self.size.1, "{} vs. {}", point.1, self.size.1);
        (point.0 + point.1 * self.size.0) as usize
    }
}
