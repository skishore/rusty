use std::cmp::max;

use rand::Rng;

//////////////////////////////////////////////////////////////////////////////

// Basics

#[macro_export]
macro_rules! cast {
    ($expr: expr, $enum:ident::$variant:ident) => {
        if let $enum::$variant(x) = $expr { x } else { panic!() }
    }
}

#[macro_export]
macro_rules! static_assert_size {
    ($x:ty, $y:expr) => {
        const _: fn() = || {
            let _ = std::mem::transmute::<$x, [u8; $y]>;
        };
    };
}

pub type RNG = rand::rngs::StdRng;
pub type HashSet<K> = fxhash::FxHashSet<K>;
pub type HashMap<K, V> = fxhash::FxHashMap<K, V>;

pub fn sample<'a, T>(xs: &'a [T], rng: &mut RNG) -> &'a T {
    assert!(!xs.is_empty());
    &xs[rng.gen::<usize>() % xs.len()]
}

pub fn weighted<'a, T>(xs: &'a [(i32, T)], rng: &mut RNG) -> &'a T {
    let total = xs.iter().fold(0, |acc, x| acc + x.0);
    assert!(total > 0);
    let mut value = (rng.gen::<usize>() % total as usize) as i32;
    for (weight, choice) in xs {
        value -= weight;
        if value <= 0 { return choice; }
    }
    assert!(false);
    &xs[xs.len() - 1].1
}

//////////////////////////////////////////////////////////////////////////////

// Rendering helpers

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Char(pub u16);
static_assert_size!(Char, 2);

impl Char {
    pub fn is_wide(&self) -> bool { self.0 >= 0xff00 }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Color(pub u8);
static_assert_size!(Color, 1);

impl Default for Color {
    fn default() -> Self { Self(0xff) }
}

impl From<i32> for Color {
    fn from(val: i32) -> Self {
        let r = (val >> 8) & 0xf;
        let g = (val >> 4) & 0xf;
        let b = val & 0xf;
        Color((16 + b + 6 * g + 36 * r) as u8)
    }
}

impl Color {
    pub fn black() -> Self { Self(16) }
    pub fn gray() -> Self { Self::dark(5) }
    pub fn dark(i: u8) -> Self { Self(16 + 216 + i) }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Glyph(u32);

impl From<char> for Glyph {
    fn from(val: char) -> Self { Self::char(val) }
}

impl Glyph {
    // Constructors

    pub fn new(ch: Char, fg: Color, bg: Color) -> Self {
        Self((ch.0 as u32) | ((fg.0 as u32) << 16) | ((bg.0 as u32) << 24))
    }

    pub fn char(ch: char) -> Self {
        Self::new(Char(ch as u16), Color::default(), Color::default())
    }

    pub fn chfg<T: Into<Color>>(ch: char, fg: T) -> Self {
        Self::new(Char(ch as u16), fg.into(), Color::default())
    }

    pub fn wide(ch: char) -> Self {
        let ch = Char((ch as u16) + (0xff00 - 0x20));
        Self::new(ch, Color::default(), Color::default())
    }

    pub fn wdfg<T: Into<Color>>(ch: char, fg: T) -> Self {
        let ch = Char((ch as u16) + (0xff00 - 0x20));
        Self::new(ch, fg.into(), Color::default())
    }

    pub fn with_fg<T: Into<Color>>(&self, color: T) -> Self {
        Self((self.0 & 0xff00ffff) | ((color.into().0 as u32) << 16))
    }

    pub fn with_bg<T: Into<Color>>(&self, color: T) -> Self {
        Self((self.0 & 0x00ffffff) | ((color.into().0 as u32) << 24))
    }

    // Field getters

    pub fn ch(&self) -> Char { Char(self.0 as u16) }

    pub fn fg(&self) -> Color { Color((self.0 >> 16) as u8) }

    pub fn bg(&self) -> Color { Color((self.0 >> 24) as u8) }
}

//////////////////////////////////////////////////////////////////////////////

// Point and Direction

pub fn clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min { min } else if x > max { max } else { x }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Point(pub i32, pub i32);
static_assert_size!(Point, 8);

impl Point {
    pub fn in_l2_range(&self, range: i32) -> bool {
        self.len_l2() <= range as f64 - 0.5
    }

    pub fn len_nethack(&self) -> i32 {
        let (ax, ay) = (self.0.abs() as i64, self.1.abs() as i64);
        let (min, max) = (std::cmp::min(ax, ay), std::cmp::max(ax, ay));
        ((46 * min + 95 * max + 25) / 100) as i32
    }

    pub fn len_taxicab(&self) -> i32 {
        self.0.abs() + self.1.abs()
    }

    pub fn len_l1(&self) -> i32 {
        max(self.0.abs(), self.1.abs())
    }

    pub fn len_l2(&self) -> f64 {
        (self.len_l2_squared() as f64).sqrt()
    }

    pub fn len_l2_squared(&self) -> i64 {
        let (x, y) = (self.0 as i64, self.1 as i64);
        x * x + y * y
    }

    pub fn scale(&self, scale: i32) -> Point {
        Point(self.0 * scale, self.1 * scale)
    }
}

impl std::ops::Add for Point {
    type Output = Point;
    fn add(self, other: Point) -> Point {
        Point(self.0 + other.0, self.1 + other.1)
    }
}

impl std::ops::Sub for Point {
    type Output = Point;
    fn sub(self, other: Point) -> Point {
        Point(self.0 - other.0, self.1 - other.1)
    }
}

pub mod dirs {
    use crate::base::Point;

    pub const NONE: Point = Point( 0,  0);
    pub const N:    Point = Point( 0, -1);
    pub const S:    Point = Point( 0,  1);
    pub const E:    Point = Point( 1,  0);
    pub const W:    Point = Point(-1,  0);
    pub const NE:   Point = Point( 1, -1);
    pub const NW:   Point = Point(-1, -1);
    pub const SE:   Point = Point( 1,  1);
    pub const SW:   Point = Point(-1,  1);

    pub const ALL: [Point; 8] = [N, S, E, W, NE, NW, SE, SW];
}

//////////////////////////////////////////////////////////////////////////////

// Matrix

#[derive(Clone, Default)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub size: Point,
    pub default: T,
}

impl<T: Copy> Matrix<T> {
    pub fn new(size: Point, value: T) -> Self {
        assert!(0 < size.0);
        assert!(0 < size.1);
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
        0 <= px && px < sx && 0 <= py && py < sy
    }

    #[inline(always)]
    pub fn index(&self, point: Point) -> usize {
        (point.0 + point.1 * self.size.0) as usize
    }
}

//////////////////////////////////////////////////////////////////////////////

pub type Buffer = Matrix<Glyph>;

#[derive(Clone, Copy)]
pub struct Rect { pub root: Point, pub size: Point }

pub struct Slice<'a> {
    buffer: &'a mut Buffer,
    bounds: Rect,
    cursor: Point,
    fg: Option<Color>,
    bg: Option<Color>,
}

impl<'a> From<&'a mut Buffer> for Slice<'a> {
    fn from(buffer: &'a mut Buffer) -> Slice<'a> {
        let (root, size) = (Point::default(), buffer.size);
        Slice::new(buffer, Rect { root, size })
    }
}

impl<'a> Slice<'a> {
    pub fn new(buffer: &'a mut Buffer, bounds: Rect) -> Self {
        Self { buffer, bounds, cursor: Point::default(), fg: None, bg: None }
    }

    // Basic API

    pub fn get(&self, point: Point) -> Glyph {
        if !self.contains(point) { return self.buffer.default; }
        self.buffer.get(self.bounds.root + point)
    }

    pub fn set(&mut self, point: Point, glyph: Glyph) {
        if !self.contains(point) { return; }
        let glyph = self.fg.map(|x| glyph.with_fg(x)).unwrap_or(glyph);
        let glyph = self.bg.map(|x| glyph.with_bg(x)).unwrap_or(glyph);
        self.buffer.set(self.bounds.root + point, glyph);
    }

    pub fn contains(&self, point: Point) -> bool {
        let Point(px, py) = point;
        let Point(sx, sy) = self.bounds.size;
        0 <= px && px < sx && 0 <= py && py < sy
    }

    pub fn size(&self) -> Point { self.bounds.size }

    // Cursor API

    pub fn get_cursor(&self) -> Point { self.cursor }

    pub fn newline(&mut self) -> &mut Self {
        self.newlines(1)
    }

    pub fn newlines(&mut self, n: usize) -> &mut Self {
        self.cursor = Point(0, self.cursor.1 + n as i32);
        self.set_fg(None).set_bg(None)
    }

    pub fn space(&mut self) -> &mut Self {
        self.spaces(1)
    }

    pub fn spaces(&mut self, n: usize) -> &mut Self {
        self.cursor.0 += n as i32;
        self
    }

    pub fn write_chr<T: Into<Glyph>>(&mut self, t: T) -> &mut Self {
        let glyph = t.into();
        self.set(self.cursor, glyph);
        self.spaces(if glyph.ch().is_wide() { 2 } else { 1 })
    }

    pub fn write_str(&mut self, text: &str) -> &mut Self {
        text.chars().for_each(|x| { self.write_chr(x); });
        self
    }

    pub fn set_fg(&mut self, c: Option<Color>) -> &mut Self { self.fg = c; self }

    pub fn set_bg(&mut self, c: Option<Color>) -> &mut Self { self.bg = c; self }
}

//////////////////////////////////////////////////////////////////////////////

// Field-of-vision helpers

#[allow(non_snake_case)]
pub fn LOS(a: Point, b: Point) -> Vec<Point> {
    let x_diff = (a.0 - b.0).abs();
    let y_diff = (a.1 - b.1).abs();
    let x_sign = if b.0 < a.0 { -1 } else { 1 };
    let y_sign = if b.1 < a.1 { -1 } else { 1 };

    let size = (max(x_diff, y_diff) + 1) as usize;
    let mut result = vec![];
    result.reserve_exact(size);
    result.push(a);

    let mut test = 0;
    let mut current = a;

    if x_diff >= y_diff {
        test = (x_diff + test) / 2;
        for _ in 0..x_diff {
            current.0 += x_sign;
            test -= y_diff;
            if test < 0 {
                current.1 += y_sign;
                test += x_diff;
            }
            result.push(current);
        }
    } else {
        test = (y_diff + test) / 2;
        for _ in 0..y_diff {
            current.1 += y_sign;
            test -= x_diff;
            if test < 0 {
                current.0 += x_sign;
                test += y_diff;
            }
            result.push(current);
        }
    }

    assert!(result.len() == size);
    result
}

#[derive(Default)]
struct FOVNode {
    next: Point,
    prev: Point,
    children: Vec<i32>,
}

pub struct FOV {
    radius: i32,
    nodes: Vec<FOVNode>,
    cache: Vec<i32>,
}

impl FOV {
    pub fn new(radius: i32) -> Self {
        let mut result = Self { radius, nodes: vec![], cache: vec![] };
        result.nodes.push(FOVNode::default());
        for i in 0..=radius {
            for j in 0..8 {
                let (xa, ya) = if j & 1 == 0 { (radius, i) } else { (i, radius) };
                let xb = xa * if j & 2 == 0 { 1 } else { -1 };
                let yb = ya * if j & 4 == 0 { 1 } else { -1 };
                result.update(0, &LOS(Point::default(), Point(xb, yb)), 0);
            }
        }
        result
    }

    pub fn apply<F: FnMut(Point, Option<&Point>) -> bool>(&mut self, mut blocked: F) {
        let mut index = 0;
        self.cache.push(0);
        while index < self.cache.len() {
            let node = &self.nodes[self.cache[index] as usize];
            let prev = if index > 0 { Some(&node.prev) } else { None };
            if blocked(node.next, prev) { index += 1; continue; }
            for x in &node.children { self.cache.push(*x); }
            index += 1;
        }
        self.cache.clear();
    }

    fn update(&mut self, node: usize, los: &Vec<Point>, i: usize) {
        let prev = los[i];
        assert!(self.nodes[node].next == prev);
        if !prev.in_l2_range(self.radius) { return; }
        if i + 1 >= los.len() { return; }

        let next = los[i + 1];
        let child = (|| {
            for x in &self.nodes[node].children {
                if self.nodes[*x as usize].next == next { return *x as usize; }
            }
            let result = self.nodes.len();
            self.nodes.push(FOVNode { next, prev, children: vec![] });
            self.nodes[node].children.push(result as i32);
            result
        })();
        self.update(child, los, i + 1);
    }
}
