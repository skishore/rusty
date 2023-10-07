use std::cmp::max;

//////////////////////////////////////////////////////////////////////////////

// Basics

#[macro_export]
macro_rules! assert_eq_size {
    ($x:ty, $y:expr) => {
        const _: fn() = || {
            let _ = std::mem::transmute::<$x, [u8; $y]>;
        };
    };
}

pub type HashSet<K> = fxhash::FxHashSet<K>;
pub type HashMap<K, V> = fxhash::FxHashMap<K, V>;

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
        let ch = Char((ch as u16) + (0xff00 - 0x20));
        Glyph { ch, fg: Glyph::color(fg), bg: Color::default() }
    }

    pub fn gray(&self) -> Glyph {
        Glyph { ch: self.ch, fg: Color(16 + 216 + 4), bg: self.bg }
    }

    fn color(c: i32) -> Color {
        let r = (c >> 8) & 0xf;
        let g = (c >> 4) & 0xf;
        let b = (c >> 0) & 0xf;
        Color((16 + b + 6 * g + 36 * r) as u8)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Geometry helpers

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Point(pub i32, pub i32);
assert_eq_size!(Point, 8);

impl Point {
    pub fn len_l2(&self) -> f64 { (self.len_l2_squared() as f64).sqrt() }
    pub fn len_l2_squared(&self) -> i32 { self.0 * self.0 + self.1 * self.1 }
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
}

impl FOV {
    pub fn new(radius: i32) -> Self {
        let mut result = Self { radius, nodes: vec![] };
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

    pub fn apply<F: FnMut(Point, Option<&Point>) -> bool>(
            &self, mut blocked: F, scratch: &mut Vec<i32>) {
        let mut index = 0;
        scratch.clear();
        scratch.push(0);
        while index < scratch.len() {
            let node = &self.nodes[scratch[index] as usize];
            let prev = if index > 0 { Some(&node.prev) } else { None };
            if blocked(node.next, prev) { index += 1; continue; }
            for x in &node.children { scratch.push(*x); }
            index += 1;
        }
    }

    fn update(&mut self, node: usize, los: &Vec<Point>, i: usize) {
        let prev = los[i];
        assert!(self.nodes[node].next == prev);
        if prev.len_l2() > (self.radius as f64) - 0.5 { return; }
        if !(i + 1 < los.len()) { return; }

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
