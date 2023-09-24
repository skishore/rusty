use std::cmp::{max, min};

use lazy_static::lazy_static;
use rand::random;

use crate::assert_eq_size;
use crate::base::{HashMap, Glyph, Matrix, Point};

// Tile

const FLAG_NONE: u32 = 0;
const FLAG_BLOCKED: u32 = 1 << 0;
const FLAG_OBSCURE: u32 = 1 << 1;

struct Tile {
    flags: u32,
    glyph: Glyph,
    description: &'static str,
}
assert_eq_size!(Tile, 24);

lazy_static! {
    static ref TILES: HashMap<char, Tile> = {
        let items = [
            ('.', (FLAG_NONE,    Glyph::wide('.'),        "grass")),
            ('"', (FLAG_OBSCURE, Glyph::wdfg('"', 0x231), "tall grass")),
            ('#', (FLAG_BLOCKED, Glyph::wdfg('#', 0x010), "a tree")),
        ];
        let mut result = HashMap::new();
        for (ch, (flags, glyph, description)) in items {
            result.insert(ch, Tile { flags, glyph, description });
        }
        result
    };
}

// Board

struct Board {
    map: Matrix<&'static Tile>,
}

impl Board {
    fn init(&mut self) {
        self.map.fill(self.map.default);
        let d100 = || random::<i32>().rem_euclid(100);
        let size = self.map.size;

        let automata = || -> Matrix<bool> {
            let mut result = Matrix::new(size, false);
            for x in 0..size.0 {
                result.set(Point(x, 0), true);
                result.set(Point(x, size.1 - 1), true);
            }
            for y in 0..size.1 {
                result.set(Point(0, y), true);
                result.set(Point(size.0 - 1, y), true);
            }

            for y in 0..size.1 {
                for x in 0..size.0 {
                    if d100() < 45 { result.set(Point(x, y),  true); }
                }
            }

            for i in 0..3 {
                let mut next = result.clone();
                for y in 1..size.1 - 1 {
                    for x in 1..size.0 - 1 {
                        let (mut adj1, mut adj2) = (0, 0);
                        for dy in -2_i32..=2 {
                            for dx in -2_i32..=2 {
                                if dx == 0 && dy == 0 { continue; };
                                if min(dx.abs(), dy.abs()) == 2 { continue; };
                                let next = Point(x + dx, y + dy);
                                if !result.get(next) { continue; }
                                let distance = max(dx.abs(), dy.abs());
                                if distance <= 1 { adj1 += 1; }
                                if distance <= 2 { adj2 += 1; }
                            }
                        }
                        let blocked = adj1 >= 5 || (i < 2 && adj2 <= 1);
                        next.set(Point(x, y), blocked);
                    }
                }
                std::mem::swap(&mut result, &mut next);
            }
            result
        };

        let walls = automata();
        let grass = automata();
        let wt = TILES.get(&'#').unwrap();
        let gt = TILES.get(&'"').unwrap();
        for y in 0..size.1 {
            for x in 0..size.0 {
                let point = Point(x, y);
                if walls.get(point) {
                    self.map.set(point, wt);
                } else if grass.get(point) {
                    self.map.set(point, gt);
                }
            }
        }
    }
}

// State

pub struct State {
    board: Board,
}

impl State {
    pub fn new(size: Point) -> Self {
        let tile = TILES.get(&'.').unwrap();
        let mut board = Board { map: Matrix::new(size, tile) };
        let start = Point(size.0 / 2, size.1 / 2);

        loop {
            board.init();
            if board.map.get(start).flags & FLAG_BLOCKED == 0 { break; }
        }

        Self { board }
    }

    pub fn update(&mut self) {}

    pub fn render(&self, buffer: &mut Matrix<Glyph>) {
        for y in 0..buffer.size.1 {
            for x in 0..(buffer.size.0 / 2) {
                let glyph = self.board.map.get(Point(x, y)).glyph;
                buffer.set(Point(2 * x, y), glyph);
            }
        }
    }
}
