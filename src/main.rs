mod base;
mod cell;

use std::io::{self, Write};

use game_loop::game_loop;
use rand::random;
use termion::color::{self};
use termion::cursor::{Goto, Hide, Show};
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use termion::screen::{ToAlternateScreen, ToMainScreen};

use base::{Char, Color, Glyph, Matrix, Point};

struct Screen {
    extent: Point,
    output: termion::raw::RawTerminal<io::Stdout>,
    next: Matrix<Glyph>,
    prev: Matrix<Glyph>,
}

impl Screen {
    fn new(size: Point) -> Self {
        let prev = Matrix::new(size, Glyph::wide(' '));
        let next = Matrix::new(size, Glyph::wide(' '));
        let (x, y) = termion::terminal_size().unwrap();
        let output = io::stdout().into_raw_mode().unwrap();
        Self { extent: Point(x as i32, y as i32), output, next, prev }
    }

    fn render(&mut self) -> io::Result<()> {
        let mut lines_changed = 0;
        let Point(sx, sy) = self.next.size;
        for y in 0..sy {
            let mut start = sx;
            let mut limit = 0;
            for x in 0..sx {
                let next = self.next.get(Point(x, y));
                let prev = self.prev.get(Point(x, y));
                if  next == prev { continue; }
                start = std::cmp::min(start, x);
                limit = std::cmp::max(limit, x);
            }
            if start > limit { continue; }

            lines_changed += 1;
            let mx = ((self.extent.0 - sx) / 2 + start + 1) as u16;
            let my = ((self.extent.1 - sy) / 2 + y + 1) as u16;
            write!(self.output, "{}", Goto(mx, my))?;

            let mut x = start;
            let (mut fg, mut bg) = (Color(0), Color(0));
            self.set_foreground(fg)?;
            self.set_background(fg)?;
            while x <= limit {
                let glyph = self.next.get(Point(x, y));
                if glyph.fg != fg { self.set_foreground(glyph.fg)?; }
                if glyph.bg != bg { self.set_background(glyph.bg)?; }
                (fg, bg) = (glyph.fg, glyph.bg);
                x += self.write_char(glyph.ch)?;
            }
        }
        std::mem::swap(&mut self.next, &mut self.prev);
        if lines_changed > 0 { self.output.flush() } else { Ok(()) }
    }

    fn enter_alt_screen(&mut self) -> io::Result<()> {
        write!(self.output, "{}{}{}", ToAlternateScreen, Hide, termion::clear::All)?;
        self.output.flush()
    }

    fn exit_alt_screen(&mut self) -> io::Result<()> {
        write!(self.output, "{}{}", ToMainScreen, Show)?;
        self.output.flush()
    }

    fn set_foreground(&mut self, color: Color) -> io::Result<()> {
        if color.0 > 0 {
            write!(self.output, "{}", color::Fg(color::AnsiValue(color.0 - 1)))
        } else {
            write!(self.output, "{}", color::Fg(color::Reset))
        }
    }

    fn set_background(&mut self, color: Color) -> io::Result<()> {
        if color.0 > 0 {
            write!(self.output, "{}", color::Bg(color::AnsiValue(color.0 - 1)))
        } else {
            write!(self.output, "{}", color::Bg(color::Reset))
        }
    }

    fn write_char(&mut self, ch: Char) -> io::Result<i32> {
        if ch.0 == 0xff00 {
            write!(self.output, "  ")?;
            Ok(2)
        } else {
            write!(self.output, "{}", char::from_u32(ch.0 as u32).unwrap())?;
            Ok(if ch.0 <= 0xff { 1 } else { 2 })
        }
    }
}

struct Entity { pos: Point, vel: Point }

struct Game { entities: Vec<Entity>, size: Point }

impl Game {
    fn check(&self) {
        for entity in self.entities.iter() {
            assert!(0 <= entity.pos.0 && entity.pos.0 < self.size.0);
            assert!(0 <= entity.pos.1 && entity.pos.1 < self.size.1);
        }
    }

    fn update(&mut self) {
        for entity in self.entities.iter_mut() {
            entity.pos.0 = entity.pos.0 + entity.vel.0;
            entity.pos.1 = entity.pos.1 + entity.vel.1;
            if entity.pos.0 < 0 {
                entity.pos.0 = -entity.pos.0;
                entity.vel.0 = -entity.vel.0;
            } else if entity.pos.0 >= self.size.0 {
                entity.pos.0 = 2 * self.size.0 - entity.pos.0 - 2;
                entity.vel.0 = -entity.vel.0;
            }
            if entity.pos.1 < 1 {
                entity.pos.1 = -entity.pos.1;
                entity.vel.1 = -entity.vel.1;
            } else if entity.pos.1 >= self.size.1 {
                entity.pos.1 = 2 * self.size.1 - entity.pos.1 - 2;
                entity.vel.1 = -entity.vel.1;
            }
        }
        self.check();
    }

    fn render(&self, buffer: &mut Matrix<Glyph>) {
        buffer.fill(Glyph::wide('.'));
        let mut blocked = Glyph::wide('X');
        blocked.fg = Color(2);
        for entity in self.entities.iter() {
            buffer.set(Point(2 * entity.pos.0, entity.pos.1), blocked);
        }
    }
}

fn main() {
    let size = Point(40, 40);
    let mut inputs = termion::async_stdin().keys();
    let mut screen = Screen::new(Point(2 * size.0, size.1));
    screen.enter_alt_screen().unwrap();

    let mut game = Game { entities: vec![], size };

    for _ in 0..3 {
        let x = random::<i32>().rem_euclid(game.size.0);
        let y = random::<i32>().rem_euclid(game.size.1);
        let vx = if random::<bool>() { 1 } else { -1 };
        let vy = if random::<bool>() { 1 } else { -1 };
        game.entities.push(Entity { pos: Point(x, y), vel: Point(vx, vy) });
        game.check();
    }

    game_loop(game, 60, 0.01, |g| {
        if let Some(Ok(key)) = inputs.next() {
            if key == Key::Char('q') || key == Key::Ctrl('c') {
                g.exit();
            }
        }
        g.game.update()
    }, |g| {
        g.game.render(&mut screen.next);
        screen.render().unwrap();
        std::thread::sleep(std::time::Duration::from_micros(800));
    });

    screen.exit_alt_screen().unwrap();
}
