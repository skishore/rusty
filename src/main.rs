mod base;
mod cell;

use std::io::{self, Write};

use game_loop::{game_loop, TimeTrait};
use rand::random;
use termion::{clear, color};
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

    fn render(&mut self, stats: &Stats, delta: f64) -> io::Result<()> {
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
            let (mut fg, mut bg) = (Color::default(), Color::default());
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

        if delta > 1.0 {
            let out = format!(
                   "Update: {:.2}% / Render: {:.2}% / FPS: {:.2}",
                   100.0 * stats.update_total.get() / delta,
                   100.0 * stats.render_total.get() / delta,
                   stats.ticks.get() as f64 / delta);
            self.write_status_message(&out)?;
            lines_changed += 1;
            stats.clear();
        }

        if lines_changed > 0 { self.output.flush() } else { Ok(()) }
    }

    fn enter_alt_screen(&mut self) -> io::Result<()> {
        write!(self.output, "{}{}{}", ToAlternateScreen, Hide, clear::All)?;
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

    fn write_status_message(&mut self, msg: &str) -> io::Result<()> {
        self.set_foreground(Color::default())?;
        self.set_background(Color::default())?;
        let x = (self.extent.0 - msg.len() as i32) as u16;
        let y = self.extent.1 as u16;
        write!(self.output, "{}{}{}", Goto(x, y), clear::CurrentLine, msg)
    }
}

struct Entity { pos: Point, vel: Point }

struct Game { entities: Vec<Entity>, size: Point }

impl Game {
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

#[derive(Default)]
struct Stats {
    render_total: std::cell::Cell<f64>,
    update_total: std::cell::Cell<f64>,
    ticks: std::cell::Cell<usize>,
}

impl Stats {
    fn clear(&self) {
        self.render_total.set(0.0);
        self.update_total.set(0.0);
        self.ticks.set(0);
    }

    fn on_render(&self, delta: f64) {
        self.render_total.set(self.render_total.get() + delta);
    }

    fn on_update(&self, delta: f64) {
        self.update_total.set(self.update_total.get() + delta);
    }
}

fn main() {
    let size = Point(40, 40);
    let mut inputs = termion::async_stdin().keys();
    let mut screen = Screen::new(Point(2 * size.0, size.1));
    screen.enter_alt_screen().unwrap();
    screen.write_status_message("<calculating FPS...>").unwrap();
    screen.output.flush().unwrap();

    let mut game = Game { entities: vec![], size };
    let mut time = game_loop::Time::now();
    let stats = Stats::default();

    for _ in 0..10 {
        let x = random::<i32>().rem_euclid(game.size.0);
        let y = random::<i32>().rem_euclid(game.size.1);
        let vx = if random::<bool>() { 1 } else { -1 };
        let vy = if random::<bool>() { 1 } else { -1 };
        game.entities.push(Entity { pos: Point(x, y), vel: Point(vx, vy) });
    }

    game_loop(game, 60, 0.01, |g| {
        let start = game_loop::Time::now();
        if let Some(Ok(key)) = inputs.next() {
            if key == Key::Char('q') || key == Key::Ctrl('c') {
                g.exit();
            }
        }
        g.game.update();
        let limit = game_loop::Time::now();
        stats.on_update(limit.sub(&start));

        // It's roguelike: without an update(), render() will not change, and
        // render() is a fast operation. We frame-lock renders to updates.
        let start = game_loop::Time::now();
        let delta = start.sub(&time);
        g.game.render(&mut screen.next);
        screen.render(&stats, delta).unwrap();
        let limit = game_loop::Time::now();
        stats.on_render(limit.sub(&start));

        // Reset stats after they're printed as a status message.
        let ticks = stats.ticks.get();
        stats.ticks.set(ticks + 1);
        if ticks == 0 {
            time = start;
        }
    }, |_| {
        std::thread::sleep(std::time::Duration::from_micros(1000));
    });

    screen.exit_alt_screen().unwrap();
    screen.output.suspend_raw_mode().unwrap();
}
