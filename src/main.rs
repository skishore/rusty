mod base;

use std::io::{self, Write};

use game_loop::game_loop;
use rand::random;
use termion::cursor::{Goto, Hide, Show};
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use termion::screen::{ToAlternateScreen, ToMainScreen};

type Screen = termion::raw::RawTerminal<io::Stdout>;

#[derive(Eq, Hash, PartialEq)]
struct Point(i32, i32);

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
                entity.pos.0 = 2 * self.size.0 - entity.pos.0 - 1;
                entity.vel.0 = -entity.vel.0;
            }
            if entity.pos.1 < 1 {
                entity.pos.1 = -entity.pos.1;
                entity.vel.1 = -entity.vel.1;
            } else if entity.pos.1 >= self.size.1 {
                entity.pos.1 = 2 * self.size.1 - entity.pos.1 - 1;
                entity.vel.1 = -entity.vel.1;
            }
        }
    }

    fn render(&self, screen: &mut Screen) {
        for y in 0..self.size.1 {
            self.move_cursor(Point(0, y), screen);
            for x in 0..self.size.0 {
                let point = Point(x, y);
                let blocked = self.entities.iter().any(|x| x.pos == point);
                if blocked { write!(screen, "XX") } else { write!(screen, "  ") }.unwrap();
            }
        }
        screen.flush().ok();
    }

    fn move_cursor(&self, point: Point, out: &mut Screen) {
        let (x, y) = (point.0, point.1);
        write!(out, "{}", Goto((x + 1) as u16, (y + 1) as u16)).unwrap();
    }
}

fn enter_alt_screen(screen: &mut Screen) {
    write!(screen, "{}{}{}", ToAlternateScreen, Hide, termion::clear::All).unwrap();
    write!(screen, "{}", termion::clear::All).unwrap();
    screen.flush().unwrap();
}

fn exit_alt_screen(screen: &mut Screen) {
    write!(screen, "{}{}", ToMainScreen, Show).unwrap();
    screen.flush().unwrap();
}

fn main() {
    let mut inputs = termion::async_stdin().keys();
    let mut screen = io::stdout().into_raw_mode().unwrap();
    enter_alt_screen(&mut screen);

    let mut game = Game { entities: vec![], size: Point(40, 40) };

    for _ in 0..3 {
        let x = random::<i32>() % game.size.0;
        let y = random::<i32>() % game.size.1;
        let vx = if random::<bool>() { 1 } else { -1 };
        let vy = if random::<bool>() { 1 } else { -1 };
        game.entities.push(Entity { pos: Point(x, y), vel: Point(vx, vy) });
    }

    game_loop(game, 6, 0.1, |g| {
        if let Some(Ok(key)) = inputs.next() {
            if key == Key::Char('q') || key == Key::Ctrl('c') {
                g.exit();
            }
        }
        g.game.update()
    }, |g| {
        g.game.render(&mut screen);
        std::thread::sleep(std::time::Duration::from_micros(800));
    });

    exit_alt_screen(&mut screen);
}
