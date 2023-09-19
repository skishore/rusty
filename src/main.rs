mod base;

use base::{Cell, Token};
use std::io::{self, StdoutLock, Write};

use game_loop::game_loop;
use rand::random;

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

    fn render(&self) {
        let mut lock = io::stdout().lock();
        self.clear_screen(&mut lock);
        self.move_cursor(Point(0, 0), &mut lock);
        for y in 0..self.size.1 {
            for x in 0..self.size.0 {
                let point = Point(x, y);
                let blocked = self.entities.iter().any(|x| x.pos == point);
                if blocked { write!(lock, "XX") } else { write!(lock, "  ") }.ok();
            }
            write!(lock, "\n").ok();
        }
    }

    fn clear_screen(&self, lock: &mut StdoutLock) {
        write!(lock, "\x1b[2J").ok();
    }

    fn move_cursor(&self, point: Point, lock: &mut StdoutLock) {
        write!(lock, "\x1b[{};{}H", point.1 + 1, point.0 + 1).ok();
    }
}

fn main() {
    let mut game = Game { entities: vec![], size: Point(40, 40) };

    for _ in 0..3 {
        let x = random::<i32>() % game.size.0;
        let y = random::<i32>() % game.size.1;
        let vx = if random::<bool>() { 1 } else { -1 };
        let vy = if random::<bool>() { 1 } else { -1 };
        game.entities.push(Entity { pos: Point(x, y), vel: Point(vx, vy) });
    }

    game_loop(game, 6, 0.1, |g| {
        g.game.update()
    }, |g| {
        g.game.render();
        std::thread::sleep(std::time::Duration::from_micros(800));
    });
}
