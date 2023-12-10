use crate::base::{Color, HashSet, Glyph, LOS, Point, dirs, sample};

//////////////////////////////////////////////////////////////////////////////

pub trait BoardLike {
    fn get_glyph_at(&self, p: Point) -> Glyph;
    fn get_underlying_glyph_at(&self, p: Point) -> Glyph;
}

//////////////////////////////////////////////////////////////////////////////

// Types

#[derive(Copy, Clone)]
pub enum FT { Fire, Ice, Hit, Summon, Withdraw }

pub enum Event {
    Callback { frame: i32, callback: Box<dyn Fn()> },
    Other { frame: i32, point: Point, what: FT },
}

#[derive(Copy, Clone)]
pub struct Particle {
    pub point: Point,
    pub glyph: Glyph,
}

pub type Frame = Vec<Particle>;

#[derive(Default)]
pub struct Effect {
    pub events: Vec<Event>,
    pub frames: Vec<Frame>,
}

impl Effect {
    pub fn new(frames: Vec<Frame>) -> Self {
        Self { events: Vec::new(), frames }
    }

    pub fn and(self, other: Effect) -> Effect {
        Effect::parallel(vec![self, other])
    }

    pub fn then(self, other: Effect) -> Effect {
        Effect::serial(vec![self, other])
    }

    pub fn delay(self, n: i32) -> Effect {
        Effect::pause(n).then(self)
    }

    pub fn scale(self, s: f64) -> Effect {
        let mut result = Effect::default();
        let scale = |f: i32| (s * f as f64).round() as i32;

        for event in self.events.into_iter() {
            result.events.push(event.update_frame(scale));
        }
        for i in 0..(self.frames.len() as i32) {
            let (start, limit) = (scale(i), scale(i + 1));
            let frame: &Frame = &self.frames[i as usize];
            for _ in start..limit {
                result.frames.push(frame.clone());
            }
        }
        result
    }

    // Mutate the Effect in-place

    pub fn add_event(&mut self, event: Event) {
        self.events.push(event);
        self.events.sort_by_key(|e| e.frame());
    }

    pub fn add_particle(&mut self, frame: i32, particle: Particle) {
        while (self.frames.len() as i32) <= frame {
            self.frames.push(Vec::new());
        }
        self.frames[frame as usize].push(particle);
    }

    // Implementations for Effect::Constant, Effect::Pause, etc.

    pub fn constant(particle: Particle, n: i32) -> Effect {
        if n <= 0 { return Effect::default(); }
        Effect::new(vec![vec![particle]; n as usize])
    }

    pub fn pause(n: i32) -> Effect {
        if n <= 0 { return Effect::default(); }
        Effect::new(vec![vec![]; n as usize])
    }

    pub fn parallel(effects: Vec<Effect>) -> Effect {
        let mut result = Effect::default();
        for effect in effects.into_iter() {
            result.events.extend(effect.events);
            for (i, frame) in effect.frames.iter().enumerate() {
                if i >= result.frames.len() {
                    result.frames.push(Vec::new());
                }
                result.frames[i].extend(frame.clone());
            }
        }
        result.events.sort_by_key(|e| e.frame());
        result
    }

    pub fn serial(effects: Vec<Effect>) -> Effect {
        let mut offset = 0;
        let mut result = Effect::default();
        for effect in effects.into_iter() {
            for event in effect.events.into_iter() {
                result.events.push(event.update_frame(|x| x + offset));
            }
            result.frames.extend(effect.frames.clone());
            offset += effect.frames.len() as i32;
        }
        result
    }
}

impl Event {
    fn frame(&self) -> i32 {
        match self {
            Event::Callback { frame, .. } => *frame,
            Event::Other { frame, .. } => *frame,
        }
    }

    fn update_frame<F: FnOnce(i32) -> i32>(self, update: F) -> Event {
        match self {
            Event::Callback { frame, callback } =>
                Event::Callback { frame: update(frame), callback },
            Event::Other { frame, point, what } =>
                Event::Other { frame: update(frame), point, what },
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

type Sparkle<'a> = Vec<(i32, &'a str, Color)>;

fn add_sparkle(effect: &mut Effect, sparkle: &Sparkle,
               mut frame: i32, point: Point) -> i32 {
    for (delay, chars, color) in sparkle {
        for _ in 0..*delay {
            let index = rand::random::<usize>() % chars.chars().count();
            let glyph = Glyph::wdfg(chars.chars().nth(index).unwrap(), *color);
            effect.add_particle(frame, Particle { point, glyph });
            frame += 1;
        }
    }
    frame
}

fn random_delay(n: i32) -> i32 {
    let mut count = 1;
    let limit = (1.5 * n as f64).floor() as i32;
    while count < limit && (rand::random::<i32>() % n) != 0 { count += 1; }
    count
}

fn ray_character(source: Point, target: Point) -> char {
    let Point(x, y) = source - target;
    let (ax, ay) = (x.abs(), y.abs());
    if ax > 2 * ay { return '-'; }
    if ay > 2 * ax { return '|'; }
    if (x > 0) == (y > 0) { '\\' } else { '/' }
}

#[allow(non_snake_case)]
fn OverlayEffect(effect: Effect, particle: Particle) -> Effect {
    let constant = Effect::constant(particle, effect.frames.len() as i32);
    Effect::parallel(vec![effect, constant])
}

#[allow(non_snake_case)]
fn UnderlayEffect(effect: Effect, particle: Particle) -> Effect {
    let constant = Effect::constant(particle, effect.frames.len() as i32);
    Effect::parallel(vec![constant, effect])
}

#[allow(non_snake_case)]
fn ExplosionEffect(point: Point) -> Effect {
    let glyph = |ch: char| Glyph::wdfg(ch, 0x400);
    let base = vec![
        vec![Particle { point, glyph: glyph('*') }],
        vec![
            Particle { point, glyph: glyph('+') },
            Particle { point: point + dirs::N,  glyph: glyph('|') },
            Particle { point: point + dirs::S,  glyph: glyph('|') },
            Particle { point: point + dirs::E,  glyph: glyph('-') },
            Particle { point: point + dirs::W,  glyph: glyph('-') },
        ],
        vec![
            Particle { point: point + dirs::N,  glyph: glyph('-') },
            Particle { point: point + dirs::S,  glyph: glyph('-') },
            Particle { point: point + dirs::E,  glyph: glyph('|') },
            Particle { point: point + dirs::W,  glyph: glyph('|') },
            Particle { point: point + dirs::NE, glyph: glyph('\\') },
            Particle { point: point + dirs::SW, glyph: glyph('\\') },
            Particle { point: point + dirs::NW, glyph: glyph('/') },
            Particle { point: point + dirs::SE, glyph: glyph('/') },
        ],
    ];
    Effect::new(base).scale(4.0)
}

#[allow(non_snake_case)]
fn ImplosionEffect(point: Point) -> Effect {
    let glyph = |ch: char| Glyph::wdfg(ch, 0x400);
    let base = vec![
        vec![
            Particle { point, glyph: glyph('*') },
            Particle { point: point + dirs::NE, glyph: glyph('/') },
            Particle { point: point + dirs::SW, glyph: glyph('/') },
            Particle { point: point + dirs::NW, glyph: glyph('\\') },
            Particle { point: point + dirs::SE, glyph: glyph('\\') },
        ],
        vec![
            Particle { point, glyph: glyph('#') },
            Particle { point: point + dirs::NE, glyph: glyph('/') },
            Particle { point: point + dirs::SW, glyph: glyph('/') },
            Particle { point: point + dirs::NW, glyph: glyph('\\') },
            Particle { point: point + dirs::SE, glyph: glyph('\\') },
        ],
        vec![Particle { point, glyph: glyph('*') }],
        vec![Particle { point, glyph: glyph('#') }],
    ];
    Effect::new(base).scale(3.0)
}

#[allow(non_snake_case)]
fn RayEffect(source: Point, target: Point, speed: i32) -> Effect {
    let line = LOS(source, target);
    if line.len() <= 2 { return Effect::default(); }

    let mut result = Vec::new();
    let glyph = Glyph::wdfg(ray_character(source, target), 0x400);
    let denom = ((line.len() - 2 + speed as usize) % speed as usize) as i32;
    let start = if denom == 0 { speed } else { denom } as usize;
    for i in (start..line.len() - 1).step_by(speed as usize) {
        result.push((0..i).map(|j| Particle { point: line[j], glyph }).collect());
    }
    Effect::new(result)
}

#[allow(non_snake_case)]
pub fn SummonEffect(source: Point, target: Point) -> Effect {
    let mut effect = Effect::default();
    let line = LOS(source, target);
    let ball = Glyph::wdfg('*', 0x400);
    for i in 1..line.len() - 1 {
        effect.frames.push(vec![Particle { point: line[i], glyph: ball }]);
    }
    let frame = effect.frames.len() as i32 + 8;
    effect.add_event(Event::Other { frame, point: target, what: FT::Summon });
    effect.then(ExplosionEffect(target))
}

#[allow(non_snake_case)]
pub fn WithdrawEffect(source: Point, target: Point) -> Effect {
    let mut implode  = ImplosionEffect(target);
    let frame = std::cmp::max(implode.frames.len() as i32 - 6, 0);
    implode.add_event(Event::Other { frame, point: target, what: FT::Withdraw });

    let base = RayEffect(source, target, 4);
    let undo = base.frames.clone().into_iter().rev().collect();
    let last = match base.frames.last() {
        Some(x) => x.clone(),
        None => return implode,
    };

    Effect::serial(vec![
        base,
        Effect::new(vec![last]).scale(implode.frames.len() as f64).and(implode),
        Effect::new(undo),
    ])
}

#[allow(non_snake_case)]
fn SwitchEffect(source: Point, target: Point) -> Effect {
    Effect::serial(vec![
        WithdrawEffect(source, target),
        Effect::pause(4),
        SummonEffect(source, target),
    ])
}

#[allow(non_snake_case)]
pub fn EmberEffect(_: &impl BoardLike, source: Point, target: Point) -> Effect {
    let mut effect = Effect::default();
    let line = LOS(source, target);

    let trail = || vec![
        (random_delay(0), "*^^",   0x400.into()),
        (random_delay(1), "*^",    0x420.into()),
        (random_delay(2), "**^",   0x440.into()),
        (random_delay(3), "**^#%", 0x420.into()),
        (random_delay(4), "#%",    0x400.into()),
    ];

    let flame = || vec![
        (random_delay(0), "*^^",   0x400.into()),
        (random_delay(1), "*^",    0x420.into()),
        (random_delay(2), "**^#%", 0x440.into()),
        (random_delay(3), "*^#%",  0x420.into()),
        (random_delay(4), "*^#%",  0x400.into()),
    ];

    for i in 1..line.len() - 1 {
        let frame = (i - 1) / 2;
        add_sparkle(&mut effect, &trail(), frame as i32, line[i]);
    }

    let mut hit: i32 = 0;
    for direction in vec![dirs::NONE].into_iter().chain(dirs::ALL) {
        let norm = direction.len_taxicab();
        let frame = 2 * norm + (line.len() as i32 - 1) / 2;
        let finish = add_sparkle(&mut effect, &flame(), frame, target + direction);
        if norm == 0 { hit = finish; }
    }
    effect.add_event(Event::Other { frame: hit, point: target, what: FT::Hit });
    effect
}

#[allow(non_snake_case)]
pub fn IceBeamEffect(_: &impl BoardLike, source: Point, target: Point) -> Effect {
    let mut effect = Effect::default();
    let line = LOS(source, target);
    let ray = ray_character(source, target).to_string();

    let trail: Sparkle = vec![
        (2, &ray, 0x555.into()),
        (2, &ray, 0x044.into()),
        (2, &ray, 0x004.into()),
    ];

    let flame: Sparkle = vec![
        (2, "*", 0x555.into()),
        (2, "*", 0x044.into()),
        (2, "*", 0x004.into()),
        (2, "*", 0x555.into()),
        (2, "*", 0x044.into()),
        (2, "*", 0x004.into()),
    ];

    for i in 1..line.len() {
        let frame = ((i - 1) / 2) as i32;
        add_sparkle(&mut effect, &trail, frame, line[i]);
    }

    let frame = ((line.len() - 1) / 2) as i32;
    let hit = add_sparkle(&mut effect, &flame, frame, target);
    effect.add_event(Event::Other { frame: hit, point: target, what: FT::Hit, });
    effect
}

#[allow(non_snake_case)]
pub fn BlizzardEffect(_: &impl BoardLike, source: Point, target: Point) -> Effect {
    let mut effect = Effect::default();
    let ray = ray_character(source, target).to_string();

    let mut points = vec![target];
    let mut used = HashSet::default();
    while points.len() < 3 {
        let alt = target + *sample(&dirs::ALL);
        if !used.insert(alt) { continue; }
        points.push(alt);
    }

    let trail: Sparkle = vec![
        (1, &ray, 0x555.into()),
        (1, &ray, 0x044.into()),
        (1, &ray, 0x004.into()),
    ];

    let flame: Sparkle = vec![
        (1, "*", 0x555.into()),
        (1, "*", 0x044.into()),
        (1, "*", 0x004.into()),
        (1, "*", 0x555.into()),
        (1, "*", 0x044.into()),
        (1, "*", 0x004.into()),
    ];

    let mut hit: i32 = 0;
    for (p, next) in points.iter().enumerate() {
        let d = 9 * p as i32;
        let line = LOS(source, *next);
        let size = line.len() as i32;
        for i in 1..size - 1 {
            add_sparkle(&mut effect, &trail, d + i, line[i as usize]);
        }
        let finish = add_sparkle(&mut effect, &flame, d + size - 1, *next);
        if p == 0 { hit = finish; }
    }
    effect.add_event(Event::Other { frame: hit, point: target, what: FT::Hit });
    effect.scale(2.0 / 3.0)
}

#[allow(non_snake_case)]
pub fn HeadbuttEffect(board: &impl BoardLike, source: Point, target: Point) -> Effect {
    let glyph = board.get_glyph_at(source);
    let underlying = board.get_underlying_glyph_at(source);

    let trail: Sparkle = vec![
        (2, "#", 0x444.into()),
        (2, "#", 0x333.into()),
        (2, "#", 0x222.into()),
        (2, "#", 0x111.into()),
    ];

    let move_along_line = |line: &[Point], glyph: Glyph| {
        let mut effect = Effect::default();
        for i in 1..line.len() as i32 - 1 {
            let point = line[i as usize];
            effect.add_particle(i - 1, Particle { point, glyph });
            add_sparkle(&mut effect, &trail, i, point);
        }
        effect
    };

    let line = LOS(source, target);
    let back = line.iter().rev().cloned().collect::<Vec<_>>();

    let move_length = std::cmp::max(line.len() as i32 - 2, 0);
    let hold_point  = line[move_length as usize];
    let hold_effect = Effect::constant(Particle { point: hold_point, glyph }, 32);

    let to   = move_along_line(&line, glyph);
    let hold = hold_effect.delay(move_length);
    let from = move_along_line(&back, glyph);

    let back_length = hold.frames.len() as i32 + move_length;
    let back_effect = Effect::constant(Particle { point: source, glyph },
                                       from.frames.len() as i32 - move_length);
    let back = back_effect.delay(back_length);

    let hit  = move_length;
    let mut effect = UnderlayEffect(to.and(hold).then(from).and(back),
                                    Particle { point: source, glyph: underlying });
    effect.add_event(Event::Other { frame: hit, point: target, what: FT::Hit });
    effect.scale(0.5)
}
