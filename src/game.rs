use std::cmp::{max, min};

use lazy_static::lazy_static;
use rand::random;

use crate::assert_eq_size;
use crate::base::{FOV, Glyph, HashMap, Matrix, Point};
use crate::entity::{Entity, Token, Trainer};

//////////////////////////////////////////////////////////////////////////////

// Constants

const FOV_RADIUS: i32 = 15;
const VISION_RADIUS: i32 = 3;

const MOVE_TIMER: i32 = 960;
const TURN_TIMER: i32 = 120;

pub enum Input { Escape, Char(char) }

//////////////////////////////////////////////////////////////////////////////

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

//////////////////////////////////////////////////////////////////////////////

// Board

enum Status { Free, Blocked, Occupied }

struct Board {
    fov: FOV,
    map: Matrix<&'static Tile>,
    entity_index: usize,
    entity_at_pos: HashMap<Point, Entity>,
    entities: Vec<Entity>,
}

impl Board {
    fn new(size: Point) -> Self {
        Self {
            fov: FOV::new(FOV_RADIUS),
            map: Matrix::new(size, TILES.get(&'.').unwrap()),
            entity_index: 0,
            entity_at_pos: HashMap::new(),
            entities: vec![],
        }
    }

    // Reads

    fn get_active_entity(&self) -> &Entity {
        &self.entities[self.entity_index]
    }

    fn get_status(&self, point: Point) -> Status {
        if self.entity_at_pos.contains_key(&point) { return Status::Occupied; }
        let blocked = self.map.get(point).flags & FLAG_BLOCKED != 0;
        if blocked { Status::Blocked } else { Status::Free }
    }

    // Writes

    fn add_entity(&mut self, e: &Entity, t: &Token) {
        self.entities.push(e.clone());
        let collider = self.entity_at_pos.insert(e.base(t).pos, e.clone());
        assert!(collider.is_none());
    }

    fn advance_entity(&mut self, t: &mut Token) {
        charge(self.get_active_entity(), t);
        self.entity_index += 1;
        if self.entity_index == self.entities.len() {
            self.entity_index = 0;
        }
    }

    fn move_entity(&mut self, e: &Entity, t: &mut Token, to: Point) {
        let existing = self.entity_at_pos.remove(&e.base(t).pos).unwrap();
        assert!(existing.same(&e));
        let collider = self.entity_at_pos.insert(to, existing);
        assert!(collider.is_none());
        e.base_mut(t).pos = to;
    }

    fn remove_entity(&mut self, e: &Entity, t: &mut Token) {
        // The player is just tagged "removed", so we always have an entity.
        let entity = e.base_mut(t);
        entity.removed = true;
        if entity.player { return; }

        // Remove entities other than the player.
        let existing = self.entity_at_pos.remove(&entity.pos).unwrap();
        assert!(existing.same(&e));
        let index = self.entities.iter().position(|x| x.same(&e)).unwrap();
        self.entities.remove(index);

        // Fix up entity_index after removing the entity.
        if self.entity_index > index {
            self.entity_index -= 1;
        } else if self.entity_index == self.entities.len() {
            self.entity_index = 0;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Map generation

fn mapgen(map: &mut Matrix<&'static Tile>) {
    map.fill(map.default);
    let d100 = || random::<i32>().rem_euclid(100);
    let size = map.size;

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
                    let point = Point(x, y);
                    let (mut adj1, mut adj2) = (0, 0);
                    for dy in -2_i32..=2 {
                        for dx in -2_i32..=2 {
                            if dx == 0 && dy == 0 { continue; }
                            if min(dx.abs(), dy.abs()) == 2 { continue; }
                            let next = point + Point(dx, dy);
                            if !result.get(next) { continue; }
                            let distance = max(dx.abs(), dy.abs());
                            if distance <= 1 { adj1 += 1; }
                            if distance <= 2 { adj2 += 1; }
                        }
                    }
                    let blocked = adj1 >= 5 || (i < 2 && adj2 <= 1);
                    next.set(point, blocked);
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
                map.set(point, wt);
            } else if grass.get(point) {
                map.set(point, gt);
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Game logic

pub struct MoveData { dir: Point }

pub enum Action {
    Idle,
    Move(MoveData),
    WaitForInput,
}

struct ActionResult {
    success: bool,
    moves: f64,
    turns: f64,
}

impl ActionResult {
    fn failure() -> Self { Self { success: false, moves: 0., turns: 1. } }
    fn success() -> Self { Self { success: true,  moves: 0., turns: 1. } }
}

fn charge(e: &Entity, t: &mut Token) {
    let entity = e.base_mut(t);
    let charge = (TURN_TIMER as f64 * entity.speed).round() as i32;
    if entity.move_timer > 0 { entity.move_timer -= charge; }
    if entity.turn_timer > 0 { entity.turn_timer -= charge; }
}

fn move_ready(e: &Entity, t: &Token) -> bool { e.base(t).move_timer <= 0 }

fn turn_ready(e: &Entity, t: &Token) -> bool { e.base(t).turn_timer <= 0 }

fn wait(e: &Entity, t: &mut Token, result: &ActionResult) {
    let entity = e.base_mut(t);
    entity.move_timer += (MOVE_TIMER as f64 * result.moves).round() as i32;
    entity.turn_timer += (TURN_TIMER as f64 * result.turns).round() as i32;
}

fn plan(board: &Board, e: &Entity, t: &Token, input: &mut Option<Action>) -> Action {
    let entity = e.base(t);
    if entity.player {
        input.take().unwrap_or(Action::WaitForInput)
    } else {
        Action::Idle
    }
}

fn act(state: &mut State, e: &Entity, action: Action) -> ActionResult {
    match action {
        Action::Idle => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::Move(MoveData { dir }) => {
            if dir == Point::default() { return ActionResult::success(); }
            let target = e.base(&state.t).pos + dir;
            if let Status::Free = state.board.get_status(target) {
                state.board.move_entity(&state.player, &mut state.t, target);
                return ActionResult::success();
            }
            ActionResult::failure()
        }
    }
}

fn process_input(state: &mut State, input: Input) {
    let dir = match input {
        Input::Char('h') => Some(Point(-1,  0)),
        Input::Char('j') => Some(Point( 0,  1)),
        Input::Char('k') => Some(Point( 0, -1)),
        Input::Char('l') => Some(Point( 1,  0)),
        Input::Char('y') => Some(Point(-1, -1)),
        Input::Char('u') => Some(Point( 1, -1)),
        Input::Char('b') => Some(Point(-1,  1)),
        Input::Char('n') => Some(Point( 1,  1)),
        Input::Char('.') => Some(Point( 0,  0)),
        _ => None,
    };
    state.input = dir.map(|x| Action::Move(MoveData { dir: x }));
}

fn update_state(state: &mut State) {
    let player_alive = |state: &State| {
        !state.player.base(&state.t).removed
    };

    let needs_input = |state: &State| {
        if state.input.is_some() { return false; }
        let active = state.board.get_active_entity();
        if !active.same(&state.player) { return false; }
        player_alive(state)
    };

    while !state.inputs.is_empty() && needs_input(state) {
        let input = state.inputs.remove(0);
        process_input(state, input);
    }

    while player_alive(&state) {
        let e = state.board.get_active_entity();
        if !turn_ready(e, &state.t) {
            state.board.advance_entity(&mut state.t);
            continue;
        }
        let entity = e.clone();
        let action = plan(&state.board, &entity, &state.t, &mut state.input);
        let result = act(state, &entity, action);
        if entity.base(&state.t).player && !result.success { break; }
        wait(&entity, &mut state.t, &result);
    }
}

//////////////////////////////////////////////////////////////////////////////

// State

pub struct State {
    board: Board,
    input: Option<Action>,
    inputs: Vec<Input>,
    player: Trainer,
    t: Token,
}

impl State {
    pub fn new(size: Point) -> Self {
        let mut board = Board::new(size);
        let pos = Point(size.0 / 2, size.1 / 2);

        loop {
            mapgen(&mut board.map);
            if board.map.get(pos).flags & FLAG_BLOCKED == 0 { break; }
        }

        let t = unsafe { Token::new() };
        let player = Trainer::new(pos, true);
        board.add_entity(&player, &t);
        Self { board, input: None, inputs: vec![], player, t }
    }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Matrix<Glyph>) {
        let pos = self.player.base(&self.t).pos;
        let offset = Point(FOV_RADIUS, FOV_RADIUS);
        let vision = self.compute_vision(pos);
        let unseen = Glyph::wide(' ');

        for y in 0..buffer.size.1 {
            for x in 0..(buffer.size.0 / 2) {
                let point = Point(x, y);
                let glyph = self.board.map.get(point).glyph;
                let sight = vision.get(point - pos + offset);
                buffer.set(Point(2 * x, y), if sight < 0 { unseen } else { glyph });
            }
        }

        for entity in &self.board.entities {
            let Point(x, y) = entity.base(&self.t).pos;
            let seen = buffer.get(Point(2 * x, y)) != unseen;
            if seen { buffer.set(Point(2 * x, y), entity.base(&self.t).glyph) }
        }
    }

    pub fn compute_vision(&self, pos: Point) -> Matrix<i32> {
        let side = 2 * FOV_RADIUS + 1;
        let offset = Point(FOV_RADIUS, FOV_RADIUS);
        let mut vision = Matrix::new(Point(side, side), -1);

        let blocked = |p: Point, prev: Option<&Point>| {
            let visibility = (|| {
                // These constant values come from Point.distanceNethack.
                // They are chosen such that, in a field of tall grass, we'll
                // only see cells at a distanceNethack <= kVisionRadius.
                if prev.is_none() { return 100 * (VISION_RADIUS + 1) - 95 - 46 - 25; }

                let tile = self.board.map.get(p + pos);
                if tile.flags & FLAG_BLOCKED != 0 { return 0; }

                let parent = prev.unwrap();
                let obscure = tile.flags & FLAG_OBSCURE != 0;
                let diagonal = p.0 != parent.0 && p.1 != parent.1;
                let loss = if obscure { 95 + if diagonal { 46 } else { 0 } } else { 0 };
                let prev = vision.get(*parent + offset);
                max(prev - loss, 0)
            })();

            vision.set(p + offset, visibility);
            visibility <= 0
        };
        self.board.fov.apply(blocked);
        vision
    }
}
