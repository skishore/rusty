use std::cmp::{max, min};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use lazy_static::lazy_static;
use rand::random;

use crate::assert_eq_size;
use crate::base::{Color, FOV, Glyph, HashMap, Matrix, Point};
use crate::entity::{Entity, ETRef, Token, Pokemon, Trainer, WeakEntity};
use crate::pathing::{DIRECTIONS, AStar, BFS, BFSResult, Status};

//////////////////////////////////////////////////////////////////////////////

// Constants

const MAP_SIZE: i32 = 100;
const MAX_MEMORY: i32 = 1024;

const FOV_RADIUS_SMALL: i32 = 12;
const FOV_RADIUS_LARGE: i32 = 15;
const VISION_ANGLE: f64 = std::f64::consts::TAU / 3.;
const VISION_RADIUS: i32 = 3;

const MOVE_TIMER: i32 = 960;
const TURN_TIMER: i32 = 120;

const WANDER_TURNS: f64 = 3.;

const ASTAR_LIMIT_ATTACK: i32 = 32;
const ASTAR_LIMIT_WANDER: i32 = 256;
const BFS_LIMIT_ATTACK: i32 = 8;
const BFS_LIMIT_WANDER: i32 = 64;

#[derive(Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char) }

type StdCell<T> = std::cell::Cell<T>;

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

impl Tile {
    fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    fn blocked(&self) -> bool { self.flags & FLAG_BLOCKED != 0 }
    fn obscure(&self) -> bool { self.flags & FLAG_OBSCURE != 0 }
}

lazy_static! {
    static ref TILES: HashMap<char, Tile> = {
        let items = [
            ('.', (FLAG_NONE,    Glyph::wide('.'),        "grass")),
            ('"', (FLAG_OBSCURE, Glyph::wdfg('"', 0x231), "tall grass")),
            ('#', (FLAG_BLOCKED, Glyph::wdfg('#', 0x010), "a tree")),
        ];
        let mut result = HashMap::default();
        for (ch, (flags, glyph, description)) in items {
            result.insert(ch, Tile { flags, glyph, description });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

struct CellKnowledge {
    age: i32,
    entity: Option<Rc<EntityKnowledge>>,
    tile: &'static Tile,
    visibility: i32,
}
assert_eq_size!(CellKnowledge, 24);

struct EntityKnowledge {
    age: StdCell<i32>,
    pos: StdCell<Point>,
    glyph: StdCell<Glyph>,
    moved: StdCell<bool>,
    rival: bool,
    _weak: WeakEntity,
}
assert_eq_size!(EntityKnowledge, 32);

#[derive(Default)]
struct Knowledge {
    map: HashMap<Point, CellKnowledge>,
    entities: HashMap<usize, Rc<EntityKnowledge>>,
}

impl Knowledge {
    // Reads

    fn get_cell(&self, p: Point) -> Option<&CellKnowledge> { self.map.get(&p) }

    fn get_status(&self, p: Point) -> Option<Status> {
        self.get_cell(p).map(|x| {
            if x.entity.is_some() { return Status::Occupied; }
            if x.tile.blocked() { Status::Blocked } else { Status::Free }
        })
    }

    fn can_see_now(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| x.age == 0).unwrap_or(false)
    }

    fn remembers(&self, p: Point) -> bool {
        self.map.contains_key(&p)
    }

    fn blocked(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| x.tile.blocked()).unwrap_or(false)
    }

    fn unblocked(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| !x.tile.blocked()).unwrap_or(false)
    }

    // Writes

    fn update(&mut self, board: &BaseBoard, e: &Entity, t: &Token, vision: &Vision) {
        let my_entity = e.base(t);
        self.forget(my_entity.player);
        let offset = vision.offset - my_entity.pos;
        let my_trainer = trainer(e, t);

        for point in &vision.cells_seen {
            let visibility = vision.visibility.get(*point + offset);
            assert!(visibility >= 0);

            let entity = (|| {
                let entity = board.get_entity_at(*point)?;
                let rival = !trainers_match(&my_trainer, &trainer(entity, t));
                let glyph = entity.base(t).glyph;
                let known = self.entities.entry(entity.id()).and_modify(|x| {
                    let old_pos = x.pos.get();
                    if !x.moved.get() && old_pos != *point {
                        self.map.entry(old_pos).and_modify(|x| {
                            assert!(x.entity.is_some());
                            x.entity = None;
                        });
                    }
                    x.age.set(0);
                    x.pos.set(*point);
                    x.glyph.set(glyph);
                    x.moved.set(false);
                }).or_insert_with(|| {
                    let known = EntityKnowledge {
                        age: 0.into(),
                        pos: (*point).into(),
                        glyph: glyph.into(),
                        moved: false.into(),
                        rival,
                        _weak: entity.into(),
                    };
                    Rc::new(known)
                });
                Some(Rc::clone(known))
            })();

            let tile = board.map.get(*point);
            let cell = CellKnowledge { age: 0, entity, tile, visibility };
            let prev = self.map.insert(*point, cell);
            if let Some(CellKnowledge { entity: Some(x), .. }) = prev {
                assert!(x.pos.get() == *point);
                if x.age.get() > 0 { x.moved.set(true); }
            }
        }
    }

    fn forget(&mut self, player: bool) {
        if player {
            self.map.iter_mut().for_each(|x| x.1.age = 1);
            self.entities.iter_mut().for_each(|x| x.1.age.set(1));
            return;
        }

        let mut removed: Vec<Point> = vec![];
        for (key, val) in self.map.iter_mut() {
            val.age += 1;
            if val.age >= MAX_MEMORY { removed.push(*key); }
        }
        removed.iter().for_each(|x| { self.map.remove(x); });

        let mut removed: Vec<usize> = vec![];
        for (key, val) in self.entities.iter_mut() {
            let age = val.age.get() + 1;
            if age >= MAX_MEMORY { removed.push(*key); } else { val.age.set(age); }
        }
        removed.iter().for_each(|x| { self.entities.remove(x); });
    }
}

//////////////////////////////////////////////////////////////////////////////

// Vision

struct Vision {
    cells_seen: Vec<Point>,
    visibility: Matrix<i32>,
    offset: Point,
}

impl Default for Vision {
    fn default() -> Self {
        let radius = max(FOV_RADIUS_SMALL, FOV_RADIUS_LARGE);
        let vision_side = 2 * radius + 1;
        let vision_size = Point(vision_side, vision_side);
        Self {
            cells_seen: vec![],
            visibility: Matrix::new(vision_size, -1),
            offset: Point(radius, radius),
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Board

struct BaseBoard {
    map: Matrix<&'static Tile>,
    entity_index: usize,
    entity_at_pos: HashMap<Point, Entity>,
    entities: Vec<Entity>,
}

struct Board {
    base: BaseBoard,
    fov_large: FOV,
    fov_small: FOV,
    vision: Vision,
    known: HashMap<usize, Box<Knowledge>>,
}

impl Deref for Board {
    type Target = BaseBoard;
    fn deref(&self) -> &Self::Target { &self.base }
}

impl DerefMut for Board {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.base }
}

impl BaseBoard {
    fn new(size: Point) -> Self {
        Self {
            map: Matrix::new(size, Tile::get('#')),
            entity_index: 0,
            entity_at_pos: HashMap::default(),
            entities: vec![],
        }
    }

    // Reads

    fn get_active_entity(&self) -> &Entity {
        &self.entities[self.entity_index]
    }

    fn get_entity_at(&self, p: Point) -> Option<&Entity> {
        self.entity_at_pos.get(&p)
    }

    fn get_status(&self, p: Point) -> Status {
        if self.entity_at_pos.contains_key(&p) { return Status::Occupied; }
        if self.map.get(p).blocked() { Status::Blocked } else { Status::Free }
    }

    // Field-of-vision

    fn compute_vision(&self, e: &Entity, t: &Token, fov: &mut FOV, vision: &mut Vision) {
        let entity = e.base(t);
        let omni = entity.player;
        let pos = entity.pos;
        let dir = entity.dir;

        vision.visibility.fill(-1);
        vision.cells_seen.clear();

        let blocked = |p: Point, prev: Option<&Point>| {
            if !omni && !Self::in_vision_cone(dir, p) { return true; }

            let lookup = p + vision.offset;
            let cached = vision.visibility.get(lookup);

            let visibility = (|| {
                // These constant values come from Point.distanceNethack.
                // They are chosen such that, in a field of tall grass, we'll
                // only see cells at a distanceNethack <= kVisionRadius.
                if prev.is_none() { return 100 * (VISION_RADIUS + 1) - 95 - 46 - 25; }

                let tile = self.map.get(p + pos);
                if tile.blocked() { return 0; }

                let parent = prev.unwrap();
                let diagonal = p.0 != parent.0 && p.1 != parent.1;
                let loss = if tile.obscure() { 95 + if diagonal { 46 } else { 0 } } else { 0 };
                let prev = vision.visibility.get(*parent + vision.offset);
                max(prev - loss, 0)
            })();

            if visibility > cached {
                vision.visibility.set(lookup, visibility);
                if cached < 0 && 0 <= visibility {
                    vision.cells_seen.push(p + pos);
                }
            }
            visibility <= 0
        };
        fov.apply(blocked);
    }

    fn in_vision_cone(pos: Point, dir: Point) -> bool {
        if pos == Point::default() || dir == Point::default() { return true; }
        let dot = (pos.0 as i64 * dir.0 as i64 + pos.1 as i64 * dir.1 as i64) as f64;
        let l2_product = (pos.len_l2_squared() as f64 * dir.len_l2_squared() as f64).sqrt();
        dot / l2_product > (0.5 * VISION_ANGLE).cos()
    }
}

impl Board {
    fn new(size: Point) -> Self {
        Self {
            base: BaseBoard::new(size),
            fov_large: FOV::new(FOV_RADIUS_LARGE),
            fov_small: FOV::new(FOV_RADIUS_SMALL),
            vision: Vision::default(),
            known: HashMap::default(),
        }
    }

    // Knowledge

    fn get_known(&self, e: &Entity) -> &Knowledge {
        self.known.get(&e.id()).unwrap()
    }

    fn update_known(&mut self, e: &Entity, t: &Token) -> &Knowledge {
        let player = e.base(t).player;
        let fov = if player { &mut self.fov_large } else { &mut self.fov_small };
        self.base.compute_vision(e, t, fov, &mut self.vision);
        let known = self.known.get_mut(&e.id()).unwrap();
        known.update(&self.base, e, t, &self.vision);
        known
    }

    // Writes

    fn add_entity(&mut self, e: &Entity, t: &Token) {
        self.entities.push(e.clone());
        self.known.insert(e.id(), Box::default());
        let collider = self.entity_at_pos.insert(e.base(t).pos, e.clone());
        assert!(collider.is_none());
        self.update_known(e, t);
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
        assert!(existing.same(e));
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
        assert!(existing.same(e));
        let index = self.entities.iter().position(|x| x.same(e)).unwrap();
        self.entities.remove(index);
        self.known.remove(&e.id());

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
    let ft = Tile::get('.');
    let wt = Tile::get('#');
    let gt = Tile::get('"');

    map.fill(ft);
    let d100 = || random::<u32>() % 100;
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

pub struct MoveData { dir: Point, turns: f64 }

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
    fn success() -> Self { Self::success_turns(1.) }
    fn success_turns(turns: f64) -> Self { Self { success: true,  moves: 0., turns } }
}

fn sample<T>(xs: &[T]) -> &T {
    assert!(!xs.is_empty());
    &xs[random::<usize>() % xs.len()]
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

fn trainer(e: &Entity, t: &Token) -> Option<Trainer> {
    match e.test_ref(t) {
        ETRef::Pokemon(x) => x.data(t).individual.trainer.upgrade(),
        ETRef::Trainer(x) => Some(x.clone()),
    }
}

fn trainers_match(a: &Option<Trainer>, b: &Option<Trainer>) -> bool {
    if let (Some(aa), Some(bb)) = (a, b) { return aa.same(bb); }
    a.is_none() && b.is_none()
}

fn explore_near(known: &Knowledge, e: &Entity, t: &Token,
                source: Point, age: i32, turns: f64) -> Action {
    let pos = e.base(t).pos;
    let check = |p: Point| {
        if p == pos { return Status::Free; }
        known.get_status(p).unwrap_or(Status::Free)
    };
    let done1 = |p: Point| {
        let cell = known.get_cell(p);
        if cell.map(|x| x.age <= age).unwrap_or(false) { return false; }
        DIRECTIONS.iter().any(|x| known.unblocked(p + *x))
    };
    let done0 = |p: Point| {
        done1(p) && DIRECTIONS.iter().all(|x| !known.blocked(p + *x))
    };

    let result = BFS(source, done0, BFS_LIMIT_WANDER, check).or_else(||
                 BFS(source, done1, BFS_LIMIT_WANDER, check));

    let dir = (|| {
        let BFSResult { dirs, mut targets } = result?;
        if dirs.is_empty() || targets.is_empty() { return None; }
        let target = (|| {
            if source == pos { return *sample(&targets); }
            targets.sort_by_cached_key(|x| (*x - pos).len_l2_squared());
            targets[0]
        })();
        let path = AStar(pos, target, ASTAR_LIMIT_WANDER, check).unwrap_or(vec![]);
        if path.is_empty() { return Some(*sample(&dirs)); }
        Some(path[0] - pos)
    })();

    let dir = dir.unwrap_or_else(|| *sample(&DIRECTIONS));
    Action::Move(MoveData { dir, turns })
}

fn wander(known: &Knowledge, e: &Pokemon, t: &mut Token) -> Action {
    let wander = &mut e.data_mut(t).wander;
    wander.time = wander.time - 1;
    if wander.time < 0 {
        wander.wait = !wander.wait;
        let multiplier = if wander.wait { WANDER_TURNS } else { 1. };
        let limit = max(1, (16. * multiplier).round() as i32);
        wander.time = random::<i32>().rem_euclid(limit);
    }
    if wander.wait { return Action::Idle; }
    explore_near(known, e, t, e.base(t).pos, 9999, WANDER_TURNS)
}

fn plan_pokemon(known: &Knowledge, e: &Pokemon, t: &mut Token) -> Action {
    let mut targets: Vec<(i32, Point)> = vec![];
    for entity in known.entities.values() {
        if !entity.rival { continue; }
        targets.push((entity.age.get(), entity.pos.get()));
    }
    if !targets.is_empty() {
        targets.sort_by_cached_key(|(age, _)| *age);
        let (age, pos) = targets[0];
        if age == 0 { return Action::Idle; }
        return explore_near(known, e, t, pos, age, 1.);
    }
    wander(known, e, t)
}

fn plan(known: &Knowledge, e: &Entity, t: &mut Token, input: &mut Option<Action>) -> Action {
    let entity = e.base(t);
    if entity.player {
        return input.take().unwrap_or(Action::WaitForInput)
    }
    match e.test_ref(t) {
        ETRef::Pokemon(x) => plan_pokemon(known, x, t),
        ETRef::Trainer(_) => Action::Idle,
    }
}

fn act(state: &mut State, e: &Entity, action: Action) -> ActionResult {
    match action {
        Action::Idle => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::Move(MoveData { dir ,turns }) => {
            if dir == Point::default() { return ActionResult::success_turns(turns); }
            e.base_mut(&mut state.t).dir = dir;
            let target = e.base(&state.t).pos + dir;
            if state.board.get_status(target) == Status::Free {
                state.board.move_entity(e, &mut state.t, target);
                return ActionResult::success_turns(turns);
            }
            ActionResult::failure()
        }
    }
}

fn process_input(state: &mut State, input: Input) {
    if input == Input::Char('\t') || input == Input::BackTab {
        let length = state.board.entities.len();
        let shift = if input == Input::Char('\t') { 1 } else { length - 1 };
        state.index = (state.index + shift) % length;
        return;
    }

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
    state.input = dir.map(|x| Action::Move(MoveData { dir: x, turns: 1. }));
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

    let mut update = false;

    while player_alive(state) {
        let entity = state.board.get_active_entity();
        if !turn_ready(entity, &state.t) {
            state.board.advance_entity(&mut state.t);
            continue;
        } else if needs_input(state) {
            break;
        }

        let entity = entity.clone();
        let known = state.board.update_known(&entity, &state.t);
        let action = plan(known, &entity, &mut state.t, &mut state.input);

        let result = act(state, &entity, action);
        if entity.base(&state.t).player && !result.success { break; }
        wait(&entity, &mut state.t, &result);
        update = true;
    }

    if update {
        state.board.update_known(&state.player, &state.t);
        let entity = state.board.entities[state.index].clone();
        state.board.update_known(&entity, &state.t);
    }
}

//////////////////////////////////////////////////////////////////////////////

// State

pub struct State {
    board: Board,
    index: usize,
    input: Option<Action>,
    inputs: Vec<Input>,
    player: Trainer,
    t: Token,
}

impl State {
    pub fn new() -> Self {
        let size = Point(MAP_SIZE, MAP_SIZE);
        let mut board = Board::new(size);
        let pos = Point(size.0 / 2, size.1 / 2);

        loop {
            mapgen(&mut board.map);
            if !board.map.get(pos).blocked() { break; }
        }

        let t = unsafe { Token::new() };
        let player = Trainer::new(pos, true);
        board.add_entity(&player, &t);

        let rng = |n: i32| random::<i32>().rem_euclid(n);
        let pos = |board: &Board| {
            for _ in 0..100 {
                let p = Point(rng(size.0), rng(size.1));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        let options = ["Pidgey", "Ratatta"];
        for _ in 0..20 {
            if let Some(pos) = pos(&board) {
                let (dir, species) = (*sample(&DIRECTIONS), *sample(&options));
                board.add_entity(&Pokemon::new(pos, dir, species, None), &t);
            }
        }

        Self { board, index: 0, input: None, inputs: vec![], player, t }
    }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Matrix<Glyph>) {
        let entity = &self.board.entities[self.index];
        let known = self.board.get_known(entity);
        let offset = Point(0, 0);
        let unseen = Glyph::wide(' ');

        for y in 0..buffer.size.1 {
            for x in 0..(buffer.size.0 / 2) {
                let point = Point(x, y);
                let glyph = match known.get_cell(point + offset) {
                    Some(cell) => if cell.age > 0 {
                        cell.tile.glyph.fg(Color::gray())
                    } else {
                        cell.entity.as_ref().map(|x| x.glyph.get()).unwrap_or(cell.tile.glyph)
                    },
                    None => unseen,
                };
                buffer.set(Point(2 * x, y), glyph);
            }
        }
        for entity in known.entities.values() {
            let Point(x, y) = entity.pos.get();
            let color = if entity.age.get() > 0 {
                if entity.moved.get() { 0x400 } else { 0x440 }
            } else {
                0x040
            };
            buffer.set(Point(2 * x, y), entity.glyph.get().fg(Color::black()).bg(color));
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

#[allow(soft_unstable)]
#[cfg(test)]
mod tests {
    use super::*;
    extern crate test;

    #[bench]
    fn bench_state_update(b: &mut test::Bencher) {
        let mut state = State::new();
        b.iter(|| {
            state.inputs.push(Input::Char('.'));
            state.update();
        });
    }
}
