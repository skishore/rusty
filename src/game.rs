use std::cmp::{max, min};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::ops::{Deref, DerefMut};

use lazy_static::lazy_static;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::{cast, static_assert_size};
use crate::base::{Buffer, Color, Glyph, Slice};
use crate::base::{HashMap, HashSet, LOS, Matrix, Point, Rect, dirs};
use crate::base::{sample, weighted, RNG};
use crate::effect::{Effect, Event, Frame, FT, ray_character, self};
use crate::entity::{EID, PID, TID, EntityMap, EntityMapKey};
use crate::entity::{Entity, Pokemon, Trainer};
use crate::entity::{PokemonArgs, SummonArgs, TrainerArgs};
use crate::entity::{PokemonEdge, PokemonIndividualData, PokemonSpeciesData};
use crate::knowledge::{Knowledge, Timestamp, Vision, VisionArgs, get_pp, get_hp};
use crate::knowledge::{EntityKnowledge, EntityView, PokemonView};
use crate::pathing::{AStar, BFS, BFSResult, Dijkstra, DijkstraMap, Status};

//////////////////////////////////////////////////////////////////////////////

// Constants

pub const MOVE_TIMER: i32 = 960;
pub const TURN_TIMER: i32 = 120;

const ATTACK_DAMAGE: i32 = 40;
const ATTACK_RANGE: i32 = 8;

const ASSESS_TIME: i32 = 17;
const ASSESS_ANGLE: f64 = TAU / 3.;

const MAX_ASSESS: i32 = 32;
const MAX_HUNGER: i32 = 1024;
const MAX_THIRST: i32 = 256;

const ASSESS_STDEV: f64 = TAU / 18.;
const ASSESS_STEPS: i32 = 4;
const ASSESS_TURNS: i32 = 32;

const MIN_FLIGHT_TURNS: i32 = 8;
const MAX_FLIGHT_TURNS: i32 = 64;
const MAX_FOLLOW_TURNS: i32 = 64;
const TURN_TIMES_LIMIT: usize = 64;

const FOV_RADIUS_NPC: i32 = 12;
const FOV_RADIUS_PC_: i32 = 21;

const SUMMON_RANGE: i32 = 12;
const WANDER_STEPS: i32 = 16;
const WANDER_TURNS: f64 = 3.;
const WORLD_SIZE: i32 = 100;

const ASTAR_LIMIT_ATTACK: i32 = 32;
const ASTAR_LIMIT_WANDER: i32 = 256;
const BFS_LIMIT_ATTACK: i32 = 8;
const BFS_LIMIT_WANDER: i32 = 64;
const FLIGHT_MAP_LIMIT: i32 = 1024;

const PLAYER_KEY: char = 'a';
const RETURN_KEY: char = 'r';
const PARTY_KEYS: [char; 6] = ['a', 's', 'd', 'f', 'g', 'h'];
const ATTACK_KEYS: [char; 4] = ['a', 's', 'd', 'f'];
const SUMMON_KEYS: [char; 3] = ['s', 'd', 'f'];

#[derive(Eq, PartialEq)]
pub enum Input { Escape, BackTab, Char(char) }

//////////////////////////////////////////////////////////////////////////////

// Tile

const FLAG_NONE: u32 = 0;
const FLAG_BLOCKED: u32 = 1 << 0;
const FLAG_OBSCURE: u32 = 1 << 1;

pub struct Tile {
    pub flags: u32,
    pub glyph: Glyph,
    pub description: &'static str,
}
static_assert_size!(Tile, 24);

impl Tile {
    pub fn get(ch: char) -> &'static Tile { TILES.get(&ch).unwrap() }
    pub fn blocked(&self) -> bool { self.flags & FLAG_BLOCKED != 0 }
    pub fn obscure(&self) -> bool { self.flags & FLAG_OBSCURE != 0 }
}

impl PartialEq for &'static Tile {
    fn eq(&self, next: &&'static Tile) -> bool {
        *self as *const Tile == *next as *const Tile
    }
}

impl Eq for &'static Tile {}

lazy_static! {
    static ref TILES: HashMap<char, Tile> = {
        let items = [
            ('.', (FLAG_NONE,    Glyph::wide('.'),        "grass")),
            ('"', (FLAG_OBSCURE, Glyph::wdfg('"', 0x231), "tall grass")),
            ('#', (FLAG_BLOCKED, Glyph::wdfg('#', 0x010), "a tree")),
            ('%', (FLAG_NONE,    Glyph::wdfg('%', 0x400), "flowers")),
            ('~', (FLAG_NONE,    Glyph::wdfg('~', 0x015), "water")),
        ];
        let mut result = HashMap::default();
        for (ch, (flags, glyph, description)) in items {
            result.insert(ch, Tile { flags, glyph, description });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// BoardView - a read-only Board

pub struct BoardView {
    _map: Matrix<&'static Tile>,
    active_entity_index: usize,
    entity_at_pos: HashMap<Point, EID>,
    entity_order: Vec<EID>,
    entities: EntityMap,
}

impl BoardView {
    fn new(size: Point) -> Self {
        Self {
            _map: Matrix::new(size, Tile::get('#')),
            active_entity_index: 0,
            entity_at_pos: HashMap::default(),
            entity_order: vec![],
            entities: EntityMap::default(),
        }
    }

    // Reads

    pub fn get_active_entity(&self) -> EID {
        self.entity_order[self.active_entity_index]
    }

    pub fn get_entity<T: EntityMapKey>(&self, id: T) -> Option<&T::ValueType> {
        self.entities.get(id)
    }

    pub fn get_entity_at(&self, p: Point) -> Option<EID> {
        self.entity_at_pos.get(&p).map(|x| *x)
    }

    pub fn get_size(&self) -> Point {
        self._map.size
    }

    pub fn get_status(&self, p: Point) -> Status {
        if self.entity_at_pos.contains_key(&p) { return Status::Occupied; }
        if self._map.get(p).blocked() { Status::Blocked } else { Status::Free }
    }

    pub fn get_tile_at(&self, p: Point) -> &'static Tile {
        self._map.get(p)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Board

struct LogLine {
    color: Color,
    menu: bool,
    text: String,
}

pub struct Board {
    base: BoardView,
    known: Option<Box<Knowledge>>,
    npc_vision: Vision,
    _pc_vision: Vision,
    _effect: Effect,
    log: Vec<LogLine>,
}

impl Deref for Board {
    type Target = BoardView;
    fn deref(&self) -> &Self::Target { &self.base }
}

impl DerefMut for Board {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.base }
}

impl Board {
    fn new(size: Point) -> Self {
        Self {
            base: BoardView::new(size),
            known: Some(Box::default()),
            npc_vision: Vision::new(FOV_RADIUS_NPC),
            _pc_vision: Vision::new(FOV_RADIUS_PC_),
            _effect: Effect::default(),
            log: vec![],
        }
    }

    // Writes

    fn add_effect(&mut self, effect: Effect, rng: &mut RNG) {
        let mut existing = Effect::default();
        std::mem::swap(&mut self._effect, &mut existing);
        self._effect = existing.and(effect);
        self._execute_effect_callbacks(rng);
    }

    fn advance_effect(&mut self, rng: &mut RNG) -> bool {
        if self._effect.frames.is_empty() {
            assert!(self._effect.events.is_empty());
            return false;
        }
        self._effect.frames.remove(0);
        self._effect.events.iter_mut().for_each(|x| x.update_frame(|y| y - 1));
        self._execute_effect_callbacks(rng);
        true
    }

    fn _execute_effect_callbacks(&mut self, rng: &mut RNG) {
        while self._execute_one_effect_callback(rng) {}
    }

    fn _execute_one_effect_callback(&mut self, rng: &mut RNG) -> bool {
        if self._effect.events.is_empty() { return false; }
        let event = &self._effect.events[0];
        if !self._effect.frames.is_empty() && event.frame() > 0 { return false; }
        match self._effect.events.remove(0) {
            Event::Callback { callback, .. } => callback(self, rng),
            Event::Other { .. } => (),
        }
        true
    }

    fn fill_map(&mut self, t: &'static Tile) {
        self._map.fill(t)
    }

    fn set_tile_at(&mut self, p: Point, t: &'static Tile) {
        self._map.set(p, t)
    }

    // Logging

    fn log<S: Into<String>>(&mut self, text: S) {
        self.log_color(text, Color::default());
    }

    fn log_color<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        let (color, text) = (color.into(), text.into());
        self.log.push(LogLine { color, menu: false, text });
        if self.log.len() as i32 > UI_LOG_SIZE { self.log.remove(0); }
    }

    fn log_menu<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        let (color, text) = (color.into(), text.into());
        if self.log.last().map(|x| x.menu).unwrap_or(false) { self.log.pop(); }
        self.log.push(LogLine { color, menu: true, text });
        if self.log.len() as i32 > UI_LOG_SIZE { self.log.remove(0); }
    }

    fn end_menu_logging(&mut self) {
        self.log.last_mut().map(|x| x.menu = false);
    }

    // Knowledge

    fn get_current_frame(&self) -> Option<&Frame> {
        self._effect.frames.iter().next()
    }

    fn set_focus(&mut self, eid: EID, focus: Option<EID>) {
        self.entities[eid].known.focus = focus;
    }

    fn update_known(&mut self, eid: EID) {
        let mut known = self.known.take().unwrap_or_default();
        std::mem::swap(&mut self.base.entities[eid].known, &mut known);

        let entity = &self.base.entities[eid];
        let (player, pos, dir) = (entity.player, entity.pos, entity.dir);
        let vision = if player { &mut self._pc_vision } else { &mut self.npc_vision };
        vision.compute(&VisionArgs { player, pos, dir }, |p| self.base.get_tile_at(p));
        known.update(&entity, &self.base, vision);

        std::mem::swap(&mut self.base.entities[eid].known, &mut known);
        self.known = Some(known);
    }

    // Writes

    fn add_pokemon(&mut self, args: &PokemonArgs) -> PID {
        let pid = self.entities.add_pokemon(args);
        self._add_to_caches(pid.eid(), args.pos);
        pid
    }

    fn add_summoned_pokemon(&mut self, args: SummonArgs) -> PID {
        let pos = args.pos;
        let pid = self.entities.add_summons(args);
        self._add_to_caches(pid.eid(), pos);
        pid
    }

    fn add_trainer(&mut self, args: &TrainerArgs) -> TID {
        let tid = self.entities.add_trainer(args);
        self._add_to_caches(tid.eid(), args.pos);
        tid
    }

    fn _add_to_caches(&mut self, eid: EID, pos: Point) {
        let collider = self.entity_at_pos.insert(pos, eid);
        assert!(collider.is_none());
        self.entity_order.push(eid);
        self.update_known(eid);
    }

    fn advance_entity(&mut self) {
        let eid = self.get_active_entity();
        charge(&mut self.entities[eid]);
        self.active_entity_index += 1;
        if self.active_entity_index == self.entity_order.len() {
            self.active_entity_index = 0;
        }
    }

    fn move_entity(&mut self, eid: EID, to: Point) {
        let entity = &mut self.base.entities[eid];
        let existing = self.base.entity_at_pos.remove(&entity.pos).unwrap();
        assert!(existing == eid);
        let collider = self.base.entity_at_pos.insert(to, existing);
        assert!(collider.is_none());
        entity.pos = to;
    }

    fn swap_entities(&mut self, a: Point, b: Point) {
        assert!(a != b);
        let ea = self.base.entity_at_pos.remove(&a).unwrap();
        let eb = self.base.entity_at_pos.insert(b, ea).unwrap();
        self.base.entity_at_pos.insert(a, eb);
        self.base.entities[ea].pos = b;
        self.base.entities[eb].pos = a;
    }

    fn remove_entity(&mut self, eid: EID) {
        // The player is just tagged "removed", so we always have an entity.
        let entity = &mut self.base.entities[eid];
        entity.removed = true;
        if entity.player { return; }

        // Remove the entity from the entity_at_pos lookup table.
        let existing = self.base.entity_at_pos.remove(&entity.pos).unwrap();
        assert!(existing == eid);

        // Remove the entity from the entity_order list.
        let index = self.entity_order.iter().position(|x| *x == eid).unwrap();
        self.entity_order.remove(index);
        if self.active_entity_index > index {
            self.active_entity_index -= 1;
        } else if self.active_entity_index == self.entity_order.len() {
            self.active_entity_index = 0;
        }

        // Remove the entity from the slotmap.
        let removed = self.entities.remove_entity(eid).unwrap();

        // Delete hard edges from this entity to others.
        match removed {
            Entity::Pokemon(x) => {
                if let Some(tid) = x.data.me.trainer {
                    let trainer = &mut self.entities[tid];
                    trainer.data.summons.retain(|x| x.eid() != eid);
                    for edge in &mut trainer.data.pokemon {
                        if let PokemonEdge::Out(z) = edge && z.eid() == eid {
                            *edge = PokemonEdge::In(x.data.me);
                            break;
                        }
                    }
                }
            }
            Entity::Trainer(x) => {
                for pokemon in x.data.pokemon {
                    if let PokemonEdge::Out(y) = pokemon {
                        self.entities[y].data.me.trainer = None;
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Map generation

fn mapgen(board: &mut Board, rng: &mut RNG) {
    let ft = Tile::get('.');
    let wt = Tile::get('#');
    let gt = Tile::get('"');
    let fl = Tile::get('%');
    let wa = Tile::get('~');

    board.fill_map(ft);
    let size = board.get_size();

    let automata = |rng: &mut RNG| -> Matrix<bool> {
        let mut d100 = || rng.gen::<u32>() % 100;
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

    let walls = automata(rng);
    let grass = automata(rng);
    for y in 0..size.1 {
        for x in 0..size.0 {
            let point = Point(x, y);
            if walls.get(point) {
                board.set_tile_at(point, wt);
            } else if grass.get(point) {
                board.set_tile_at(point, gt);
            }
        }
    }

    let die = |n: i32, rng: &mut RNG| rng.gen::<i32>().rem_euclid(n);
    let mut river = vec![Point::default()];
    for i in 1..size.1 {
        let last = river.iter().last().unwrap().0;
        let next = last + die(3, rng) - 1;
        river.push(Point(next, i));
    }
    let target = river[0] + *river.iter().last().unwrap();
    let offset = Point((size - target).0 / 2, 0);
    for x in &river { board.set_tile_at(*x + offset, wa); }

    let pos = |board: &Board, rng: &mut RNG| {
        for _ in 0..100 {
            let p = Point(die(size.0, rng), die(size.1, rng));
            if let Status::Free = board.get_status(p) { return Some(p); }
        }
        None
    };
    for _ in 0..5 { if let Some(pos) = pos(board, rng) { board.set_tile_at(pos, fl); } }
}

//////////////////////////////////////////////////////////////////////////////

// Targeting UI

fn can_target(entity: &EntityKnowledge) -> bool {
    entity.age == 0 && !entity.friend
}

fn init_target(data: TargetData, source: Point, target: Point) -> Box<Target> {
    Box::new(Target { data, error: "".into(), frame: 0, path: vec![], source, target })
}

fn init_summon_target(player: &Trainer, data: TargetData) -> Box<Target> {
    let (known, pos, dir) = (&*player.known, player.pos, player.dir);
    let mut target = init_target(data, pos, pos);

    if let Some(x) = defend_at_pos(pos, player) {
        let line = LOS(pos, x);
        for p in line.iter().skip(1).rev() {
            update_target(known, &mut target, *p);
            if target.error.is_empty() { return target; }
        }
    }

    let mut okay = |p: Point| {
        if !check_follower_square(known, player, p, false) { return false; }
        update_target(known, &mut target, p);
        target.error.is_empty()
    };

    let best = pos + dir.scale(2);
    let next = pos + dir.scale(1);
    if okay(best) { return target; }
    if okay(next) { return target; }

    let mut options: Vec<Point> = vec![];
    for dx in -2..=2 {
        for dy in -2..=2 {
            let p = pos + Point(dx, dy);
            if okay(p) { options.push(p); }
        }
    }

    let update = (|| {
        if options.is_empty() { return pos; }
        *options.select_nth_unstable_by_key(0, |x| (*x - best).len_l2_squared()).1
    })();
    update_target(known, &mut target, update);
    target
}

fn outside_map(player: &Trainer, point: Point) -> bool {
    let delta = point - player.pos;
    let limit_x = (UI_MAP_SIZE_X - 1) / 2;
    let limit_y = (UI_MAP_SIZE_Y - 1) / 2;
    delta.0.abs() > limit_x || delta.1.abs() > limit_y
}

fn update_target(known: &Knowledge, target: &mut Target, update: Point) {
    let mut okay_until = 0;
    let los = LOS(target.source, update);

    target.error = "".into();
    target.frame = 0;
    target.path = los.into_iter().skip(1).map(|x| (x, true)).collect();
    target.target = update;

    match &target.data {
        TargetData::FarLook => {
            for (i, x) in target.path.iter().enumerate() {
                if known.get(x.0).visible() { okay_until = i + 1; }
            }
            if okay_until < target.path.len() {
                target.error = "You can't see a clear path there.".into();
            }
        }
        TargetData::Summon { range, .. } => {
            if target.path.is_empty() {
                target.error = "There's something in the way.".into();
            }
            for (i, x) in target.path.iter().enumerate() {
                let cell = known.get(x.0);
                if cell.status().unwrap_or(Status::Free) != Status::Free {
                    target.error = "There's something in the way.".into();
                } else if !(x.0 - target.source).in_l2_range(*range) {
                    target.error = "You can't throw that far.".into();
                } else if !cell.visible() {
                    target.error = "You can't see a clear path there.".into();
                }
                if !target.error.is_empty() { break; }
                okay_until = i + 1;
            }
        }
    }

    for i in okay_until..target.path.len() {
        target.path[i].1 = false;
    }
}

fn select_valid_target(state: &mut State) -> Option<EID> {
    let known = &*state.board.entities[state.player].known;
    let target = state.target.as_ref()?;
    let entity = known.get(target.target).entity();

    match &target.data {
        TargetData::FarLook => {
            let entity = entity?;
            if can_target(entity) { Some(entity.eid) } else { None }
        }
        TargetData::Summon { index, .. } => {
            state.input = Action::Summon(*index, target.target);
            known.focus
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Game logic

pub enum Command {
    Return,
}

pub struct MoveData { dir: Point, look: Point, turns: f64 }

pub enum Action {
    Attack(Point),
    Idle,
    Look(Point),
    Move(MoveData),
    Shout(PID, Command),
    Summon(usize, Point),
    WaitForInput,
    Withdraw(PID),
}

struct ActionResult {
    success: bool,
    moves: f64,
    turns: f64,
}

impl ActionResult {
    fn failure() -> Self { Self { success: false, moves: 0., turns: 1. } }
    fn success() -> Self { Self::success_turns(1.) }
    fn success_moves(moves: f64) -> Self { Self { success: true,  moves, turns: 1. } }
    fn success_turns(turns: f64) -> Self { Self { success: true,  moves: 0., turns } }
}

fn charge(entity: &mut Entity) {
    let charge = (TURN_TIMER as f64 * entity.speed).round() as i32;
    if entity.move_timer > 0 { entity.move_timer -= charge; }
    if entity.turn_timer > 0 { entity.turn_timer -= charge; }
}

fn name(me: &PokemonIndividualData) -> &'static str {
    me.species.name
}

fn shout(board: &mut Board, tid: TID, text: &str) {
    let entity = &board.entities[tid];
    if entity.player {
        board.log_menu(format!("You shout: \"{}\"", text), 0x231);
        board.end_menu_logging();
    } else {
        board.log_menu(format!("{} shouts: \"{}\"", entity.data.name, text), 0x234);
    }
}

fn move_ready(entity: &Entity) -> bool { entity.move_timer <= 0 }

fn turn_ready(entity: &Entity) -> bool { entity.turn_timer <= 0 }

fn wait(entity: &mut Entity, result: &ActionResult) {
    entity.move_timer += (MOVE_TIMER as f64 * result.moves).round() as i32;
    entity.turn_timer += (TURN_TIMER as f64 * result.turns).round() as i32;
}

fn rivals<'a>(trainer: &'a Trainer) -> Vec<(&'a EntityKnowledge, &'a PokemonView)> {
    let mut rivals = vec![];
    for entity in &trainer.known.entities {
        if entity.age > 0 { continue; }
        if let EntityView::Pokemon(x) = &entity.view && !x.trainer {
            rivals.push((entity, x));
        }
    }
    let pos = trainer.pos;
    rivals.sort_by_cached_key(
        |(x, _)| ((x.pos - pos).len_l2_squared(), x.pos.0, x.pos.1));
    rivals
}

//////////////////////////////////////////////////////////////////////////////

// AI state definitions:

//#[derive(Debug)]
//pub struct Pause {
//    time: i32,
//}
//
//#[derive(Debug)]
//pub struct Fight {
//    age: i32,
//    target: Point,
//}
//
//#[derive(Debug)]
//pub struct Flight {
//    age: i32,
//    switch: i32,
//    target: Vec<Point>,
//}
//
//#[derive(Debug)]
//pub struct Wander {
//    time: i32,
//}
//
//#[derive(Debug)]
//pub struct Assess {
//    switch: i32,
//    target: Point,
//    time: i32,
//}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Goal { Assess, Chase, Drink, Eat, Explore, Flee }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StepKind { Calm, Drink, Eat, Look, Move }

#[derive(Clone, Copy, Debug)]
struct Step { kind: StepKind, target: Point }

type Hint = (Goal, &'static Tile);

#[derive(Debug)]
pub struct FightState {
    age: i32,
    bias: Point,
    target: Point,
    searching: bool,
}

#[derive(Debug, Default)]
pub struct FlightState {
    distances: HashMap<Point, i32>,
    threats: Vec<Point>,
    fleeing: bool,
}

#[derive(Debug)]
pub struct AIState {
    goal: Goal,
    plan: Vec<Step>,
    time: Timestamp,
    hints: HashMap<Goal, Point>,
    fight: Option<FightState>,
    flight: FlightState,
    till_assess: i32,
    till_hunger: i32,
    till_thirst: i32,
    turn_times: VecDeque<Timestamp>,
}

impl AIState {
    fn new(rng: &mut RNG) -> Self {
        Self {
            goal: Goal::Explore,
            plan: vec![],
            time: Timestamp::default(),
            hints: HashMap::default(),
            fight: None,
            flight: FlightState::default(),
            till_assess: rng.gen::<i32>().rem_euclid(MAX_ASSESS),
            till_hunger: rng.gen::<i32>().rem_euclid(MAX_HUNGER),
            till_thirst: rng.gen::<i32>().rem_euclid(MAX_THIRST),
            turn_times: VecDeque::with_capacity(TURN_TIMES_LIMIT),
        }
    }

    fn age_at_turn(&self, turn: i32) -> i32 {
        if self.turn_times.is_empty() { return 0; }
        self.time - self.turn_times[min(self.turn_times.len() - 1, turn as usize)]
    }

    fn record_turn(&mut self, time: Timestamp) {
        if self.turn_times.len() == TURN_TIMES_LIMIT { self.turn_times.pop_back(); }
        self.turn_times.push_front(self.time);
        self.time = time;
    }
}

//////////////////////////////////////////////////////////////////////////////

// AI routines

fn step(dir: Point, turns: f64) -> Action {
    Action::Move(MoveData { dir, look: dir, turns })
}

fn wander(dir: Point) -> Action {
    step(dir, WANDER_TURNS)
}

fn explore(entity: &Entity) -> Option<BFSResult> {
    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| {
        if p == pos { return Status::Free; }
        entity.known.get(p).status().unwrap_or(Status::Free)
    };
    let done1 = |p: Point| {
        if known.get(p).tile().is_some() { return false };
        dirs::ALL.iter().any(|x| known.get(p + *x).unblocked())
    };
    let done0 = |p: Point| {
        done1(p) && dirs::ALL.iter().all(|x| !known.get(p + *x).blocked())
    };

    BFS(pos, done0, BFS_LIMIT_WANDER, check).or_else(||
    BFS(pos, done1, BFS_LIMIT_WANDER, check))
}

fn search_around(entity: &Entity, source: Point, age: i32, bias: Point) -> Option<BFSResult> {
    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| {
        if p == pos { return Status::Free; }
        entity.known.get(p).status().unwrap_or(Status::Free)
    };
    let done1 = |p: Point| {
        if known.get(p).age() < age { return false };
        dirs::ALL.iter().any(|x| known.get(p + *x).unblocked())
    };
    let done0 = |p: Point| {
        done1(p) && dirs::ALL.iter().all(|x| !known.get(p + *x).blocked())
    };
    let heuristic = |p: Point| { 8 * (pos + bias - p).len_nethack() };

    let path = Dijkstra(source, done0, ASTAR_LIMIT_WANDER, check, heuristic).or_else(||
               Dijkstra(source, done1, ASTAR_LIMIT_WANDER, check, heuristic))?;
    if path.is_empty() {
        Some(BFSResult { dirs: vec![Point::default()], targets: vec![source] })
    } else {
        Some(BFSResult { dirs: vec![path[0] - source], targets: vec![path[path.len() - 1]] })
    }
}

fn attack_target(entity: &Entity, target: Point, rng: &mut RNG) -> Action {
    let (known, source) = (&*entity.known, entity.pos);
    if source == target { return Action::Idle; }

    let range = ATTACK_RANGE;
    let valid = |x| has_line_of_sight(x, target, known, range);
    if !valid(source) {
        return path_to_target(entity, target, known, range, valid, rng);
    } else if !move_ready(entity) {
        return Action::Look(target - source);
    }
    //if 1 == 1 { return Action::Look(target - source); }
    Action::Attack(target)
}

fn assess_direction(dir: Point, ai: &mut AIState, rng: &mut RNG) {
    let Point(dx, dy) = dir;

    for _ in 0..ASSESS_STEPS {
        let scale = 1000;
        let steps = rng.gen::<i32>().rem_euclid(ASSESS_TURNS);
        let angle = Normal::new(0., ASSESS_STDEV).unwrap().sample(rng);
        let (sin, cos) = (angle.sin(), angle.cos());
        let rx = (cos * (scale * dx) as f64) + (sin * (scale * dy) as f64);
        let ry = (cos * (scale * dy) as f64) - (sin * (scale * dx) as f64);
        let target = Point(rx as i32, ry as i32);
        for _ in 0..steps {
            ai.plan.push(Step { kind: StepKind::Look, target })
        }
    }
}

fn assess_threats(source: Point, ai: &mut AIState) {
    if !ai.flight.fleeing { return; }
    if ai.flight.threats.is_empty() { return; }

    let target = ai.flight.threats[0];
    let Point(dx, dy) = target - source;

    for i in 0..ASSESS_TIME {
        let a = 1 * ASSESS_TIME / 4;
        let b = 2 * ASSESS_TIME / 4;
        let c = 3 * ASSESS_TIME / 4;
        let depth = if i < a {
            -i
        } else if i < c {
            i - 2 * a
        } else {
            -i + 2 * c - 2 * a
        };
        let scale = 1000;
        let angle = ASSESS_ANGLE * depth as f64 / b as f64;
        let (sin, cos) = (angle.sin(), angle.cos());
        let rx = (cos * (scale * dx) as f64) + (sin * (scale * dy) as f64);
        let ry = (cos * (scale * dy) as f64) - (sin * (scale * dx) as f64);
        let target = Point(rx as i32, ry as i32);
        ai.plan.push(Step { kind: StepKind::Look, target });
    }
    ai.plan.push(Step { kind: StepKind::Calm, target: source });
}

fn flee_from_threats(entity: &Entity, ai: &mut AIState) -> Option<BFSResult> {
    if !ai.flight.fleeing { return None; }
    if ai.flight.threats.is_empty() { return None; }

    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| {
        if p == pos { return Status::Free; }
        if known.get(p).unblocked() { Status::Free } else { Status::Blocked }
    };

    let mut map = HashMap::default();
    map.insert(pos, 0);
    DijkstraMap(check, FLIGHT_MAP_LIMIT, &mut map);

    let score = |p: Point, v: i32| {
        let mut threat = Point::default();
        let mut threat_distance = std::i32::MAX;
        for x in &ai.flight.threats {
            let y = (*x - p).len_nethack();
            if y < threat_distance { (threat, threat_distance) = (*x, y); }
        }
        let blocked = {
            let los = LOS(p, threat);
            let len = max(los.len(), 2) - 2;
            los.iter().skip(1).take(len).any(|x| known.get(*x).blocked())
        };
        let frontier = dirs::ALL.iter().any(|x| !known.get(*x).tile().is_some());

        let source_distance = 0.0625 * v as f64;
        let threat_distance = threat_distance as f64;
        1.2 * threat_distance +
        -1.0 * source_distance +
        if blocked { 16.0 } else { 0.0 } +
        if frontier { 64.0 } else { 0.0 }
    };

    let mut best_point = pos;
    let mut best_score = std::f64::MIN;
    let mut best_nearby_point = pos;
    let mut best_nearby_score = std::f64::MIN;
    for (k, v) in &map {
        let s = score(*k, *v);
        if s > best_score {
            (best_point, best_score) = (*k, s);
        }
        if s > best_nearby_score && (*k - pos).len_l1() == 1 {
            (best_nearby_point, best_nearby_score) = (*k, s);
        }
    }
    Some(BFSResult { dirs: vec![best_nearby_point - pos], targets: vec![best_point] })
}

fn update_ai_state(entity: &Entity, hints: &[Hint], ai: &mut AIState) {
    ai.till_assess = max(0, ai.till_assess - 1);
    ai.till_hunger = max(0, ai.till_hunger - 1);
    ai.till_thirst = max(0, ai.till_thirst - 1);

    let (known, pos) = (&*entity.known, entity.pos);
    let last_turn_age = known.time - ai.time;
    let mut seen = HashSet::default();
    for cell in &known.cells {
        if (ai.time - cell.time) >= 0 { break; }
        for (goal, tile) in hints {
            if cell.tile == tile && seen.insert(goal) {
                ai.hints.insert(*goal, cell.point);
            }
        }
    }
    ai.record_turn(known.time);

    // TODO(shaunak): For now, only wild Pokemon have predator/prey relations
    // so we can skip this whole threat() branch based on this check. However,
    // the rest of the logic goes through if we have a more complex way to
    // compute these threats.
    //
    // "rival" means that we have a hostile relationship with that entity.
    // We'll end up with three states - Friendly, Neutral, or Rival - or more.
    // An entity is a "threat" if its a rival and our combat analysis against
    // it shows that we'd probably lose. These predicates can be generalized
    // to all entities.
    //
    // The threshold for a rival being a threat may also depend on some other
    // parameter like "aggressiveness". A maximally-aggressive entity will
    // stand and fight even in hopeless situations.
    let prey = match entity {
        Entity::Pokemon(x) => x.data.me.species.name == "Pidgey",
        Entity::Trainer(_) => true,
    };

    // We're a predator, and we should chase and attack rivals.
    if !prey {
        let fight = std::mem::take(&mut ai.fight);
        let limit = ai.age_at_turn(MAX_FOLLOW_TURNS);
        let mut targets = known.entities.iter().filter(
            |x| x.rival).collect::<Vec<_>>();
        if !targets.is_empty() {
            targets.sort_unstable_by_key(|x| x.age);
            let EntityKnowledge { age, pos: target, .. } = *targets[0];
            let restart = age < last_turn_age;
            let (mut bias, mut searching) = (target - pos, false);
            if !restart && let Some(x) = fight { (bias, searching) = (x.bias, x.searching) };
            if age < limit { ai.fight = Some(FightState { age, bias, searching, target }); }
            if restart { ai.plan.clear(); }
        }
        return;
    }

    // We're prey, and we should treat rivals as threats.
    let limit = ai.age_at_turn(MAX_FLIGHT_TURNS);
    let mut threats: Vec<_> = known.entities.iter().filter_map(
        |x| if x.age < limit && x.rival { Some(x.pos) } else { None }).collect();
    threats.sort_unstable_by_key(|x| (x.0, x.1));
    if threats == ai.flight.threats { return; }

    ai.plan.clear();
    let fleeing = !threats.is_empty();
    ai.flight = FlightState { distances: HashMap::default(), threats, fleeing };
    if !fleeing { return; }

    let check = |p: Point| {
        if p == pos { return Status::Free; }
        if known.get(p).unblocked() { Status::Free } else { Status::Blocked }
    };
    for x in &ai.flight.threats { ai.flight.distances.insert(*x, 0); }
    DijkstraMap(check, FLIGHT_MAP_LIMIT, &mut ai.flight.distances);
}

fn plan_from_cached(entity: &Entity, hints: &[Hint],
                    ai: &mut AIState, rng: &mut RNG) -> Option<Action> {
    if ai.plan.is_empty() { return None; }

    // Check whether we can execute the immediate next step in the plan.
    let (known, pos) = (&*entity.known, entity.pos);
    let next = *ai.plan.iter().last().unwrap();
    let dir = next.target - pos;
    if next.kind != StepKind::Look && dir.len_l1() > 1 { return None; }

    // Check whether the plan's goal is still a top priority for us.
    let mut goals: Vec<Goal> = vec![];
    if ai.flight.fleeing {
        goals.push(Goal::Flee);
    } else if ai.fight.is_some() {
        goals.push(Goal::Chase);
    } else if ai.goal == Goal::Assess {
        goals.push(Goal::Assess);
    } else {
        if ai.till_hunger == 0 && ai.hints.contains_key(&Goal::Eat) {
            goals.push(Goal::Eat);
        }
        if ai.till_thirst == 0 && ai.hints.contains_key(&Goal::Drink) {
            goals.push(Goal::Drink);
        }
    }
    if goals.is_empty() { goals.push(Goal::Explore); }
    if !goals.contains(&ai.goal) { return None; }

    // Check if we got specific information that invalidates the plan.
    let point_matches_goal = |goal: Goal, point: Point| {
        let tile = hints.iter().find_map(
            |x| if x.0 == goal { Some(x.1) } else { None });
        if tile.is_none() { return false; }
        known.get(point).tile() == tile
    };
    let step_valid = |Step { kind, target }| match kind {
        StepKind::Calm => true,
        StepKind::Drink => point_matches_goal(Goal::Drink, target),
        StepKind::Eat => point_matches_goal(Goal::Eat, target),
        StepKind::Look => true,
        StepKind::Move => match known.get(target).status().unwrap_or(Status::Free) {
            Status::Occupied => target != next.target,
            Status::Blocked  => false,
            Status::Free     => true,
        }
    };
    if !ai.plan.iter().all(|x| step_valid(*x)) { return None; }

    // The plan is good! Execute the next step.
    ai.plan.pop();
    let wait = Some(wander(Point::default()));
    match next.kind {
        StepKind::Calm => { ai.flight.fleeing = false; None }
        StepKind::Drink => { ai.till_thirst = MAX_THIRST; wait }
        StepKind::Eat => { ai.till_hunger = MAX_HUNGER; wait }
        StepKind::Look => {
            if ai.plan.is_empty() && ai.goal == Goal::Assess {
                ai.till_assess = rng.gen::<i32>().rem_euclid(MAX_ASSESS);
            }
            Some(Action::Look(next.target))
        }
        StepKind::Move => {
            let mut target = next.target;
            for next in ai.plan.iter().rev().take(8) {
                if next.kind == StepKind::Look { break; }
                for point in LOS(pos, next.target) {
                    if known.get(point).blocked() { break; }
                }
                target = next.target;
            }
            let look = if target == pos { entity.dir } else { target - pos };
            let quick = ai.goal == Goal::Flee || ai.goal == Goal::Chase;
            let turns = WANDER_TURNS * if quick { 0.5 } else { 1.0 };
            Some(Action::Move(MoveData { dir, look, turns }))
        }
    }
}

fn plan_from_state(pokemon: &Pokemon, ai: &mut AIState, rng: &mut RNG) -> Action {
    pokemon.data.debug.take();
    let hints = [
        (Goal::Drink, Tile::get('~')),
        (Goal::Eat, Tile::get('%')),
    ];
    update_ai_state(pokemon, &hints, ai);
    if let Some(x) = plan_from_cached(pokemon, &hints, ai, rng) { return x; }

    ai.plan.clear();
    ai.goal = Goal::Explore;
    pokemon.data.target.take();
    pokemon.data.debug.set("Recomputing plan!".into());
    let (mut dirs, mut targets) = (vec![], vec![]);
    let (known, pos) = (&pokemon.known, pokemon.pos);

    let fallback = {
        let check = |p: Point| {
            if p == pos { return Status::Free; }
            known.get(p).status().unwrap_or(Status::Free)
        };

        if let Some(x) = flee_from_threats(pokemon, ai) {
            (dirs, targets) = (x.dirs, x.targets);
            ai.goal = Goal::Flee;
        } else if let Some(x) = &mut ai.fight {
            let source = if x.searching { pokemon.pos } else { x.target };
            if x.age == 0 {
                (ai.goal, x.searching) = (Goal::Chase, false);
                return attack_target(pokemon, x.target, rng);
            } else if let Some(y) = search_around(pokemon, source, x.age, x.bias) {
                (ai.goal, x.searching) = (Goal::Chase, true);
                (dirs, targets) = (y.dirs, y.targets);
            }
        }

        let mut add_candidates = |ai: &mut AIState, goal: Goal| {
            if ai.goal != Goal::Explore { return; }

            let tile = hints.iter().find_map(
                |x| if x.0 == goal { Some(x.1) } else { None });
            if tile.is_none() { return; }

            let target = |p: Point| known.get(p).tile() == tile;
            if target(pos) {
                (dirs, targets) = (vec![Point::default()], vec![pos]);
                ai.goal = goal;
            } else if let Some(x) = BFS(pos, target, BFS_LIMIT_WANDER, check) {
                (dirs, targets) = (x.dirs, x.targets);
                ai.goal = goal;
            }
        };
        if ai.till_thirst == 0 { add_candidates(ai, Goal::Drink); }
        if ai.till_hunger == 0 { add_candidates(ai, Goal::Eat); }

        if dirs.is_empty() && ai.till_assess == 0 {
            (dirs, targets) = (vec![Point::default()], vec![pos]);
            ai.goal = Goal::Assess;
        } else if dirs.is_empty() && let Some(x) = explore(pokemon) {
            (dirs, targets) = (x.dirs, x.targets);
        } else if dirs.is_empty() {
            return wander(*sample(&dirs::ALL, rng));
        }
        wander(*sample(&dirs, rng))
    };

    if targets.is_empty() { return fallback; }

    let mut target = *targets.select_nth_unstable_by_key(
        0, |x| (*x - pos).len_l2_squared()).1;
    if ai.goal == Goal::Explore {
        for _ in 0..64 {
            let candidate = target + *sample(&dirs::ALL, rng);
            if known.get(candidate).tile().is_none() { target = candidate; }
        }
    }
    pokemon.data.target.set(targets);

    let check = |p: Point| {
        if p == pos { return Status::Free; }
        known.get(p).status().unwrap_or(Status::Free)
    };
    if let Some(path) = AStar(pos, target, ASTAR_LIMIT_WANDER, check) {
        let kind = StepKind::Move;
        ai.plan = path.iter().map(|x| Step { kind, target: *x }).collect();
        match ai.goal {
            Goal::Assess => assess_direction(pokemon.dir, ai, rng),
            Goal::Chase => {}
            Goal::Drink => ai.plan.push(Step { kind: StepKind::Drink, target }),
            Goal::Eat => ai.plan.push(Step { kind: StepKind::Eat, target }),
            Goal::Explore => {}
            Goal::Flee => assess_threats(target, ai),
        }
        ai.plan.reverse();
        if let Some(x) = plan_from_cached(pokemon, &hints, ai, rng) { return x; }
        ai.plan.clear();
    }
    fallback
}

fn plan_wild_pokemon(pokemon: &Pokemon, rng: &mut RNG) -> Action {
    let mut ai = pokemon.data.ai.take().unwrap_or(Box::new(AIState::new(rng)));
    let result = plan_from_state(pokemon, &mut ai, rng);
    pokemon.data.ai.set(Some(ai));
    result
}

fn check_follower_square(known: &Knowledge, trainer: &Trainer,
                         target: Point, ignore_occupant: bool) -> bool {
    let free = match known.get(target).status().unwrap_or(Status::Blocked) {
        Status::Free => true,
        Status::Blocked => false,
        Status::Occupied => ignore_occupant,
    };
    if !free { return false }

    let source = trainer.pos;
    let length = (source - target).len_nethack();
    if length > 2 { return false; }
    if length < 2 { return true; }

    known.get(source).visibility() == known.get(target).visibility()
}

fn defend_at_pos(source: Point, trainer: &Trainer) -> Option<Point> {
    let rivals = rivals(trainer);
    if rivals.is_empty() { return None; }

    // TODO(shaunak): Combine our knowledge and our leader's here.
    let known = &*trainer.known;
    let defended = |p: Point| {
        if p == source || p == trainer.pos { return false; }
        let cell = known.get(p);
        if cell.entity().map(|x| x.friend).unwrap_or(false) { return true; }
        !cell.unblocked()
    };

    let mut scores = HashMap::default();
    for (rival, _) in &rivals {
        let mut marked = HashSet::default();
        let los = LOS(rival.pos, trainer.pos);

        let diff = rival.pos - trainer.pos;
        let shift_a = if diff.0.abs() > diff.1.abs() {
            Point(0, if diff.1 == 0 { 1 } else { diff.1.signum() })
        } else {
            Point(if diff.0 == 0 { 1 } else { diff.0.signum() }, 0)
        };
        let shift_b = Point::default() - shift_a;
        let shifts: [(Point, f64); 3] =
            [(Point::default(), 64.), (shift_a, 8.), (shift_b, 1.)];

        for (shift, score) in &shifts {
            if los.iter().any(|x| defended(*x + *shift)) { continue; }
            for point in &los {
                let delta = *point + *shift - trainer.pos;
                if delta.0.abs() > 2 || delta.1.abs() > 2 { continue; }
                if !marked.insert(delta) { continue; }
                *scores.entry(delta).or_insert(0.) += *score;
            }
        }
    }

    let (mut best_score, mut best_point) = (f64::NEG_INFINITY, None);
    for x in -2..=2 {
        for y in -2..=2 {
            if x == 0 && y == 0 { continue; }
            let (d, p) = (Point(x, y), Point(x, y) + trainer.pos);
            if !check_follower_square(known, trainer, p, p == source) { continue; }

            let mut score = scores.get(&d).cloned().unwrap_or(f64::NEG_INFINITY);
            if score == f64::NEG_INFINITY { continue; }
            score += 0.0625 * d.len_l2_squared() as f64;
            score -= 0.015625 * (p - source).len_l2_squared() as f64;
            if score > best_score { (best_score, best_point) = (score, Some(p)); }
        }
    }
    best_point
}

fn has_line_of_sight(source: Point, target: Point, known: &Knowledge, range: i32) -> bool {
    if (source - target).len_nethack() > range { return false; }
    if !known.get(target).visible() { return false; }
    let los = LOS(source, target);
    let last = los.len() - 1;
    los.iter().enumerate().all(|(i, p)| {
        if i == 0 || i == last { return true; }
        known.get(*p).status() == Some(Status::Free)
    })
}

fn path_to_target<F: Fn(Point) -> bool>(
        entity: &Entity, target: Point, known: &Knowledge,
        range: i32, valid: F, rng: &mut RNG) -> Action {
    let check = |p: Point| {
        if p == entity.pos { return Status::Free; }
        known.get(p).status().unwrap_or(Status::Free)
    };
    let source = entity.pos;
    let result = BFS(source, &valid, BFS_LIMIT_ATTACK, check);
    let mut dirs = result.map(|x| x.dirs).unwrap_or_default();
    if valid(source) { dirs.push(Point::default()); }

    let step = |dir: Point| {
        let look = target - source - dir;
        Action::Move(MoveData { dir, look, turns: 1. })
    };

    if !dirs.is_empty() {
        let scores: Vec<_> = dirs.iter().map(
            |x| ((*x + source - target).len_nethack() - range).abs()).collect();
        let best = *scores.iter().reduce(|acc, x| min(acc, x)).unwrap();
        let opts: Vec<_> = dirs.iter().enumerate().filter(|(i, _)| scores[*i] == best).collect();
        return step(*sample(&opts, rng).1);
    }

    let path = AStar(source, target, ASTAR_LIMIT_ATTACK, check);
    let dir = path.and_then(|x| if x.is_empty() { None } else { Some(x[0] - source) });
    step(dir.unwrap_or_else(|| *sample(&dirs::ALL, rng)))
}

fn defend_leader(pokemon: &Pokemon, trainer: &Trainer) -> Option<Action> {
    // TODO(shaunak): Combine our knowledge and our leader's here.
    let known = &*trainer.known;
    let check = |p: Point| known.get(p).status().unwrap_or(Status::Occupied);
    let goal = defend_at_pos(pokemon.pos, trainer)?;
    let path = AStar(pokemon.pos, goal, ASTAR_LIMIT_ATTACK, check)?;
    if path.is_empty() { return Some(Action::Idle); }
    Some(step(path[0] - pokemon.pos, 1.))
}

fn follow_command(pokemon: &Pokemon, trainer: &Trainer,
                  command: &Command, rng: &mut RNG) -> Action {
    // TODO(shaunak): Combine our knowledge and our leader's here.
    let known = &*trainer.known;
    match command {
        Command::Return => {
            let range = SUMMON_RANGE;
            let valid = |x| has_line_of_sight(trainer.pos, x, known, range);
            path_to_target(pokemon, trainer.pos, known, range, valid, rng)
        }
    }
}

fn follow_leader(pokemon: &Pokemon, trainer: &Trainer, rng: &mut RNG) -> Action {
    // TODO(shaunak): Combine our knowledge and our leader's here.
    let known = &*trainer.known;
    let (pp, tp) = (pokemon.pos, trainer.pos);
    let okay = |p: Point| check_follower_square(known, trainer, p, p == pp);

    if (pp - tp).len_nethack() <= 3 {
        let mut moves: Vec<_> = dirs::ALL.iter().filter_map(
            |x| if okay(pp + *x) { Some((1, *x)) } else { None }).collect();
        if okay(pp) { moves.push((16, Point::default())); }
        if !moves.is_empty() { return step(*weighted(&moves, rng), 1.); }
    }

    let check = |p: Point| known.get(p).status().unwrap_or(Status::Occupied);
    let path = AStar(pp, tp, ASTAR_LIMIT_ATTACK, check).unwrap_or_default();
    let dir = if !path.is_empty() { path[0] - pp } else { *sample(&dirs::ALL, rng) };
    step(dir, 1.)
}

fn plan_pokemon(board: &Board, pokemon: &Pokemon, rng: &mut RNG) -> Action {
    let trainer = match pokemon.data.me.trainer {
        Some(tid) => &board.entities[tid],
        None => return plan_wild_pokemon(pokemon, rng),
    };

    let commands = pokemon.data.commands.take();
    while !commands.is_empty() {
        let action = follow_command(pokemon, trainer, &commands[0], rng);
        pokemon.data.commands.set(commands);
        return action;
    }
    if 1 == 1 { return plan_wild_pokemon(pokemon, rng); }

    //let ready = move_ready(pokemon);
    //if !ready && let Some(x) = defend_leader(pokemon, trainer) { return x;  }
    // (use attacks here!)
    //if ready && let Some(x) = defend_leader(pokemon, trainer) { return x;  }
    if let Some(x) = defend_leader(pokemon, trainer) { return x; }
    follow_leader(pokemon, trainer, rng)
}

fn plan(board: &Board, eid: EID, input: &mut Action, rng: &mut RNG) -> Action {
    let entity = &board.entities[eid];
    if entity.player {
        for pid in &cast!(entity, Entity::Trainer).data.summons {
            let summon = &board.entities[*pid];
            let mut commands = summon.data.commands.take();
            let mut result = None;
            if let Some(Command::Return) = commands.iter().next() &&
               has_line_of_sight(entity.pos, summon.pos, &*entity.known, SUMMON_RANGE) {
                result = Some(Action::Withdraw(summon.id()));
                commands.remove(0);
            }
            summon.data.commands.set(commands);
            if let Some(x) = result { return x; }
        }
        return std::mem::replace(input, Action::WaitForInput);
    }
    match entity {
        Entity::Pokemon(x) => plan_pokemon(board, x, rng),
        Entity::Trainer(_) => Action::Idle,
    }
}

//////////////////////////////////////////////////////////////////////////////

// Apply an Action to our state

type CB = Box<dyn Fn(&mut Board, &mut RNG)>;

fn apply_damage(board: &Board, target: Point, callback: CB) -> Effect {
    let eid = match board.get_entity_at(target) {
        None => return Effect::default(),
        Some(x) => x,
    };

    let glyph = board.entities[eid].glyph;
    let flash = glyph.with_fg(Color::black()).with_bg(0x400);
    let particle = effect::Particle { glyph: flash, point: target };
    let mut effect = Effect::serial(vec![
        Effect::constant(particle, UI_DAMAGE_FLASH),
        Effect::pause(UI_DAMAGE_TICKS),
    ]);
    let frame = effect.frames.len() as i32;
    effect.add_event(Event::Callback { frame, callback });
    effect
}

fn apply_effect(mut effect: Effect, what: FT, callback: CB) -> Effect {
    let frame = effect.events.iter().find_map(
        |x| if x.what() == Some(what) { Some(x.frame()) } else { None });
    if let Some(frame) = frame {
        effect.add_event(Event::Callback { frame, callback });
    }
    effect
}

fn apply_summon(source: Point, target: Point, callback: CB) -> Effect {
    let effect = effect::SummonEffect(source, target);
    apply_effect(effect, FT::Summon, callback)
}

fn apply_withdraw(source: Point, target: Point, callback: CB) -> Effect {
    let effect = effect::WithdrawEffect(source, target);
    apply_effect(effect, FT::Withdraw, callback)
}

fn act(state: &mut State, eid: EID, action: Action) -> ActionResult {
    match action {
        Action::Idle => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::Look(dir) => {
            state.board.entities[eid].dir = dir;
            ActionResult::success()
        }
        Action::Move(MoveData { dir, look, turns }) => {
            if dir == Point::default() {
                return ActionResult::success_turns(turns);
            }
            if look != Point::default() {
                state.board.entities[eid].dir = look;
            }
            let entity = &state.board.entities[eid];
            let (source, target) = (entity.pos, entity.pos + dir);
            match state.board.get_status(target) {
                Status::Blocked => {
                    state.board.entities[eid].dir = dir;
                    ActionResult::failure()
                }
                Status::Occupied => {
                    let other = state.board.get_entity_at(target);
                    let other = other.map(|x| &state.board.entities[x]);
                    if let Some(Entity::Pokemon(x)) = other &&
                       x.data.me.trainer.map(|x| x.eid()) == Some(eid) {
                        if entity.player {
                            state.board.log(format!(
                                "You swap places with {}.", name(&x.data.me)));
                        }
                        state.board.swap_entities(source, target);
                        return ActionResult::success_turns(turns);
                    }
                    state.board.entities[eid].dir = dir;
                    ActionResult::failure()
                }
                Status::Free => {
                    state.board.move_entity(eid, target);
                    ActionResult::success_turns(turns)
                }
            }
        }
        Action::Shout(pid, command) => {
            let trainer = match &state.board.entities[eid] {
                Entity::Trainer(x) => x,
                _ => return ActionResult::failure(),
            };
            let pokemon = match state.board.entities.get(pid) {
                Some(x) if x.data.me.trainer.map(|x| x.eid()) == Some(eid) => x,
                _ => return ActionResult::failure(),
            };
            let (tid, name) = (trainer.id(), name(&pokemon.data.me));
            let (source, target) = (trainer.pos, pokemon.pos);
            if matches!(command, Command::Return) &&
               has_line_of_sight(source, target, &trainer.known, SUMMON_RANGE) {
                shout(&mut state.board, tid, &format!("{}, return!", name));
                let cb = move |board: &mut Board, _: &mut RNG| board.remove_entity(pid.eid());
                state.add_effect(apply_withdraw(source, target, Box::new(cb)));
                return ActionResult::success();
            }
            shout(&mut state.board, tid, &format!("{}, return!", name));
            let pokemon = &mut state.board.entities[pid];
            pokemon.data.commands.get_mut().push(command);
            ActionResult::success()
        }
        Action::Attack(target) => {
            let pokemon = match &state.board.entities[eid] {
                Entity::Pokemon(x) => x,
                _ => return ActionResult::failure(),
            };
            let (known, source) = (&pokemon.known, pokemon.pos);
            if !has_line_of_sight(source, target, known, ATTACK_RANGE) {
                return ActionResult::failure();
            }
            let oid = state.board.get_entity_at(target);
            let cb = move |board: &mut Board, rng: &mut RNG| {
                let oid = if let Some(x) = oid { x } else { return; };
                let other = &mut board.entities[oid];
                other.dir = source - target;
                let removed = match other {
                    Entity::Pokemon(x) => {
                        let damage = rng.gen::<i32>().rem_euclid(ATTACK_DAMAGE);
                        x.data.me.cur_hp = max(0, x.data.me.cur_hp - damage);
                        x.data.me.cur_hp == 0
                    }
                    Entity::Trainer(x) => {
                        x.data.cur_hp = max(0, x.data.cur_hp - 1);
                        x.data.cur_hp == 0
                    }
                };
                let cb = move |board: &mut Board, _: &mut RNG| {
                    if removed { board.remove_entity(oid); };
                };
                board.add_effect(apply_damage(board, target, Box::new(cb)), rng);
            };
            let effect = effect::HeadbuttEffect(&state.board, &mut state.rng, source, target);
            state.add_effect(apply_effect(effect, FT::Hit, Box::new(cb)));
            ActionResult::success_moves(1.)
        }
        Action::Summon(index, target) => {
            let trainer = match &state.board.entities[eid] {
                Entity::Trainer(x) => x,
                _ => return ActionResult::failure(),
            };
            let (tid, source) = (trainer.id(), trainer.pos);
            let name = match trainer.data.pokemon.get(index) {
                Some(PokemonEdge::In(x)) if x.cur_hp > 0 => name(x),
                _ => return ActionResult::failure(),
            };
            if state.board.get_status(target) != Status::Free {
                return ActionResult::failure();
            }

            shout(&mut state.board, tid, &format!("Go! {}!", name));
            let cb = move |board: &mut Board, _: &mut RNG| {
                let trainer = &mut board.entities[tid];
                let me = cast!(trainer.data.pokemon.remove(index), PokemonEdge::In);
                let dir = target - trainer.pos;
                let arg = SummonArgs { pos: target, dir, me };
                let pid = board.add_summoned_pokemon(arg);
                let trainer = &mut board.entities[tid];
                trainer.data.pokemon.insert(index, PokemonEdge::Out(pid));
                trainer.data.summons.push(pid);
            };
            state.add_effect(apply_summon(source, target, Box::new(cb)));
            ActionResult::success()
        }
        Action::Withdraw(pid) => {
            let trainer = match &state.board.entities[eid] {
                Entity::Trainer(x) => x,
                _ => return ActionResult::failure(),
            };
            let pokemon = match state.board.entities.get(pid) {
                Some(x) if x.data.me.trainer.map(|x| x.eid()) == Some(eid) => x,
                _ => return ActionResult::failure(),
            };
            let (source, target) = (trainer.pos, pokemon.pos);
            if !has_line_of_sight(source, target, &trainer.known, SUMMON_RANGE) {
                return ActionResult::failure();
            }
            let name = name(&pokemon.data.me);
            let message = if trainer.player {
                format!("You withdraw {}.", name)
            } else {
                format!("{} withdraws {}.", trainer.data.name, name)
            };
            state.board.log_color(message, 0x234);
            let cb = move |board: &mut Board, _: &mut RNG| board.remove_entity(pid.eid());
            state.add_effect(apply_withdraw(source, target, Box::new(cb)));
            ActionResult::success()
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Top-level game update

fn get_direction(ch: char) -> Option<Point> {
    match ch {
        'h' => Some(dirs::W),
        'j' => Some(dirs::S),
        'k' => Some(dirs::N),
        'l' => Some(dirs::E),
        'y' => Some(dirs::NW),
        'u' => Some(dirs::NE),
        'b' => Some(dirs::SW),
        'n' => Some(dirs::SE),
        '.' => Some(dirs::NONE),
        _ => None,
    }
}

fn process_input(state: &mut State, input: Input) {
    let player = &state.board.entities[state.player];
    let (known, eid) = (&*player.known, player.id().eid());

    let tab = input == Input::Char('\t') || input == Input::BackTab;
    let enter = input == Input::Char('\n') || input == Input::Char('.');

    if let Some(x) = &mut state.menu {
        let summon = &state.board.entities[player.data.summons[x.summon as usize]];
        let attack = summon.data.me.attacks.len() as i32;
        let count = ATTACK_KEYS.len() as i32 + 1;
        let valid = |x: i32| { x == count - 1 || 0 <= x && x < attack };
        let chosen = if enter { x.index } else {
            ATTACK_KEYS.iter().position(|x| input == Input::Char(*x)).map(|x| x as i32)
                .unwrap_or(if input == Input::Char(RETURN_KEY) { count - 1} else { -1})
        };
        let dir = if let Input::Char(x) = input { get_direction(x) } else { None };

        if let Some(dir) = dir && dir.0 == 0 {
            loop {
                x.index += dir.1;
                if x.index >= count { x.index = 0; }
                if x.index < 0 { x.index = max(count - 1, 0); }
                if valid(x.index) { break; }
            }
        } else if chosen >= 0 {
            if chosen == count - 1 {
                state.input = Action::Shout(summon.id(), Command::Return);
                state.menu = None;
            } else {
                state.board.log_menu("Canceled.", 0x234);
                state.menu = None;
            }
        } else if input == Input::Escape {
            state.board.log_menu("Canceled.", 0x234);
            state.menu = None;
        }
        return;
    }

    if let Some(x) = &mut state.choice {
        let choice = if enter {
            Some(*x as usize)
        } else {
            PARTY_KEYS.iter().position(|x| input == Input::Char(*x))
        };
        let count = player.data.pokemon.len() as i32;
        let dir = if let Input::Char(x) = input { get_direction(x) } else { None };
        if let Some(dir) = dir && dir.0 == 0 {
            *x += dir.1;
            if *x >= count { *x = 0; }
            if *x < 0 { *x = max(count - 1, 0); }
        } else if let Some(choice) = choice {
            let pokemon = &player.data.pokemon;
            let length = pokemon.len();
            if choice >= length {
                let error = format!("You are only carrying {} Pokemon!", length);
                state.board.log_menu(error, 0x422);
            } else if let PokemonEdge::Out(x) = &pokemon[choice] {
                let pokemon = &state.board.entities[*x];
                let error = format!("{} is already out!", name(&pokemon.data.me));
                state.board.log_menu(error, 0x422);
            } else if let PokemonEdge::In(x) = &pokemon[choice] && x.cur_hp == 0 {
                let error = format!("{} has no strength left!", name(x));
                state.board.log_menu(error, 0x422);
            } else if let PokemonEdge::In(x) = &pokemon[choice] {
                let data = TargetData::Summon { index: choice, range: SUMMON_RANGE };
                let target = init_summon_target(player, data);
                let message = format!("Choose where to send out {}:", name(x));
                state.board.log_menu(message, 0x234);
                state.target = Some(target);
                state.choice = None;
            }
        } else if input == Input::Escape {
            state.board.log_menu("Canceled.", 0x234);
            state.choice = None;
        }
        return;
    }

    let apply_tab = |player: &Trainer, prev: Option<EID>, off: bool| -> Option<EID> {
        let rivals = rivals(player);
        if rivals.is_empty() { return None; }

        let t = input == Input::Char('\t');
        let n = rivals.len() + if off { 1 } else { 0 };

        let next = prev.and_then(|x| rivals.iter().position(|y| y.0.eid == x));
        let start = next.or_else(|| if off { Some(n - 1) } else { None });
        let index = start.map(|x| if t { x + n + 1 } else { x + n - 1 } % n)
                         .unwrap_or_else(|| if t { 0 } else { n - 1 });
        if index < rivals.len() { Some(rivals[index].0.eid) } else { None }
    };

    let get_initial_target = |player: &Trainer, source: Point| -> Point {
        let focus = known.focus.and_then(|x| known.entity(x));
        if let Some(target) = focus && target.age == 0 { return target.pos; }
        let rival = rivals(player).into_iter().next();
        if let Some(rival) = rival { return rival.0.pos; }
        source
    };

    let get_updated_target = |player: &Trainer, target: Point| -> Option<Point> {
        if tab {
            let old_eid = known.get(target).entity().map(|x| x.eid);
            let new_eid = apply_tab(player, old_eid, false);
            return Some(known.entity(new_eid?)?.pos);
        }

        let ch = if let Input::Char(x) = input { Some(x) } else { None }?;
        let dir = get_direction(ch.to_lowercase().next().unwrap_or(ch))?;
        let scale = if ch.is_uppercase() { 4 } else { 1 };

        let mut prev = target;
        for _ in 0..scale {
            let next = prev + dir;
            if outside_map(player, prev + dir) { break; }
            prev = next;
        }
        Some(prev)
    };

    if let Some(x) = &state.target {
        let update = get_updated_target(player, x.target);
        if let Some(update) = update && update != x.target {
            let mut target = state.target.take();
            target.as_mut().map(|x| update_target(known, x, update));
            state.target = target;
        } else if enter {
            if x.error.is_empty() {
                let focus = select_valid_target(state);
                state.board.set_focus(eid, focus);
                state.target = None;
            } else {
                state.board.log_menu(&x.error, 0x422);
            }
        } else if input == Input::Escape {
            if let TargetData::FarLook = x.data {
                let valid = x.error.is_empty();
                let focus = if valid { select_valid_target(state) } else { None };
                state.board.set_focus(eid, focus);
            }
            state.board.log_menu("Canceled.", 0x234);
            state.target = None;
        }
        return;
    }

    if tab {
        state.board.set_focus(eid, apply_tab(player, known.focus, true));
        return;
    } else if input == Input::Escape {
        state.board.set_focus(eid, None);
        return;
    }

    if input == Input::Char('x') {
        let source = player.pos;
        let update = get_initial_target(player, source);
        let mut target = init_target(TargetData::FarLook, source, update);
        update_target(&*player.known, &mut target, update);
        state.board.log_menu("Use the movement keys to examine a location:", 0x234);
        state.target = Some(target);
        return;
    }

    let index = SUMMON_KEYS.iter().position(|x| input == Input::Char(*x));
    if let Some(i) = index && i >= player.data.summons.len() {
        state.board.log_menu("Choose a Pokemon to send out with J/K:", 0x234);
        state.choice = Some(0);
        return;
    } else if let Some(i) = index {
        state.board.log_menu("Choose a command with J/K:", 0x234);
        state.menu = Some(Menu { index: 0, summon: i as i32 });
        return;
    }

    if input == Input::Char('q') || input == Input::Char('w') {
        let board = &state.board;
        let i = board.entity_order.iter().enumerate().find_map(|x| {
            let okay = Some(board.entities[*x.1].id()) == state.point_of_view;
            if okay { Some(x.0) } else { None }
        }).unwrap_or(0);
        let l = board.entity_order.len();
        let j = (i + if input == Input::Char('q') { l - 1 } else { 1 }) % l;
        state.point_of_view = if j == 0 { None } else { Some(board.entity_order[j]) };
        return;
    }

    let dir = if let Input::Char(x) = input { get_direction(x) } else { None };
    state.input = dir.map(|x| step(x, 1.)).unwrap_or(Action::WaitForInput);
}

fn update_focus(state: &mut State) {
    let known = &*state.board.entities[state.player].known;
    let focus = match &state.target {
        Some(x) => known.get(x.target).entity(),
        None => known.focus.and_then(|x| known.entity(x)),
    };
    if let Some(entity) = focus && can_target(entity) {
        let floor = Tile::get('.');
        let (player, pos, dir) = (entity.player, entity.pos, entity.dir);
        let lookup = |p: Point| known.get(p).tile().unwrap_or(floor);
        state.focus.vision.compute(&VisionArgs { player, pos, dir }, lookup);
        state.focus.active = true;
    } else {
        state.focus.active = false;
    }
}

fn update_state(state: &mut State) {
    if state.board.advance_effect(&mut state.rng) {
        state.board.update_known(state.player.eid());
        return;
    }

    let player_alive = |state: &State| {
        !state.board.entities[state.player].removed
    };

    let needs_input = |state: &State| {
        if !matches!(state.input, Action::WaitForInput) { return false; }
        let active = state.board.get_active_entity();
        if active != state.player.eid() { return false; }
        player_alive(state) && state.board.get_current_frame().is_none()
    };

    while !state.inputs.is_empty() && needs_input(state) {
        let input = state.inputs.remove(0);
        process_input(state, input);
    }
    if let Some(target) = state.target.as_mut() {
        target.frame = (target.frame + 1) % UI_TARGET_FRAMES;
        update_focus(state);
        return;
    }

    let mut update = false;

    while player_alive(state) && state.board.get_current_frame().is_none() {
        let eid = state.board.get_active_entity();
        let entity = &state.board.entities[eid];
        if !turn_ready(entity) {
            state.board.advance_entity();
            continue;
        } else if needs_input(state) {
            break;
        }

        // Update the trainer's view, too, since owned Pokemon use it to plan.
        // TODO(shaunak): Combine both views so we don't need to do that.
        let tid = entity.trainer();
        state.board.update_known(eid);
        if let Some(x) = tid { state.board.update_known(x.eid()); }

        let player = eid == state.player.eid();
        let action = plan(&state.board, eid, &mut state.input, &mut state.rng);
        let result = act(state, eid, action);
        update = true;

        if player && !result.success { break; }
        if let Some(x) = state.board.entities.get_mut(eid) { wait(x, &result); }
    }

    if update {
        state.board.update_known(state.player.eid());
        if let Some(x) = state.point_of_view && state.board.entity_order.contains(&x) {
            state.board.update_known(x);
        }
    }
    update_focus(state);
}

//////////////////////////////////////////////////////////////////////////////

// UI

const UI_COL_SPACE: i32 = 2;
const UI_ROW_SPACE: i32 = 1;
const UI_KEY_SPACE: i32 = 4;

const UI_LOG_SIZE: i32 = 4;
const UI_MAP_SIZE_X: i32 = WORLD_SIZE;
const UI_MAP_SIZE_Y: i32 = WORLD_SIZE;
const UI_CHOICE_SIZE: i32 = 40;
const UI_STATUS_SIZE: i32 = 30;
const UI_COLOR: i32 = 0x430;

const UI_DAMAGE_FLASH: i32 = 6;
const UI_DAMAGE_TICKS: i32 = 6;

const UI_TARGET_FRAMES: i32 = 20;

struct UI {
    log: Rect,
    map: Rect,
    choice: Rect,
    rivals: Rect,
    status: Rect,
    target: Rect,
    bounds: Point,
}

impl Default for UI {
    fn default() -> Self {
        let kl = Self::render_key(PLAYER_KEY).chars().count() as i32;
        assert!(kl == UI_KEY_SPACE);

        let ss = UI_STATUS_SIZE;
        let (x, y) = (UI_MAP_SIZE_X, UI_MAP_SIZE_Y);
        let (col, row) = (UI_COL_SPACE, UI_ROW_SPACE);
        let w = 2 * x + 2 + 2 * (ss + kl + 2 * col);
        let h = y + 2 + row + UI_LOG_SIZE + row + 1;

        let status = Rect {
            root: Point(col, row + 1),
            size: Point(ss + kl, y - 2 * row),
        };
        let map = Rect {
            root: Point(status.root.0 + status.size.0 + col + 1, 1),
            size: Point(2 * x, y),
        };
        let target = Rect {
            root: Point(map.root.0 + map.size.0 + col + 1, status.root.1),
            size: Point(ss + kl, 9),
        };
        let log = Rect {
            root: Point(0, map.root.1 + map.size.1 + row + 1),
            size: Point(w, UI_LOG_SIZE),
        };

        let (cw, ch) = (UI_CHOICE_SIZE, 6 * 5);
        let mut choice = Rect {
            root: Point((w - cw) / 2, (h - ch) / 2),
            size: Point(cw, ch),
        };
        if map.root.0 % 2 == choice.root.0 % 2 {
            choice.root.0 -= 1;
            choice.size.0 += 2;
        }
        if choice.size.0 % 2 != 0 { choice.size.0 += 1; }

        let ry = target.root.1 + target.size.1 + 2 * row + 1;
        let rivals = Rect {
            root: Point(target.root.0, ry),
            size: Point(ss + kl, status.root.1 + status.size.1 - ry),
        };

        Self { log, map, choice, rivals, status, target, bounds: Point(w, h) }
    }
}

impl UI {
    // Public entry points

    fn render_map(&self, entity: &Entity, frame: Option<&Frame>,
                  offset: Point, range: Option<i32>, slice: &mut Slice) {
        let (known, pos) = (&*entity.known, entity.pos);
        let in_range = |p: Point| range.map(|x| (p - pos).in_l2_range(x)).unwrap_or(true);
        let unseen = Glyph::wide(' ');

        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let point = Point(x, y) + offset;
                let value = known.get(point);
                let glyph = match value.tile() {
                    Some(x) => if value.visible() {
                        let glyph = value.entity().map(|y| y.glyph).unwrap_or(x.glyph);
                        if in_range(point) { glyph } else { glyph.with_fg(Color::gray()) }
                    } else {
                        x.glyph.with_fg(Color::gray())
                    }
                    None => unseen
                };
                slice.set(Point(2 * x, y), glyph);
            }
        }

        for eid in entity.friends() {
            if let Some(friend) = known.entity(eid) && friend.age > 0 {
                let Point(x, y) = friend.pos - offset;
                slice.set(Point(2 * x, y), Glyph::wide('?'));
            }
        }

        if let Some(frame) = frame {
            for effect::Particle { point, glyph } in frame {
                if !known.get(*point).visible() { continue; }
                let Point(x, y) = *point - offset;
                slice.set(Point(2 * x, y), *glyph);
            }
        }
    }

    fn render_log(&self, buffer: &mut Buffer, log: &Vec<LogLine>) {
        let slice = &mut Slice::new(buffer, self.log);
        for line in log {
            slice.set_fg(Some(line.color)).write_str(&line.text).newline();
        }
    }

    fn render_rivals(&self, buffer: &mut Buffer, trainer: &Trainer, target: Option<&Target>) {
        let slice = &mut Slice::new(buffer, self.rivals);
        let mut rivals = rivals(trainer);
        rivals.truncate(max(slice.size().1, 0) as usize / 2);

        for (rival, pokemon) in rivals {
            let PokemonSpeciesData { glyph, name, .. } = pokemon.species;
            let (hp, hp_color) = (pokemon.hp, Self::hp_color(pokemon.hp));
            let hp_text = format!("{}%", max((100.0 * hp).floor() as i32, 1));
            let (sn, sh) = (name.chars().count(), hp_text.chars().count());
            let ss = max(16 - sn as i32 - sh as i32, 0) as usize;

            slice.newline();
            slice.write_chr(*glyph).space().write_str(name);
            slice.spaces(ss).set_fg(Some(hp_color)).write_str(&hp_text).newline();

            let targeted = match &target {
                Some(x) => x.target == rival.pos,
                None => trainer.known.focus == Some(rival.eid),
            };
            if targeted {
                let start = slice.get_cursor() - Point(0, 1);
                let (fg, bg) = (Color::default(), Color::gray());
                for x in 0..UI_STATUS_SIZE {
                    let p = start + Point(x, 0);
                    slice.set(p, Glyph::new(slice.get(p).ch(), fg, bg));
                }
            }
        }
    }

    fn render_status(&self, buffer: &mut Buffer, trainer: &Trainer,
                     menu: Option<&Menu>, summon: Option<&Pokemon>) {
        assert!(menu.is_some() == summon.is_some());
        let slice = &mut Slice::new(buffer, self.status);
        let known = &*trainer.known;
        if let Some(view) = known.entity(trainer.id().eid()) {
            let key = if menu.is_some() { '-' } else { PLAYER_KEY };
            self.render_entity(Some(key), None, view, slice);
        }
        for (i, key) in SUMMON_KEYS.iter().enumerate() {
            let key = if menu.is_some() { '-' } else { *key };
            let eid = trainer.data.summons.get(i).map(|x| x.eid());
            if let Some(view) = eid.and_then(|x| known.entity(x)) {
                self.render_entity(Some(key), None, view, slice);
                if let Some(x) = menu && x.summon == i as i32 {
                    self.render_menu(slice, x.index, summon.unwrap());
                }
            } else {
                self.render_empty_option(key, 0, slice);
            }
        }
    }

    fn render_target(&self, buffer: &mut Buffer, trainer: &Trainer, target: Option<&Target>) {
        let slice = &mut Slice::new(buffer, self.target);
        let known = &*trainer.known;
        if target.is_none() && known.focus.is_none() {
            let fg = Some(0x111.into());
            slice.newline();
            slice.set_fg(fg).write_str("No target selected.").newline();
            slice.newline();
            slice.set_fg(fg).write_str("[x] examine your surroundings").newline();
            return;
        }

        let (cell, view, header, seen) = match &target {
            Some(x) => {
                let cell = known.get(x.target);
                let (seen, view) = (cell.visible(), cell.entity());
                let header = match &x.data {
                    TargetData::FarLook => "Examining...".into(),
                    TargetData::Summon { index, .. } => {
                        let name = match &trainer.data.pokemon[*index] {
                            PokemonEdge::In(y) => name(y),
                            PokemonEdge::Out(_) => "?",
                        };
                        format!("Sending out {}...", name)
                    }
                };
                (cell, view, header, seen)
            }
            None => {
                let view = known.focus.and_then(|x| known.entity(x));
                let seen = view.map(|x| x.age == 0).unwrap_or(false);
                let cell = view.map(|x| known.get(x.pos)).unwrap_or(known.default());
                let header = if seen {
                    "Last target:"
                } else {
                    "Last target: (remembered)"
                }.into();
                (cell, view, header, seen)
            },
        };

        let fg = if target.is_some() || seen { None } else { Some(0x111.into()) };
        let text = if view.is_some() {
            if seen { "Standing on: " } else { "Stood on: " }
        } else {
            if seen { "You see: " } else { "You saw: " }
        };

        slice.newline();
        slice.set_fg(fg).write_str(&header).newline();

        if let Some(view) = view {
            self.render_entity(None, fg, view, slice);
        } else {
            slice.newline();
        }

        slice.set_fg(fg).write_str(text);
        if let Some(x) = cell.tile() {
            slice.write_chr(x.glyph).space();
            slice.write_chr('(').write_str(x.description).write_chr(')').newline();
        } else {
            slice.write_str("(unseen location)").newline();
        }
    }

    fn render_choice(&self, buffer: &mut Buffer, trainer: &Trainer,
                     summons: Vec<&Pokemon>, choice: i32) {
        self.render_dialog(buffer, &self.choice);
        let slice = &mut Slice::new(buffer, self.choice);
        let options = &trainer.data.pokemon;
        for (i, key) in PARTY_KEYS.iter().enumerate() {
            let selected = choice == i as i32;
            match if i < options.len() { Some(&options[i]) } else { None } {
                Some(PokemonEdge::Out(x)) => {
                    let pokemon = *summons.iter().find(|y| y.id() == *x).unwrap();
                    let (me, pp) = (&*pokemon.data.me, get_pp(pokemon));
                    self.render_option(*key, 1, selected, me, pp, slice);
                },
                Some(PokemonEdge::In(x)) =>
                    self.render_option(*key, 0, selected, x, 1.0, slice),
                None => self.render_empty_option(*key, UI_COL_SPACE + 1, slice),
            }
        }
    }

    // High-level private helpers

    fn render_menu(&self, slice: &mut Slice, index: i32, summon: &Pokemon) {
        let spaces = UI::render_key('-').chars().count();

        for (i, key) in ATTACK_KEYS.iter().enumerate() {
            let prefix = if index == i as i32 { " > " } else { "  " };
            let attack = summon.data.me.attacks.get(i);
            let name = attack.map(|x| x.name).unwrap_or("---");
            let fg = if attack.is_some() { None } else { Some(0x111.into()) };
            slice.set_fg(fg).spaces(spaces).write_str(prefix);
            slice.write_str(&Self::render_key(*key)).write_str(name);
            slice.newline().newline();
        }

        let prefix = if index == ATTACK_KEYS.len() as i32 { " > " } else { "  " };
        slice.spaces(spaces).write_str(prefix);
        slice.write_str(&Self::render_key(RETURN_KEY)).write_str("Call back");
        slice.newline().newline();
    }

    fn render_empty_option(&self, key: char, space: i32, slice: &mut Slice) {
        let n = space as usize;
        let fg = Some(0x111.into());
        let prefix = UI::render_key(key);

        slice.newline();
        slice.set_fg(fg).spaces(n).write_str(&prefix).write_str("---").newline();
        slice.newline().newline().newline();
    }

    fn render_option(&self, key: char, out: i32, selected: bool,
                     me: &PokemonIndividualData, pp: f64, slice: &mut Slice) {
        let hp = get_hp(me);
        let (hp_color, pp_color) = (Self::hp_color(hp), 0x123.into());
        let fg = if out == 0 && hp > 0. { None } else { Some(0x111.into()) };

        let x = if selected { 1 } else { 0 };
        let arrow = if selected { '>' } else { ' ' };

        let prefix = UI::render_key(key);
        let n = prefix.chars().count() + (UI_COL_SPACE + 1) as usize;
        let w = slice.size().0 - (n as i32) - 2 * UI_COL_SPACE - 6;
        let status_bar_line = |p: &str, v: f64, c: Color, s: &mut Slice| {
            s.set_fg(fg).spaces(n + x).write_str(p);
            self.render_bar(v, c, w, s);
            s.newline();
        };

        slice.newline();
        slice.spaces(UI_COL_SPACE as usize).write_chr(arrow).spaces(x);
        slice.set_fg(fg).write_str(&prefix).write_str(me.species.name).newline();
        status_bar_line("HP: ", hp, hp_color, slice);
        status_bar_line("PP: ", pp, pp_color, slice);
        slice.newline();
    }

    fn render_entity(&self, key: Option<char>, fg: Option<Color>,
                     entity: &EntityKnowledge, slice: &mut Slice) {
        let prefix = key.map(|x| UI::render_key(x)).unwrap_or(String::default());
        let n = prefix.chars().count();
        let w = UI_STATUS_SIZE - 6;
        let status_bar_line = |p: &str, v: f64, c: Color, s: &mut Slice| {
            self.render_bar(v, c, w, s.set_fg(fg).spaces(n).write_str(p));
            s.newline();
        };

        slice.newline();
        match &entity.view {
            EntityView::Pokemon(x) => {
                let (hp_color, pp_color) = (Self::hp_color(x.hp), 0x123.into());
                slice.set_fg(fg).write_str(&prefix).write_str(x.species.name).newline();
                status_bar_line("HP: ", x.hp, hp_color, slice);
                status_bar_line("PP: ", x.pp, pp_color, slice);
            }
            EntityView::Trainer(x) => {
                let hp_color = Self::hp_color(x.hp);
                let name = if entity.player { "You" } else { &x.name };
                slice.set_fg(fg).write_str(&prefix).write_str(name).newline();
                status_bar_line("HP: ", x.hp, hp_color, slice);
                slice.set_fg(fg).spaces(n + 5);
                for status in &x.status {
                    let color = if *status { Color::default() } else { 0x111.into() };
                    slice.write_chr(Glyph::chfg('*', color));
                    slice.spaces(1);
                }
                slice.newline();
            }
        }
        slice.newline();
    }

    // Private implementation details

    fn render_bar(&self, value: f64, color: Color, width: i32, slice: &mut Slice) {
        let count = if value > 0. { max(1, (width as f64 * value) as i32) } else { 0 };
        let glyph = Glyph::chfg('=', color);

        slice.write_chr('[');
        for _ in 0..count { slice.write_chr(glyph); }
        for _ in count..width { slice.write_chr(' '); }
        slice.write_chr(']');
    }

    fn render_box(&self, buffer: &mut Buffer, rect: &Rect) {
        let Point(w, h) = rect.size;
        let color: Color = UI_COLOR.into();
        buffer.set(rect.root + Point(-1, -1), Glyph::chfg('┌', color));
        buffer.set(rect.root + Point( w, -1), Glyph::chfg('┐', color));
        buffer.set(rect.root + Point(-1,  h), Glyph::chfg('└', color));
        buffer.set(rect.root + Point( w,  h), Glyph::chfg('┘', color));

        let tall = Glyph::chfg('│', color);
        let flat = Glyph::chfg('─', color);
        for x in 0..w {
            buffer.set(rect.root + Point(x, -1), flat);
            buffer.set(rect.root + Point(x,  h), flat);
        }
        for y in 0..h {
            buffer.set(rect.root + Point(-1, y), tall);
            buffer.set(rect.root + Point( w, y), tall);
        }
    }

    fn render_dialog(&self, buffer: &mut Buffer, rect: &Rect) {
        for x in 0..rect.size.0 {
            for y in 0..rect.size.1 {
                buffer.set(rect.root + Point(x, y), buffer.default);
            }
        }
        self.render_box(buffer, rect);
    }

    fn render_frame(&self, buffer: &mut Buffer) {
        let ml = self.map.root.0 - 1;
        let mw = self.map.size.0 + 2;
        let mh = self.map.size.1 + 2;
        let tt = self.target.root.1;
        let th = self.target.size.1;
        let rt = tt + th + UI_ROW_SPACE;
        let uw = self.bounds.0;
        let uh = self.bounds.1;

        self.render_title(buffer, ml, Point(0, 0), "Party");
        self.render_title(buffer, ml, Point(ml + mw, 0), "Target");
        self.render_title(buffer, ml, Point(ml + mw, rt), "Wild Pokemon");
        self.render_title(buffer, ml, Point(0, mh - 1), "Log");
        self.render_title(buffer, ml, Point(ml + mw, mh - 1), "");
        self.render_title(buffer, uw, Point(0, uh - 1), "");

        self.render_box(buffer, &self.map);
    }

    fn render_title(&self, buffer: &mut Buffer, width: i32, pos: Point, text: &str) {
        let shift = 2;
        let color: Color = UI_COLOR.into();
        let dashes = Glyph::chfg('-', color);
        let prefix_width = shift + text.chars().count() as i32;
        assert!(prefix_width <= width);
        for x in 0..shift {
            buffer.set(pos + Point(x, 0), dashes);
        }
        for (i, c) in text.chars().enumerate() {
            buffer.set(pos + Point(i as i32 + shift, 0), Glyph::chfg(c, color));
        }
        for x in prefix_width..width {
            buffer.set(pos + Point(x, 0), dashes);
        }
    }

    // Static helpers

    fn hp_color(hp: f64) -> Color {
        (if hp <= 0.25 { 0x300 } else if hp <= 0.50 { 0x330 } else { 0x020 }).into()
    }

    fn render_key(key: char) -> String {
        format!("[{}] ", key)
    }
}

//////////////////////////////////////////////////////////////////////////////

// State

struct Focus {
    active: bool,
    vision: Vision,
}

struct Menu {
    index: i32,
    summon: i32,
}

enum TargetData {
    FarLook,
    Summon { index: usize, range: i32 },
}

struct Target {
    data: TargetData,
    error: String,
    frame: i32,
    path: Vec<(Point, bool)>,
    source: Point,
    target: Point,
}

pub struct State {
    board: Board,
    focus: Focus,
    input: Action,
    inputs: Vec<Input>,
    choice: Option<i32>,
    target: Option<Box<Target>>,
    point_of_view: Option<EID>,
    player: TID,
    menu: Option<Menu>,
    rng: RNG,
    ui: UI,
}

impl State {
    pub fn new(seed: Option<u64>) -> Self {
        let size = Point(WORLD_SIZE, WORLD_SIZE);
        let pos = Point(size.0 / 2, size.1 / 2);
        let rng = seed.map(|x| RNG::seed_from_u64(x));
        let mut rng = rng.unwrap_or_else(|| RNG::from_entropy());
        let mut board = Board::new(size);

        loop {
            mapgen(&mut board, &mut rng);
            if !board.get_tile_at(pos).blocked() { break; }
        }

        let name = "skishore";
        let args = TrainerArgs { pos, dir: Point(0, 1), player: true, name };
        let tid = board.add_trainer(&args);
        let player = &mut board.entities[tid];
        player.register_pokemon("Bulbasaur");
        player.register_pokemon("Charmander");
        player.register_pokemon("Squirtle");
        player.register_pokemon("Eevee");
        player.register_pokemon("Pikachu");

        let die = |n: i32, rng: &mut RNG| rng.gen::<i32>().rem_euclid(n);
        let pos = |board: &Board, rng: &mut RNG| {
            for _ in 0..100 {
                let p = Point(die(size.0, rng), die(size.1, rng));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        let options = ["Pidgey", "Ratatta"];
        for i in 0..20 {
            if let Some(pos) = pos(&board, &mut rng) {
                let index = if i % 10 == 0 { 1 } else { 0 };
                let (dir, species) = (*sample(&dirs::ALL, &mut rng), options[index]);
                board.add_pokemon(&PokemonArgs { pos, dir, species });
            }
        }
        board.update_known(tid.eid());

        let point_of_view = Some(board.entity_order[1]);

        Self {
            board,
            focus: Focus {
                active: false,
                vision: Vision::new(FOV_RADIUS_NPC),
            },
            ui: UI::default(),
            input: Action::WaitForInput,
            inputs: vec![],
            choice: None,
            target: None,
            point_of_view,
            player: tid,
            menu: None,
            rng,
        }
    }

    pub fn add_effect(&mut self, x: Effect) { self.board.add_effect(x, &mut self.rng) }

    pub fn add_input(&mut self, input: Input) { self.inputs.push(input) }

    pub fn update(&mut self) { update_state(self); }

    pub fn render(&self, buffer: &mut Buffer) {
        if buffer.data.is_empty() {
            let size = self.ui.bounds;
            let mut overwrite = Matrix::new(size, ' '.into());
            std::mem::swap(buffer, &mut overwrite);
        }
        buffer.fill(buffer.default);
        self.ui.render_frame(buffer);

        let player = &self.board.entities[self.player];
        let offset = player.pos - Point(UI_MAP_SIZE_X / 2, UI_MAP_SIZE_Y / 2);

        let frame = self.board.get_current_frame();
        let slice = &mut Slice::new(buffer, self.ui.map);
        let range = self.target.as_ref().and_then(|x| match &x.data {
            TargetData::Summon { range, .. } => Some(*range),
            _ => None,
        });

        let board = &self.board;
        let offset = Point::default();
        let entity = board.entity_order.iter().find_map(|x| {
            let okay = Some(board.entities[*x].id()) == self.point_of_view;
            if okay { Some(&board.entities[*x]) } else { None }
        });
        if let Some(entity) = entity {
            self.ui.render_map(entity, frame, Point::default(), None, slice);
            let (debug, extra) = if let Entity::Pokemon(x) = entity &&
                                    let Some(mut ai) = x.data.ai.take() {
                let debug = {
                    let a = std::mem::take(&mut ai.plan);
                    let b = std::mem::take(&mut ai.turn_times);
                    let c = std::mem::take(&mut ai.flight.distances);
                    let debug = format!("{:?}", ai);
                    ai.flight.distances = c;
                    ai.turn_times = b;
                    ai.plan = a;
                    debug
                };

                if 0 == 1 {
                    for (point, value) in &ai.flight.distances {
                        let point = Point(2 * point.0, point.1);
                        let ch = std::char::from_digit((10000 + *value) as u32 % 10, 10).unwrap();
                        slice.set(point, Glyph::wdfg(ch, slice.get(point).fg()));
                    }
                }
                for step in &ai.plan {
                    if frame.is_some() { continue; }
                    if step.kind == StepKind::Look { continue; }
                    let point = Point(2 * step.target.0, step.target.1);
                    let mut glyph = slice.get(point).with_fg(0x400);
                    if glyph.ch() == Glyph::wide(' ').ch() { glyph = Glyph::wdfg('.', 0x400); }
                    slice.set(point, glyph);
                }
                x.data.ai.set(Some(ai));

                let target = x.data.target.take();
                for t in &target {
                    if frame.is_some() { continue; }
                    let point = Point(2 * t.0, t.1);
                    let glyph = slice.get(point).with_fg(Color::black()).with_bg(0x400);
                    slice.set(point, glyph);
                }
                x.data.target.set(target);

                let extra = x.data.debug.take();
                x.data.debug.set(extra.clone());
                (debug, extra)
            } else {
                ("<none>".into(), "".into())
            };
            for eid in &self.board.entity_order {
                let other = &self.board.entities[*eid];
                let point = Point(2 * other.pos.0, other.pos.1);
                slice.set(point, other.glyph);
            }
            for other in &entity.known.entities {
                if frame.is_some() { continue; }
                let color = if other.age == 0 { 0x040 } else {
                    if other.moved { 0x400 } else { 0x440 }
                };
                let glyph = other.glyph.with_fg(Color::black()).with_bg(color);
                let point = Point(2 * other.pos.0, other.pos.1);
                slice.set(point, glyph);
            };
            if let Some(frame) = frame {
                for effect::Particle { point, glyph } in frame {
                    slice.set(Point(2 * point.0, point.1), *glyph);
                }
            }
            let slice = &mut Slice::new(buffer, self.ui.log);
            slice.write_str(&debug).newline().write_str(&extra);
            return;
        }

        self.ui.render_map(player, frame, offset, range, slice);

        let set = |slice: &mut Slice, p: Point, glyph: Glyph| {
            let point = Point(2 * (p.0 - offset.0), p.1 - offset.1);
            slice.set(point, glyph);
        };

        let recolor = |slice: &mut Slice, p: Point, fg: Option<Color>, bg: Option<Color>| {
            let point = Point(2 * (p.0 - offset.0), p.1 - offset.1);
            if !slice.contains(point) { return; }

            let mut glyph = slice.get(point);
            if let Some(fg) = fg {
                glyph = glyph.with_fg(fg);
            }
            if glyph.bg() == Color::default() && let Some(bg) = bg {
                glyph = glyph.with_bg(bg);
            }
            slice.set(point, glyph);
        };

        if let Some(target) = &self.target {
            let color = if target.error.is_empty() { 0x440 } else { 0x400 };
            recolor(slice, target.source, Some(Color::black()), Some(0x222.into()));
            recolor(slice, target.target, Some(Color::black()), Some(color.into()));

            let frame = target.frame >> 1;
            let count = UI_TARGET_FRAMES >> 1;
            let limit = target.path.len() as i32 - 1;
            let ch = ray_character(target.source, target.target);
            for i in 0..limit {
                if !((i + count - frame) % count < 2) { continue; }
                let (point, okay) = target.path[i as usize];
                let color = if okay { 0x440 } else { 0x400 };
                set(slice, point, Glyph::wdfg(ch, color));
            }
        }

        if self.focus.active {
            let aware = self.focus.vision.can_see_now(player.pos);
            let color = if aware { 0x100 } else { 0x000 };
            let slice = &mut Slice::new(buffer, self.ui.map);

            for (i, point) in self.focus.vision.points_seen.iter().enumerate() {
                if i == 0 {
                    recolor(slice, *point, Some(Color::black()), Some(0x222.into()));
                } else {
                    recolor(slice, *point, None, Some(color.into()));
                }
            }
        }

        let menu = self.menu.as_ref();
        let target = self.target.as_ref().map(|x| x.as_ref());
        let summon = menu.map(|x| &self.board.entities[player.data.summons[x.summon as usize]]);

        self.ui.render_status(buffer, player, menu, summon);
        self.ui.render_rivals(buffer, player, target);
        self.ui.render_target(buffer, player, target);
        self.ui.render_log(buffer, &self.board.log);

        if let Some(x) = self.choice {
            let summons = player.data.summons.iter().map(
                    |x| &self.board.entities[*x]).collect();
            self.ui.render_choice(buffer, player, summons, x);
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
        let mut state = State::new(Some(17));
        b.iter(|| {
            state.inputs.push(Input::Char('.'));
            state.update();
        });
    }
}
