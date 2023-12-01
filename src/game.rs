use std::cmp::{max, min};
use std::f64::consts::TAU;
use std::ops::{Deref, DerefMut};

use lazy_static::lazy_static;
use rand::random;

use crate::static_assert_size;
use crate::base::{Buffer, Color, Glyph, Slice};
use crate::base::{HashMap, LOS, Matrix, Point, Rect, clamp};
use crate::entity::{EID, PID, TID, EntityMap, EntityMapKey};
use crate::entity::{Entity, Pokemon, PokemonArgs, Trainer, TrainerArgs};
use crate::entity::{PokemonEdge, PokemonIndividualData, PokemonSpeciesData};
use crate::knowledge::{Knowledge, Vision, VisionArgs, get_pp, get_hp};
use crate::knowledge::{CellKnowledge, EntityKnowledge, EntityView, PokemonView};
use crate::pathing::{AStar, DijkstraMap, DijkstraMode};
use crate::pathing::{DIRECTIONS, BFS, BFSResult, Status};

//////////////////////////////////////////////////////////////////////////////

// Constants

pub const MOVE_TIMER: i32 = 960;
pub const TURN_TIMER: i32 = 120;

const ASSESS_TIME: i32 = 17;
const ASSESS_ANGLE: f64 = TAU / 3.;

const MIN_FLIGHT_TIME: i32 = 8;
const MAX_FLIGHT_TIME: i32 = 64;
const MAX_FOLLOW_TIME: i32 = 64;

const FOV_RADIUS_NPC: i32 = 12;
const FOV_RADIUS_PC_: i32 = 21;

const WANDER_TURNS: f64 = 3.;
const WORLD_SIZE: i32 = 100;

const ASTAR_LIMIT_ATTACK: i32 = 32;
const ASTAR_LIMIT_WANDER: i32 = 256;
const BFS_LIMIT_ATTACK: i32 = 8;
const BFS_LIMIT_WANDER: i32 = 64;

const PLAYER_KEY: char = 'a';
const PARTY_KEYS: [char; 6] = ['a', 's', 'd', 'f', 'g', 'h'];
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

// BoardView - a read-only Board

pub struct BoardView {
    map: Matrix<&'static Tile>,
    active_entity_index: usize,
    entity_at_pos: HashMap<Point, EID>,
    entity_order: Vec<EID>,
    entities: EntityMap,
}

impl BoardView {
    fn new(size: Point) -> Self {
        Self {
            map: Matrix::new(size, Tile::get('#')),
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

    pub fn get_entity(&self, eid: EID) -> Option<&Entity> {
        self.entities.get(eid)
    }

    pub fn get_entity_at(&self, p: Point) -> Option<EID> {
        self.entity_at_pos.get(&p).map(|x| *x)
    }

    pub fn get_status(&self, p: Point) -> Status {
        if self.entity_at_pos.contains_key(&p) { return Status::Occupied; }
        if self.map.get(p).blocked() { Status::Blocked } else { Status::Free }
    }

    pub fn get_tile_at(&self, p: Point) -> &'static Tile {
        self.map.get(p)
    }
}

//////////////////////////////////////////////////////////////////////////////

// Board

struct LogLine {
    color: Color,
    menu: bool,
    text: String,
}

struct Board {
    base: BoardView,
    known: Option<Box<Knowledge>>,
    npc_vision: Vision,
    _pc_vision: Vision,
    logs: Vec<LogLine>,
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
            logs: vec![],
        }
    }

    // Logging

    fn log<S: Into<String>>(&mut self, text: S) {
        self.log_color(text, Color::default());
    }

    fn log_color<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        let (color, text) = (color.into(), text.into());
        self.logs.push(LogLine { color, menu: false, text });
        if self.logs.len() as i32 > UI_LOG_SIZE { self.logs.remove(0); }
    }

    fn log_menu<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        let (color, text) = (color.into(), text.into());
        if self.logs.last().map(|x| x.menu).unwrap_or(false) { self.logs.pop(); }
        self.logs.push(LogLine { color, menu: true, text });
        if self.logs.len() as i32 > UI_LOG_SIZE { self.logs.remove(0); }
    }

    // Knowledge

    fn set_focus(&mut self, eid: EID, focus: Option<EID>) {
        self.entities[eid].known.focus = focus;
    }

    fn update_known(&mut self, eid: EID) {
        let mut known = self.known.take().unwrap_or_default();
        std::mem::swap(&mut self.base.entities[eid].known, &mut known);

        let entity = &self.base.entities[eid];
        let (player, pos, dir) = (entity.player, entity.pos, entity.dir);
        let vision = if player { &mut self._pc_vision } else { &mut self.npc_vision };
        vision.compute(&VisionArgs { player, pos, dir }, |p| self.base.map.get(p));
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
                    for edge in &mut self.entities[tid].data.pokemon {
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

// Targeting UI

fn check_follower_square(known: &Knowledge, leader: &Trainer,
                         target: Point, ignore_occupant: bool) -> bool {
    let free = match known.get_status(target).unwrap_or(Status::Blocked) {
        Status::Free => true,
        Status::Blocked => false,
        Status::Occupied => ignore_occupant,
    };
    if !free { return false }

    let source = leader.pos;
    let length = (source - target).len_nethack();
    if length > 2 { return false; }
    if length < 2 { return true; }

    let vision = |p: Point| { known.get_cell(p).map(|x| x.visibility).unwrap_or(-1) };
    vision(source) == vision(target)
}

fn can_target(entity: &EntityKnowledge) -> bool {
    entity.age == 0 && !entity.friend
}

fn init_target(data: TargetData, source: Point, target: Point) -> Box<Target> {
    Box::new(Target { data, error: "".into(), frame: 0, path: vec![], source, target })
}

fn init_summon_target(player: &Trainer, data: TargetData) -> Box<Target> {
    let (known, pos, dir) = (&*player.known, player.pos, player.dir);
    let mut target = init_target(data, pos, pos);

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

    options.sort_by_cached_key(|x| (*x - best).len_l2_squared());
    let update = options.into_iter().next().unwrap_or(pos);
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
                if known.can_see_now(x.0) { okay_until = i + 1; }
            }
            if okay_until < target.path.len() {
                target.error = "You can't see a clear path there.".into();
            }
        }
        TargetData::Summon(_) => {
            if target.path.is_empty() {
                target.error = "There's something in the way.".into();
            }
            for (i, x) in target.path.iter().enumerate() {
                if known.get_status(x.0).unwrap_or(Status::Free) != Status::Free {
                    target.error = "There's something in the way.".into();
                } else if !known.can_see_now(x.0) {
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
    let entity = known.get_entity_at(target.target);

    match &target.data {
        TargetData::FarLook => {
            let entity = entity?;
            if can_target(entity) { Some(entity.eid) } else { None }
        }
        TargetData::Summon(_) => {
            known.focus
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Game logic

pub struct MoveData { dir: Point, turns: f64 }

pub enum Action {
    Idle,
    Look(Point),
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

fn charge(entity: &mut Entity) {
    let charge = (TURN_TIMER as f64 * entity.speed).round() as i32;
    if entity.move_timer > 0 { entity.move_timer -= charge; }
    if entity.turn_timer > 0 { entity.turn_timer -= charge; }
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

#[derive(Debug)]
pub struct Fight {
    pub age: i32,
    pub target: Point,
}

#[derive(Debug)]
pub struct Flight {
    pub age: i32,
    pub switch: i32,
    pub target: Point,
}

#[derive(Debug)]
pub struct Wander {
    pub time: i32,
}

#[derive(Debug)]
pub struct Assess {
    pub switch: i32,
    pub target: Point,
    pub time: i32,
}

#[derive(Debug)]
pub enum AIState {
    Fight(Fight),
    Flight(Flight),
    Wander(Wander),
    Assess(Assess),
}

impl Default for Wander {
    fn default() -> Self { Self { time: 0 } }
}

impl Default for AIState {
    fn default() -> Self { Self::Wander(Wander::default()) }
}

//////////////////////////////////////////////////////////////////////////////

// AI routines

fn explore_near(entity: &Entity, source: Point, age: i32, turns: f64) -> Action {
    let (known, pos) = (&*entity.known, entity.pos);
    let check = |p: Point| {
        if p == pos { return Status::Free; }
        entity.known.get_status(p).unwrap_or(Status::Free)
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

fn assess(entity: &Entity, state: &mut Assess) -> Action {
    state.time -= 1;
    let time = state.time;
    let a = 1 * ASSESS_TIME / 4;
    let b = 2 * ASSESS_TIME / 4;
    let c = 3 * ASSESS_TIME / 4;
    let depth = if time < a {
        -time
    } else if time < c {
        time - 2 * a
    } else {
        -time + 2 * c - 2 * a
    };
    let scale = 1000;
    let angle = ASSESS_ANGLE * depth as f64 / b as f64;
    let (sin, cos) = (angle.sin(), angle.cos());
    let Point(dx, dy) = state.target - entity.pos;
    let rx = (cos * (scale * dx) as f64) + (sin * (scale * dy) as f64);
    let ry = (cos * (scale * dy) as f64) - (sin * (scale * dx) as f64);
    Action::Look(Point(rx as i32, ry as i32))
}

fn flight(entity: &Entity, source: Point) -> Option<Action> {
    let mut map: HashMap<Point, i32> = HashMap::default();
    map.insert(source, 0);

    let limit = 1024;
    let (known, pos) = (&*entity.known, entity.pos);

    let check = |p: Point| {
        if p == pos { return Status::Free; }
        known.get_status(p).unwrap_or(Status::Blocked)
    };
    DijkstraMap(DijkstraMode::Expand(limit), check, 1, &mut map);

    for (pos, val) in map.iter_mut() {
        let frontier = DIRECTIONS.iter().any(|x| !known.remembers(*pos + *x));
        if frontier { *val += FOV_RADIUS_NPC; }
        *val *= -10;
    }
    DijkstraMap(DijkstraMode::Update, check, 1, &mut map);

    let lookup = |x: &Point| map.get(x).map(|x| *x).unwrap_or(-9999);
    let mut best_steps = vec![Point::default()];
    let mut best_score = lookup(&pos);

    for dir in &DIRECTIONS {
        let option = pos + *dir;
        if check(option) != Status::Free { continue; }
        let score = lookup(&option);
        if score > best_score { continue; }
        if score < best_score { best_steps.clear(); }
        best_steps.push(*dir);
        best_score = score;
    }

    if best_steps[0] == Point::default() { return None; }
    Some(Action::Move(MoveData { dir: *sample(&best_steps), turns: 1.5 }))
}

fn plan_pokemon(pokemon: &Pokemon) -> Action {
    let mut ai = pokemon.data.ai.take().unwrap_or(Box::default());
    let prey = pokemon.data.me.species.name == "Pidgey";

    let mut targets: Vec<(i32, Point)> = vec![];
    for entity in pokemon.known.entities.iter() {
        if !entity.rival { continue; }
        targets.push((entity.age, entity.pos));
    }

    if !targets.is_empty() {
        targets.sort_by_cached_key(|(age, _)| *age);
        let (age, target) = targets[0];

        if prey {
            let switch = match ai.as_ref() {
                AIState::Flight(x) => x.switch,
                AIState::Assess(x) => x.switch,
                _ => MIN_FLIGHT_TIME,
            };
            let flight = matches!(*ai, AIState::Flight(_));
            if age >= switch && flight {
                *ai = AIState::Assess(Assess { switch, target, time: ASSESS_TIME });
            } else if age < switch && !flight {
                let switch = clamp(2 * switch, MIN_FLIGHT_TIME, MAX_FLIGHT_TIME);
                *ai = AIState::Flight(Flight { age, switch, target });
            }
            if let AIState::Flight(x) = ai.as_mut() {
                (x.age, x.target) = (age, target);
            }
        } else if age < MAX_FOLLOW_TIME || matches!(*ai, AIState::Fight(_)) {
            *ai = AIState::Fight(Fight { age, target });
        }
    }
    if let AIState::Fight(x) = ai.as_ref() && x.age >= MAX_FOLLOW_TIME {
        let (switch, target) = (MIN_FLIGHT_TIME, x.target);
        *ai = AIState::Assess(Assess { switch, target, time: ASSESS_TIME })
    }
    if let AIState::Wander(x) = ai.as_ref() && x.time <= 0 {
        let (switch, target) = (MIN_FLIGHT_TIME, pokemon.pos + pokemon.dir);
        *ai = AIState::Assess(Assess { switch, target, time: ASSESS_TIME })
    }
    if let AIState::Assess(x) = ai.as_ref() && x.time <= 0 {
        *ai = AIState::Wander(Wander { time: random::<i32>().rem_euclid(16) });
    }

    let mut replacement = None;
    let action = match ai.as_mut() {
        AIState::Fight(x) => if x.age == 0 {
            Action::Look(x.target - pokemon.pos)
        } else {
            explore_near(pokemon.base(), x.target, x.age, 1.)
        }
        AIState::Flight(x) => flight(pokemon.base(), x.target).unwrap_or_else(||{
            let (target, time) = (x.target, ASSESS_TIME);
            let mut x = Assess { switch: min(x.age + 1, x.switch), target, time };
            let result = assess(pokemon.base(), &mut x);
            replacement = Some(AIState::Assess(x));
            result
        }),
        AIState::Wander(x) => {
            x.time -= 1;
            explore_near(pokemon.base(), pokemon.pos, 9999, WANDER_TURNS)
        }
        AIState::Assess(ref mut x) => assess(pokemon.base(), x),
    };
    if let Some(x) = replacement { *ai = x; }
    pokemon.data.ai.set(Some(ai));
    action
}

fn plan(entity: &Entity, input: &mut Action) -> Action {
    if entity.player {
        return std::mem::replace(input, Action::WaitForInput);
    }
    match entity {
        Entity::Pokemon(x) => plan_pokemon(x),
        Entity::Trainer(_) => Action::Idle,
    }
}

fn act(state: &mut State, eid: EID, action: Action) -> ActionResult {
    match action {
        Action::Idle => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::Look(dir) => {
            state.board.entities[eid].dir = dir;
            ActionResult::success()
        }
        Action::Move(MoveData { dir ,turns }) => {
            if dir == Point::default() {
                return ActionResult::success_turns(turns);
            }
            let entity = &mut state.board.entities[eid];
            let target = entity.pos + dir;
            entity.dir = dir;
            if state.board.get_status(target) == Status::Free {
                state.board.move_entity(eid, target);
                return ActionResult::success_turns(turns);
            }
            ActionResult::failure()
        }
    }
}

fn get_direction(ch: char) -> Option<Point> {
    match ch {
        'h' => Some(Point(-1,  0)),
        'j' => Some(Point( 0,  1)),
        'k' => Some(Point( 0, -1)),
        'l' => Some(Point( 1,  0)),
        'y' => Some(Point(-1, -1)),
        'u' => Some(Point( 1, -1)),
        'b' => Some(Point(-1,  1)),
        'n' => Some(Point( 1,  1)),
        '.' => Some(Point( 0,  0)),
        _ => None,
    }
}

fn process_input(state: &mut State, input: Input) {
    let player = &state.board.entities[state.player];
    let (known, eid) = (&*player.known, player.id().eid());

    let tab = input == Input::Char('\t') || input == Input::BackTab;
    let enter = input == Input::Char('\n') || input == Input::Char('.');

    let name = |me: &PokemonIndividualData| -> &str { me.species.name };

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
                let target = init_summon_target(player, TargetData::Summon(choice));
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
        let focus = known.focus.and_then(|x| known.get_view_of(x));
        if let Some(target) = focus && target.age == 0 { return target.pos; }
        let rival = rivals(player).into_iter().next();
        if let Some(rival) = rival { return rival.0.pos; }
        source
    };

    let get_updated_target = |player: &Trainer, target: Point| -> Option<Point> {
        if tab {
            let old_eid = known.get_entity_at(target).map(|x| x.eid);
            let new_eid = apply_tab(player, old_eid, false);
            return Some(known.get_view_of(new_eid?)?.pos);
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
    }

    let dir = if let Input::Char(x) = input { get_direction(x) } else { None };
    state.input = dir.map(|x| Action::Move(MoveData { dir: x, turns: 1. }))
                     .unwrap_or(Action::WaitForInput);
}

fn update_focus(state: &mut State) {
    let known = &*state.board.entities[state.player].known;
    let focus = match &state.target {
        Some(x) => known.get_entity_at(x.target),
        None => known.focus.and_then(|x| known.get_view_of(x)),
    };
    if let Some(entity) = focus && can_target(entity) {
        let floor = Tile::get('.');
        let (player, pos, dir) = (entity.player, entity.pos, entity.dir);
        let lookup = |p: Point| known.get_cell(p).map(|x| x.tile).unwrap_or(floor);
        state.focus.vision.compute(&VisionArgs { player, pos, dir }, lookup);
        state.focus.active = true;
    } else {
        state.focus.active = false;
    }
}

fn update_state(state: &mut State) {
    let player_alive = |state: &State| {
        !state.board.entities[state.player].removed
    };

    let needs_input = |state: &State| {
        if !matches!(state.input, Action::WaitForInput) { return false; }
        let active = state.board.get_active_entity();
        if active != state.player.eid() { return false; }
        player_alive(state)
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

    while player_alive(state) {
        let eid = state.board.get_active_entity();
        let entity = &state.board.entities[eid];
        if !turn_ready(entity) {
            state.board.advance_entity();
            continue;
        } else if needs_input(state) {
            break;
        }

        state.board.update_known(eid);
        let player = eid == state.player.eid();
        let entity = &state.board.entities[eid];
        let action = plan(&entity, &mut state.input);
        let result = act(state, eid, action);
        update = true;

        if player && !result.success { break; }
        if let Some(x) = state.board.entities.get_mut(eid) { wait(x, &result); }
    }

    if update {
        state.board.update_known(state.player.eid());
    }
    update_focus(state);
}

//////////////////////////////////////////////////////////////////////////////

// UI

const UI_COL_SPACE: i32 = 2;
const UI_ROW_SPACE: i32 = 1;
const UI_KEY_SPACE: i32 = 4;

const UI_LOG_SIZE: i32 = 4;
const UI_MAP_SIZE_X: i32 = 43;
const UI_MAP_SIZE_Y: i32 = 43;
const UI_CHOICE_SIZE: i32 = 40;
const UI_STATUS_SIZE: i32 = 30;
const UI_COLOR: i32 = 0x430;

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
    fn render_bar(&self, buffer: &mut Buffer, width: i32, pos: Point, text: &str) {
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

        self.render_bar(buffer, ml, Point(0, 0), "Party");
        self.render_bar(buffer, ml, Point(ml + mw, 0), "Target");
        self.render_bar(buffer, ml, Point(ml + mw, rt), "Wild Pokemon");
        self.render_bar(buffer, ml, Point(0, mh - 1), "Log");
        self.render_bar(buffer, ml, Point(ml + mw, mh - 1), "");
        self.render_bar(buffer, uw, Point(0, uh - 1), "");

        self.render_box(buffer, &self.map);
    }

    fn render_key(key: char) -> String { format!("[{}] ", key) }
}

//////////////////////////////////////////////////////////////////////////////

// State

struct Focus {
    active: bool,
    vision: Vision,
}

enum TargetData {
    FarLook,
    Summon(usize),
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
    player: TID,
    ui: UI,
}

impl State {
    pub fn new() -> Self {
        let size = Point(WORLD_SIZE, WORLD_SIZE);
        let mut board = Board::new(size);
        let pos = Point(size.0 / 2, size.1 / 2);

        loop {
            mapgen(&mut board.map);
            if !board.map.get(pos).blocked() { break; }
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

        let rng = |n: i32| random::<i32>().rem_euclid(n);
        let pos = |board: &Board| {
            for _ in 0..100 {
                let p = Point(rng(size.0), rng(size.1));
                if let Status::Free = board.get_status(p) { return Some(p); }
            }
            None
        };
        let options = ["Pidgey", "Ratatta"];
        for i in 0..20 {
            if let Some(pos) = pos(&board) {
                let index = if i % 4 == 0 { 1 } else { 0 };
                let (dir, species) = (*sample(&DIRECTIONS), options[index]);
                board.add_pokemon(&PokemonArgs { pos, dir, species, trainer: None });
            }
        }

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
            player: tid,
        }
    }

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
        let (known, pos) = (&*player.known, player.pos);
        let offset = pos - Point(UI_MAP_SIZE_X / 2, UI_MAP_SIZE_Y / 2);
        let unseen = Glyph::wide(' ');

        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let point = Point(x, y);
                let glyph = match known.get_cell(point + offset) {
                    Some(x) => if x.age > 0 {
                        x.tile.glyph.with_fg(Color::gray())
                    } else {
                        known.get_entity(x).map(|y| y.glyph).unwrap_or(x.tile.glyph)
                    }
                    None => unseen
                };
                buffer.set(self.ui.map.root + Point(2 * x, y), glyph);
            }
        }

        let set = |buffer: &mut Buffer, p: Point, glyph: Glyph| {
            let Point(x, y) = p - offset;
            if !(0 <= x && x < UI_MAP_SIZE_X) { return; }
            if !(0 <= y && y < UI_MAP_SIZE_Y) { return; }
            let point = self.ui.map.root + Point(2 * x, y);
            buffer.set(point, glyph);
        };

        let recolor = |buffer: &mut Buffer, p: Point,
                       fg: Option<Color>, bg: Option<Color>| {
            let Point(x, y) = p - offset;
            if !(0 <= x && x < UI_MAP_SIZE_X) { return; }
            if !(0 <= y && y < UI_MAP_SIZE_Y) { return; }
            let point = self.ui.map.root + Point(2 * x, y);

            let mut glyph = buffer.get(point);
            if let Some(fg) = fg {
                glyph = glyph.with_fg(fg);
            }
            if glyph.bg() == Color::default() && let Some(bg) = bg {
                glyph = glyph.with_bg(bg);
            }
            buffer.set(point, glyph);
        };

        if let Some(target) = &self.target {
            let color = if target.error.is_empty() { 0x440 } else { 0x400 };
            recolor(buffer, target.source, Some(Color::black()), Some(0x222.into()));
            recolor(buffer, target.target, Some(Color::black()), Some(color.into()));

            let frame = target.frame >> 1;
            let count = UI_TARGET_FRAMES >> 1;
            let limit = target.path.len() as i32 - 1;
            let ch = self.ray_character(target.source, target.target);
            for i in 0..limit {
                if !((i + count - frame) % count < 2) { continue; }
                let (point, okay) = target.path[i as usize];
                let color = if okay { 0x440 } else { 0x400 };
                set(buffer, point, Glyph::wdfg(ch, color));
            }
        }

        if self.focus.active {
            let aware = self.focus.vision.get_visibility_at(player.pos) >= 0;
            let color = if aware { 0x100 } else { 0x000 };

            for (i, point) in self.focus.vision.points_seen.iter().enumerate() {
                if i == 0 {
                    recolor(buffer, *point, Some(Color::black()), Some(0x222.into()));
                } else {
                    recolor(buffer, *point, None, Some(color.into()));
                }
            }
        }

        self.render_log(&mut Slice::new(buffer, self.ui.log));
        self.render_rivals(player, &mut Slice::new(buffer, self.ui.rivals));
        self.render_status(player, &mut Slice::new(buffer, self.ui.status));
        self.render_target(player, &mut Slice::new(buffer, self.ui.target));

        if let Some(x) = self.choice {
            self.ui.render_dialog(buffer, &self.ui.choice);
            self.render_choice(player, x, &mut Slice::new(buffer, self.ui.choice));
        }
    }

    fn render_log(&self, slice: &mut Slice) {
        for line in &self.board.logs {
            slice.set_fg(Some(line.color)).write_str(&line.text);
        }
    }

    fn render_rivals(&self, trainer: &Trainer, slice: &mut Slice) {
        let mut rivals = rivals(trainer);
        rivals.truncate(max(slice.size().1, 0) as usize / 2);

        for (entity, pokemon) in rivals {
            let PokemonSpeciesData { glyph, name, .. } = pokemon.species;
            let (hp, hp_color) = (pokemon.hp, self.hp_color(pokemon.hp));
            let hp_text = format!("{}%", max((100.0 * hp).floor() as i32, 1));
            let (sn, sh) = (name.chars().count(), hp_text.chars().count());
            let ss = max(16 - sn as i32 - sh as i32, 0) as usize;

            slice.newline();
            slice.write_chr(*glyph).space().write_str(name);
            slice.spaces(ss).set_fg(Some(hp_color)).write_str(&hp_text).newline();

            let targeted = match &self.target {
                Some(x) => x.target == entity.pos,
                None => trainer.known.focus == Some(entity.eid),
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

    fn render_status(&self, trainer: &Trainer, slice: &mut Slice) {
        if let Some(entity) = trainer.known.get_view_of(trainer.id().eid()) {
            self.render_entity(Some(PLAYER_KEY), None, entity, slice);
        }
        for (_, key) in SUMMON_KEYS.iter().enumerate() {
            self.render_empty_option(*key, 0, slice);
        }
    }

    fn render_target(&self, trainer: &Trainer, slice: &mut Slice) {
        let known = &*trainer.known;
        let name = |me: &PokemonIndividualData| -> &str { me.species.name };

        if self.target.is_none() && known.focus.is_none() {
            let fg = Some(0x111.into());
            slice.newline();
            slice.set_fg(fg).write_str("No target selected.").newline();
            slice.newline();
            slice.set_fg(fg).write_str("[x] examine your surroundings").newline();
            return;
        }

        let (cell, entity, header, seen) = match &self.target {
            Some(x) => {
                let cell = known.get_cell(x.target);
                let seen = cell.map(|x| x.age == 0).unwrap_or(false);
                let entity = cell.and_then(|x| known.get_entity(x));
                let header = match &x.data {
                    TargetData::FarLook => "Examining...".into(),
                    TargetData::Summon(x) => {
                        let name = match &trainer.data.pokemon[*x] {
                            PokemonEdge::In(y) => name(y),
                            PokemonEdge::Out(y) =>
                                name(&self.board.entities[*y].data.me),
                        };
                        format!("Sending out {}...", name)
                    }
                };
                (cell, entity, header, seen)
            }
            None => {
                let entity = known.focus.and_then(|x| known.get_view_of(x));
                let seen = entity.map(|x| x.age == 0).unwrap_or(false);
                let cell = entity.and_then(|x| known.get_cell(x.pos));
                let header = if seen {
                    "Last target:"
                } else {
                    "Last target: (remembered)"
                }.into();
                (cell, entity, header, seen)
            },
        };

        let fg = if self.target.is_some() || seen { None } else { Some(0x111.into()) };
        let text = if entity.is_some() {
            if seen { "Standing on: " } else { "Stood on: " }
        } else {
            if seen { "You see: " } else { "You saw: " }
        };

        slice.newline();
        slice.set_fg(fg).write_str(&header).newline();

        if let Some(x) = entity {
            self.render_entity(None, fg, x, slice);
        } else {
            slice.newline();
        }

        slice.set_fg(fg).write_str(text);
        if let Some(CellKnowledge { tile: x, .. }) = cell {
            slice.write_chr(x.glyph).space();
            slice.write_chr('(').write_str(x.description).write_chr(')').newline();
        } else {
            slice.write_str("(unseen location)").newline();
        }
    }

    fn render_choice(&self, trainer: &Trainer, choice: i32, slice: &mut Slice) {
        let options = &trainer.data.pokemon;
        for (i, key) in PARTY_KEYS.iter().enumerate() {
            let selected = choice == i as i32;
            match if i < options.len() { Some(&options[i]) } else { None } {
                Some(PokemonEdge::Out(x)) => {
                    let pokemon = &self.board.entities[*x];
                    let (me, pp) = (&*pokemon.data.me, get_pp(pokemon.base()));
                    self.render_option(*key, 1, selected, me, pp, slice);
                },
                Some(PokemonEdge::In(x)) =>
                    self.render_option(*key, 0, selected, x, 1.0, slice),
                None => self.render_empty_option(*key, UI_COL_SPACE + 1, slice),
            }
        }
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
        let (hp_color, pp_color) = (self.hp_color(hp), 0x123.into());
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
                let (hp_color, pp_color) = (self.hp_color(x.hp), 0x123.into());
                slice.set_fg(fg).write_str(&prefix).write_str(x.species.name).newline();
                status_bar_line("HP: ", x.hp, hp_color, slice);
                status_bar_line("PP: ", x.pp, pp_color, slice);
            }
            EntityView::Trainer(x) => {
                let hp_color = self.hp_color(x.hp);
                let name = if entity.player { "You" } else { &x.name };
                slice.set_fg(fg).write_str(&prefix).write_str(name).newline();
                status_bar_line("HP: ", x.hp, hp_color, slice);
            }
        }
        slice.newline();
    }

    fn render_bar<'a>(&self, value: f64, color: Color, width: i32, slice: &mut Slice) {
        let count = if value > 0. { max(1, (width as f64 * value) as i32) } else { 0 };
        let glyph = Glyph::chfg('=', color);

        slice.write_chr('[');
        for _ in 0..count { slice.write_chr(glyph); }
        for _ in count..width { slice.write_chr(' '); }
        slice.write_chr(']');
    }

    fn ray_character(&self, source: Point, target: Point) -> char {
        let Point(x, y) = source - target;
        let (ax, ay) = (x.abs(), y.abs());
        if ax > 2 * ay { return '-'; }
        if ay > 2 * ax { return '|'; }
        if (x > 0) == (y > 0) { '\\' } else { '/' }
    }

    fn hp_color(&self, hp: f64) -> Color {
        (if hp <= 0.25 { 0x300 } else if hp <= 0.50 { 0x330 } else { 0x020 }).into()
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
