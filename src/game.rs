use std::cmp::{max, min};
use std::f64::consts::TAU;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

use lazy_static::lazy_static;
use rand::random;

use crate::assert_eq_size;
use crate::base::{Buffer, Color, Glyph, Slice};
use crate::base::{FOV, HashMap, LOS, Matrix, Point, Rect, clamp};
use crate::entity::{AIState, Assess, Fight, Flight, Wander};
use crate::entity::{Pokemon, PokemonEdge};
use crate::entity::{PokemonIndividualData as PID, PokemonSpeciesData as PSD};
use crate::entity::{Entity, EID, ET, Token, Trainer, WeakEntity};
use crate::pathing::{AStar, DijkstraMap, DijkstraMode};
use crate::pathing::{DIRECTIONS, BFS, BFSResult, Status};

//////////////////////////////////////////////////////////////////////////////

// Constants

const MAX_MEMORY: i32 = 1024;
const WORLD_SIZE: i32 = 100;

const ASSESS_TIME: i32 = 17;
const ASSESS_ANGLE: f64 = TAU / 3.;

const MIN_FLIGHT_TIME: i32 = 8;
const MAX_FLIGHT_TIME: i32 = 64;
const MAX_FOLLOW_TIME: i32 = 64;

const FOV_RADIUS_SMALL: i32 = 12;
const FOV_RADIUS_LARGE: i32 = 21;
const VISION_ANGLE: f64 = TAU / 3.;
const VISION_RADIUS: i32 = 3;

const MOVE_TIMER: i32 = 960;
const TURN_TIMER: i32 = 120;

const WANDER_TURNS: f64 = 3.;

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

struct PokemonView {
    species: &'static PSD,
    trainer: bool,
    hp: f64,
    pp: f64,
}

struct TrainerView {
    hp: f64,
    name: Rc<str>,
}

enum EntityView {
    Pokemon(PokemonView),
    Trainer(TrainerView),
}

impl Default for EntityView {
    fn default() -> Self { Self::Trainer(TrainerView { hp: 0., name: "".into() }) }
}

fn get_hp(me: &PID) -> f64 {
    me.cur_hp as f64 / max(me.max_hp, 1) as f64
}

fn get_pp(e: &Entity, t: &Token) -> f64 {
    1. - clamp(e.base(t).move_timer as f64 / MOVE_TIMER as f64, 0., 1.)
}

fn get_view(e: &Entity, t: &Token) -> EntityView {
    match e.test(t) {
        ET::Pokemon(x) => {
            let species = x.data(t).me.species;
            let trainer = x.data(t).me.trainer.is_some();
            let (hp, pp) = (get_hp(&x.data(t).me), get_pp(e, t));
            EntityView::Pokemon(PokemonView { species, trainer, hp, pp })
        }
        ET::Trainer(x) => {
            let data = &x.data(t);
            let name = data.name.clone();
            let hp = data.cur_hp as f64 / max(data.max_hp, 1) as f64;
            EntityView::Trainer(TrainerView { hp, name })
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

#[derive(Clone, Copy, Eq, PartialEq)] struct EntityIndex(i32);

struct CellKnowledge {
    age: i32,
    eid: Option<EntityIndex>,
    tile: &'static Tile,
    visibility: i32,
}
assert_eq_size!(CellKnowledge, 24);

#[derive(Default)]
struct EntityKnowledge {
    age: i32,
    pos: Point,
    dir: Point,
    glyph: Glyph,
    moved: bool,
    rival: bool,
    friend: bool,
    player: bool,
    weak: WeakEntity,
    view: EntityView,
}
assert_eq_size!(EntityKnowledge, 72);

#[derive(Default)]
struct Knowledge {
    _map: HashMap<Point, CellKnowledge>,
    _entity_by_id: HashMap<EID, EntityIndex>,
    entities: Vec<EntityKnowledge>,
    focus: Option<EID>,
}

impl Knowledge {
    // Reads

    fn get_cell(&self, p: Point) -> Option<&CellKnowledge> { self._map.get(&p) }

    fn get_entity(&self, cell: &CellKnowledge) -> Option<&EntityKnowledge> {
        cell.eid.map(|eid| self._entity_raw(eid))
    }

    fn get_entity_at(&self, p: Point) -> Option<&EntityKnowledge> {
        self.get_entity(self.get_cell(p)?)
    }

    fn get_status(&self, p: Point) -> Option<Status> {
        self.get_cell(p).map(|x| {
            if x.eid.is_some() { return Status::Occupied; }
            if x.tile.blocked() { Status::Blocked } else { Status::Free }
        })
    }

    fn get_view_of(&self, eid: EID) -> Option<&EntityKnowledge> {
        self._entity_by_id.get(&eid).map(|x| self._entity_raw(*x))
    }

    fn can_see_now(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| x.age == 0).unwrap_or(false)
    }

    fn remembers(&self, p: Point) -> bool {
        self._map.contains_key(&p)
    }

    fn blocked(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| x.tile.blocked()).unwrap_or(false)
    }

    fn unblocked(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| !x.tile.blocked()).unwrap_or(false)
    }

    fn _entity_raw(&self, eid: EntityIndex) -> &EntityKnowledge {
        &self.entities[eid.0 as usize]
    }

    fn _entity_mut(&mut self, eid: EntityIndex) -> &mut EntityKnowledge {
        &mut self.entities[eid.0 as usize]
    }

    // Writes

    fn update(&mut self, board: &BaseBoard, e: &Entity, t: &Token, vision: &Vision) {
        let my_entity = e.base(t);
        let my_species = species(e, t);
        let my_trainer = trainer(e, t);
        self._forget(my_entity.player);
        let offset = vision.offset - my_entity.pos;

        for point in &vision.cells_seen {
            let visibility = vision.visibility.get(*point + offset);
            assert!(visibility >= 0);

            let eid = (|| {
                let entity = board.get_entity_at(*point)?;
                let eid = *self._entity_by_id.entry(entity.id()).and_modify(|x| {
                    let existing = &mut self.entities[x.0 as usize];
                    if !existing.moved && existing.pos != *point {
                        self._map.entry(existing.pos).and_modify(|y| {
                            assert!(y.eid == Some(*x));
                            y.eid = None;
                        });
                    };
                }).or_insert_with(|| {
                    self.entities.push(EntityKnowledge::default());
                    self.entities.last_mut().unwrap().weak = entity.into();
                    EntityIndex(self.entities.len() as i32 - 1)
                });

                let species = species(entity, t);
                let entry = self._entity_mut(eid);

                entry.age = 0;
                entry.pos = *point;
                entry.dir = entity.base(t).dir;
                entry.moved = false;
                entry.glyph = entity.base(t).glyph;
                entry.rival = species.is_some() && species != my_species;
                entry.friend = trainer(entity, t) == my_trainer;
                entry.player = entity.base(t).player;
                entry.view = get_view(entity, t);

                Some(eid)
            })();

            let tile = board.map.get(*point);
            let cell = CellKnowledge { age: 0, eid, tile, visibility };
            let prev = self._map.insert(*point, cell);
            if let Some(x) = prev && x.eid != eid && let Some(other) = x.eid {
                self._mark_entity_moved(other, *point);
            }
        }
    }

    fn _forget(&mut self, player: bool) {
        if player {
            self._map.iter_mut().for_each(|x| x.1.age = 1);
            self.entities.iter_mut().for_each(|x| x.age = 1);
            return;
        }

        let mut removed: Vec<(Point, Option<EntityIndex>)> = vec![];
        for (key, val) in self._map.iter_mut() {
            val.age += 1;
            if val.age >= MAX_MEMORY { removed.push((*key, val.eid)); }
        }
        removed.iter().for_each(|x| {
            self._map.remove(&x.0);
            if let Some(eid) = x.1 { self._mark_entity_moved(eid, x.0); }
        });

        let mut removed: Vec<EntityIndex> = vec![];
        for (i, val) in self.entities.iter_mut().enumerate() {
            val.age += 1;
            if val.age >= MAX_MEMORY { removed.push(EntityIndex(i as i32)); }
        }
        removed.iter().rev().for_each(|x| { self._remove_entity(*x); });
    }

    fn _mark_entity_moved(&mut self, eid: EntityIndex, pos: Point) {
        let entity = self._entity_mut(eid);
        assert!(entity.pos == pos);
        entity.moved = true;
    }

    fn _remove_entity(&mut self, eid: EntityIndex) {
        let id = self._entity_raw(eid).weak.id();
        let popped = self.entities.pop().unwrap();
        let popped_eid = EntityIndex(self.entities.len() as i32);
        let swap = eid != popped_eid;

        if !popped.moved {
            let cell = self._map.get_mut(&popped.pos).unwrap();
            assert!(cell.eid == Some(popped_eid));
            cell.eid = if swap { Some(eid) } else { None };
        }

        let deleted = if swap {
            self._entity_by_id.remove(&id);
            self._entity_by_id.insert(popped.weak.id(), eid)
        } else {
            self._entity_by_id.remove(&popped.weak.id())
        };
        assert!(deleted == Some(popped_eid));

        if self.focus == Some(id) { self.focus = None; }

        if swap { *self._entity_mut(eid) = popped; }
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

struct LogLine {
    color: Color,
    menu: bool,
    text: String,
}

struct Board {
    base: BaseBoard,
    fov_large: FOV,
    fov_small: FOV,
    vision: Vision,
    known: HashMap<EID, Box<Knowledge>>,
    logs: Vec<LogLine>,
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

    fn compute_vision<F: Fn(Point) -> &'static Tile>(
            &self, player: bool, pos: Point, dir: Point, f: F,
            fov: &mut FOV, vision: &mut Vision) {
        vision.visibility.fill(-1);
        vision.cells_seen.clear();

        let blocked = |p: Point, prev: Option<&Point>| {
            if !player && !Self::in_vision_cone(dir, p) { return true; }

            let lookup = p + vision.offset;
            let cached = vision.visibility.get(lookup);

            let visibility = (|| {
                // These constant values come from Point.distanceNethack.
                // They are chosen such that, in a field of tall grass, we'll
                // only see cells at a distanceNethack <= kVisionRadius.
                if prev.is_none() { return 100 * (VISION_RADIUS + 1) - 95 - 46 - 25; }

                let tile = f(p + pos);
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
            logs: vec![],
        }
    }

    // Logging

    fn log<S: Into<String>>(&mut self, text: S) {
        self.log_color(text, Color::default());
    }

    fn log_color<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        self.logs.push(LogLine { color: color.into(), menu: false, text: text.into() });
        if self.logs.len() as i32 > UI_LOG_SIZE { self.logs.remove(0); }
    }

    fn log_menu<S: Into<String>, T: Into<Color>>(&mut self, text: S, color: T) {
        if self.logs.last().map(|x| x.menu).unwrap_or(false) { self.logs.pop(); }
        self.logs.push(LogLine { color: color.into(), menu: true, text: text.into() });
        if self.logs.len() as i32 > UI_LOG_SIZE { self.logs.remove(0); }
    }

    // Knowledge

    fn get_known(&self, e: &Entity) -> &Knowledge {
        self.known.get(&e.id()).unwrap()
    }

    fn set_focus(&mut self, e: &Entity, focus: Option<EID>) {
        self.known.get_mut(&e.id()).unwrap().focus = focus;
    }

    fn update_known(&mut self, e: &Entity, t: &Token) -> &Knowledge {
        let base = &self.base;
        let (player, pos, dir) = (e.base(t).player, e.base(t).pos, e.base(t).dir);
        let fov = if player { &mut self.fov_large } else { &mut self.fov_small };
        base.compute_vision(player, pos, dir, |p| base.map.get(p), fov, &mut self.vision);
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
        assert!(&existing == e);
        let collider = self.entity_at_pos.insert(to, existing);
        assert!(collider.is_none());
        e.base_mut(t).pos = to;
    }

    fn remove_entity(&mut self, e: &Entity, t: &mut Token) {
        // Delete hard edges from this entity to others.
        match e.test(t) {
            ET::Pokemon(x) => {
                if let Some(trainer) = x.data(t).me.trainer.clone() {
                    let me = x.data(t).me.clone();
                    let trainer = trainer.data_mut(t);
                    let index = trainer.pokemon.iter().position(
                        |y| if let PokemonEdge::Out(z) = y && z == x { true } else { false });
                    trainer.pokemon[index.unwrap()] = PokemonEdge::In(me);
                }
            }
            ET::Trainer(x) => {
                let pokemon = x.data(t).pokemon.iter().filter_map(|y| match y {
                    PokemonEdge::Out(y) => Some(y.clone()),
                    PokemonEdge::In(_) => None,
                }).collect::<Vec<_>>();
                pokemon.iter().for_each(|y| y.data_mut(t).me.trainer = None);
            }
        }

        // The player is just tagged "removed", so we always have an entity.
        let entity = e.base_mut(t);
        entity.removed = true;
        if entity.player { return; }

        // Remove entities other than the player.
        let existing = self.entity_at_pos.remove(&entity.pos).unwrap();
        assert!(&existing == e);
        let index = self.entities.iter().position(|x| x == e).unwrap();
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

// Targeting UI

fn check_follower_square(known: &Knowledge, leader: &Entity, t: &Token,
                         target: Point, ignore_occupant: bool) -> bool {
    let free = match known.get_status(target).unwrap_or(Status::Blocked) {
        Status::Free => true,
        Status::Blocked => false,
        Status::Occupied => ignore_occupant,
    };
    if !free { return false }

    let source = leader.base(t).pos;
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

fn init_summon_target(data: TargetData, state: &State) -> Box<Target> {
    let known = state.board.get_known(&state.player);
    let source = state.player.base(&state.t).pos;
    let mut target = init_target(data, source, source);

    let mut okay = |p: Point| {
        if !check_follower_square(known, &state.player, &state.t, p, false) { return false; }
        update_target(state, &mut target, p);
        target.error.is_empty()
    };

    let best = source + state.player.base(&state.t).dir.scale(2);
    let next = source + state.player.base(&state.t).dir.scale(1);
    if okay(best) { return target; }
    if okay(next) { return target; }

    let mut options: Vec<Point> = vec![];
    for dx in -2..=2 {
        for dy in -2..=2 {
            let pos = source + Point(dx, dy);
            if okay(pos) { options.push(pos); }
        }
    }

    options.sort_by_cached_key(|x| (*x - best).len_l2_squared());
    let update = options.into_iter().next().unwrap_or(source);
    update_target(state, &mut target, update);
    target
}

fn outside_map(state: &State, point: Point) -> bool {
    let delta = point - state.player.base(&state.t).pos;
    let limit_x = (UI_MAP_SIZE_X - 1) / 2;
    let limit_y = (UI_MAP_SIZE_Y - 1) / 2;
    delta.0.abs() > limit_x || delta.1.abs() > limit_y
}

fn update_target(state: &State, target: &mut Target, update: Point) {
    let mut okay_until = 0;
    let known = state.board.get_known(&state.player);
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
    let known = state.board.get_known(&state.player);
    let target = state.target.as_ref()?;
    let entity = known.get_entity_at(target.target);

    match &target.data {
        TargetData::FarLook => {
            let entity = entity?;
            if can_target(entity) { Some(entity.weak.id()) } else { None }
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

fn rivals<'a>(known: &'a Knowledge, e: &Trainer, t: &Token)
        -> Vec<(&'a EntityKnowledge, &'a PokemonView)> {
    let mut rivals = vec![];
    for entity in &known.entities {
        if entity.age > 0 { continue; }
        if let EntityView::Pokemon(x) = &entity.view && !x.trainer {
            rivals.push((entity, x));
        }
    }
    let pos = e.base(t).pos;
    rivals.sort_by_cached_key(
        |(x, _)| ((x.pos - pos).len_l2_squared(), x.pos.0, x.pos.1));
    rivals
}

fn species(e: &Entity, t: &Token) -> Option<&'static PSD> {
    match e.test(t) {
        ET::Pokemon(x) => Some(x.data(t).me.species),
        ET::Trainer(_) => None,
    }
}

fn trainer(e: &Entity, t: &Token) -> Option<Trainer> {
    match e.test(t) {
        ET::Pokemon(x) => x.data(t).me.trainer.clone(),
        ET::Trainer(x) => Some(x.clone()),
    }
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

fn assess(_: &Knowledge, e: &Entity, t: &Token, s: &mut Assess) -> Action {
    s.time -= 1;
    let a = 1 * ASSESS_TIME / 4;
    let b = 2 * ASSESS_TIME / 4;
    let c = 3 * ASSESS_TIME / 4;
    let depth = if s.time < a {
        -s.time
    } else if s.time < c {
        s.time - 2 * a
    } else {
        -s.time + 2 * c - 2 * a
    };
    let scale = 1000;
    let angle = ASSESS_ANGLE * depth as f64 / b as f64;
    let (sin, cos) = (angle.sin(), angle.cos());
    let Point(dx, dy) = s.target - e.base(t).pos;
    let rx = (cos * (scale * dx) as f64) + (sin * (scale * dy) as f64);
    let ry = (cos * (scale * dy) as f64) - (sin * (scale * dx) as f64);
    Action::Look(Point(rx as i32, ry as i32))
}

fn flight(known: &Knowledge, e: &Entity, t: &Token,
          source: Point) -> Option<Action> {
    let mut map: HashMap<Point, i32> = HashMap::default();
    map.insert(source, 0);

    let limit = 1024;
    let pos = e.base(t).pos;

    let check = |p: Point| {
        if p == pos { return Status::Free; }
        known.get_status(p).unwrap_or(Status::Blocked)
    };
    DijkstraMap(DijkstraMode::Expand(limit), check, 1, &mut map);

    for (pos, val) in map.iter_mut() {
        let frontier = DIRECTIONS.iter().any(|x| !known.remembers(*pos + *x));
        if frontier { *val += FOV_RADIUS_SMALL; }
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

fn plan_pokemon(known: &Knowledge, e: &Pokemon, t: &Token) -> Action {
    let mut ai = e.data(t).ai.take().unwrap_or(Box::default());
    let prey = e.data(t).me.species.name == "Pidgey";
    let mut targets: Vec<(i32, Point)> = vec![];
    for entity in known.entities.iter() {
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
        let (switch, target) = (MIN_FLIGHT_TIME, e.base(t).pos + e.base(t).dir);
        *ai = AIState::Assess(Assess { switch, target, time: ASSESS_TIME })
    }
    if let AIState::Assess(x) = ai.as_ref() && x.time <= 0 {
        *ai = AIState::Wander(Wander { time: random::<i32>().rem_euclid(16) });
    }

    let mut replacement = None;
    let action = match ai.as_mut() {
        AIState::Fight(x) => if x.age == 0 {
            Action::Look(x.target - e.base(t).pos)
        } else {
            explore_near(known, e, t, x.target, x.age, 1.)
        }
        AIState::Flight(x) => flight(known, e, t, x.target).unwrap_or_else(||{
            let (target, time) = (x.target, ASSESS_TIME);
            let mut x = Assess { switch: min(x.age + 1, x.switch), target, time };
            let result = assess(known, e, t, &mut x);
            replacement = Some(AIState::Assess(x));
            result
        }),
        AIState::Wander(x) => {
            x.time -= 1;
            explore_near(known, e, t, e.base(t).pos, 9999, WANDER_TURNS)
        }
        AIState::Assess(ref mut x) => assess(known, e, t, x),
    };
    if let Some(x) = replacement { *ai = x; }
    e.data(t).ai.set(Some(ai));
    action
}

fn plan(known: &Knowledge, e: &Entity, t: &Token, input: &mut Action) -> Action {
    let entity = e.base(t);
    if entity.player {
        return std::mem::replace(input, Action::WaitForInput);
    }
    match e.test(t) {
        ET::Pokemon(x) => plan_pokemon(known, x, t),
        ET::Trainer(_) => Action::Idle,
    }
}

fn act(state: &mut State, e: &Entity, action: Action) -> ActionResult {
    match action {
        Action::Idle => ActionResult::success(),
        Action::WaitForInput => ActionResult::failure(),
        Action::Look(dir) => {
            e.base_mut(&mut state.t).dir = dir;
            ActionResult::success()
        }
        Action::Move(MoveData { dir ,turns }) => {
            if dir == Point::default() {
                return ActionResult::success_turns(turns);
            }
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
    let tab = input == Input::Char('\t') || input == Input::BackTab;
    let enter = input == Input::Char('\n') || input == Input::Char('.');
    let name = |pid: &PID| -> &str { pid.species.name };

    if let Some(x) = &mut state.choice {
        let choice = if enter {
            Some(*x as usize)
        } else {
            PARTY_KEYS.iter().position(|x| input == Input::Char(*x))
        };
        let count = state.player.data(&state.t).pokemon.len() as i32;
        let dir = if let Input::Char(x) = input { get_direction(x) } else { None };
        if let Some(dir) = dir && dir.0 == 0 {
            *x += dir.1;
            if *x >= count { *x = 0; }
            if *x < 0 { *x = max(count - 1, 0); }
        } else if let Some(choice) = choice {
            let pokemon = &state.player.data(&state.t).pokemon;
            if choice >= pokemon.len() {
                let error = format!("You are only carrying {} Pokemon!", pokemon.len());
                state.board.log_menu(error, 0x422);
            } else if let PokemonEdge::Out(x) = &pokemon[choice] {
                let error = format!("{} is already out!", name(&x.data(&state.t).me));
                state.board.log_menu(error, 0x422);
            } else if let PokemonEdge::In(x) = &pokemon[choice] && x.cur_hp == 0 {
                let error = format!("{} has no strength left!", name(x));
                state.board.log_menu(error, 0x422);
            } else if let PokemonEdge::In(x) = &pokemon[choice] {
                let message = format!("Choose where to send out {}:", name(x));
                state.board.log_menu(message, 0x234);
                let target = init_summon_target(TargetData::Summon(choice), state);
                state.target = Some(target);
                state.choice = None;
            }
        } else if input == Input::Escape {
            state.board.log_menu("Canceled.", 0x234);
            state.choice = None;
        }
        return;
    }

    let get_initial_target = |state: &State, source: Point| -> Point {
        let known = state.board.get_known(&state.player);
        let focus = known.focus.and_then(|x| known.get_view_of(x));
        if let Some(target) = focus && target.age == 0 { return target.pos; }
        let rival = rivals(known, &state.player, &state.t).into_iter().next();
        if let Some(rival) = rival { return rival.0.pos; }
        source
    };

    let get_tabbed_target = |prev: Option<EID>, tab_off: bool| -> Option<EID> {
        let known = state.board.get_known(&state.player);
        let rivals = rivals(known, &state.player, &state.t);
        if rivals.is_empty() { return None; }

        let t = input == Input::Char('\t');
        let n = rivals.len() + if tab_off { 1 } else { 0 };

        let next = prev.and_then(|x| rivals.iter().position(|y| y.0.weak.id() == x));
        let start = next.or_else(|| if tab_off { Some(n - 1) } else { None });
        let index = start.map(|x| if t { x + n + 1 } else { x + n - 1 } % n)
                         .unwrap_or_else(|| if t { 0 } else { n - 1 });
        if index < rivals.len() { Some(rivals[index].0.weak.id()) } else { None }
    };

    let get_updated_target = |target: Point| -> Option<Point> {
        if tab {
            let known = state.board.get_known(&state.player);
            let entity = known.get_entity_at(target);
            let old_eid = entity.map(|x| x.weak.id());
            let new_eid = get_tabbed_target(old_eid, false);
            return Some(known.get_view_of(new_eid?)?.pos);
        }

        let ch = if let Input::Char(x) = input { Some(x) } else { None }?;
        let dir = get_direction(ch.to_lowercase().next().unwrap_or(ch))?;
        let scale = if ch.is_uppercase() { 4 } else { 1 };

        let mut prev = target;
        for _ in 0..scale {
            let next = prev + dir;
            if outside_map(state, prev + dir) { break; }
            prev = next;
        }
        Some(prev)
    };

    if let Some(x) = &state.target {
        let update = get_updated_target(x.target);
        if let Some(update) = update && update != x.target {
            let mut target = state.target.take();
            target.as_mut().map(|x| update_target(state, x, update));
            state.target = target;
        } else if enter {
            if x.error.is_empty() {
                let focus = select_valid_target(state);
                state.board.set_focus(&state.player, focus);
                state.target = None;
            } else {
                state.board.log_menu(&x.error, 0x422);
            }
        } else if input == Input::Escape {
            if let TargetData::FarLook = x.data {
                let valid = x.error.is_empty();
                let focus = if valid { select_valid_target(state) } else { None };
                state.board.set_focus(&state.player, focus);
            }
            state.board.log_menu("Canceled.", 0x234);
            state.target = None;
        }
        return;
    }

    if tab {
        let prev = state.board.get_known(&state.player).focus;
        state.board.set_focus(&state.player, get_tabbed_target(prev, true));
    } else if input == Input::Escape {
        state.board.set_focus(&state.player, None);
    }

    if input == Input::Char('x') {
        state.board.log_menu("Use the movement keys to examine a location:", 0x234);
        let source = state.player.base(&state.t).pos;
        let update = get_initial_target(state, source);
        let mut target = init_target(TargetData::FarLook, source, update);
        update_target(state, &mut target, update);
        state.target = Some(target);
        return;
    }

    let index = SUMMON_KEYS.iter().position(|x| input == Input::Char(*x));
    if let Some(i) = index && i >= state.player.data(&state.t).summons.len() {
        state.board.log_menu("Choose a Pokemon to send out with J/K:", 0x234);
        state.choice = Some(0);
        return;
    }

    let dir = if let Input::Char(x) = input { get_direction(x) } else { None };
    state.input = dir.map(|x| Action::Move(MoveData { dir: x, turns: 1. }))
                     .unwrap_or(Action::WaitForInput);
}

fn update_focus(state: &mut State) {
    let known = state.board.get_known(&state.player);
    let focus = match &state.target {
        Some(x) => known.get_entity_at(x.target),
        None => known.focus.and_then(|x| known.get_view_of(x)),
    };
    if let Some(entity) = focus && can_target(entity) {
        let floor = Tile::get('.');
        let Focus { fov, vision, .. } = &mut state.focus;
        let (player, pos, dir) = (entity.player, entity.pos, entity.dir);
        let lookup = |p: Point| known.get_cell(p).map(|x| x.tile).unwrap_or(floor);
        state.board.compute_vision(player, pos, dir, lookup, fov, vision);
        state.focus.target = Some(pos);
    } else {
        state.focus.target = None;
    }
}

fn update_state(state: &mut State) {
    let player_alive = |state: &State| {
        !state.player.base(&state.t).removed
    };

    let needs_input = |state: &State| {
        if !matches!(state.input, Action::WaitForInput) { return false; }
        let active = state.board.get_active_entity();
        if active != state.player.deref() { return false; }
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
        let entity = state.board.get_active_entity();
        if !turn_ready(entity, &state.t) {
            state.board.advance_entity(&mut state.t);
            continue;
        } else if needs_input(state) {
            break;
        }

        let entity = entity.clone();
        let known = state.board.update_known(&entity, &state.t);
        let action = plan(known, &entity, &state.t, &mut state.input);

        let result = act(state, &entity, action);
        if entity.base(&state.t).player && !result.success { break; }
        wait(&entity, &mut state.t, &result);
        update = true;
    }

    if update {
        state.board.update_known(&state.player, &state.t);
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
    fov: FOV,
    vision: Vision,
    target: Option<Point>,
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
    player: Trainer,
    t: Token,
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

        let mut t = unsafe { Token::new() };
        let mut player = Trainer::new(pos, true, "skishore");
        player.register_pokemon(&mut t, "Bulbasaur");
        player.register_pokemon(&mut t, "Charmander");
        player.register_pokemon(&mut t, "Squirtle");
        player.register_pokemon(&mut t, "Eevee");
        player.register_pokemon(&mut t, "Pikachu");
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
        for i in 0..20 {
            if let Some(pos) = pos(&board) {
                let index = if i % 4 == 0 { 1 } else { 0 };
                let (dir, species) = (*sample(&DIRECTIONS), options[index]);
                board.add_entity(&Pokemon::new(pos, dir, species, None), &t);
            }
        }

        Self {
            board,
            focus: Focus {
                fov: FOV::new(FOV_RADIUS_SMALL),
                vision: Vision::default(),
                target: None,
            },
            ui: UI::default(),
            input: Action::WaitForInput,
            inputs: vec![],
            choice: None,
            target: None,
            player,
            t,
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

        let e = &self.player;
        let known = self.board.get_known(e);
        let pos = e.base(&self.t).pos;
        let offset = pos - Point(UI_MAP_SIZE_X / 2, UI_MAP_SIZE_Y / 2);
        let unseen = Glyph::wide(' ');

        for y in 0..UI_MAP_SIZE_Y {
            for x in 0..UI_MAP_SIZE_X {
                let point = Point(x, y);
                let glyph = match known.get_cell(point + offset) {
                    Some(cell) => if cell.age > 0 {
                        cell.tile.glyph.with_fg(Color::gray())
                    } else {
                        known.get_entity(cell).map(|x| x.glyph).unwrap_or(cell.tile.glyph)
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

        if let Some(target) = self.focus.target {
            let adjusted = pos - target + self.focus.vision.offset;
            let aware = self.focus.vision.visibility.get(adjusted) >= 0;
            let color = if aware { 0x100 } else { 0x000 };

            for (i, point) in self.focus.vision.cells_seen.iter().enumerate() {
                if i == 0 {
                    recolor(buffer, *point, Some(Color::black()), Some(0x222.into()));
                } else {
                    recolor(buffer, *point, None, Some(color.into()));
                }
            }
        }

        self.render_log(&mut Slice::new(buffer, self.ui.log));
        self.render_rivals(known, e, &mut Slice::new(buffer, self.ui.rivals));
        self.render_status(known, e, &mut Slice::new(buffer, self.ui.status));
        self.render_target(known, &mut Slice::new(buffer, self.ui.target));

        if let Some(choice) = self.choice {
            self.ui.render_dialog(buffer, &self.ui.choice);
            self.render_choice(e, choice, &mut Slice::new(buffer, self.ui.choice));
        }
    }

    fn render_log(&self, slice: &mut Slice) {
        for line in &self.board.logs {
            slice.set_fg(Some(line.color)).write_str(&line.text);
        }
    }

    fn render_rivals(&self, known: &Knowledge, e: &Trainer, slice: &mut Slice) {
        let mut rivals = rivals(known, e, &self.t);
        rivals.truncate(max(slice.size().1, 0) as usize / 2);

        for (entity, pokemon) in rivals {
            let PSD {glyph, name, ..} = pokemon.species;
            let (hp, hp_color) = (pokemon.hp, self.hp_color(pokemon.hp));
            let hp_text = format!("{}%", max((100.0 * hp).floor() as i32, 1));
            let (sn, sh) = (name.chars().count(), hp_text.chars().count());
            let ss = max(16 - sn as i32 - sh as i32, 0) as usize;

            slice.newline();
            slice.write_chr(*glyph).space().write_str(name);
            slice.spaces(ss).set_fg(Some(hp_color)).write_str(&hp_text).newline();

            let targeted = match &self.target {
                Some(x) => x.target == entity.pos,
                None => known.focus == Some(entity.weak.id()),
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

    fn render_status(&self, known: &Knowledge, e: &Trainer, slice: &mut Slice) {
        if let Some(entity) = known.get_view_of(e.id()) {
            self.render_entity(Some(PLAYER_KEY), None, entity, slice);
        }
        for (_, key) in SUMMON_KEYS.iter().enumerate() {
            self.render_empty_option(*key, 0, slice);
        }
    }

    fn render_target(&self, known: &Knowledge, slice: &mut Slice) {
        let name = |pid: &PID| -> &str { pid.species.name };

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
                        let name = match &self.player.data(&self.t).pokemon[*x] {
                            PokemonEdge::In(x) => name(x),
                            PokemonEdge::Out(x) => name(&x.data(&self.t).me),
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

    fn render_choice(&self, e: &Trainer, choice: i32, slice: &mut Slice) {
        let options = &e.data(&self.t).pokemon;
        for (i, key) in PARTY_KEYS.iter().enumerate() {
            let selected = choice == i as i32;
            match if i < options.len() { Some(&options[i]) } else { None } {
                Some(PokemonEdge::Out(x)) => {
                    let (me, pp) = (&*x.data(&self.t).me, get_pp(e, &self.t));
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
                     option: &PID, pp: f64, slice: &mut Slice) {
        let hp = get_hp(option);
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
        slice.set_fg(fg).write_str(&prefix).write_str(option.species.name).newline();
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
