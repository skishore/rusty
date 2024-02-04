use std::cmp::max;
use std::f64::consts::TAU;
use std::rc::Rc;

use crate::static_assert_size;
use crate::base::{FOV, Glyph, HashMap, Matrix, Point, clamp};
use crate::entity::{EID, Entity, PokemonEdge};
use crate::entity::{PokemonIndividualData, PokemonSpeciesData};
use crate::game::{BoardView, Tile, MOVE_TIMER};
use crate::list::{Handle, List};
use crate::pathing::Status;

//////////////////////////////////////////////////////////////////////////////

// Constants

const MAX_ENTITY_MEMORY: usize = 32;
const MAX_TILE_MEMORY: usize = 4096;

const VISION_ANGLE: f64 = TAU / 3.;
const VISION_RADIUS: i32 = 3;

//////////////////////////////////////////////////////////////////////////////

// Timestamp

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Timestamp(u32);

impl std::ops::Sub for Timestamp {
    type Output = i32;
    fn sub(self, other: Timestamp) -> Self::Output {
        (self.0 - other.0) as Self::Output
    }
}

//////////////////////////////////////////////////////////////////////////////

// Vision

pub struct VisionArgs {
    pub player: bool,
    pub pos: Point,
    pub dir: Point,
}

pub struct Vision {
    fov: FOV,
    center: Point,
    offset: Point,
    visibility: Matrix<i32>,
    pub points_seen: Vec<Point>,
}

impl Vision {
    pub fn new(radius: i32) -> Self {
        let vision_side = 2 * radius + 1;
        let vision_size = Point(vision_side, vision_side);
        Self {
            fov: FOV::new(radius),
            center: Point(radius, radius),
            offset: Point::default(),
            visibility: Matrix::new(vision_size, -1),
            points_seen: vec![],
        }
    }

    pub fn can_see_now(&self, p: Point) -> bool {
        self.get_visibility_at(p) >= 0
    }

    pub fn get_visibility_at(&self, p: Point) -> i32 {
        self.visibility.get(p + self.offset)
    }

    pub fn compute<F: Fn(Point) -> &'static Tile>(&mut self, args: &VisionArgs, f: F) {
        let VisionArgs { player, pos, dir } = *args;
        self.offset = self.center - pos;
        self.visibility.fill(-1);
        self.points_seen.clear();

        let blocked = |p: Point, prev: Option<&Point>| {
            if !player && !Self::in_vision_cone(dir, p) { return true; }

            let lookup = p + self.center;
            let cached = self.visibility.get(lookup);

            let visibility = (|| {
                // These constant values come from Point.distanceNethack.
                // They are chosen such that, in a field of tall grass, we'll
                // only see cells at a distanceNethack <= kVisionRadius.
                if prev.is_none() { return 100 * (VISION_RADIUS + 1) - 95 - 46 - 25; }

                let tile = f(p + pos);
                if tile.blocked() { return 0; }

                let parent = prev.unwrap();
                let obscure = tile.obscure();
                let diagonal = p.0 != parent.0 && p.1 != parent.1;
                let loss = if obscure { 95 + if diagonal { 46 } else { 0 } } else { 0 };
                let prev = self.visibility.get(*parent + self.center);
                max(prev - loss, 0)
            })();

            if visibility > cached {
                self.visibility.set(lookup, visibility);
                if cached < 0 && 0 <= visibility {
                    self.points_seen.push(p + pos);
                }
            }
            visibility <= 0
        };
        self.fov.apply(blocked);
    }

    fn in_vision_cone(pos: Point, dir: Point) -> bool {
        if pos == Point::default() || dir == Point::default() { return true; }
        let dot = (pos.0 as i64 * dir.0 as i64 + pos.1 as i64 * dir.1 as i64) as f64;
        let (l2p, l2d) = (pos.len_l2_squared() as f64, dir.len_l2_squared() as f64);
        dot / (l2p * l2d).sqrt() > (0.5 * VISION_ANGLE).cos()
    }
}

//////////////////////////////////////////////////////////////////////////////

// Views

pub struct PokemonView {
    pub species: &'static PokemonSpeciesData,
    pub trainer: bool,
    pub hp: f64,
    pub pp: f64,
}

pub struct TrainerView {
    pub hp: f64,
    pub name: Rc<str>,
    pub status: Vec<bool>,
}

pub enum EntityView {
    Pokemon(PokemonView),
    Trainer(TrainerView),
}

impl Default for EntityView {
    fn default() -> Self {
        Self::Trainer(TrainerView { hp: 0., name: "".into(), status: vec![] })
    }
}

pub fn get_hp(me: &PokemonIndividualData) -> f64 {
    me.cur_hp as f64 / max(me.max_hp, 1) as f64
}

pub fn get_pp(entity: &Entity) -> f64 {
    1. - clamp(entity.move_timer as f64 / MOVE_TIMER as f64, 0., 1.)
}

pub fn get_view(entity: &Entity, view: &BoardView) -> EntityView {
    match entity {
        Entity::Pokemon(x) => {
            let species = x.data.me.species;
            let trainer = x.data.me.trainer.is_some();
            let (hp, pp) = (get_hp(&x.data.me), get_pp(entity));
            EntityView::Pokemon(PokemonView { species, trainer, hp, pp })
        }
        Entity::Trainer(x) => {
            let name = x.data.name.clone();
            let hp = x.data.cur_hp as f64 / max(x.data.max_hp, 1) as f64;
            let status = x.data.pokemon.iter().map(|x| {
                let me = match &x {
                    PokemonEdge::In(x) => Some(&**x),
                    PokemonEdge::Out(x) => view.get_entity(*x).map(|x| &*x.data.me),
                };
                me.map(|x| x.cur_hp > 0).unwrap_or(false)
            }).collect();
            EntityView::Trainer(TrainerView { hp, name, status })
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

type CellHandle = Handle<CellKnowledge>;
type EntityHandle = Handle<EntityKnowledge>;

pub struct CellKnowledge {
    handle: Option<EntityHandle>,
    pub point: Point,
    pub tile: &'static Tile,
    pub time: Timestamp,
    pub visibility: i32,
}
static_assert_size!(CellKnowledge, 32);

pub struct EntityKnowledge {
    pub eid: EID,
    pub age: i32,
    pub pos: Point,
    pub dir: Point,
    pub glyph: Glyph,
    pub moved: bool,
    pub rival: bool,
    pub friend: bool,
    pub player: bool,
    pub view: EntityView,
}

#[derive(Default)]
pub struct Knowledge {
    cell_by_point: HashMap<Point, CellHandle>,
    entity_by_eid: HashMap<EID, EntityHandle>,
    pub cells: List<CellKnowledge>,
    pub entities: List<EntityKnowledge>,
    pub focus: Option<EID>,
    pub time: Timestamp,
}

pub struct CellResult<'a> {
    root: &'a Knowledge,
    cell: Option<&'a CellKnowledge>,
}

impl<'a> CellResult<'a> {
    // Field lookups

    pub fn age(&self) -> i32 {
        self.cell.map(|x| self.root.time - x.time).unwrap_or(std::i32::MAX)
    }

    pub fn tile(&self) -> Option<&'static Tile> {
        self.cell.map(|x| x.tile)
    }

    pub fn visibility(&self) -> i32 {
        self.cell.map(|x| x.visibility).unwrap_or(-1)
    }

    // Derived fields

    pub fn entity(&self) -> Option<&'a EntityKnowledge> {
        self.cell.and_then(|x| x.handle.map(|y| &self.root.entities[y]))
    }

    pub fn status(&self) -> Option<Status> {
        self.cell.map(|x| {
            if x.handle.is_some() { return Status::Occupied; }
            if x.tile.blocked() { Status::Blocked } else { Status::Free }
        })
    }

    // Predicates

    pub fn blocked(&self) -> bool {
        self.cell.map(|x| x.tile.blocked()).unwrap_or(false)
    }

    pub fn unblocked(&self) -> bool {
        self.cell.map(|x| !x.tile.blocked()).unwrap_or(false)
    }

    pub fn visible(&self) -> bool {
        self.cell.map(|x| self.root.time == x.time).unwrap_or(false)
    }
}

impl Knowledge {
    // Reads

    pub fn default(&self) -> CellResult {
        CellResult { root: self, cell: None }
    }

    pub fn entity(&self, eid: EID) -> Option<&EntityKnowledge> {
        self.entity_by_eid.get(&eid).map(|x| &self.entities[*x])
    }

    pub fn get(&self, p: Point) -> CellResult {
        let cell_handle = self.cell_by_point.get(&p);
        CellResult { root: self, cell: cell_handle.map(|x| &self.cells[*x]) }
    }

    // Writes

    pub fn update(&mut self, me: &Entity, view: &BoardView, vision: &Vision) {
        self.age_out(me.player);
        let time = self.time;

        // Entities have approximate knowledge about friends, even if unseen.
        for eid in me.friends() {
            if let Some(friend) = view.get_entity(eid) && !vision.can_see_now(friend.pos) {
                self.update_entity(me, view, friend, false);
            }
        }

        // Entities have exact knowledge about anything they can see.
        for point in &vision.points_seen {
            let point = *point;
            let visibility = vision.get_visibility_at(point);
            assert!(visibility >= 0);

            let handle = (|| {
                let other = view.get_entity(view.get_entity_at(point)?)?;
                Some(self.update_entity(me, view, other, true))
            })();

            let mut prev_handle = None;
            let tile = view.get_tile_at(point);
            self.cell_by_point.entry(point).and_modify(|x| {
                self.cells.move_to_front(*x);
                let cell = CellKnowledge { handle, point, tile, time, visibility };
                prev_handle = std::mem::replace(&mut self.cells[*x], cell).handle;
            }).or_insert_with(|| {
                let cell = CellKnowledge { handle, point, tile, time, visibility };
                self.cells.push_front(cell)
            });

            if prev_handle != handle && let Some(other) = prev_handle {
                self.mark_entity_moved(other, point);
            };
        }

        self.forget(me.player);
    }

    // Private helpers

    fn update_entity(&mut self, me: &Entity, view: &BoardView,
                     other: &Entity, seen: bool) -> EntityHandle {
        let handle = *self.entity_by_eid.entry(other.id()).and_modify(|x| {
            self.entities.move_to_front(*x);
            let existing = &mut self.entities[*x];
            if !existing.moved && !(seen && existing.pos == other.pos) {
                let cell_handle = self.cell_by_point.get(&existing.pos);
                let cell = &mut self.cells[*cell_handle.unwrap()];
                assert!(cell.handle == Some(*x));
                cell.handle = None;
            };
        }).or_insert_with(|| {
            self.entities.push_front(EntityKnowledge {
                eid: other.id(),
                age: Default::default(),
                pos: Default::default(),
                dir: Default::default(),
                moved: Default::default(),
                glyph: Default::default(),
                rival: Default::default(),
                friend: Default::default(),
                player: Default::default(),
                view: Default::default(),
            })
        });

        let species = other.species();
        let trainer = other.trainer();
        let entry = &mut self.entities[handle];

        entry.age = if seen { 0 } else { 1 };
        entry.pos = other.pos;
        entry.dir = other.dir;
        entry.moved = !seen;
        entry.glyph = other.glyph;
        entry.rival = !trainer.is_some() && species != me.species() && false;
        entry.friend = trainer.is_some() && trainer == me.trainer();
        entry.player = other.player;
        entry.view = get_view(other, view);

        handle
    }

    fn age_out(&mut self, _player: bool) {
        for x in &mut self.entities { x.age += 1; }
        self.time.0 += 1;
    }

    fn forget(&mut self, player: bool) {
        if player { return; }

        while self.cell_by_point.len() > MAX_TILE_MEMORY {
            // We don't need to check age, here; we can only see a bounded
            // number of cells per turn, much less than MAX_TILE_MEMORY.
            let CellKnowledge { point, handle, .. } = self.cells.pop_back().unwrap();
            if let Some(x) = handle { self.mark_entity_moved(x, point); }
            self.cell_by_point.remove(&point);
        }

        while self.entity_by_eid.len() > MAX_ENTITY_MEMORY {
            let entity = self.entities.back().unwrap();
            if entity.age == 0 || entity.friend { break; }

            let handle = self.entity_by_eid.remove(&entity.eid).unwrap();
            if !entity.moved {
                let cell_handle = self.cell_by_point.get(&entity.pos);
                let cell = &mut self.cells[*cell_handle.unwrap()];
                assert!(cell.handle == Some(handle));
                cell.handle = None;
            }
            self.entities.pop_back();
        }
    }

    fn mark_entity_moved(&mut self, handle: EntityHandle, pos: Point) {
        let entity = &mut self.entities[handle];
        assert!(entity.pos == pos);
        assert!(!entity.moved);
        entity.moved = true;
    }
}
