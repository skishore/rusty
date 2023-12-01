use std::cmp::max;
use std::f64::consts::TAU;
use std::rc::Rc;

use crate::static_assert_size;
use crate::base::{FOV, Glyph, HashMap, Matrix, Point, clamp};
use crate::entity::{EID, Entity, PokemonIndividualData, PokemonSpeciesData};
use crate::game::{BoardView, Tile, MOVE_TIMER};
use crate::pathing::Status;

//////////////////////////////////////////////////////////////////////////////

// Constants

const MAX_MEMORY: i32 = 1024;

const VISION_ANGLE: f64 = TAU / 3.;
const VISION_RADIUS: i32 = 3;

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
}

pub enum EntityView {
    Pokemon(PokemonView),
    Trainer(TrainerView),
}

impl Default for EntityView {
    fn default() -> Self { Self::Trainer(TrainerView { hp: 0., name: "".into() }) }
}

pub fn get_hp(me: &PokemonIndividualData) -> f64 {
    me.cur_hp as f64 / max(me.max_hp, 1) as f64
}

pub fn get_pp(entity: &Entity) -> f64 {
    1. - clamp(entity.move_timer as f64 / MOVE_TIMER as f64, 0., 1.)
}

fn get_view(entity: &Entity) -> EntityView {
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
            EntityView::Trainer(TrainerView { hp, name })
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

// Knowledge

#[derive(Clone, Copy, Eq, PartialEq)] pub struct EntityIndex(i32);

pub struct CellKnowledge {
    index: Option<EntityIndex>,
    pub age: i32,
    pub tile: &'static Tile,
    pub visibility: i32,
}
static_assert_size!(CellKnowledge, 24);

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
static_assert_size!(EntityKnowledge, 72);

#[derive(Default)]
pub struct Knowledge {
    map: HashMap<Point, CellKnowledge>,
    entity_by_id: HashMap<EID, EntityIndex>,
    pub entities: Vec<EntityKnowledge>,
    pub focus: Option<EID>,
}

impl Knowledge {
    // Reads

    pub fn get_cell(&self, p: Point) -> Option<&CellKnowledge> {
        self.map.get(&p)
    }

    pub fn get_entity(&self, cell: &CellKnowledge) -> Option<&EntityKnowledge> {
        cell.index.map(|eid| self.entity_raw(eid))
    }

    pub fn get_entity_at(&self, p: Point) -> Option<&EntityKnowledge> {
        self.get_entity(self.get_cell(p)?)
    }

    pub fn get_status(&self, p: Point) -> Option<Status> {
        self.get_cell(p).map(|x| {
            if x.index.is_some() { return Status::Occupied; }
            if x.tile.blocked() { Status::Blocked } else { Status::Free }
        })
    }

    pub fn get_view_of(&self, eid: EID) -> Option<&EntityKnowledge> {
        self.entity_by_id.get(&eid).map(|x| self.entity_raw(*x))
    }

    pub fn can_see_now(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| x.age == 0).unwrap_or(false)
    }

    pub fn remembers(&self, p: Point) -> bool {
        self.map.contains_key(&p)
    }

    pub fn blocked(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| x.tile.blocked()).unwrap_or(false)
    }

    pub fn unblocked(&self, p: Point) -> bool {
        self.get_cell(p).map(|x| !x.tile.blocked()).unwrap_or(false)
    }

    // Writes

    pub fn update(&mut self, entity: &Entity, view: &BoardView, vision: &Vision) {
        let my_species = entity.species();
        let my_trainer = entity.trainer();
        self.forget(entity.player);

        for point in &vision.points_seen {
            let visibility = vision.get_visibility_at(*point);
            assert!(visibility >= 0);

            let index = (|| {
                let other = view.get_entity(view.get_entity_at(*point)?)?;
                let index = *self.entity_by_id.entry(other.id()).and_modify(|x| {
                    let existing = &mut self.entities[x.0 as usize];
                    if !existing.moved && existing.pos != *point {
                        self.map.entry(existing.pos).and_modify(|y| {
                            assert!(y.index == Some(*x));
                            y.index = None;
                        });
                    };
                }).or_insert_with(|| {
                    self.entities.push(EntityKnowledge {
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
                    });
                    EntityIndex(self.entities.len() as i32 - 1)
                });

                let species = other.species();
                let entry = self.entity_mut(index);

                entry.age = 0;
                entry.pos = *point;
                entry.dir = other.dir;
                entry.moved = false;
                entry.glyph = other.glyph;
                entry.rival = species.is_some() && species != my_species;
                entry.friend = other.trainer() == my_trainer;
                entry.player = other.player;
                entry.view = get_view(other);

                Some(index)
            })();

            let tile = view.get_tile_at(*point);
            let cell = CellKnowledge { age: 0, index, tile, visibility };
            let prev = self.map.insert(*point, cell);
            if let Some(x) = prev && x.index != index && let Some(other) = x.index {
                self.mark_entity_moved(other, *point);
            }
        }
    }

    // Private helpers

    fn entity_raw(&self, index: EntityIndex) -> &EntityKnowledge {
        &self.entities[index.0 as usize]
    }

    fn entity_mut(&mut self, index: EntityIndex) -> &mut EntityKnowledge {
        &mut self.entities[index.0 as usize]
    }

    fn forget(&mut self, player: bool) {
        if player {
            self.map.iter_mut().for_each(|x| x.1.age = 1);
            self.entities.iter_mut().for_each(|x| x.age = 1);
            return;
        }

        let mut removed: Vec<(Point, Option<EntityIndex>)> = vec![];
        for (key, val) in self.map.iter_mut() {
            val.age += 1;
            if val.age >= MAX_MEMORY { removed.push((*key, val.index)); }
        }
        removed.iter().for_each(|x| {
            self.map.remove(&x.0);
            if let Some(eid) = x.1 { self.mark_entity_moved(eid, x.0); }
        });

        let mut removed: Vec<EntityIndex> = vec![];
        for (i, val) in self.entities.iter_mut().enumerate() {
            val.age += 1;
            if val.age >= MAX_MEMORY { removed.push(EntityIndex(i as i32)); }
        }
        removed.iter().rev().for_each(|x| { self.remove_entity(*x); });
    }

    fn mark_entity_moved(&mut self, index: EntityIndex, pos: Point) {
        let entity = self.entity_mut(index);
        assert!(entity.pos == pos);
        entity.moved = true;
    }

    fn remove_entity(&mut self, index: EntityIndex) {
        let eid = self.entity_raw(index).eid;
        let popped = self.entities.pop().unwrap();
        let popped_index = EntityIndex(self.entities.len() as i32);
        let swap = index != popped_index;

        if !popped.moved {
            let cell = self.map.get_mut(&popped.pos).unwrap();
            assert!(cell.index == Some(popped_index));
            cell.index = if swap { Some(index) } else { None };
        }

        let deleted = if swap {
            self.entity_by_id.remove(&eid);
            self.entity_by_id.insert(popped.eid, index)
        } else {
            self.entity_by_id.remove(&popped.eid)
        };
        assert!(deleted == Some(popped_index));

        if self.focus == Some(eid) { self.focus = None; }

        if swap { *self.entity_mut(index) = popped; }
    }
}