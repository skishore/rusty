use std::ops::Deref;
use std::rc::Rc;

use crate::base::{Glyph, Point};
use crate::cell::{Cell, Token};

//////////////////////////////////////////////////////////////////////////////

// Actual data definitions:

#[derive(Debug)]
pub struct EntityBase {
    pub player: bool,
    pub removed: bool,
    pub glyph: Glyph,
    pub pos: Point,
}

#[derive(Debug)]
pub struct Entity {
    base: EntityBase,
    data: EntityData
}

#[derive(Debug)]
enum EntityData {
    Pokemon(PokemonData),
    Trainer(TrainerData),
}

#[derive(Debug)]
pub struct PokemonData {
    pub species: String,
}

#[derive(Debug)]
pub struct TrainerData {}

//////////////////////////////////////////////////////////////////////////////

// Boilerplate follows...

#[derive(Clone)]
pub struct EntityRef(Rc<Cell<Entity>>);

impl EntityRef {
    pub fn base<'a>(&'a self, t: &'a Token<Entity>) -> &'a EntityBase {
        &self.0.get(t).base
    }

    pub fn base_mut<'a>(&'a self, t: &'a mut Token<Entity>) -> &'a mut EntityBase {
        &mut self.0.get_mut(t).base
    }

    pub fn same(&self, other: &EntityRef) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    pub fn test(self, t: &Token<Entity>) -> MatchRef {
        match &self.0.get(t).data {
            EntityData::Pokemon(_) => MatchRef::Pokemon(PokemonRef(self)),
            EntityData::Trainer(_) => MatchRef::Trainer(TrainerRef(self)),
        }
    }
}

pub enum MatchRef {
    Pokemon(PokemonRef),
    Trainer(TrainerRef),
}

#[derive(Clone)]
pub struct PokemonRef(EntityRef);

impl Deref for PokemonRef {
    type Target = EntityRef;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl PokemonRef {
    pub fn data<'a>(&'a self, t: &'a Token<Entity>) -> &'a PokemonData {
        let data = &self.0.0.get(t).data;
        if let EntityData::Pokemon(x) = data { return &x; }
        panic!("PokemonRef contained a Trainer: {:?}", data);
    }

    pub fn data_mut<'a>(&'a self, t: &'a mut Token<Entity>) -> &'a mut PokemonData {
        let data = &mut self.0.0.get_mut(t).data;
        if let EntityData::Pokemon(ref mut x) = data { return x; }
        panic!("PokemonRef contained a Trainer: {:?}", data);
    }
}

#[derive(Clone)]
pub struct TrainerRef(EntityRef);

impl Deref for TrainerRef {
    type Target = EntityRef;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl TrainerRef {
    pub fn new(pos: Point, player: bool) -> TrainerRef {
        let base = EntityBase { player, removed: false, glyph: Glyph::wide('@'), pos };
        let entity = Entity { base, data: EntityData::Trainer(TrainerData {}) };
        TrainerRef(EntityRef(Rc::new(Cell::new(entity))))
    }

    pub fn data<'a>(&'a self, t: &'a Token<Entity>) -> &'a TrainerData {
        let data = &self.0.0.get(t).data;
        if let EntityData::Trainer(x) = data { return &x; }
        panic!("TrainerRef contained a Pokemon: {:?}", data);
    }

    pub fn data_mut<'a>(&'a self, t: &'a mut Token<Entity>) -> &'a mut TrainerData {
        let data = &mut self.0.0.get_mut(t).data;
        if let EntityData::Trainer(ref mut x) = data { return x; }
        panic!("TrainerRef contained a Pokemon: {:?}", data);
    }
}
