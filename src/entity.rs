use std::ops::Deref;
use std::rc::Rc;

use crate::base::{Glyph, Point};
use crate::cell::{self, Cell};

//////////////////////////////////////////////////////////////////////////////

// Actual data definitions:

#[derive(Debug)]
pub struct EntityData {
    pub player: bool,
    pub removed: bool,
    pub glyph: Glyph,
    pub pos: Point,
}

#[derive(Debug)]
pub struct PokemonData {
    pub species: String,
}

#[derive(Debug)]
pub struct TrainerData {}

fn trainer(pos: Point, player: bool) -> EntityRepr {
    let glyph = Glyph::wide('@');
    let base = EntityData { player, removed: false, glyph, pos };
    EntityRepr { base, data: EntityType::Trainer(TrainerData {}) }
}

//////////////////////////////////////////////////////////////////////////////

// Boilerplate follows...

#[derive(Debug)]
pub struct EntityRepr {
    base: EntityData,
    data: EntityType,
}

#[derive(Debug)]
enum EntityType {
    Pokemon(PokemonData),
    Trainer(TrainerData),
}

pub enum ET {
    Pokemon(Pokemon),
    Trainer(Trainer),
}

pub enum ETRef<'a> {
    Pokemon(&'a Pokemon),
    Trainer(&'a Trainer),
}

pub type Token = cell::Token<EntityRepr>;

#[derive(Clone)]
pub struct Entity(Rc<Cell<EntityRepr>>);

impl Entity {
    pub fn base<'a>(&'a self, t: &'a Token) -> &'a EntityData {
        &self.0.get(t).base
    }

    pub fn base_mut<'a>(&'a self, t: &'a mut Token) -> &'a mut EntityData {
        &mut self.0.get_mut(t).base
    }

    pub fn same(&self, other: &Entity) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    pub fn test(self, t: &Token) -> ET {
        match &self.0.get(t).data {
            EntityType::Pokemon(_) => ET::Pokemon(Pokemon(self)),
            EntityType::Trainer(_) => ET::Trainer(Trainer(self)),
        }
    }

    pub fn test_ref<'a>(&'a self, t: &Token) -> ETRef<'a> {
        let p = self as *const Entity;
        match &self.0.get(t).data {
            EntityType::Pokemon(_) => ETRef::Pokemon(unsafe { &*(p as *const Pokemon) }),
            EntityType::Trainer(_) => ETRef::Trainer(unsafe { &*(p as *const Trainer) }),
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct Pokemon(Entity);

impl Deref for Pokemon {
    type Target = Entity;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl Pokemon {
    pub fn data<'a>(&'a self, t: &'a Token) -> &'a PokemonData {
        let data = &self.0.0.get(t).data;
        if let EntityType::Pokemon(x) = data { return &x; }
        unsafe { std::hint::unreachable_unchecked() }
    }

    pub fn data_mut<'a>(&'a self, t: &'a mut Token) -> &'a mut PokemonData {
        let data = &mut self.0.0.get_mut(t).data;
        if let EntityType::Pokemon(ref mut x) = data { return x; }
        unsafe { std::hint::unreachable_unchecked() }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct Trainer(Entity);

impl Deref for Trainer {
    type Target = Entity;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl Trainer {
    pub fn new(pos: Point, player: bool) -> Trainer {
        Trainer(Entity(Rc::new(Cell::new(trainer(pos, player)))))
    }

    pub fn data<'a>(&'a self, t: &'a Token) -> &'a TrainerData {
        let data = &self.0.0.get(t).data;
        if let EntityType::Trainer(x) = data { return &x; }
        unsafe { std::hint::unreachable_unchecked() }
    }

    pub fn data_mut<'a>(&'a self, t: &'a mut Token) -> &'a mut TrainerData {
        let data = &mut self.0.0.get_mut(t).data;
        if let EntityType::Trainer(ref mut x) = data { return x; }
        unsafe { std::hint::unreachable_unchecked() }
    }
}
