use std::ops::Deref;
use std::rc::{Rc, Weak};

use lazy_static::lazy_static;

use crate::base::{Glyph, HashMap, Point};
use crate::cell::{self, Cell};

//////////////////////////////////////////////////////////////////////////////

// Constants

const TRAINER_SPEED: f64 = 0.10;

lazy_static! {
    static ref POKEMON_SPECIES: HashMap<&'static str, PokemonSpeciesData> = {
        let items = [
            ("Pidgey",  30, 0.33, Glyph::wide('P')),
            ("Ratatta", 60, 0.25, Glyph::wide('R')),
        ];
        let mut result = HashMap::default();
        for (name, hp, speed, glyph) in items {
            result.insert(name, PokemonSpeciesData { name, glyph, speed, hp });
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Actual data definitions:

pub struct EntityData {
    pub player: bool,
    pub removed: bool,
    pub move_timer: i32,
    pub turn_timer: i32,
    pub glyph: Glyph,
    pub speed: f64,
    pub dir: Point,
    pub pos: Point,
}

pub struct PokemonSpeciesData {
    pub name: &'static str,
    pub glyph: Glyph,
    pub speed: f64,
    pub hp: i32,
}

pub struct PokemonIndividualData {
    pub species: &'static PokemonSpeciesData,
}

pub struct PokemonData {
    pub data: Box<PokemonIndividualData>,
    pub cur_hp: i32,
    pub max_hp: i32,
}

pub struct TrainerData {}

//////////////////////////////////////////////////////////////////////////////

// Constructors

fn pokemon(pos: Point, dir: Point, species: &str) -> EntityRepr {
    let species = POKEMON_SPECIES.get(species).unwrap();
    let data = PokemonData {
        data: Box::new(PokemonIndividualData { species }),
        cur_hp: species.hp,
        max_hp: species.hp,
    };
    let base = EntityData {
        player: false,
        removed: false,
        move_timer: 0,
        turn_timer: 0,
        glyph: species.glyph,
        speed: species.speed,
        dir,
        pos,
    };
    EntityRepr { base, data: EntityType::Pokemon(data) }
}

fn trainer(pos: Point, player: bool) -> EntityRepr {
    let base = EntityData {
        player,
        removed: false,
        move_timer: 0,
        turn_timer: 0,
        glyph: Glyph::wide('@'),
        speed: TRAINER_SPEED,
        dir: Point(1, 0),
        pos,
    };
    EntityRepr { base, data: EntityType::Trainer(TrainerData {}) }
}

//////////////////////////////////////////////////////////////////////////////

// Boilerplate follows...

pub struct EntityRepr {
    base: EntityData,
    data: EntityType,
}

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

#[derive(Clone)]
pub struct WeakEntity(Weak<Cell<EntityRepr>>);

impl From<&Entity> for WeakEntity {
    fn from(val: &Entity) -> Self { WeakEntity(Rc::downgrade(&val.0)) }
}

impl Entity {
    pub fn base<'a>(&'a self, t: &'a Token) -> &'a EntityData {
        &self.0.get(t).base
    }

    pub fn base_mut<'a>(&'a self, t: &'a mut Token) -> &'a mut EntityData {
        &mut self.0.get_mut(t).base
    }

    pub fn id(&self) -> usize { Rc::as_ptr(&self.0) as usize }

    pub fn same(&self, other: &Entity) -> bool { Rc::ptr_eq(&self.0, &other.0) }

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
    pub fn new(pos: Point, dir: Point, species: &str) -> Pokemon {
        Pokemon(Entity(Rc::new(Cell::new(pokemon(pos, dir, species)))))
    }

    pub fn data<'a>(&'a self, t: &'a Token) -> &'a PokemonData {
        let data = &self.0.0.get(t).data;
        if let EntityType::Pokemon(x) = data { return x; }
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
        if let EntityType::Trainer(x) = data { return x; }
        unsafe { std::hint::unreachable_unchecked() }
    }

    pub fn data_mut<'a>(&'a self, t: &'a mut Token) -> &'a mut TrainerData {
        let data = &mut self.0.0.get_mut(t).data;
        if let EntityType::Trainer(ref mut x) = data { return x; }
        unsafe { std::hint::unreachable_unchecked() }
    }
}
