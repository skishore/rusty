use std::ops::Deref;
use std::rc::{Rc, Weak};

use lazy_static::lazy_static;

use crate::base::{Glyph, HashMap, Point};
use crate::cell::{self, Cell};

//////////////////////////////////////////////////////////////////////////////

// Constants

const TRAINER_HP: i32 = 8;
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

// Pathing state definitions:

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

// Entity data definitions:

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
    pub trainer: WeakTrainer,
    pub cur_hp: i32,
    pub max_hp: i32,
}

pub struct PokemonData {
    pub ai: AIState,
    pub individual: Box<PokemonIndividualData>,
}

pub struct TrainerData {
    pub cur_hp: i32,
    pub max_hp: i32,
}

//////////////////////////////////////////////////////////////////////////////

// Constructors

fn pokemon(pos: Point, dir: Point, species: &str, trainer: Option<&Trainer>) -> EntityRepr {
    let species = POKEMON_SPECIES.get(species).unwrap();
    let individual = PokemonIndividualData {
        species,
        trainer: trainer.map(|x| x.into()).unwrap_or_default(),
        cur_hp: species.hp,
        max_hp: species.hp,
    };
    let data = PokemonData {
        ai: AIState::default(),
        individual: Box::new(individual),
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
    let data = TrainerData { cur_hp: TRAINER_HP, max_hp: TRAINER_HP };
    EntityRepr { base, data: EntityType::Trainer(data) }
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

pub enum ET<'a> {
    Pokemon(&'a Pokemon),
    Trainer(&'a Trainer),
}

pub type Token = cell::Token<EntityRepr>;

#[derive(Clone)]
#[repr(transparent)]
pub struct Entity(Rc<Cell<EntityRepr>>);

#[derive(Clone, Default)]
#[repr(transparent)]
pub struct WeakEntity(Weak<Cell<EntityRepr>>);

impl From<&Entity> for WeakEntity {
    fn from(val: &Entity) -> Self { Self(Rc::downgrade(&val.0)) }
}

impl WeakEntity {
    pub fn id(&self) -> usize { Weak::as_ptr(&self.0) as usize }

    pub fn upgrade(&self) -> Option<Entity> { self.0.upgrade().map(|x| Entity(x)) }
}

impl Entity {
    pub fn base<'a>(&'a self, t: &'a Token) -> &'a EntityData {
        &self.0.get(t).base
    }

    pub fn base_mut<'a>(&'a self, t: &'a mut Token) -> &'a mut EntityData {
        &mut self.0.get_mut(t).base
    }

    pub fn id(&self) -> usize { Rc::as_ptr(&self.0) as usize }

    pub fn test<'a>(&'a self, t: &Token) -> ET<'a> {
        let p = self as *const Entity;
        match &self.0.get(t).data {
            EntityType::Pokemon(_) => ET::Pokemon(unsafe { &*(p as *const Pokemon) }),
            EntityType::Trainer(_) => ET::Trainer(unsafe { &*(p as *const Trainer) }),
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct Pokemon(Entity);

#[derive(Clone, Default)]
#[repr(transparent)]
pub struct WeakPokemon(WeakEntity);

impl From<&Pokemon> for WeakPokemon {
    fn from(val: &Pokemon) -> Self { Self((&val.0).into()) }
}

impl WeakPokemon {
    pub fn upgrade(&self) -> Option<Pokemon> { self.0.upgrade().map(|x| Pokemon(x)) }
}

impl Deref for Pokemon {
    type Target = Entity;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl Deref for WeakPokemon {
    type Target = WeakEntity;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl Pokemon {
    pub fn new(pos: Point, dir: Point, species: &str, trainer: Option<&Trainer>) -> Pokemon {
        Pokemon(Entity(Rc::new(Cell::new(pokemon(pos, dir, species, trainer)))))
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

#[derive(Clone, Default)]
#[repr(transparent)]
pub struct WeakTrainer(WeakEntity);

impl From<&Trainer> for WeakTrainer {
    fn from(val: &Trainer) -> Self { Self((&val.0).into()) }
}

impl WeakTrainer {
    pub fn upgrade(&self) -> Option<Trainer> { self.0.upgrade().map(|x| Trainer(x)) }
}

impl Deref for Trainer {
    type Target = Entity;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl Deref for WeakTrainer {
    type Target = WeakEntity;
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

//////////////////////////////////////////////////////////////////////////////

// More boilerplate: equality operators

impl PartialEq for &'static PokemonSpeciesData {
    fn eq(&self, next: &&'static PokemonSpeciesData) -> bool {
        *self as *const PokemonSpeciesData ==
        *next as *const PokemonSpeciesData
    }
}

impl Eq for &'static PokemonSpeciesData {}

impl PartialEq for Entity {
    fn eq(&self, other: &Entity) -> bool { Rc::ptr_eq(&self.0, &other.0) }
}

impl Eq for Entity {}

impl PartialEq for Pokemon {
    fn eq(&self, other: &Pokemon) -> bool { Rc::ptr_eq(&self.0.0, &other.0.0) }
}

impl Eq for Pokemon {}

impl PartialEq for Trainer {
    fn eq(&self, other: &Trainer) -> bool { Rc::ptr_eq(&self.0.0, &other.0.0) }
}

impl Eq for Trainer {}
