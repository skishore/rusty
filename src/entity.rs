use std::boxed::Box;
use std::cell::Cell;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::num::NonZeroU64;
use std::rc::Rc;

use lazy_static::lazy_static;
use slotmap::{DefaultKey, Key, KeyData, SlotMap};

use crate::{cast, static_assert_size};
use crate::base::{Glyph, HashMap, Point, RNG};
use crate::effect::{Effect, self};
use crate::game::{AIState, Board, Command};
use crate::knowledge::Knowledge;

//////////////////////////////////////////////////////////////////////////////

// Constants

const TRAINER_HP: i32 = 8;
const TRAINER_SPEED: f64 = 0.10;
const TYPED_ENTITY_OFFSET: isize = 8;

lazy_static! {
    static ref ATTACKS: HashMap<&'static str, Attack> = {
        let items: Vec<(&str, i32, i32, GenEffect)> = vec![
            ("Ember",    12, 40, Box::new(&effect::EmberEffect)),
            ("Ice Beam", 12, 60, Box::new(&effect::IceBeamEffect)),
            ("Blizzard", 12, 80, Box::new(&effect::BlizzardEffect)),
            ("Headbutt",  8, 80, Box::new(&effect::HeadbuttEffect)),
            ("Tackle",    4, 40, Box::new(&effect::HeadbuttEffect)),
        ];
        let mut result = HashMap::default();
        for (name, range, damage, effect) in items {
            result.insert(name, Attack { name, range, damage, effect });
        }
        result
    };
}

lazy_static! {
    static ref POKEMON_SPECIES:
            HashMap<&'static str, (PokemonSpeciesData, Vec<&'static Attack>)> = {
        let items = [
            ("Bulbasaur",  90, 0.17, Glyph::wdfg('B', 0x020), vec![]),
            ("Charmander", 80, 0.20, Glyph::wdfg('C', 0x410), vec!["Ember"]),
            ("Squirtle",   70, 0.25, Glyph::wdfg('S', 0x234), vec!["Ice Beam"]),
            ("Eevee",      80, 0.20, Glyph::wdfg('E', 0x420), vec!["Headbutt"]),
            ("Pikachu",    60, 0.25, Glyph::wdfg('P', 0x440), vec![]),
            ("Pidgey",     80, 0.33, Glyph::wide('P'), vec![]),
            ("Ratatta",    60, 0.25, Glyph::wide('R'), vec!["Headbutt"]),
        ];
        let mut result = HashMap::default();
        for (name, hp, speed, glyph, attacks) in items {
            let attacks = attacks.iter().chain(&["Tackle"])
                .map(|x| ATTACKS.get(x).unwrap_or_else(
                        || panic!("Unknown attack: {}", x)))
                .collect::<Vec<_>>();
            let species = PokemonSpeciesData { name, glyph, speed, hp };
            result.insert(name, (species, attacks));
        }
        result
    };
}

//////////////////////////////////////////////////////////////////////////////

// Entity data definitions

#[repr(C)]
pub enum Entity {
    Pokemon(Pokemon),
    Trainer(Trainer),
}
static_assert_size!(Entity, 72);

#[repr(C)]
pub struct EntityData {
    eid: EID,
    pub player: bool,
    pub removed: bool,
    pub move_timer: i32,
    pub turn_timer: i32,
    pub glyph: Glyph,
    pub speed: f64,
    pub dir: Point,
    pub pos: Point,
    pub known: Box<Knowledge>,
}
static_assert_size!(EntityData, 56);

#[repr(C)]
pub struct Pokemon {
    entity: EntityData,
    pub data: Box<PokemonData>,
}
static_assert_size!(Pokemon, 64);

#[repr(C)]
pub struct Trainer {
    entity: EntityData,
    pub data: Box<TrainerData>,
}
static_assert_size!(Trainer, 64);

pub type GenEffect = Box<dyn Fn(&Board, &mut RNG, Point, Point) -> Effect + Send + Sync>;

pub struct Attack {
    pub name: &'static str,
    pub range: i32,
    pub damage: i32,
    pub effect: GenEffect,
}

pub struct PokemonSpeciesData {
    pub name: &'static str,
    pub glyph: Glyph,
    pub speed: f64,
    pub hp: i32,
}

pub struct PokemonIndividualData {
    pub attacks: Vec<&'static Attack>,
    pub species: &'static PokemonSpeciesData,
    pub trainer: Option<TID>,
    pub cur_hp: i32,
    pub max_hp: i32,
}

pub struct PokemonData {
    pub ai: Cell<Option<Box<AIState>>>,
    pub me: Box<PokemonIndividualData>,
    pub commands: Cell<Vec<Command>>,
    pub debug: Cell<String>,
    pub flight: Cell<HashMap<Point, i32>>,
    pub target: Cell<Vec<Point>>,
}

pub struct TrainerData {
    pub cur_hp: i32,
    pub max_hp: i32,
    pub name: Rc<str>,
    pub pokemon: Vec<PokemonEdge>,
    pub summons: Vec<PID>,
}

pub enum PokemonEdge {
    Out(PID),
    In(Box<PokemonIndividualData>),
}

//////////////////////////////////////////////////////////////////////////////

// Constructors

pub struct PokemonArgs<'a> {
    pub pos: Point,
    pub dir: Point,
    pub species: &'a str,
}

pub struct SummonArgs {
    pub pos: Point,
    pub dir: Point,
    pub me: Box<PokemonIndividualData>,
}

pub struct TrainerArgs<'a> {
    pub pos: Point,
    pub dir: Point,
    pub name: &'a str,
    pub player: bool,
}

fn individual(species: &str, trainer: Option<TID>) -> Box<PokemonIndividualData> {
    let (species, attacks) = POKEMON_SPECIES.get(species).unwrap_or_else(
        || panic!("Unknown Pokemon species: {}", species));
    let me = PokemonIndividualData {
        attacks: attacks.clone(),
        species,
        trainer,
        cur_hp: species.hp,
        max_hp: species.hp,
    };
    Box::new(me)
}

fn pokemon(eid: EID, args: &PokemonArgs) -> Entity {
    let me = individual(args.species, None);
    summons(eid, SummonArgs { pos: args.pos, dir: args.dir, me })
}

fn summons(eid: EID, args: SummonArgs) -> Entity {
    let data = PokemonData {
        ai: Cell::default(),
        me: args.me,
        commands: Cell::default(),
        debug: Cell::default(),
        flight: Cell::default(),
        target: Cell::default(),
    };
    let entity = EntityData {
        eid,
        player: false,
        removed: false,
        move_timer: 0,
        turn_timer: 0,
        glyph: data.me.species.glyph,
        speed: data.me.species.speed,
        dir: args.dir,
        pos: args.pos,
        known: Box::default(),
    };
    Entity::Pokemon(Pokemon { entity, data: Box::new(data) })
}

fn trainer(eid: EID, args: &TrainerArgs) -> Entity {
    let entity = EntityData {
        eid,
        player: args.player,
        removed: false,
        move_timer: 0,
        turn_timer: 0,
        glyph: Glyph::wide('@'),
        speed: TRAINER_SPEED,
        dir: args.dir,
        pos: args.pos,
        known: Box::default(),
    };
    let (cur_hp, max_hp) = (TRAINER_HP, TRAINER_HP);
    let (name, pokemon, summons) = (args.name.into(), vec![], vec![]);
    let data = TrainerData { cur_hp, max_hp, name, pokemon, summons };
    Entity::Trainer(Trainer { entity, data: Box::new(data) })
}

//////////////////////////////////////////////////////////////////////////////

// Impls and references

impl Deref for Entity {
    type Target = EntityData;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        let base = self as *const Self as *const u8;
        unsafe { &*(base.offset(TYPED_ENTITY_OFFSET) as *const EntityData) }
    }
}

impl DerefMut for Entity {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let base = self as *mut Self as *mut u8;
        unsafe { &mut *(base.offset(TYPED_ENTITY_OFFSET) as *mut EntityData) }
    }
}

impl Deref for Pokemon {
    type Target = Entity;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        let base = self as *const Self as *const u8;
        unsafe { &*(base.offset(-TYPED_ENTITY_OFFSET) as *const Entity) }
    }
}

impl Deref for Trainer {
    type Target = Entity;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        let base = self as *const Self as *const u8;
        unsafe { &*(base.offset(-TYPED_ENTITY_OFFSET) as *const Entity) }
    }
}

impl DerefMut for Pokemon {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let base = self as *mut Self as *mut u8;
        unsafe { &mut *(base.offset(-TYPED_ENTITY_OFFSET) as *mut Entity) }
    }
}

impl DerefMut for Trainer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let base = self as *mut Self as *mut u8;
        unsafe { &mut *(base.offset(-TYPED_ENTITY_OFFSET) as *mut Entity) }
    }
}

impl Entity {
    pub fn id(&self) -> EID { self.eid }

    pub fn friends(&self) -> Vec<EID> {
        match self {
            Entity::Trainer(x) => x.data.summons.iter().map(|x| x.eid()).collect(),
            Entity::Pokemon(x) => x.data.me.trainer.iter().map(|x| x.eid()).collect(),
        }
    }

    pub fn species(&self) -> Option<&'static PokemonSpeciesData> {
        match self {
            Entity::Pokemon(x) => Some(x.data.me.species),
            Entity::Trainer(_) => None,
        }
    }

    pub fn trainer(&self) -> Option<TID> {
        match self {
            Entity::Pokemon(x) => x.data.me.trainer,
            Entity::Trainer(x) => Some(x.id()),
        }
    }
}

impl Pokemon {
    pub fn id(&self) -> PID { PID(self.eid) }
}

impl Trainer {
    pub fn id(&self) -> TID { TID(self.eid) }

    pub fn register_pokemon(&mut self, species: &str) {
        let me = individual(species, Some(self.id()));
        self.data.pokemon.push(PokemonEdge::In(me));
    }
}

impl PartialEq for &'static PokemonSpeciesData {
    fn eq(&self, next: &&'static PokemonSpeciesData) -> bool {
        *self as *const PokemonSpeciesData ==
        *next as *const PokemonSpeciesData
    }
}

impl Eq for &'static PokemonSpeciesData {}

//////////////////////////////////////////////////////////////////////////////

// Slotmap support

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct EID(NonZeroU64);
static_assert_size!(Option<EID>, 8);

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct PID(EID);
static_assert_size!(Option<PID>, 8);

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct TID(EID);
static_assert_size!(Option<TID>, 8);

fn to_key(eid: EID) -> DefaultKey {
    KeyData::from_ffi(eid.0.get()).into()
}

fn to_eid(key: DefaultKey) -> EID {
    EID(NonZeroU64::new(key.data().as_ffi()).unwrap())
}

pub trait EntityMapKey {
    type ValueType;
    fn eid(self) -> EID;
    fn as_ref(x: &Entity) -> &Self::ValueType;
    fn as_mut(x: &mut Entity) -> &mut Self::ValueType;
}

impl EntityMapKey for EID {
    type ValueType = Entity;
    fn eid(self) -> EID { self }
    fn as_ref(x: &Entity) -> &Entity { x }
    fn as_mut(x: &mut Entity) -> &mut Entity { x }
}

impl EntityMapKey for PID {
    type ValueType = Pokemon;
    fn eid(self) -> EID { self.0 }
    fn as_ref(x: &Entity) -> &Pokemon { cast!(x, Entity::Pokemon) }
    fn as_mut(x: &mut Entity) -> &mut Pokemon { cast!(x, Entity::Pokemon) }
}

impl EntityMapKey for TID {
    type ValueType = Trainer;
    fn eid(self) -> EID { self.0 }
    fn as_ref(x: &Entity) -> &Trainer { cast!(x, Entity::Trainer) }
    fn as_mut(x: &mut Entity) -> &mut Trainer { cast!(x, Entity::Trainer) }
}

#[derive(Default)]
pub struct EntityMap {
    map: SlotMap<DefaultKey, Entity>,
}

impl EntityMap {
    pub fn get<T: EntityMapKey>(&self, id: T) -> Option<&T::ValueType> {
        self.map.get(to_key(id.eid())).map(|x| T::as_ref(x))
    }

    pub fn get_mut<T: EntityMapKey>(&mut self, id: T) -> Option<&mut T::ValueType> {
        self.map.get_mut(to_key(id.eid())).map(|x| T::as_mut(x))
    }

    pub fn add_pokemon(&mut self, args: &PokemonArgs) -> PID {
        PID(to_eid(self.map.insert_with_key(|x| pokemon(to_eid(x), args))))
    }

    pub fn add_summons(&mut self, args: SummonArgs) -> PID {
        PID(to_eid(self.map.insert_with_key(|x| summons(to_eid(x), args))))
    }

    pub fn add_trainer(&mut self, args: &TrainerArgs) -> TID {
        TID(to_eid(self.map.insert_with_key(|x| trainer(to_eid(x), args))))
    }

    pub fn remove_entity(&mut self, eid: EID) -> Option<Entity> {
        self.map.remove(to_key(eid))
    }
}

impl<T: EntityMapKey> Index<T> for EntityMap {
    type Output = T::ValueType;
    fn index(&self, id: T) -> &Self::Output { self.get(id).unwrap() }
}

impl<T: EntityMapKey> IndexMut<T> for EntityMap {
    fn index_mut(&mut self, id: T) -> &mut Self::Output { self.get_mut(id).unwrap() }
}
