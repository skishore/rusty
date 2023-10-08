use std::cell::UnsafeCell;
use std::marker::PhantomData;

use crate::assert_eq_size;

pub struct Token<T: ?Sized> { _x: PhantomData<T> }

impl<T> Token<T> {
    pub unsafe fn new() -> Self { Self { _x: PhantomData {} } }
}
assert_eq_size!(Token<i32>, 0);

pub struct Cell<T: ?Sized> {
    value: UnsafeCell<T>,
}

impl<T> Cell<T> {
    pub fn new(v: impl Into<T>) -> Self {
        Cell { value: UnsafeCell::new(v.into()) }
    }

    #[inline(always)]
    pub fn get<'a>(&self, _: &'a Token<T>) -> &'a T {
        unsafe { &*self.value.get() }
    }

    #[inline(always)]
    pub fn get_mut<'a>(&self, _: &'a mut Token<T>) -> &'a mut T {
        unsafe { &mut *self.value.get() }
    }
}
