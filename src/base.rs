mod cell {
    use std::cell::UnsafeCell;

    pub struct Token { _x: () }

    impl Token {
        pub unsafe fn new() -> Self { Self { _x: () } }
    }

    pub struct Cell<T: ?Sized> {
        value: UnsafeCell<T>,
    }

    impl<T> Cell<T> {
        pub fn new(v: impl Into<T>) -> Self {
            Cell { value: UnsafeCell::new(v.into()) }
        }

        #[inline]
        pub fn get<'a>(&self, _: &'a Token) -> &'a T {
            return unsafe { &*self.value.get() }
        }

        #[inline]
        pub fn get_mut<'a>(&self, _: &'a mut Token) -> &'a mut T {
            return unsafe { &mut *self.value.get() }
        }
    }
}

pub use cell::{Cell, Token};
