mod base;

use base::{Cell, Token};

fn main() {
    let mut t = unsafe { Token::new() };
    let x: Cell<i32> = Cell::new(17);
    let y: Cell<i32> = Cell::new(34);
    *x.get_mut(&mut t) = 51;
    *y.get_mut(&mut t) = 64;
    let z = {
        let a = x.get(&t);
        let b = y.get(&t);
        a + b
    };
    println!("Final value: {:?}", z);
}
