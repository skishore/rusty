use crate::base::{Matrix, Point};

//////////////////////////////////////////////////////////////////////////////

#[derive(Eq, PartialEq)]
pub enum Status { Free, Blocked, Occupied }

pub const DIRECTIONS: [Point; 8] = [
    Point(-1,  0),
    Point( 0,  1),
    Point( 0, -1),
    Point( 1,  0),
    Point(-1, -1),
    Point( 1, -1),
    Point(-1,  1),
    Point( 1,  1),
];

//////////////////////////////////////////////////////////////////////////////

// BFS (breadth-first search)

pub struct BFSResult {
    pub dirs: Vec<Point>,
    pub targets: Vec<Point>,
}

#[allow(non_snake_case)]
pub fn BFS<F: Fn(Point) -> bool, G: Fn(Point) -> Status>(
        source: Point, target: F, limit: i32, check: G) -> Option<BFSResult> {
    let kUnknown = -1;
    let kBlocked = -2;

    let n = 2 * limit + 1;
    let initial = Point(limit, limit);
    let offset = source - initial;
    let mut distances = Matrix::new(Point(n, n), kUnknown);
    distances.set(initial, 0);

    let mut i = 1;
    let mut prev: Vec<Point> = vec![initial];
    let mut next: Vec<Point> = vec![];
    let mut targets: Vec<Point> = vec![];

    while i <= limit {
        for pp in &prev {
            for dir in &DIRECTIONS {
                let np = *pp + *dir;
                let distance = distances.get(np);
                if distance != kUnknown { continue; }

                let point = np + offset;
                let free = check(point) == Status::Free;
                let done = free && target(point);

                distances.set(np, if free { i } else { kBlocked });
                if done { targets.push(np); }
                if free { next.push(np); }
            }
        }
        if next.is_empty() || !targets.is_empty() { break; }
        std::mem::swap(&mut next, &mut prev);
        next.clear();
        i += 1;
    }

    if targets.is_empty() { return None; }

    let mut result = BFSResult { dirs: vec![], targets: vec![] };
    result.targets = targets.iter().map(|x| *x + offset).collect();
    prev = targets;
    next.clear();
    i -= 1;

    while i > 0 {
        for pp in &prev {
            for dir in &DIRECTIONS {
                let np = *pp + *dir;
                let distance = distances.get(np);
                if distance != i { continue; }

                distances.set(np, kUnknown);
                next.push(np);
            }
        }
        std::mem::swap(&mut next, &mut prev);
        next.clear();
        i -= 1;
    }

    assert!(!prev.is_empty());
    result.dirs = prev.iter().map(|x| *x - initial).collect();
    Some(result)
}
