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

//////////////////////////////////////////////////////////////////////////////

// Heap, used for A*

#[derive(Clone, Copy)] struct AStarHeapIndex(i32);
#[derive(Clone, Copy)] struct AStarNodeIndex(i32);

struct AStarNode {
    distance: i32,
    score: i32,
    index: AStarHeapIndex,
    predecessor: Point,
    pos: Point,
}

#[derive(Default)]
struct AStarHeap {
    nodes: Vec<AStarNode>,
    heap: Vec<AStarNodeIndex>,
}

impl AStarHeap {

    // Heap operations

    fn is_empty(&self) -> bool { self.nodes.is_empty() }

    fn extract_min(&mut self) -> AStarNodeIndex {
        let mut index = AStarHeapIndex(0);
        let result = self.get_heap(index);
        self.mut_node(result).index = AStarHeapIndex(-1);

        let node = self.heap.pop().unwrap();
        if self.heap.is_empty() { return result; }

        let limit = self.heap.len() as i32;
        let score = self.get_node(node).score;
        let (mut c0, mut c1) = Self::children(index);

        while c0.0 < limit {
            let mut child_index = c0;
            let mut child_score = self.heap_score(c0);
            if c1.0 < limit {
                let c1_score = self.heap_score(c1);
                if c1_score < child_score {
                    (child_index, child_score) = (c1, c1_score);
                }
            }
            if score <= child_score { break; }

            self.heap_move(child_index, index);
            (c0, c1) = Self::children(child_index);
            index = child_index;
        }

        self.mut_node(node).index = index;
        self.set_heap(index, node);
        result
    }

    fn heapify(&mut self, n: AStarNodeIndex) {
        let score = self.get_node(n).score;
        let mut index = self.get_node(n).index;

        while index.0 > 0 {
            let parent_index = Self::parent(index);
            let parent_score = self.heap_score(parent_index);
            if parent_score <= score { break; }

            self.heap_move(parent_index, index);
            index = parent_index;
        }

        self.mut_node(n).index = index;
        self.set_heap(index, n);
    }

    fn push(&mut self, node: AStarNode) -> AStarNodeIndex {
        assert!(node.index.0 == -1);
        assert!(self.nodes.len() == self.heap.len());

        let result = AStarNodeIndex(self.nodes.len() as i32);
        self.nodes.push(node);
        self.heap.push(result);
        self.heapify(result);
        result
    }

    // Lower-level helpers

    fn heap_score(&self, h: AStarHeapIndex) -> i32 {
        self.get_node(self.get_heap(h)).score
    }

    fn heap_move(&mut self, from: AStarHeapIndex, to: AStarHeapIndex) {
        let node = self.get_heap(from);
        self.mut_node(node).index = to;
        self.set_heap(to, node);
    }

    fn get_heap(&self, h: AStarHeapIndex) -> AStarNodeIndex {
        self.heap[h.0 as usize]
    }

    fn set_heap(&mut self, h: AStarHeapIndex, n: AStarNodeIndex) {
        self.heap[h.0 as usize] = n;
    }

    fn get_node(&self, n: AStarNodeIndex) -> &AStarNode {
        &self.nodes[n.0 as usize]
    }

    fn mut_node(&mut self, n: AStarNodeIndex) -> &mut AStarNode {
        &mut self.nodes[n.0 as usize]
    }

    fn parent(h: AStarHeapIndex) -> AStarHeapIndex {
        AStarHeapIndex((h.0 - 1) / 2)
    }

    fn children(h: AStarHeapIndex) -> (AStarHeapIndex, AStarHeapIndex) {
        (AStarHeapIndex(2 * h.0 + 1), AStarHeapIndex(2 * h.0 + 2))
    }
}
