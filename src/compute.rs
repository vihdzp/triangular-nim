//! Computes all Nim values for the board.

use crate::board::{Board, Tagged, Type, SIZE, SIZEU};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::HashSet;

/// An upper bound for the number of components of the board, after taking our optimizations into
/// account.
const MAX_COMPONENTS: usize = 8;
/// Number of entries to write in each hash table before dumping it on the main one.
const HASH_LEN: usize = 1_024;

/// Stores the info required to calculate the Nim value of a board.
#[derive(Debug)]
struct State {
    /// The (connected) board we're analyzing.
    board: Board,
    /// The connected components after we made a move, whose Nim values we want to find.
    components: SmallVec<[Board; MAX_COMPONENTS]>,

    /// The running Nim value for the components.
    nim: u8,
    /// A bitield that represents what Nim values the moves can have.
    mex_set: Type,

    /// The component index to check.
    index: u8,
    /// The move number to apply.
    mov: u8,
}

impl State {
    /// Initial state.
    fn new(board: Board) -> Self {
        Self {
            board,
            components: SmallVec::new(),

            nim: 0,
            mex_set: 0,

            index: 0,
            mov: 0,
        }
    }

    /// Advances to the next move that matches for the board. Returns the board corresponding to it.
    fn next_move(&mut self) -> Option<Board> {
        // Go to the next move.
        while (self.mov as usize) < Board::MOVE_COUNT {
            let mov_board = Board::MOVES[self.mov as usize];
            self.mov += 1;

            // We found a move that fits.
            if self.board.fits(mov_board) {
                return Some(mov_board);
            }
        }

        None
    }
}

/// Represents the stack throughout the entire calculation of all Nim values.
struct Stack {
    /// The stack of states.
    states: Vec<State>,
    /// Hash set with Nim values.
    hash: HashSet<Tagged>,
    /// Return value from the stack.
    ret: Option<u8>,
}

impl Stack {
    /// Initializes a new stack, starting at a board.
    fn new(board: Board) -> Self {
        Self {
            states: vec![State::new(board)],
            hash: HashSet::new(),
            ret: None,
        }
    }

    /// Performs one step in the procedure for [`Board::moves`]. Returns whether we're done.
    fn step_moves(&mut self, main_hash: Option<&HashSet<Tagged>>) -> bool {
        let state;
        if let Some(s) = self.states.last_mut() {
            state = s;
        } else {
            return false;
        }

        // If we're returning a value, add it to the Nim total.
        if let Some(ret) = self.ret {
            state.nim ^= ret;
        }
        self.ret = None;

        // Starting out a new move.
        if state.index == state.components.len() as u8 {
            // If we just finished analyzing a move, register the nim value we found.
            if !state.components.is_empty() {
                state.mex_set |= 1 << state.nim;
            }
            state.index = 1;

            // Go to the next move.
            while let Some(mov_board) = state.next_move() {
                state.nim = 0;
                let new_board = state.board.mov(mov_board);

                // Account for simple cases without entering the main loop again.
                if new_board == Board::EMPTY {
                    state.mex_set |= 1;
                } else if new_board.0.count_ones() == 1 {
                    state.mex_set |= 2;
                } else {
                    state.components.clear();
                    let (nim, mut components) = new_board.components_reduced();
                    state.nim = nim;

                    let mut i = 0;
                    while i < components.len() {
                        // Get cached Nim values if possible.
                        // Search both hash tables, starting with the main one.
                        let comp = components[i].into();
                        let mut cache = None;
                        if let Some(hash) = main_hash {
                            cache = hash.get(&comp);
                        }
                        if cache.is_none() {
                            cache = self.hash.get(&comp);
                        }

                        if let Some(board) = cache {
                            state.nim ^= board.tag();
                            components.swap_remove(i);
                        } else {
                            i += 1;
                        }
                    }

                    state.components = components;
                    if let Some(&fst) = state.components.first() {
                        self.states.push(State::new(fst));
                        return true;
                    }
                    state.mex_set |= 1 << state.nim;
                }
            }

            // We ran out of moves, return!
            let mut mex = 0;
            while state.mex_set % 2 == 1 {
                mex += 1;
                state.mex_set >>= 1;
            }

            self.ret = Some(mex);
            self.hash.insert(Tagged::new(state.board, mex));
            self.states.pop();
        }
        // Check the next component.
        else {
            let board = state.components[state.index as usize];
            state.index += 1;
            self.states.push(State::new(board));
        }

        true
    }
}

impl Board {
    /// Gets the components of the board after applying some trivial optimizations. Also returns an
    /// initial Nim value.
    pub fn components_reduced(self) -> (u8, SmallVec<[Board; MAX_COMPONENTS]>) {
        let mut nim = 0;
        let mut components = SmallVec::new();

        'outer: for comp in self.components() {
            if comp.0.count_ones() == 1 {
                nim ^= 1;
            } else {
                let norm = comp.normalize();

                // Delete duplicates.
                for (idx, &other) in components.iter().enumerate() {
                    if other == norm {
                        components.swap_remove(idx);
                        continue 'outer;
                    }
                }

                components.push(norm);
            }
        }

        (nim, components)
    }

    /// Evaluates a board position based on a precomputed hash table.
    ///
    /// Returns `None` if it can't find a position.
    pub fn eval(self, hash: &HashSet<Tagged>) -> Option<u8> {
        let (mut nim, components) = self.components_reduced();
        for comp in components {
            if let Some(tagged) = hash.get(&comp.into()) {
                nim ^= tagged.tag();
            } else {
                return None;
            }
        }

        Some(nim)
    }

    /// Compute the Nim values for this board size.
    pub fn moves() -> impl Iterator<Item = Tagged> {
        let mut stack = Stack::new(Board::FULL);

        // Calculate first few moves.
        // This is done sequentially to avoid constantly calculating the same thing.
        while stack.hash.len() < HASH_LEN {
            if !stack.step_moves(None) {
                return stack.hash.into_iter();
            }
        }

        // Multithread for the rest.
        // We start with different boards to hopefully find different positions across threads.
        let hash = std::sync::RwLock::new(stack.hash.clone());
        let mut stacks = Vec::with_capacity(SIZEU + 1);
        stacks.push(stack);
        for i in 0..SIZE {
            stacks.push(Stack::new(Board((1 << SIZE) - (1 << i))));
        }

        stacks.into_par_iter().for_each(|mut stack| {
            let mut finish = false;

            while !finish {
                // Compute moves in this thread.
                while stack.hash.len() < HASH_LEN {
                    if !stack.step_moves(Some(&hash.read().unwrap())) {
                        finish = true;
                        break;
                    }
                }

                // Once we've computed a handful of moves, add them to the hash table.
                let mut hash = hash.write().unwrap();
                for board in stack.hash.drain() {
                    hash.insert(board);
                }
            }
        });

        hash.into_inner().unwrap().into_iter()
    }
}
