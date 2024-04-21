//! Computes all Nim values for the board.

use crate::board::{Board, Tagged, Type};
use smallvec::SmallVec;
use std::collections::HashSet;

/// An upper bound for the number of components of the board, after taking our optimizations into
/// account.
const MAX_COMPONENTS: usize = 8;

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
        let mut hash = HashSet::<Tagged>::new();
        let mut stack = vec![State::new(Board::FULL)];
        let mut ret = None;

        'outer: while let Some(state) = stack.last_mut() {
            assert_eq!(state.board, state.board.normalize());
            // If we're returning a value, add it to the Nim total.
            if let Some(ret) = ret {
                state.nim ^= ret;
            }
            ret = None;

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
                            if let Some(board) = hash.get(&components[i].into()) {
                                state.nim ^= board.tag();
                                components.swap_remove(i);
                            } else {
                                i += 1;
                            }
                        }

                        state.components = components;
                        if let Some(&fst) = state.components.first() {
                            stack.push(State::new(fst));
                            continue 'outer;
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

                ret = Some(mex);
                hash.insert(Tagged::new(state.board, mex));
                stack.pop();
            }
            // Check the next component.
            else {
                let board = state.components[state.index as usize];
                state.index += 1;
                stack.push(State::new(board));
            }
        }

        hash.into_iter()
    }
}
