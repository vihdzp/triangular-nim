//! Defines the [`Board`] type and operations on it.

use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult, Write},
    num::IntErrorKind,
    str::FromStr,
};

/// Integer type used to store a [`Board`].
pub type Type = u64;
/// Integer type needed to store a [`Tagged`]. This is only used in auxiliary computations.
pub type TaggedType = u64;
/// Integer type used to store the move index.
pub type MovType = u16;

/// Length of the board.
pub const N: u8 = 8;
/// Length of the board.
pub const NU: usize = N as usize;
/// Size of the board.
pub const SIZE: u8 = N * (N + 1) / 2;
/// Size of the board.
pub const SIZEU: usize = SIZE as usize;
/// Size of the tagged board.
pub const TAGGED_SIZE: u8 = SIZE + SIZE.ilog2() as u8 + 1;

/// Bytes for the board type.
pub const BYTES: usize = Type::BITS as usize / 8;
/// The number of bytes needed to optimally store a [`Tagged`] for a given value of [`N`].
pub const TAGGED_BYTES: usize = (TAGGED_SIZE as usize + 7) / 8;

/// Represents the triangular grid and the counters that remain on it.
///
/// Each bit represents a different tile. The topmost bits should go unused.
///
/// ```txt
///              0
///            1   2
///          3   4   5
///        6   7   8   9
///     10  11  12  13  14
///   15  16  17  18  19  20
/// 21  22  23  24  25  26  27
/// ```
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Board(pub Type);

/// Gap between baords.
const GAP: u8 = 2;
/// Boards per row.
const BOARD_PER_ROW: usize = 80 / (2 * N + GAP) as usize;

/// Auxiliary type to define `Display` for.
struct Array<'a>(&'a [Board]);

impl<'a> Display for Array<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let array = self.0;
        for board in array {
            #[allow(clippy::print_in_format_impl)]
            if board.0 & !Board::FULL.0 != 0 {
                eprintln!("Board has extra bits!");
            }
        }

        let chunk_count = (array.len() + BOARD_PER_ROW - 1) / BOARD_PER_ROW;
        for chunk in 0..chunk_count {
            let start = BOARD_PER_ROW * chunk;
            let end = (BOARD_PER_ROW * (chunk + 1)).min(array.len());
            let slice = &array[start..end];
            let mut pos = 0;

            for row in 0..N {
                for _ in 0..(N - row - 1) {
                    f.write_char(' ')?;
                }

                for (idx, board) in slice.iter().enumerate() {
                    let mut inner_pos = pos;
                    for col in 0..=row {
                        let mask = 1 << inner_pos;
                        f.write_char(if board.0 & mask == mask { 'O' } else { '·' })?;

                        if col != row {
                            f.write_char(' ')?;
                        }

                        inner_pos += 1;
                    }

                    if idx != slice.len() - 1 {
                        for _ in 0..(2 * (N - row - 1) + GAP) {
                            f.write_char(' ')?;
                        }
                    }
                }

                pos += row + 1;

                if row != N - 1 {
                    f.write_char('\n')?;
                }
            }

            if chunk != chunk_count - 1 {
                f.write_char('\n')?;
            }
        }

        Ok(())
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", Array(std::array::from_ref(self)))
    }
}

impl Board {
    /// Prints an array of boards.
    pub fn print_array(array: &[Self]) {
        println!("{}", Array(array));
    }
}

/// Converts a row and column to a bit index.
const fn get_pos(row: u8, col: u8) -> u8 {
    row * (row + 1) / 2 + col
}

/// Rotates a position on the triangle clockwise.
const fn rotate(row: u8, col: u8) -> (u8, u8) {
    (N - 1 + col - row, N - 1 - row)
}

/// All tuples to be rotated.
const ROTATION: [[u8; 3]; SIZEU / 3] = {
    let mut rots = [[0; 3]; SIZEU / 3];
    let mut idx = 0;
    let mut pos = 0;

    let mut row = 0;
    'outer: while row < N {
        let mut col = 0;
        while col <= row {
            let (row_1, col_1) = rotate(row, col);
            let (row_2, col_2) = rotate(row_1, col_1);

            let pos_1 = get_pos(row_1, col_1);
            let pos_2 = get_pos(row_2, col_2);

            if pos < pos_1 && pos < pos_2 {
                rots[idx] = [pos, pos_1, pos_2];
                idx += 1;
                if idx == SIZEU / 3 {
                    break 'outer;
                }
            }

            col += 1;
            pos += 1;
        }

        row += 1;
    }

    rots
};

impl Board {
    /// The empty board.
    pub const EMPTY: Self = Self(0);
    /// The starting board.
    pub const FULL: Self = Self((1 << (N * (N + 1) / 2)) - 1);

    /// [`Board`]s corresponding to each row.
    pub const ROWS: [Self; NU] = {
        let mut boards = [Self(0); NU];
        let mut ones = 1;
        let mut col = 0;

        while col < N {
            let shift = col * (col + 1) / 2;
            boards[col as usize].0 = ones << shift;
            col += 1;

            ones *= 2;
            ones += 1;
        }

        boards
    };

    /// Bit masks for the right-slanting columns in a [`Board`].
    pub const R_COLS: [Self; NU] = {
        let mut boards = [Self(0); NU];
        let mut row = 0;

        while row < N {
            let mut col = 0;
            let mut entry = 0;

            while col <= row {
                let fst = N - row + col - 1;
                entry += 1 << (fst * (fst + 1) / 2 + col);
                col += 1;
            }

            boards[row as usize].0 = entry;
            row += 1;
        }

        boards
    };

    /// Bit masks for the left-slanting columns in a [`Board`].
    pub const L_COLS: [Self; NU] = {
        let mut boards = [Self(0); NU];
        let mut row = 0;

        while row < N {
            let mut col = 0;
            let mut entry = 0;

            while col <= row {
                let fst = N - row + col;
                entry += 1 << (fst * (fst + 1) / 2 - col - 1);
                col += 1;
            }

            boards[row as usize].0 = entry;
            row += 1;
        }

        boards
    };

    /// Initializes a board. This zeroes out any extra bits!
    pub fn new(num: Type) -> Self {
        Self(num & ((1 << SIZE) - 1))
    }

    /// Gets whether there's a counter at the given bit.
    pub fn get(self, pos: u8) -> bool {
        (self.0 >> pos) & 1 == 1
    }

    /// Sets a counter at the given bit.
    pub fn set(&mut self, val: bool, pos: u8) {
        let mask = 1 << pos;
        self.0 &= !mask;
        self.0 |= Type::from(val) * mask;
    }

    /// Can the given move be performed on this board?
    pub fn fits(self, mov: Self) -> bool {
        self.0 & mov.0 == mov.0
    }

    /// Performs a move.
    pub fn mov(self, mov: Self) -> Self {
        Self(self.0 & !mov.0)
    }

    /// Shifts the board northwest.
    pub fn shift_nw(self, count: u8) -> Self {
        let count = count as usize;
        Self(
            Self::ROWS
                .iter()
                .enumerate()
                .skip(count)
                .map(|(r, mask)| {
                    let shift = (count * (2 * r - count + 3) / 2) as u8;
                    ((self.0 & mask.0) >> shift) & Self::ROWS[r - count].0
                })
                .sum(),
        )
    }

    /// Shifts the board northeast.
    pub fn shift_ne(self, count: u8) -> Self {
        let count = count as usize;
        Self(
            Self::ROWS
                .iter()
                .enumerate()
                .skip(count)
                .map(|(r, mask)| {
                    let shift = ((r * (r + 1) - (r - count) * (r - count + 1)) / 2) as u8;
                    ((self.0 & mask.0) >> shift) & Self::ROWS[r - count].0
                })
                .sum(),
        )
    }

    /// Makes the board flush with the top corner.
    pub fn flush_top(mut self) -> Self {
        // Flush northwest.
        for (c, col) in Self::L_COLS.iter().rev().enumerate() {
            if self.0 & col.0 != 0 {
                if c != 0 {
                    self = self.shift_nw(c as u8);
                }
                break;
            }
        }

        // Flush northeast.
        for (c, col) in Self::R_COLS.iter().rev().enumerate() {
            if self.0 & col.0 != 0 {
                if c != 0 {
                    self = self.shift_ne(c as u8);
                }
                break;
            }
        }

        self
    }

    /// Reflects the board horizontally.
    fn mirror(self) -> Self {
        const BITS: usize = Type::BITS as usize;

        Self(
            Self::ROWS
                .iter()
                .enumerate()
                .map(|(r, row)| {
                    let t = r * (r + 2);
                    let rev = (self.0 & row.0).reverse_bits();
                    if t < BITS {
                        rev >> (BITS - 1 - t)
                    } else {
                        rev << (t - BITS + 1)
                    }
                })
                .sum(),
        )
    }

    /// Exchanges the counters on three tiles in clockwise order.
    fn rot_3(&mut self, a: u8, b: u8, c: u8) {
        let get_c = self.get(c);
        self.set(self.get(b), c);
        self.set(self.get(a), b);
        self.set(get_c, a);
    }

    /// Rotates the board clockwise.
    fn rotate(mut self) -> Self {
        for [a, b, c] in ROTATION {
            self.rot_3(a, b, c);
        }

        self
    }

    /// Returns a fixed representative for all boards with this same shape.
    pub fn normalize(self) -> Self {
        let rot_0 = self.flush_top();
        let rot_1 = self.rotate().flush_top();
        let rot_2 = rot_1.rotate().flush_top();

        [
            rot_0,
            rot_1,
            rot_2,
            rot_0.mirror(),
            rot_1.mirror(),
            rot_2.mirror(),
        ]
        .into_iter()
        .min()
        .unwrap()
    }

    /// Initializes a new iterator over connected components.
    pub fn components(self) -> Components {
        Components::new(self)
    }
}

/// Error in parsing a board.
#[derive(Debug)]
pub enum ParseBoardError {
    /// String contains digit other than 0 and 1.
    InvalidDigit,
    /// Integer was too big.
    Size,
}

impl Display for ParseBoardError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let err_str = match self {
            Self::InvalidDigit => "String can only contain digits 0 and 1.",
            Self::Size => "Integer too big.",
        };
        write!(f, "{err_str}")
    }
}
impl Error for ParseBoardError {}

impl FromStr for Board {
    type Err = ParseBoardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match Type::from_str_radix(s, 2) {
            Ok(num) => {
                if num < (1 << SIZE) {
                    Ok(Self(num))
                } else {
                    Err(ParseBoardError::Size)
                }
            }

            Err(error) => {
                match error.kind() {
                    IntErrorKind::InvalidDigit => Err(ParseBoardError::InvalidDigit),
                    IntErrorKind::PosOverflow => Err(ParseBoardError::Size),

                    // These cases should not be reachable.
                    _ => Ok(Self::EMPTY),
                }
            }
        }
    }
}

/// Gets the connected components of a board.
pub struct Components {
    /// The board being investigated.
    board: Board,
    /// First position to search the next component.
    index: u8,
}

impl Components {
    /// Initializes a new iterator over connected components.
    pub const fn new(board: Board) -> Self {
        Self { board, index: 0 }
    }
}

/// Encodes the adjacency graph of the board.
pub const ADJACENT: [u8; SIZEU * 6] = {
    let mut adj = [SIZE; SIZEU * 6];
    let mut pos = 0;

    let mut row = 0;
    while row < N {
        let mut col = 0;
        while col <= row {
            let mut idx = (6 * pos) as usize;

            // West, northwest.
            if col != 0 {
                adj[idx] = pos - 1;
                adj[idx + 1] = pos - row - 1;
                idx += 2;
            }
            // East, northeast.
            if col != row {
                adj[idx] = pos + 1;
                adj[idx + 1] = pos - row;
                idx += 2;
            }
            // Southwest, southeast.
            if row != N - 1 {
                adj[idx] = pos + row + 1;
                adj[idx + 1] = pos + row + 2;
            }

            pos += 1;
            col += 1;
        }

        row += 1;
    }

    adj
};

impl Iterator for Components {
    type Item = Board;

    fn next(&mut self) -> Option<Board> {
        // Find first unvisited position.
        loop {
            if self.index == SIZE {
                return None;
            }

            if self.board.get(self.index) {
                break;
            }

            self.index += 1;
        }

        let mut component = Board::EMPTY;
        let mut vertices = vec![self.index];

        // Depth-first search.
        while let Some(v) = vertices.pop() {
            let mask = 1 << v;
            if self.board.0 & mask != 0 {
                self.board.0 &= !mask;
                component.0 += mask;

                // Push adjacent vertices.
                let v = v as usize;
                for &w in &ADJACENT[(6 * v)..(6 * v + 6)] {
                    if w != SIZE {
                        vertices.push(w);
                    }
                }
            }
        }

        Some(component)
    }
}

impl Board {
    /// Number of possible moves.
    pub const MOVE_COUNT: usize = NU * NU * (NU + 1) / 2;

    /// All possible valid moves.
    pub const MOVES: [Self; Self::MOVE_COUNT] = {
        let mut moves = [Self::EMPTY; Self::MOVE_COUNT];
        let mut idx = 0;
        let mut init_mask = 1;

        let mut row = 0;
        while row < N {
            let mut col = 0;
            while col <= row {
                // Single counter.
                moves[idx] = Self(init_mask);
                idx += 1;

                // Row move.
                let mut length = 2;
                let mut mask = init_mask;
                let mut mov = mask;
                while length <= row - col + 1 {
                    mask <<= 1;
                    mov |= mask;
                    moves[idx] = Self(mov);

                    idx += 1;
                    length += 1;
                }

                // Column moves.
                length = 2;
                let mut l_mov = init_mask;
                let mut r_mov = init_mask;
                while length <= N - row {
                    l_mov |= 1 << get_pos(row + length - 1, col);
                    r_mov |= 1 << get_pos(row + length - 1, col + length - 1);
                    moves[idx] = Self(l_mov);
                    moves[idx + 1] = Self(r_mov);

                    idx += 2;
                    length += 1;
                }

                init_mask <<= 1;
                col += 1;
            }

            row += 1;
        }

        moves
    };
}

/// Represents a position on the board.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Position {
    /// Row number.
    row: u8,
    /// Column number.
    col: u8,
}

#[derive(Clone, Copy, Debug)]
pub enum ParsePositionError {
    /// Invalid digit in row or column.
    InvalidDigit,
    /// Position out of bounds.
    OutOfBounds,
    /// Invalid position format.
    InvalidFormat,
}

impl Display for ParsePositionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let err_str = match self {
            Self::InvalidDigit => "Invalid digit in number.",
            Self::OutOfBounds => "Position out of bounds.",
            Self::InvalidFormat => "Expected position in format 'row,column'.",
        };
        write!(f, "{err_str}")
    }
}

impl Error for ParsePositionError {}

impl FromStr for Position {
    type Err = ParsePositionError;

    fn from_str(s: &str) -> Result<Self, ParsePositionError> {
        let mut iter = s.split(',');
        let mut pos = [0u8; 2];
        for part in &mut pos {
            if let Some(p) = iter.next() {
                match p.parse() {
                    Ok(p) => *part = p,

                    Err(error) => match error.kind() {
                        IntErrorKind::Empty => return Err(ParsePositionError::InvalidFormat),
                        IntErrorKind::InvalidDigit => return Err(ParsePositionError::InvalidDigit),

                        // These cases should not be reachable.
                        _ => {}
                    },
                }
            } else {
                return Err(ParsePositionError::InvalidFormat);
            }
        }

        if iter.next().is_none() {
            if let Some(pos) = Self::new(pos[0], pos[1]) {
                Ok(pos)
            } else {
                Err(ParsePositionError::OutOfBounds)
            }
        } else {
            Err(ParsePositionError::InvalidFormat)
        }
    }
}

impl Position {
    /// Initializes a new position. Returns `None` if invalid.
    const fn new(row: u8, col: u8) -> Option<Self> {
        if col <= row && row < N {
            Some(Self { row, col })
        } else {
            None
        }
    }
}

impl From<Position> for [u8; 2] {
    fn from(value: Position) -> Self {
        [value.row, value.col]
    }
}

/// Type of move on the board.
#[derive(Clone, Copy, Debug)]
enum MoveType {
    /// Row move.
    Row,
    /// Left-leaning column move.
    LeftCol,
    /// Right-leaning column move.
    RightCol,
}

/// Represents a move on the board.
#[derive(Clone, Copy, Debug)]
pub struct Move {
    /// Start position.
    start: Position,
    /// End position.
    end: Position,
    /// Move type.
    typ: MoveType,
}

impl Move {
    /// Initializes a new move. Returns `None` if invalid.
    pub const fn new(start: Position, end: Position) -> Option<Self> {
        if start.row == end.row {
            Some(Self {
                start,
                end,
                typ: MoveType::Row,
            })
        } else if start.col == end.col {
            Some(Self {
                start,
                end,
                typ: MoveType::LeftCol,
            })
        } else if start.row + end.col == end.row + start.col {
            Some(Self {
                start,
                end,
                typ: MoveType::RightCol,
            })
        } else {
            None
        }
    }
}

/// Sort two values.
fn sort_2(a: u8, b: u8) -> (u8, u8) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

impl From<Move> for Board {
    fn from(mov: Move) -> Self {
        let [r1, c1] = mov.start.into();
        let [r2, c2] = mov.end.into();

        // Row move.
        match mov.typ {
            MoveType::Row => {
                let row = r1 * (r1 + 1) / 2;
                let (col_1, col_2) = sort_2(c1, c2);
                Self(((1 << (col_2 + 1)) - (1 << col_1)) << row)
            }

            MoveType::LeftCol => {
                let col = c1;
                let (row_1, row_2) = sort_2(r1, r2);
                let mut board = Board::EMPTY;

                for row in row_1..=row_2 {
                    board.0 |= 1 << get_pos(row, col);
                }
                board
            }

            MoveType::RightCol => {
                let (row_1, row_2, col_1) = if r1 < r2 { (r1, r2, c1) } else { (r2, r1, c2) };
                let mut board = Board::EMPTY;

                for row in row_1..=row_2 {
                    board.0 |= 1 << get_pos(row, col_1 + row - row_1);
                }
                board
            }
        }
    }
}

/// A [`Board`], compressed and with the topmost bits tagged.
///
/// The equality relation implemented is that of boards. Hashing is implemented in the same way.
#[derive(Clone, Copy, Debug)]
pub struct Tagged(pub [u8; TAGGED_BYTES]);

impl Display for Tagged {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "Value = {}\n{}", self.tag(), self.board())
    }
}

impl Tagged {
    /// Initializes a tagged board.
    pub fn new(board: Board, tag: u8) -> Self {
        Self(
            (TaggedType::from(board.0) | (TaggedType::from(tag) << SIZE)).to_le_bytes()
                [..TAGGED_BYTES]
                .try_into()
                .unwrap(),
        )
    }

    /// Recovers the board.
    pub fn board(self) -> Board {
        let mut bytes = [0; BYTES];
        for (i, &b) in self.0.iter().enumerate().take(BYTES) {
            bytes[i] = b;
        }

        Board::new(Type::from_le_bytes(bytes))
    }

    /// Gets the tag value.
    pub fn tag(self) -> u8 {
        // Case on whether the tag is contained in one or two bytes.
        if SIZE / 8 == TAGGED_SIZE / 8 {
            self.0[TAGGED_BYTES - 1] >> (SIZE % 8)
        } else {
            (u16::from_le_bytes([self.0[TAGGED_BYTES - 2], self.0[TAGGED_BYTES - 1]]) >> (SIZE % 8))
                as u8
        }
    }
}

impl From<Board> for Tagged {
    fn from(value: Board) -> Self {
        Self::new(value, 0)
    }
}

impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.board() == other.board()
    }
}

impl Eq for Tagged {}

impl std::hash::Hash for Tagged {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write(&self.board().0.to_ne_bytes());
    }
}
