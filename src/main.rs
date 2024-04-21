//! Main program loop.

#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]

mod board;
mod compute;

use board::{Board, Tagged, N, TAGGED_BYTES};
use fnv::FnvHashSet as HashSet;
use std::{
    fs::File,
    io::{BufReader, ErrorKind as IoErrorKind, Read, Result as IoResult, Write},
};

/// Default file name for saving the Nim boards.
const NIM_FILE: &str = "solutions.nim";
/// Error message when a position can't be found on the hash table.
const NIM_NOT_FOUND: &str = "Couldn't find Nim value. Have you loaded the moves yet?";

/// File argument description.
const FILE_ARG: &str = "file | The path to the file to load.";
/// Position argument.
const POS_ARG: &str = "pos | The value of the position to evaluate, as a binary string.";

/// Saves all Nim values found.
///
/// The first byte is the board size. All subsequent bytes are the tagged boards.
fn save_moves(path: &str) -> IoResult<()> {
    let mut file = File::create_new(path)?;
    file.write_all(&[N])?;
    for board in Board::moves() {
        file.write_all(&board.0)?;
    }

    Ok(())
}

/// Calls [`save_moves`] and logs the result.
fn save_moves_print(path: &str) {
    use std::time::Instant;
    let start = Instant::now();

    match save_moves(path) {
        Ok(()) => {
            let duration = start.elapsed();
            let secs = duration.as_secs();
            let millis = duration.subsec_millis();
            println!(
                "Moves succesfully saved in '{path}'. Took {:02}:{:02}.{millis}.",
                secs / 60,
                secs % 60
            );
        }

        Err(error) => {
            let err_str = match error.kind() {
                IoErrorKind::AlreadyExists => "file already exists",
                IoErrorKind::NotFound => "file path not found",
                _ => "unknown error",
            };
            println!("Moves could not be saved: {err_str}.");
        }
    }
}

/// Reads Nim moves from a file.
fn load_moves(path: &str) -> IoResult<HashSet<Tagged>> {
    let mut file = BufReader::new(File::open(path)?).bytes();
    let err = Err(IoErrorKind::InvalidData.into());

    if let Some(Ok(N)) = file.next() {
        let mut buf = [0; TAGGED_BYTES];
        let mut hash = HashSet::default();

        while let Some(byte) = file.next() {
            buf[0] = byte?;
            for buf_byte in buf.iter_mut().skip(1) {
                if let Some(file_byte) = file.next() {
                    *buf_byte = file_byte?;
                } else {
                    return err;
                }
            }

            hash.insert(Tagged(buf));
        }

        Ok(hash)
    } else {
        err
    }
}

/// Calls [`load_moves`] and logs the result.
fn load_moves_print(hash: &mut HashSet<Tagged>, file: &str) {
    match load_moves(file) {
        Ok(moves) => {
            *hash = moves;
            println!("Moves succesfully loaded from '{file}'.");
        }
        Err(error) => {
            let err_str = match error.kind() {
                IoErrorKind::NotFound => "file not found",
                IoErrorKind::InvalidData => "file is malformed",
                _ => "unknown error",
            };
            println!("Moves could not be loaded: {err_str}.");
        }
    }
}

/// Prints text for an unrecognized command.
fn unrecognized_command(c: &str) {
    println!("Command '{c}' not recognized.");
}

/// Prints help text.
fn help(command: Option<&str>) {
    match command {
        None => println!(
            "\nCommand list:
  bin            | Prints out the board in binary.
  eval [pos?]    | Evaluates the current or specified position.
  exit           | Exit program.
  help [name?]   | Get help about all commands or a specific command.
  load [file?]   | Loads all precomputed Nim values.
  move [move]    | Applies a move to the board.
  opt  [pos?]    | Finds the optimal move from a position.
  save [file?]   | Computes all Nim values and stores them on disk.
  set  [pos]     | Sets the board to a particular position.
  show [pos?]    | Shows the current or specified position.
  reset          | Resets the board."
        ),

        Some("binary" | "bin") => println!(
            "\nPrints out the board in binary.

Syntax: bin"
        ),

        Some("eval") => println!(
            "\nEvaluates the current or specified position, prints out its Nim value. This
command only works once the moves have been loaded from disk.

Syntax: eval [pos?]
  {POS_ARG}"
        ),

        Some("exit") => println!(
            "\nExits the program immediately.

Syntax: exit"
        ),

        Some("help") => println!(
            "\nShows the list of all available commands, or gives help for a specific command.

Syntax: help [name?]
  name | The name of the command to give help for."
        ),

        Some("load") => println!(
            "\nLoads the Nim solutions from disk. By default, searches for 'solutions.nim' in
the root folder the program is run from. This command runs automatically on
program start.

Syntax: load [file?]
  {FILE_ARG}"
        ),

        Some("move" | "mov") => println!(
            "\nApplies a move to the board.
        
Syntax: move [move]
  move | The move, specified by its endpoints in the format 'row,column'."
        ),

        Some("optimal" | "opt") => println!(
            "\nFinds the optimal moves from the current or specified position.

Syntax: opt [pos?]
  {POS_ARG}"
        ),

        Some("save") => println!(
            "\nSaves the Nim solutions into disk. Might take a while. By default, saves these
solutions in `solutions.nim` in the root folder the program is run from.

Syntax: save [file?]
  {FILE_ARG}"
        ),

        Some("set") => println!(
            "\nSets the board to the specified position.

Syntax: set [pos]
  {POS_ARG}"
        ),

        Some("show") => println!(
            "\nDisplays the current or specified position on screen.

Syntax: show [pos?]
  {POS_ARG}"
        ),

        Some("reset") => println!(
            "\nResets the board to its initial state.

Syntax: reset"
        ),

        Some(c) => unrecognized_command(c),
    };
}

/// Main loop.
#[allow(clippy::too_many_lines)]
fn main() {
    // Quick sanity check.
    assert!(
        board::Type::BITS as u8 >= board::SIZE,
        "Board type set incorrectly!"
    );
    assert!(
        board::TaggedType::BITS as u8 >= board::TAGGED_SIZE,
        "Tagged board type set incorrectly!"
    );

    println!(
        "Triangular Nim Solver v.{}. Type 'help' for help.\n",
        env!("CARGO_PKG_VERSION")
    );

    let mut board = Board::FULL;
    let mut buf = String::new();
    let mut optimal = Vec::new();

    println!("Loading moves...");
    let mut hash = HashSet::default();
    load_moves_print(&mut hash, NIM_FILE);
    println!("{board}");

    // Program loop.
    'main: loop {
        print!("\n> ");
        std::io::stdout().flush().expect("flush failed!");
        buf.clear();
        if let Err(error) = std::io::stdin().read_line(&mut buf) {
            println!("Error: {error}");
        }

        buf.make_ascii_lowercase();
        let mut command = buf.split(char::is_whitespace).filter(|s| !s.is_empty());

        match command.next() {
            // Prints the board in binary.
            Some("binary" | "bin") => println!("{:b}", board.0),

            // Evaluate the board.
            Some("eval") => {
                if let Some(nim) = board.eval(&hash) {
                    println!("Nim value: {nim}");
                } else {
                    println!("{NIM_NOT_FOUND}");
                }
            }

            // Exit program.
            Some("exit") => return,

            // Show help dialog.
            Some("help") => help(command.next()),

            // Read positions from file.
            Some("load") => load_moves_print(&mut hash, command.next().unwrap_or(NIM_FILE)),

            // Applies a move.
            Some("move" | "mov") => {
                if let Some(start) = command.next() {
                    match start.parse() {
                        Ok(start) => {
                            // End position defaults to start.
                            let end;
                            if let Some(new_end) = command.next() {
                                match new_end.parse() {
                                    Ok(new_end) => end = new_end,
                                    Err(error) => {
                                        println!("{error}");
                                        continue;
                                    }
                                }
                            } else {
                                end = start;
                            }

                            // Read and apply move.
                            if let Some(mov) = board::Move::new(start, end) {
                                let mov: Board = mov.into();
                                if board.fits(mov) {
                                    board = board.mov(mov);
                                    println!("{board}");
                                } else {
                                    println!("Move cannot be performed.");
                                }
                            } else {
                                println!("Invalid move.");
                            }
                        }

                        Err(error) => println!("{error}"),
                    }
                } else {
                    println!("Missing argument [move].");
                }
            }

            // Shows the optimal moves.
            Some("optimal" | "opt") => {
                let eval_board;
                if let Some(s) = command.next() {
                    match s.parse() {
                        Ok(new_board) => {
                            eval_board = new_board;
                        }

                        Err(err) => {
                            println!("{err}");
                            continue;
                        }
                    }
                } else {
                    eval_board = board;
                }

                match eval_board.eval(&hash) {
                    Some(0) => {
                        println!("This is a losing position!");
                        continue;
                    }

                    None => {
                        println!("{NIM_NOT_FOUND}");
                        continue;
                    }

                    _ => {}
                }

                optimal.clear();

                for mov in Board::MOVES {
                    if eval_board.fits(mov) {
                        let new_board = eval_board.mov(mov);
                        match new_board.eval(&hash) {
                            Some(0) => optimal.push(new_board),

                            None => {
                                println!("{NIM_NOT_FOUND}");
                                continue 'main;
                            }

                            _ => {}
                        }
                    }
                }

                Board::print_array(&optimal);
            }

            // Reset board.
            Some("reset") => {
                board = Board::FULL;
                println!("Board reset.\n{board}");
            }

            // Compute all positions, save to file.
            Some("save") => save_moves_print(command.next().unwrap_or(NIM_FILE)),

            // Sets board.
            Some("set") => {
                if let Some(s) = command.next() {
                    match s.parse() {
                        Ok(new_board) => {
                            board = new_board;
                            println!("Board updated.\n{board}");
                        }
                        Err(err) => println!("{err}"),
                    }
                } else {
                    println!("Missing argument [pos].");
                }
            }

            // Show the board.
            Some("show") => {
                if let Some(s) = command.next() {
                    match s.parse::<Board>() {
                        Ok(show_board) => println!("{show_board}"),
                        Err(err) => println!("{err}"),
                    }
                } else {
                    println!("{board}");
                }
            }

            // Unknown command.
            Some(c) => unrecognized_command(c),

            // No command.
            None => {}
        }
    }
}
