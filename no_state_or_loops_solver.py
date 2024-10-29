"""A sudoku solver with gratuitous use of map and filter.

Just to show one way that it can be done. This isn't a good approach in Python, and
I've made it worse just to make things look and work just a bit more like they would
in Haskell (in my narrow conception of Haskell, at least).

:author: Shay Hill
:created: 2024-10-29
"""


import functools as ft
import itertools as it
import operator as op
from typing import Callable, Container, Iterable, Iterator, TypeVar

# ===============================================================================
#   Create functions that return the indices of the cells in the same row, column, or
#   box given a cell index.
# ===============================================================================


def _new_row_idxer(side_len: int) -> Callable[[int], tuple[int, ...]]:
    """Return a function that returns the indices of the cells in a row."""

    def get_row_idxs(cell_idx: int) -> tuple[int, ...]:
        at_row = cell_idx // side_len * side_len
        return tuple(range(at_row, at_row + side_len))

    return get_row_idxs


def _new_col_idxer(side_len: int) -> Callable[[int], tuple[int, ...]]:
    """Return a function that returns the indices of the cells in a col."""

    def get_col_idxs(cell_idx: int) -> tuple[int, ...]:
        at_col = cell_idx % side_len
        return tuple(range(at_col, side_len**2, side_len))

    return get_col_idxs


def _new_box_idxer(side_len: int) -> Callable[[int], tuple[int, ...]]:
    """Return a function that returns the indices of the cells in a box."""
    box_side_len = int(side_len**0.5)

    def get_box_idxs(cell_idx: int) -> tuple[int, ...]:
        min_row = (cell_idx // side_len // box_side_len) * box_side_len
        min_col = (cell_idx % side_len // box_side_len) * box_side_len
        return tuple(
            map(
                lambda rc: rc[0] * side_len + rc[1],
                it.product(
                    range(min_row, min_row + box_side_len),
                    range(min_col, min_col + box_side_len),
                ),
            )
        )

    return get_box_idxs


# ===============================================================================
#   Helper functions that make the Python work in a BIT more Haskell-ish way.
# ===============================================================================


_T = TypeVar("_T")
_U = TypeVar("_U")


def op_contains_flipped(item: _T, container: Container[_T]) -> bool:
    """Reverse the arguments to operator.contains."""
    return op.contains(container, item)


def _app(arg: _T, funcs: Iterable[Callable[[_T], _U]]) -> tuple[_U, ...]:
    """Apply a function to a sequence of arguments.

    This is the inverse of map."""

    def app(f: Callable[[_T], _U]) -> _U:
        return f(arg)

    return tuple(map(app, funcs))


def _try_items(board: str, idxs: tuple[int, ...]) -> str:
    """Return the items in board at the given indices if they exist.

    :param board: The board to get items from.
    :param idxs: A monotonically increasing iterable of indices.
    :yield: The items in board at the given indices if they exist.
    """

    def try_items(idxs_: tuple[int, ...], result: str = "") -> str:
        try:
            head, tail = idxs_[0], idxs_[1:]
        except IndexError:
            return result
        try:
            return try_items(tail, result + board[head])
        except IndexError:
            return result

    return try_items(idxs)


# ===============================================================================


def _iter_candidates(side_len: int) -> Iterator[str]:
    """Yield a character for each possible value in a cell.

    123456789:;<=>?@ABCDEFG[:side_len]
    """
    return map(chr, range(49, 49 + side_len))


def new_solver(puzzle: str) -> Callable[[str], Iterator[str]]:
    """Return a function that yields solutions to the given puzzle.

    :param puzzle: The puzzle to solve. A tuple of 81 integers with known values 1
        through 9 and 0 in spots where we need to fill in.
    :return: A function that yields solutions to the given puzzle.
    """
    if not (len(puzzle) ** 0.25) ** 4 == len(puzzle):
        msg = "Puzzle must be a square of a square"
        raise ValueError(msg)

    side_len = int(len(puzzle) ** 0.5)
    idx_group_getters = _app(side_len, (_new_col_idxer, _new_row_idxer, _new_box_idxer))
    iter_candidates = ft.partial(_iter_candidates, side_len)

    def solve_from_here(board: str = "") -> Iterator[str]:
        """Yield solutions to the puzzle, starting from board."""
        if len(board) == len(puzzle):
            yield board
            return

        if puzzle[len(board)] != "0":
            yield from solve_from_here(board + puzzle[len(board)])
            return

        idx_groups = [f(len(board)) for f in idx_group_getters]

        def add_candidate_if_no_conflicts(board: str, candidate: str) -> str | None:
            """Add one candidate to the board if it doesn't conflict."""

            candidate_in_group_values = ft.partial(op_contains_flipped, candidate)

            try_puzzle = ft.partial(_try_items, puzzle)
            if any(map(candidate_in_group_values, map(try_puzzle, idx_groups))):
                return
            try_board = ft.partial(_try_items, board)
            if any(map(candidate_in_group_values, map(try_board, idx_groups))):
                return
            return board + candidate

        yield from it.chain(
            *map(
                solve_from_here,
                filter(
                    None,
                    map(
                        ft.partial(add_candidate_if_no_conflicts, board),
                        iter_candidates(),
                    ),
                ),
            )
        )

    return solve_from_here


puzzles = [
    "028057000030240070009000200013060000500091006090370050000706492082010560000520810",
    "040200105100406700009037006014065807007819402698004001430601028061940570972080610",
    "004961830023050609009207145015340790000780001090120004401000900800000400060004520",
    "000045003104208579508039240002450008407801062689327415001000954040012037000574821",
    "651000970000109060009056004100540000400091207807060015002080006006702001070005342",
    "083960754045237089679008023310546097400780231790103460020090000064375902907010006",
    "074500026030146080609200045312058000006701038790002001520003960060010572007620800",
    "001053000035000479680207100120005890056000000897600354312000068008020940000800500",
    "571040629034169578609257134120470890006390010008612005015700960800030751007521083",
    "893654271024078069500120000215430097340890120789010430430001952001902083972580610",
    "070020046060000890200800715084097000710000059000130480697002008058000060430080070",
]


def _format_solution(puzzle_and_solution: tuple[str, str]) -> str:
    """Format a solution to a puzzle."""
    puzzle, solution = puzzle_and_solution
    return f"Puzzle:\n{puzzle}\nSolution:\n{solution}\n"


if __name__ == "__main__":
    solvers = map(new_solver, puzzles)
    solutions = _app("", solvers)
    first_solutions = map(next, solutions)
    print("\n".join(map(_format_solution, zip(puzzles, first_solutions))))
