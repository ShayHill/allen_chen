"""Use logic to solve a Sudoku puzzle. Follow up with backtracking if necessary.

:author: Shay Hill
:created: 2024-11-04
"""


import functools as ft
import itertools as it
from typing import (
    Iterable,
    Iterator,
    TypeVar,
    Hashable,
    Generic,
)
from collections import Counter


def _iter_candidates(order: int) -> Iterator[str]:
    """Yield a character for each possible value in a cell.

    123456789:;<=>?@ABCDEFG[:side_len]
    """
    return map(chr, range(49, 49 + order))


_T = TypeVar("_T")
_CellT = TypeVar("_CellT")

from typing import Sequence


def _try_items(seq: Sequence[_T], idxs: Iterable[int]) -> list[_T]:
    """Get a list of multiple items from a sequence."""

    def _iter_items() -> Iterator[_T]:
        for idx in idxs:
            try:
                yield seq[idx]
            except IndexError:
                return

    return list(_iter_items())


def _try_items_multi(seq: Sequence[_T], *idxs: Iterable[int]) -> Iterator[list[_T]]:
    """Get a list of multiple items from a sequence."""
    for idx in idxs:
        yield _try_items(seq, idx)


# ===================================================================================
#   Get row, col, and box indices for a Sudoku puzzle
#
#   Indices for a 9x9 puzzle:
#
#    0  1  2   3  4  5   6  7  8
#    9 10 11  12 13 14  15 16 17
#   18 19 20  21 22 23  24 25 26
#
#   27 28 29  30 31 32  33 34 35
#   36 37 38  39 40 41  42 43 44
#   45 46 47  48 49 50  51 52 53
#
#   54 55 56  57 58 59  60 61 62
#   63 64 65  66 67 68  69 70 71
#   72 73 74  75 76 77  78 79 80
# ===================================================================================

Idxs = tuple[int, ...]


@ft.lru_cache
def irows(order: int) -> list[Idxs]:
    """Return a nested list of row indices from top row."""
    return [tuple(range(i * order, i * order + order)) for i in range(order)]


@ft.lru_cache
def icols(order: int) -> list[Idxs]:
    """Return a nested list of column indices from left column."""
    return list(zip(*irows(order)))


@ft.lru_cache
def iboxs(order: int) -> list[Idxs]:
    """Return a nested list of box indices from top left box."""
    box_d = int(order ** (1 / 2))
    boxes: list[Idxs] = []
    for bl_rows in it.batched(irows(order), box_d):
        bl_cols = it.batched(zip(*bl_rows), box_d)
        boxes.extend(tuple(it.chain(*zip(*x))) for x in bl_cols)
    return boxes


def igroups(order: int) -> Iterator[Idxs]:
    """Return row, col, and box indices."""
    return it.chain(irows(order), icols(order), iboxs(order))


def iconstellation(order: int, idx: int) -> tuple[Idxs, Idxs, Idxs]:
    """Return the row, col, and box indices of a cell.

    :param order: the length of one side of the puzzle
    :param idx: the index of the next cell to be added to a progressive solution

    This is only called to index values in a progressive solution, so a partial
    constellation (only indexes cells that exist so far is the progressive solution)
    is returned.
    """

    def _is_in_range(x: int) -> bool:
        return x < idx

    box_d = int(order ** (1 / 2))
    which_row = idx // order
    which_col = idx % order
    which_box = which_row // box_d * box_d + which_col // box_d
    return (
        tuple(it.takewhile(_is_in_range, irows(order)[which_row])),
        tuple(it.takewhile(_is_in_range, icols(order)[which_col])),
        tuple(it.takewhile(_is_in_range, iboxs(order)[which_box])),
    )


# ===================================================================================


class Sudoku(Generic[_CellT]):
    """Values in a row, column, or box.

    Abstract away all the ugliness needed to index rows, cols, and boxes.
    """

    def __init__(self, puzzle: Sequence[_CellT], len: int):
        self.puzzle = puzzle
        self.len = len
        self.sl = int(len ** (1 / 2))  # length of one side of puzzle
        self._bl = int(len ** (1 / 4))  # length of one side of a box
        if not self._bl**4 == len:
            msg = "Puzzle len must be a square of a square."
            raise ValueError(msg)
        self._row_idxs = irows(self.sl)
        self._col_idxs = icols(self.sl)
        self._box_idxs = iboxs(self.sl)

    def _try_puzzle_items(self, idxs: Iterable[int]) -> list[_CellT]:
        """Get a list of multiple items from self._puzzle."""
        return _try_items(self.puzzle, idxs)

    def _get_row(self, idx: int) -> list[_CellT]:
        """Get one row of items from self._puzzle."""
        return self._try_puzzle_items(self._row_idxs[idx])

    def _get_col(self, idx: int) -> list[_CellT]:
        """Get one col of items from self._puzzle."""
        return self._try_puzzle_items(self._col_idxs[idx])

    def _get_box(self, idx: int) -> list[_CellT]:
        """Get one box of items from self._puzzle."""
        return self._try_puzzle_items(self._box_idxs[idx])

    def get_rows(self) -> Iterator[list[_CellT]]:
        """Return puzzle values in a row."""
        return (self._get_row(i) for i in range(self.sl))

    def get_cols(self) -> Iterator[list[_CellT]]:
        """Return puzzle values in a col."""
        return (self._get_col(i) for i in range(self.sl))

    def get_boxs(self) -> Iterator[list[_CellT]]:
        """Return puzzle values in a box."""
        return (self._get_box(i) for i in range(self.sl))

    def get_groups(self) -> Iterator[list[_CellT]]:
        """Return puzzle values in a row, col, and box."""
        return it.chain(self.get_rows(), self.get_cols(), self.get_boxs())

    def get_constellation(self, idx: int) -> list[list[_CellT]]:
        """Return the row, col, and box items of a cell."""
        which_row = idx // self.sl
        which_col = idx % self.sl
        which_box = which_row // self._bl * self._bl + which_col // self._bl
        return [
            self._get_row(which_row),
            self._get_col(which_col),
            self._get_box(which_box),
        ]


def solve_group_using_logic(group: list[set[str]]) -> None:
    """Solve a group (row, col, or box) using Sudoku logic."""

    # eliminate values that conflict with fixed (no alternatives) values
    fixed: set[str] = set()
    fixed.update(*(x for x in group if len(x) == 1))
    for set_ in (x for x in group if len(x) > 1):
        set_ -= fixed

    # eliminate values trapped in "bound siblings"
    siblings = Counter(map(frozenset, group))
    if len(siblings) < 2:
        return
    for bound in (set(k) for k, v in siblings.items() if len(k) == v):
        for set_ in (x for x in group if len(x) > 1):
            if set_ != bound:
                set_ -= bound


def solve_as_far_as_possible_using_logic(puzzle: str) -> list[set[str]]:
    """Iteratively apply `solve_group_using_logic`.

    :param puzzle: a string of characters representing a Sudoku puzzle with '.' for
        unknowns.
    :return: a set of possible values for each cell in the puzzle. If a puzzle has
        only one possible solution, all of these sets will be singletons, and the
        puzzle will be solved.
    """
    order = int(len(puzzle) ** (1 / 2))
    unknown = set(_iter_candidates(order))
    puzzle_as_sets = [unknown.copy() if c == "." else {c} for c in puzzle]

    prev_score = sum(map(len, puzzle_as_sets))
    while True:
        for group in _try_items_multi(puzzle_as_sets, *igroups(order)):
            solve_group_using_logic(group)
        score = sum(map(len, puzzle_as_sets))
        if score in {len(puzzle), prev_score}:
            break
        prev_score = score
    return puzzle_as_sets


def complete_solution_using_backtracking(
    partial_solution: list[set[str]],
) -> Iterator[str]:
    """Complete the solution using backtracking.

    :param partial_solution: a set of possible values for each cell in the puzzle.
        The result of calling `solve_as_far_as_possible_using_logic`.
    :return: a generator of complete solutions, each a string.
    """
    order = int(len(partial_solution) ** 0.5)
    idxs_of_known_values = {i for i, x in enumerate(partial_solution) if len(x) == 1}

    def _add_cell(
        solution: str = "", cells: list[set[str]] = partial_solution
    ) -> Iterator[str]:
        """If mult. candidate vals exist, try out each one at the end of the solution.

        At this point, the subsets of possible values do not conflict with the given
        puzzle values, so these can be excluded from validity checks.
        """
        if not cells:
            yield solution
            return

        if len(cells[0]) == 1:
            (next_cell,) = cells[0]  # unpack the singleton set
            yield from _add_cell(solution + next_cell, cells[1:])
            return

        idxs: set[int] = set().union(*iconstellation(order, len(solution)))
        idxs -= idxs_of_known_values
        potential_conflicts = {solution[i] for i in idxs}

        for candidate in sorted(cells[0] - potential_conflicts):
            yield from _add_cell(solution + candidate, cells[1:])

    yield from _add_cell()


puzzles = [
    ".................................................................................",
    ".28.57....3.24..7...9...2...13.6....5...91..6.9.37..5....7.......................",
    ".28.57....3.24..7...9...2...13.6....5...91..6.9.37..5....7.6492.82.1.56....52.81.",
    ".4.2..1.51..4.67....9.37..6.14.658.7..78194.2698..4..143.6.1.28.6194.57.972.8.61.",
    "..496183..23.5.6.9..92.7145.1534.79....78...1.9.12...44.1...9..8.....4...6...452.",
    "....45..31.42.85795.8.3924...245...84.78.1.62689327415..1...954.4..12.37...574821",
    "651...97....1.9.6...9.56..41..54....4...912.78.7.6..15..2.8...6..67.2..1.7...5342",
    ".8396.754.45237.89679..8.2331.546.974..78.23179.1.346..2..9.....643759.29.7.1...6",
    ".745...26.3.146.8.6.92...45312.58.....67.1.3879...2..152...396..6..1.572..762.8..",
    "..1.53....35...47968.2.71..12...589..56......8976..354312....68..8.2.94....8..5..",
    "571.4.629.341695786.925713412.47.89...639..1...8612..5.157..96.8...3.751..7521.83",
    "893654271.24.78.695..12....21543..9734.89.12.789.1.43.43...1952..19.2.8397258.61.",
    ".7..2..46.6....89.2..8..715.84.97...71.....59...13.48.697..2..8.58....6.43..8..7.",
]

for puzzle in puzzles:
    partial_solution = solve_as_far_as_possible_using_logic(puzzle)
    solution = next(complete_solution_using_backtracking(partial_solution))
    print(puzzle)
    print(solution)
