"""Solve a Sudoku puzzle with Donald Knuth's Algorithm X.

The gist is to represent every possible state of a cell (value, row, column) as a row
in a sparse matrix. Each row in the sparse matrix is marked with the constraints it
satisfies, but no two rows can satisfy the same constraint. So the solution is a
subset of these cell-state rows that only satisfies each constraint once.

A 9x9 Sudoku puzzle has 9**81 possible grid states, but only  9**3 possible cell
states, so finding a subset of possible cell states is a lot faster than finding an
instance of a complete grid state.

This is plenty of literature online for this, but a lot of the code is missing. I saw
one author complaining that his work was being submitted as solutions to homework
problems. Maybe that's why the code has disappeared. Maybe the problem has just
fallen out of fashion. The algorithm is simple, and there will be plenty of ways to
accomplish it, but I just used methods (and some code) I found.

:author: Shay Hill
:created: 2024-11-04
"""
import functools as ft
import itertools as it
from typing import Iterator


def _zero_based_int_to_chr(i: int) -> str:
    """Convert a 0-based int to a char where 0 -> '1'."""
    return chr(i + 49)


def _chr_to_zero_based_int(ch: str) -> int:
    """Convert a char to a 0-based int where '1' -> 0."""
    return ord(ch) - 49


# ===================================================================================
#   Ali Assaf's sparse-matrix algorithm X solver
#   https://aliassaf.github.io/software/algorithm-x/
#
#   Not quite verbatim from his site, but *almost*.
# ===================================================================================

Col2Rows = dict[int, set[int]]
Row2Cols = dict[int, list[int]]


def algo_x_solve(
    col2rows: Col2Rows,
    row2cols: Row2Cols,
    solution: list[int] | None = None,
) -> Iterator[list[int]]:
    """Yield all solutions to the exact cover problem represented by col_to_rows.

    :param col_to_rows: A dict mapping column numbers to sets of row numbers.
        This is Sudoku.condition2included_rows.
    :param row_to_cols: A dict mapping row numbers to lists of column numbers.
        This is Sudoku.row2satisfied_conditions.
    :param solution: A list of row numbers that have been selected so far.
    :yield: A list of row numbers that form a solution.

    * Find the column with the minimum conflicts.
    * Select a row from that column.
    * Remove that row, remove any conflicting rows, and remove any constraints that
      row satisfied.
    * If all remaining constraints are still met, continue; else backtrack.
    """
    solution = solution or []
    if not col2rows:
        yield list(solution)
        return
    col = min(col2rows, key=lambda c: len(col2rows[c]))
    for row in list(col2rows[col]):
        solution.append(row)
        cached_cols = _algo_x_select(col2rows, row2cols, row)
        yield from algo_x_solve(col2rows, row2cols, solution)
        _algo_x_deselect(col2rows, row2cols, row, cached_cols)
        _ = solution.pop()


def _algo_x_select(col2rows: Col2Rows, row2cols: Row2Cols, row: int) -> list[set[int]]:
    """Remove row and requirements it satisfies from the sparse matrix."""
    cols: list[set[int]] = []
    for exclusive_col in row2cols[row]:
        for row_with_col in col2rows[exclusive_col]:
            for other_col in (x for x in row2cols[row_with_col] if x != exclusive_col):
                col2rows[other_col].remove(row_with_col)
        cols.append(col2rows.pop(exclusive_col))
    return cols


def _algo_x_deselect(
    col2rows: Col2Rows, row2cols: Row2Cols, row: int, cols: list[set[int]]
):
    """Restore the rows to the columns that were removed in _algo_x_select."""
    for exclusive_col in reversed(row2cols[row]):
        col2rows[exclusive_col] = cols.pop()
        for row_with_col in col2rows[exclusive_col]:
            for other_col in (x for x in row2cols[row_with_col] if x != exclusive_col):
                col2rows[other_col].add(row_with_col)


class Sudoku:
    def __init__(self, puzzle: str):
        self.puzzle = puzzle
        self.order = int(len(puzzle) ** 0.5)
        self.box_d = int(self.order**0.5)
        if self.box_d**4 != len(puzzle):
            msg = "Puzzle string must represent a square puzzle of square boxes."
            raise ValueError(msg)

    @ft.cached_property
    def _given_vals(self) -> list[tuple[int, int, int]]:
        """ATTENTION: THIS is where the chrs are converted to 0-based ints."""
        given_vals: list[tuple[int, int, int]] = []
        for r, c in it.product(range(self.order), repeat=2):
            puzzle_chr_as_int = _chr_to_zero_based_int(self.puzzle[r * self.order + c])
            if puzzle_chr_as_int >= 0:
                given_vals.append((puzzle_chr_as_int, r, c))
        return given_vals

    def row_from_vrc(self, vrc: tuple[int, int, int]) -> int:
        """Get row index from value, row, and column."""
        v, r, c = vrc
        return c + (self.order * r) + (self.order * self.order * v)

    def vrc_from_row(self, row: int) -> tuple[int, int, int]:
        """Get value, row, and column from a row index."""
        c = row % self.order
        r = (row // self.order) % self.order
        v = row // (self.order * self.order)
        return v, r, c

    def _conflicts_with_given(
        self, val: int, row: int, col: int, given_vals: list[tuple[int, int, int]]
    ) -> bool:
        """Return True if val, row, int conflicts with any of the given values.

        Return True if:
        * row and col match, but val is not the given value
        * val matches a given value in the same row or col
        * val matches a given value in the same box
        """
        if not given_vals:
            return False
        v, r, c = given_vals[0]
        if v != val and row == r and col == c:
            return True
        if v == val and (row == r) != (col == c):
            return True
        box_beg_row = (r // self.box_d) * self.box_d
        box_beg_col = (c // self.box_d) * self.box_d
        if (
            v == val
            and box_beg_row < row < box_beg_row + self.box_d
            and box_beg_col < col < box_beg_col + self.box_d
            and not (row == r and col == c)
        ):
            return True
        return self._conflicts_with_given(val, row, col, given_vals[1:])

    @ft.cached_property
    def row2satisfied_conditions(self):
        """Map each row index to the column numbers of the constraints it satisfies.

        This is, in effect, a sparse matrix. The full matrix would have one row for
        each combination of (value, row, and column), so N**3 rows. The Sudoky puzzle
        has four constraints (unique in row, unique in column, unique in box, unique
        in cell), each a function of two values. To map out all of these constraints,
        the full matrix would have 4 * N**2 columns.

        I will call these constraints Row, Col, Val, and One. There may be many ways
        to lay these out, I found some Java code that mapped them out
        RRRRRRRRRCCCCCCCCCVVVVVVVVVV for most of the columns followed 81 O positions
        in the last column (for an order=9 puzzle). Staying with that because it
        works.
        """
        row2satisfied_conditions: dict[int, list[int]] = {}
        for v, r, c in it.product(range(self.order), repeat=3):
            if self._conflicts_with_given(v, r, c, self._given_vals):
                continue
            row_idx = self.row_from_vrc((v, r, c))
            box_idx = (c // self.box_d) + ((r // self.box_d) * self.box_d)
            col_row = 3 * self.order * v + r
            col_col = 3 * self.order * v + self.order + c
            col_box = 3 * self.order * v + 2 * self.order + box_idx
            col_one = 3 * self.order * self.order + (c + self.order * r)
            row2satisfied_conditions[row_idx] = [col_row, col_col, col_box, col_one]
        return row2satisfied_conditions

    @ft.cached_property
    def condition2included_rows(self) -> dict[int, set[int]]:
        """Transform the row2satisfied_conditions to a column2included_rows dict.

        This is just for faster lookups.
        """
        col2included_rows: dict[int, set[int]]
        col2included_rows = {x: set() for x in range(self.order**2 * 4)}
        for k, v in self.row2satisfied_conditions.items():
            for x in v:
                col2included_rows[x].add(k)
        return col2included_rows

    def solve(self):
        """Solve with Ali Assaf's algorithm X solver."""
        solutions = algo_x_solve(
            self.condition2included_rows, self.row2satisfied_conditions
        )
        for solution in solutions:
            as_ints = [[0] * self.order for _ in range(self.order)]
            for v, r, c in map(self.vrc_from_row, solution):
                as_ints[r][c] = v
            yield "".join(map(_zero_based_int_to_chr, it.chain(*as_ints)))


puzzles = [
    ".................................................................................",
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
    solution = next(Sudoku(puzzle).solve())
    print(puzzle)
    print(solution)
