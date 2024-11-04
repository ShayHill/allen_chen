# allen_chen

## no_state_or_loops_solver

Probably the best way to do it. Simple. Fast enough. Could be expressed with less / simpler code, but I wanted to demonstrate some fp concepts (even where they're a bad fit).

## logic_solver

The way you'd do it by hand. At least, I guess so. I haven't played much (any?) Sudoku, so there might be some strategies I missed. Fast at solving puzzles, but I wanted to test with some order=25 puzzles, and I couldn't find any to copy and paste. They're 625 values each, so I don't want to copy them out.

That's where the problem lies, because solving a puzzle is fast, but *creating* a large puzzle with this method is impracticably slow. Might take several lifetimes.

Like the no_state_or_loops_solver, this has the advantage of yielding lexicographically ordered solutions (if there are multiple solutions). Blows up on empty puzzles > order == 9.

## knuth_solver

I found this idea on Wikipedia. It uses Knuth's *Algorithm X*. There are a ton of articles and a bit of code out there, so this is again a "by the book" solution (just a different book). It can create large puzzles quickly. Algorithm X is worth reading about.

[https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X](https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X)
