import sys

#   https://stackoverflow.com/questions/20034023/maximum-recursion-depth-exceeded-in-comparison
#
#
#

sys.setrecursionlimit(10000 * 10000)


def solveMaze(Maze, position, point):
    N = len(Maze)  # lin
    M = len(Maze[0])  # col

    # returns a list of the paths taken
    if position == point:
        return [point]
    x, y = position

    if y + 1 < M and x + 1 < N and Maze[x + 1][y + 1] == 0:
        c = solveMaze(Maze, (x + 1, y + 1), point)
        if c is not None:
            return [(x, y)] + c

    if x + 1 < N and Maze[x + 1][y] == 0:
        a = solveMaze(Maze, (x + 1, y), point)
        if a is not None:
            return [(x, y)] + a

    if y + 1 < M and Maze[x][y + 1] == 0:
        b = solveMaze(Maze, (x, y + 1), point)
        if b is not None:
            return [(x, y)] + b


Maze = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

print(solveMaze(Maze, (9, 4), (0, 0)))
