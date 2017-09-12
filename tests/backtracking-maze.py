def step(maze, x, y):
    # https://thetokenizer.com/2013/01/13/practicing-backtracking/

    # Accept case - we found the exit
    if maze[x][y] == 'X':
        return True

    # Reject case - we hit a wall or our path
    if maze[x][y] == '#' or maze[x][y] == '*':
        return False

    maze[x][y] = '*'

    # Try to go Right
    result = step(maze, x, y + 1)
    if result:
        return True

    # Try to go Up
    result = step(maze, x - 1, y)
    if result:
        return True

    # Try to go Left
    result = step(maze, x, y - 1)
    if result:
        return True

    # Try to go Down
    result = step(maze, x + 1, y)
    if result:
        return True

    # Deadend - this location can't be part of the solution
    # Unmark this location
    maze[x][y] = ' '

    # Go back
    return False


# Get the start location (x,y) and try to solve the maze
def solve(maze, x, y):
    if step(maze, x, y):
        maze[x][y] = 'S'


def toString():
    output = ""
    for x in range(len(maze)):
        for y in range(len(maze[0])):
            output += maze[x][y] + " "

        output += "\n"

    return output


if __name__ == '__main__':
    maze = [
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
        ['#', ' ', ' ', ' ', '#', ' ', '#', ' ', ' ', '#'],
        ['#', ' ', ' ', ' ', '#', ' ', '#', ' ', '#', '#'],
        ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
        ['#', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
        ['#', '#', '#', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
        ['#', ' ', ' ', ' ', ' ', '#', '#', '#', '#', '#'],
        ['#', ' ', ' ', ' ', ' ', ' ', '#', ' ', '#', '#'],
        ['#', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' ', '#'],
        ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#']
    ]

    maze[7][7] = 'X'

    solve(maze, 8, 1)

    print(toString())
