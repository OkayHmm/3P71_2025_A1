# Attemping the first assignment
# ZachM - 2025/09/21

import random
import unittest
from enum import Enum, auto
from collections import deque

NUM_COLORS = 10


class Axis(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()

class Degrees(Enum):
    d90 = auto()
    d180 = auto()
    d270 = auto()

class Colour(Enum):
    RED = 0
    ORANGE = 1
    YELLOW = 2
    GREEN = 3
    BLUE = 4
    INDIGO = 5
    VIOLET = 6
    PINK = 7
    BROWN = 8
    BLACK = 9


def color_change(grid, old_colour, new_colour):
    '''
    Replaces all occurrences of a specific color in a 2D grid with a new color.
    '''
    return [[new_colour if cell == old_colour else cell for cell in row] for row in grid]

def swap_colours(grid, first_colour, second_colour):
    '''
    Swap all elements of first_colour and second_colour
    '''
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if(grid[r][c] == first_colour):
                grid[r][c] = second_colour
            elif(grid[r][c] == second_colour):
                grid[r][c] = first_colour

def apply_mirror(grid, axis: Axis):
    '''
    Mirrors the grid across a specific access
    '''
    if axis == Axis.HORIZONTAL:
        return apply_mirror_horizontal(grid)
    elif axis == Axis.VERTICAL: 
        return apply_mirror_vertical(grid)
    else:
        raise ValueError("Invalid axis value to mirror")

def apply_mirror_horizontal(grid):
    '''
    Reverses the order of elements in each row
    '''
    return [row[::-1] for row in grid]

def apply_mirror_vertical(grid):
    '''
    Reverses the order of elements in each column
    '''
    return grid[::-1]

def apply_rotate(grid, degree: Degrees):
    '''
    Rotates the grid 90, 180, or 270 degrees to the right
    '''
    rows, cols = len(grid), len(grid[0])

    if degree == Degrees.d90:
        return [[grid[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]
    elif degree == Degrees.d180:
        return [[grid[rows - 1 - r][cols - 1 - c] for c in range(cols)] for r in range(rows)]
    elif degree == Degrees.d270:
        return [[grid[c][cols - 1 - r] for c in range(cols)] for r in range(rows)]
    else:
        raise ValueError("Invalid degrees to rotate")
    


def create_empty_grid(size):
    '''
    Creates an empty grid of with dimensions (size, size) filled with zeros
    '''
    return [[0 for _ in range(size)] for _ in range(size)]

def create_random_grid(size):
    '''
    Creates a grid with dimensions (size, size) where each element is a random value
    between 0 and (NUM_COLORS - 1)
    '''
    return [[random.randint(0, NUM_COLORS - 1) for _ in range(size)] for _ in range(size)]

def print_grid(grid):
    '''
    Formats and prints a provided grid out to the console
    '''
    rows, cols = len(grid), len(grid[0])
    out = ""
    for r in range(rows):
        for c in range(cols):
            out += " | " + str(grid[r][c])
        out += " |\n"
    print(out)

OPERATIONS = {
    "ColorChange": color_change,
    "Mirror": apply_mirror,
    "Rotate": apply_rotate,
}

def generate_children(program):
    children = []
    # Generate all possible colour changes
    for i in range(NUM_COLORS):
        for j in range(NUM_COLORS):
            result_grid = color_change(program.grid, i, j)
            new_seq = program.seq.copy().append( ("ColorChange", i, j) )
            assert new_seq != program.seq
            result_program = Program(result_grid, new_seq, program.complexity + 1)
            children.append(result_program)
    
    # Mirrors
    for axis in Axis:
        new_seq = program.seq.copy().append(("Mirror", axis))
        children.append(Program(apply_mirror(program.grid, axis), new_seq, program.complexity + 1))

    # Rotate
    for degrees in Degrees:
        new_seq = program.seq.copy().append(("Rotate", degrees))
        children.append(Program(apply_rotate(program.grid, degrees), new_seq, program.complexity + 1))

def is_program_acceptable(program):
    for 

def BFS(start_grid, complexity_limit):
    print("Conducting BFS...")
    initial_program = Program(start_grid)
    to_visit = deque()
    to_visit.extend(generate_children(initial_program))
    while(to_visit):
        program = to_visit.pop()
        if is_program_acceptable(program):
            return program
        if program.complexity < complexity_limit:
            to_visit.extend(generate_children(program))
    print(f"BFS finished without finding a solution (complexity limit: {complexity_limit})")

class Program():
    def __init__(self, grid=[], seq=[][], complexity=0):
        self.grid = grid
        self.sequence = seq
        self.complexity = complexity

class TestGridOperations(unittest.TestCase):
    def test_rotate_90(self):
        start = [[1,2,3],[4,5,6],[7,8,9]]
        expected = [[7,4,1],[8,5,2],[9,6,3]]
        result = apply_rotate(start, Degrees.d90)
        self.assertEqual(expected, result)

    def test_rotate_180(self):
        start = [[1,2,3],[4,5,6],[7,8,9]] 
        expected = [[9,8,7],[6,5,4],[3,2,1]]
        result = apply_rotate(start, Degrees.d180)
        self.assertEqual(expected, result)

    def test_rotate_270(self):
        start = [[1,2,3],[4,5,6],[7,8,9]] 
        expected = [[3,6,9],[2,5,8],[1,4,7]]
        result = apply_rotate(start, Degrees.d270)
        self.assertEqual(expected, result)



if __name__ == "__main__":
    unittest.main(exit=False)

    test_grid = create_random_grid(3)
    print_grid(test_grid)
    test_grid = apply_rotate(test_grid, Degrees.d90)
    test_grid = apply_rotate(test_grid, Degrees.d90)
    print_grid(test_grid)