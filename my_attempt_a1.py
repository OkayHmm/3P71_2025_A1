# Attemping the first assignment
# ZachM - 2025/09/21

import random
import unittest
import json
from enum import Enum, auto
from collections import deque

NUM_COLOURS = 10
MAX_COMPLEXITY = 3
SOLUTIONS_FILE = "C:/Users/User/Documents/School/3P71 TA 2025/3P71_2025_A1/benchmark/arc-agi_solutions.json"
CHALLENGES_FILE = "C:/Users/User/Documents/School/3P71 TA 2025/3P71_2025_A1/benchmark/arc-agi_challenges.json"

class Axis(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()

class Degrees(Enum):
    d90 = auto()
    d180 = auto()
    d270 = auto()

class Colour(Enum):
    BLANK = 0
    ORANGE = 1
    YELLOW = 2
    GREEN = 3
    BLUE = 4
    INDIGO = 5
    VIOLET = 6
    PINK = 7
    BROWN = 8
    BLACK = 9

SCALE_FACTORS = [2, 3]
SHIFTS = [-1, 0, 1]
DIMENSIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
COLOUR_MAP = [
    [(0, 4), (1, 0), (2, 3)],  # Specific mapping found in e8c4b1f6
    [(1, 3), (0, 4), (2, 0)],  # Common variations
    [(0, 1), (1, 2), (2, 0)],
    [(1, 0), (2, 4)],  # Partial mappings
    [(0, 3), (2, 4)]
]
# Scale with color mapping (for tasks like f2e9a4d1)
SCALE_COLOUR_MAPS = [
    [2, [(1, 3), (0, 0), (2, 4)]],  # 2x2 scaling with color map
    [2, [(1, 4), (0, 3), (2, 0)]],
    [3, [(1, 3), (0, 0), (2, 4)]]   # 3x3 scaling with color map
]

def colour_change(grid, old_colour, new_colour):
    '''
    Replaces all occurrences of a specific colour in the grid with a new colour.
    '''
    return [[new_colour if cell == old_colour else cell for cell in row] for row in grid]

def swap_colours(grid, first_colour, second_colour):
    '''
    Swap all elements of first_colour and second_colour
    '''
    rows, cols = len(grid), len(grid[0])
    new_grid = [[0 for c in range(len(grid[0]))] for r in range(len(grid))]
    for r in range(rows):
        for c in range(cols):
            if(grid[r][c] == first_colour):
                new_grid[r][c] = second_colour
            elif(grid[r][c] == second_colour):
                new_grid[r][c] = first_colour
            else:
                new_grid[r][c] = grid[r][c]
    return new_grid

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
        return [[grid[c][cols - 1 - r] for c in range(rows)] for r in range(cols)]
    else:
        raise ValueError("Invalid degrees to rotate")
    
def resize_irregular(grid, new_height, new_width):
    '''
    Resizes a grid, either by repeating rows & columns or by removing them until the desired dimensions are reached
    '''
    new_grid = []
    for i in range(new_height):
        new_row = []
        for j in range(new_width):
            orig_i = min(i, len(grid) - 1)
            orig_j = min(j, len(grid[0]) - 1)
            new_row.append(grid[orig_i][orig_j])
        new_grid.append(new_row)
    return new_grid

def diagonal_reflection(grid, old_colour, new_colour):
    '''
    Reflects all elements of old_colour across the diagonal, then sets them to new_colour, all old positions of old_colour are set blank
    '''
    # Reflect colour across diagonal and change it
    new_grid = [row[:] for row in grid]  # Create copy
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == old_colour:
                # Clear the original position
                new_grid[r][c] = 0
                # Reflect across main diagonal (swap r and c)
                if c < len(grid) and r < len(grid[0]):
                    new_grid[c][r] = new_colour
    return new_grid

def scale_2x2(grid):
    return scale_by(grid, 2, 2)

def scale_3x3(grid):
    return scale_by(grid, 3, 3)

def scale_2x1(grid):
    return scale_by(grid, 2, 1)

def scale_1x2(grid):
    return scale_by(grid, 1, 2)

def scale_by(grid, row_factor, col_factor):
    '''
    Enlarge each element of the grid to take up an enlarged space defined by row_factor and col_factor
    '''
    new_grid = [[0 for i in range(len(grid[0]) * col_factor)] for j in range(len(grid) * row_factor)]
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            new_r_start = r * row_factor
            new_c_start = c * col_factor
            for i in range(row_factor):
                for j in range(col_factor):
                    new_grid[new_r_start + i][new_c_start + j] = grid[r][c]
    return new_grid

def positional_shift(grid, old_colour, new_colour, r_offset, c_offset):
    new_grid = [[0 for i in range(len(grid[0]))] for j in range(len(grid))]
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == old_colour:
                new_grid[r][c] = 0
                if (r + r_offset < len(grid)) and (c + c_offset < len(grid[0])):
                    new_grid[r + r_offset][c + c_offset] = new_colour
            else:
                new_grid[r][c] = grid[r][c]

def colour_map_multiple(grid, colour_map):
    new_grid = [row[:] for row in grid]
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            val = new_grid[r][c]
            for mapping in colour_map:
                if mapping[0] == val:
                    new_grid[r][c] = mapping[1]
    return new_grid

def scale_with_colour_map(grid, scale_factor, colour_map):
    new_grid = []
    for r in range(len(grid)):
        for _ 


OPERATIONS = {
    "SwapColour": swap_colours,
    "ColourChange": colour_change,
    "Mirror": apply_mirror,
    "Rotate": apply_rotate,
    "ResizeIrregular": resize_irregular,
    "DiagonalReflection": diagonal_reflection,
    "Scale2x2": scale_2x2,
    "Scale3x3": scale_3x3,
    "Scale2x1": scale_2x1,
    "Scale1x2": scale_1x2,
    "PositionalShift": positional_shift,
    "ColourMapMultiple": colour_map_multiple,
}

#
# Grid Operations
#

def create_empty_grid(size):
    '''
    Creates an empty grid of with dimensions (size, size) filled with zeros
    '''
    return [[0 for _ in range(size)] for _ in range(size)]

def create_random_grid(size):
    '''
    Creates a grid with dimensions (size, size) where each element is a random value
    between 0 and (NUM_COLOURS - 1)
    '''
    return [[random.randint(0, NUM_COLOURS - 1) for _ in range(size)] for _ in range(size)]

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

def generate_children(program):
    children = []

    # Generate all possible colour swaps
    for i in range(1, NUM_COLOURS):
        for j in range(i+1, NUM_COLOURS): 
            children.append(create_child(program, "SwapColour", [i, j]))

    # Generate all possible colour changes
    for i in range(1, NUM_COLOURS):
        for j in range(1, NUM_COLOURS):
            children.append(create_child(program, "ColourChange", [i, j]))
    
    # Mirrors
    for axis in Axis:
        children.append(create_child(program, "Mirror", [axis]))

    # Rotate
    for degrees in Degrees:
        children.append(create_child(program, "Rotate", [degrees]))

    # Scale
    children.append(create_child(program, "Scale2x2"))
    children.append(create_child(program, "Scale3x3"))
    children.append(create_child(program, "Scale2x1"))
    children.append(create_child(program, "Scale1x2"))

    # Resize irregular
    for i in range(len(DIMENSIONS)):
        for j in range(len(DIMENSIONS)):
            children.append(create_child(program, "ResizeIrregular", [i, j]))
    
    # Positional shift
    for c1 in range(NUM_COLOURS):
        for c2 in range(NUM_COLOURS):
            for s1 in range(len(SHIFTS)):
                for s2 in range(len(SHIFTS)):
                    children.append(create_child(program, "PositionalShift", [c1, c2, s1, c2]))

    # Colour Map Multiple
    for map in COLOUR_MAP:
        children.append(create_child(program, "ColourMapMultiple", [map]))
    
    # Scale With Colour Map



    return children

def create_child(program, op_name, params=[]):
    new_seq = program.sequence.copy()
    new_seq.append((op_name, *params))
    result_grids = [OPERATIONS[op_name](program.grids[i], *params) for i in range(len(program.grids))]
    return Program(result_grids, new_seq, program.complexity + 1)

def is_program_acceptable(program, train_data):
    assert len(program.grids) == len(train_data)

    for i in range(len(program.grids)):
        if program.grids[i] != train_data[i]['output']:
            return False
    return True

#
# Search Algorithms
#

def BFS(train_data, complexity_limit):
    initial_program = Program(grids=[x['input'] for x in train_data])
    to_visit = deque()
    to_visit.extend(generate_children(initial_program))
    while(to_visit):
        program = to_visit.pop()
        if is_program_acceptable(program, train_data):
            print("BFS found a solution.")
            return program
        if program.complexity < complexity_limit:
            to_visit.extend(generate_children(program))
    print(f"BFS finished without finding a solution (complexity limit: {complexity_limit})")
    return None

class Program():
    def __init__(self, grids=[], seq=[], complexity=0):
        self.grids = grids # [training example][grid]
        self.sequence = seq # [operation, argument_1, argument_2, ... , argument_x]
        self.complexity = complexity

    # def apply_to_grid(self, grid):
    #     '''
    #     Perform all operations listed in sequence on initial grid to find final grid (state)
    #     '''
    #     for seq in self.sequence:
    #         method_name = seq[0]
    #         args = seq[1:]
    #         method_obj = globals()[method_name]
    #         method_obj(grid, *args)



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

    def test_scale_2x2(self):
        start = [[1]]
        expected = [[1, 1], [1, 1]]
        result = scale_2x2(start)
        self.assertEqual(expected, result)

if __name__ == "__main__":
    unittest.main(exit=False)

    
    #grid = create_random_grid(3);
    # grid = [[4,1,2],[7,0,1],[0,8,1]]
    # print_grid(grid)
    # test = diagonal_reflection(grid, 1, 2);
    # print_grid(test)
    # exit()

    try:
        with open(CHALLENGES_FILE, 'r') as file:
            challenges_data = json.load(file)
        with open(SOLUTIONS_FILE, 'r') as file:
            solutions_data = json.load(file)
    except FileNotFoundError:
        print("Error: The training data files were not found.")
        exit()
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file.")
        exit()

    print("Train / test data loaded.")
    count_success = 0

    for example in challenges_data:
        print(f"\nTraining example {example}")
        train_data = challenges_data[example]['train']
        test_data = challenges_data[example]['test']
        test_solution = solutions_data[example]

        BFS_result = BFS(train_data, MAX_COMPLEXITY)
        if BFS_result is not None:
            print(BFS_result.sequence)