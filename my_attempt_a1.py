# Attemping the first assignment
# ZachM - 2025/09/21

import random
import unittest
import json
import itertools
import time
from queue import PriorityQueue
from enum import Enum, auto
from collections import deque

NUM_COLOURS = 4
MAX_COMPLEXITY = 3
SOLUTIONS_FILE = "benchmark/arc-agi_solutions.json"
CHALLENGES_FILE = "benchmark/arc-agi_challenges.json"

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
#DIMENSIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DIMENSIONS = [1, 2, 3, 4]
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
    return new_grid

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
    colour_map = dict(colour_map)
    new_grid = []
    for row in grid:
        # Create scale_factor rows for each original row
        for _ in range(scale_factor):
            new_row = []
            for cell in row:
                # Map color and repeat scale_factor times
                mapped_color = colour_map.get(cell, cell)
                new_row.extend([mapped_color] * scale_factor)
            new_grid.append(new_row)
    return new_grid

def SwapColours(grid, first_colour, second_colour):
    new_grid = [[0 for c in range(len(grid[0]))] for r in range(len(grid))]
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == first_colour:
                new_grid[r][c] = second_colour
            elif grid[r][c] == second_colour:
                new_grid[r][c] = first_colour
            else:
                new_grid[r][c] = grid[r][c]
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
    "ScaleWithColourMap": scale_with_colour_map,
    "SwapColours": swap_colours,
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
        for j in range(i+1, NUM_COLOURS):
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
    for i in DIMENSIONS:
        for j in DIMENSIONS:
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
    for scale_factor, colour_map in SCALE_COLOUR_MAPS:
        children.append(create_child(program, "ScaleWithColourMap", [scale_factor, colour_map]))

    # Swap Colours
    for c1 in range(1, NUM_COLOURS):
        for c2 in range(c1+1, NUM_COLOURS):
            children.append(create_child(program, "SwapColours", [c1, c2]))

    # Diagonal Reflection
    for c1 in range(1, NUM_COLOURS):
        for c2 in range(c1+1, NUM_COLOURS):
            children.append(create_child(program, "DiagonalReflection", [c1, c2]))

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
    to_visit.append(initial_program)
    while(to_visit):
        program = to_visit.popleft()
        if is_program_acceptable(program, train_data):
            print("BFS found a solution.")
            return program
        if program.complexity < complexity_limit:
            to_visit.extend(generate_children(program))
    print(f"BFS finished without finding a solution (complexity limit: {complexity_limit})")
    return None

def AStar(train_data, complexity_limit):
    '''
    Performs an A* search...
    '''
    frontier_pq = PriorityQueue()
    seen_costs = {}
    came_from = {}

    expected_grids = [x['output'] for x in train_data]
    start_grids = [x['input'] for x in train_data]
    counter = itertools.count()
    eval = misplaced_tiles_heuristic(start_grids, expected_grids)
    initial_program = Program(grids=start_grids)
    frontier_pq.put((eval, next(counter), initial_program))
    seen_costs[initial_program] = 0
    came_from[initial_program] = None
    
    
    while(frontier_pq.qsize() > 0):
        current_program = frontier_pq.get(block=False)[2]
        if is_program_acceptable(current_program, train_data):
            print("AStar found a solution.")
            return current_program
        if current_program.complexity < complexity_limit:
            children = generate_children(current_program)   
            for child in children:
                new_path_cost = seen_costs[current_program] + 1
                eval = misplaced_tiles_heuristic(child.grids, expected_grids) + new_path_cost # f(n) = h(n) + g(n)
                if child not in seen_costs or new_path_cost < seen_costs[child]:
                    seen_costs[child] = new_path_cost
                    came_from[child] = current_program
                    frontier_pq.put((eval, next(counter), child))                    

    print(f"AStar finished without finding a solution (complexity limit: {complexity_limit})")
    return None


def GFS(train_data, complexity_limit):
    '''
    Greedy-first Search
    '''
    visited = set()
    frontier_pq = PriorityQueue()

    expected_grids = [x['output'] for x in train_data]
    start_grids = [x['input'] for x in train_data]
    counter = itertools.count()
    eval = misplaced_tiles_heuristic(start_grids, expected_grids)
    initial_program = Program(grids=start_grids)
    frontier_pq.put((eval, next(counter), initial_program))

    while(frontier_pq.qsize() > 0):
        current_program = frontier_pq.get()[2]

        if current_program in visited:
            continue

        if is_program_acceptable(current_program, train_data):
            print("GFS found a solution.")
            return current_program
        
        visited.add(current_program)
        
        if current_program.complexity < complexity_limit:
            children = generate_children(current_program)   
            for child in children:
                if child not in visited:
                    eval = misplaced_tiles_heuristic(child.grids, expected_grids)
                    frontier_pq.put((eval, next(counter), child))

    print(f"GFS finished without finding a solution (complexity limit: {complexity_limit})")
    return None
    

# Heuristics

def misplaced_tiles_heuristic(grids, expected_grids):
    total = 0
    assert len(expected_grids) == len(grids)
    for i in range(len(expected_grids)):
        total += count_misplaced_tiles(grids[i], expected_grids[i])
    return total

def count_misplaced_tiles(grid, expected_grid):
    rows, cols = len(grid), len(grid[0])
    count = 0

    for r in range(rows):
        for c in range(cols):
            if r >= len(expected_grid) or c >= len(expected_grid[0]) or grid[r][c] != expected_grid[r][c]:
                count += 1
    return count

def apply_sequence_to_grid(grid, sequence):
    '''
    Apply all operations in the sequence on the grid and return the output
    '''
    for seq in sequence:
        method_name = seq[0]
        args = seq[1:]
        method_obj = OPERATIONS[method_name]
        grid = method_obj(grid, *args)
    return grid

class Program():
    def __init__(self, grids, seq=[], complexity=0):
        self.grids = grids # [training example][grid]
        self.sequence = seq # [operation, argument_1, argument_2, ... , argument_x]
        self.complexity = complexity
        self.uid = hash(str(grids))

    def __hash__(self):
        return self.uid
    
    def __eq__(self, other):
        return isinstance(other, Program) and self.grids == other.grids

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
    #unittest.main(exit=False)

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
    total_examples = len(challenges_data)
    algorithms = {"BFS", "GFS", "AStar"}
    count_train_success = {}
    count_test_success = {}
    solutions = {}
    for algo in algorithms:
        count_train_success[algo] = 0
        count_test_success[algo] = 0
        solutions[algo] = []

    for example in challenges_data:
        print(f"\nTraining example {example}")
        train_data = challenges_data[example]['train']
        test_data = challenges_data[example]['test']
        test_solution = solutions_data[example]

        for algo in algorithms:
            timestamp = time.time_ns()
            method = globals()[algo]
            result = method(train_data, MAX_COMPLEXITY)
            if result is not None:
                print(result.sequence)
                count_train_success[algo] += 1 
                solutions[algo].append(result.sequence if result is not None else None)
            print(f"{algo} time: {(time.time_ns() - timestamp) / 1_000_000_000:.4f} seconds")

            # Now let's check the test case
            if result is not None:
                result_grid = apply_sequence_to_grid(test_data[0]['input'], result.sequence)
                if(result_grid == test_solution[0]):
                    print(f"{algo}'s solution passed the test case.")
                    count_test_success[algo] += 1
                else:
                    print(f"{algo}'s solution failed the test case.")
        

    print("Training data finding a solution success rates:")
    for algo in algorithms:
        print(f"{algo}: {count_train_success[algo]}/{total_examples} ({count_train_success[algo]/total_examples:.2f})")
    print("Test case pass rates:")
    for algo in algorithms:
        print(f"{algo}: {count_test_success[algo]}/{total_examples} ({count_test_success[algo]/total_examples:.2f})")

