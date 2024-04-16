from functools import cache
import numpy as np

@cache
def get_shift_indices(magnitude, exclude_center = True):
    magnitude = abs(magnitude)
    shifts = list()
    for y_shift in range(-magnitude, magnitude+1):
        for x_shift in range(-magnitude, magnitude+1):
            if exclude_center and x_shift == 0 and y_shift == 0:
                continue
            shifts.append((y_shift, x_shift))
#     print(shifts)
    return shifts

@cache
def compute_minesweeper_values(game_map):
    assert len(game_map) > 0 and all([len(segment) == len(game_map[0]) for segment in game_map]), "ERROR: MAP MUST BE 2 DIMENSIONAL AND FULL"
    full_map = [['*' if element == True or element == 1 else 0 for element in seq] for seq in game_map]
    shape = (len(full_map), len(full_map[0]))
    for y, seq in enumerate(full_map):
            for x, element in enumerate(seq):
                if element == '*':
                    continue
                for y_shift, x_shift in get_shift_indices(1):
                    if full_map[(y+y_shift)%shape[0]][(x+x_shift)%shape[1]] == '*':
                        full_map[y][x] += 1
    return tuple([tuple(seq) for seq in full_map]), shape

@cache
def count_reachable_cells(immutable_map, source):
    frontier = [source]
    visited = {source}
    mines = 0
    open_spaces = 0
    while frontier:
        location = frontier.pop()
        y, x = location
        if immutable_map[y][x] == '*':
            mines += 1
            continue
        else:
            open_spaces += 1
            for y_shift, x_shift in get_shift_indices(1):
                new_location = (y+y_shift)%len(immutable_map), (x+x_shift)%len(immutable_map[0])
                new_y, new_x = new_location
                if new_location in visited:
                    continue
                else:
                    frontier.append(new_location)
                    visited.add(new_location)
    return open_spaces, mines
    

class MinesweeperGame():
    def __init__(self, game_map, start_location=None, enforce_reachability=False, **kwargs):
        
        self._full_map, self.shape = compute_minesweeper_values(tuple([tuple(seq) for seq in game_map]))
        
        if start_location == None:
            # default start location is the middle of game map
            self.start = tuple([dim//2 for dim in self.shape])
        else:
            self.start = start_location
        assert len(self.start) == len(self.shape), "ERROR: START LOCATION MUST MATCH DIMENSIONALITY OF MAP"
        assert all([abs(self.start[dim]) < boundary for dim, boundary in enumerate(self.shape)]), "ERROR: START LOCATION NOT WITHIN MAP BOUNDS"
        assert self._full_map[self.start[0]][self.start[1]] != '*', "ERROR: START LOCATION MUST NOT CONTAIN A MINE"
                        
        
        if enforce_reachability:
            self.spaces, self.mines = count_reachable_cells(self._full_map, self.start)
        else:
            self.mines = 0
            for seq in self._full_map:
                self.mines += seq.count('*')
            self.spaces = self.shape[0]*self.shape[1] - self.mines
        self._mine_set = set()
        for y in range(len(self._full_map)):
            for x in range(len(self._full_map[y])):
                if self._full_map[y][x] == '*':
                    self._mine_set.add((y,x))
        self.reset()
        
    def reset(self):
        self.score = 0
        self.player_map = [[' ' for _ in range(self.shape[1])] for __ in range(self.shape[0])]
        self.visible = [[0 for _ in range(self.shape[1])] for __ in range(self.shape[0])]
        self.revealed_spaces = 0
        self._actions = dict()
        self._action_keys = list()
        self._visited = set()
        self._reveal_space(self.start)
        self.done = False
        return self.get_observations(), self.score, self.done
        
    def _reveal_space(self, source):
        frontier = [source]
        self._visited.add(source)
        observation_updates = list()
        while frontier:
            location = frontier.pop()
            y, x = location
            self.visible[y][x] = 1
            self.player_map[y][x] = self._full_map[y][x]
            
            if location in self._actions:
                del self._actions[location]
                self._action_keys.remove(location)
            if self.player_map[y][x] == '*':
                self.done = True
                break
            else:
                self.revealed_spaces += 1
            observation_updates.append((y, x))
            if self.player_map[y][x] != 0:
                continue
            else:
                for y_shift, x_shift in get_shift_indices(1):
                    new_location = (y+y_shift)%self.shape[0], (x+x_shift)%self.shape[1]
                    new_y, new_x = new_location
                    if new_location in self._visited or self.visible[new_y][new_x] == 1 or self._full_map[new_y][new_x] == '*':
                        continue
                    else:
                        frontier.append(new_location)
                        self._visited.add(new_location)
        self._update_observations(observation_updates)
        self.update_score()
    
    def _update_observations(self, observation_updates):
        to_update = set()
        # iterate over all modified cells
        for location in observation_updates:
            y, x = location
            # iterate over all locations 2 cells away from a modified cell
            for y_shift, x_shift in get_shift_indices(2):
                new_location = (y+y_shift)%self.shape[0], (x+x_shift)%self.shape[1]
                new_y, new_x = new_location
                # skip visible cells and cells 2 away not already registered for actions
                if self.visible[new_y][new_x] == 1 or ((abs(y_shift) > 1 or abs(x_shift) > 1) and new_location not in self._actions):
                    continue
                # add adjacent non-visible cells and non-adjacent cells registered as actions
                else:
                    to_update.add(new_location)
        # iterate over all cells requiring action creation/update
        for location in to_update:
            y, x = location
            scores = list()
            visible = list()
            # gather observation of cells 2 away from selected location
            for y_shift, x_shift in get_shift_indices(2):
                new_y, new_x = (y+y_shift)%self.shape[0], (x+x_shift)%self.shape[1]
                assert self.player_map[new_y][new_x] != '*', "Encountered bomb during update. Game should have ended."
                if self.visible[new_y][new_x] == 0:
                    scores.append(0)
                else:
                    scores.append(self.player_map[new_y][new_x]/8)
                visible.append(self.visible[new_y][new_x])
            if location not in self._actions:
                self._action_keys.append(location)
            self._actions[location] = np.array([scores, visible], dtype=float)
        # check if no available actions exist
        if len(self._actions) == 0:
            self.done = True
        
        # check if all remaing actions are mine selection
        elif set(self._action_keys) <= self._mine_set:
            self.done = True


    def update_score(self):
        self.score = self.revealed_spaces / self.spaces

    def get_observations(self):
        return np.array([self._actions[key] for key in self._action_keys])
    
    def step(self, action):
        self._reveal_space(self._action_keys[action])
        return self.get_observations(), self.score, self.done

def render_map(game_map, compact=True):
    for line in game_map:
        for char in line:
            if char == ' ':
                print(chr(0x25A0), end='')
            else:
                print(char, end='')
            if not compact: print(' ', end='')
        print()