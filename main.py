import numpy as np

class GridWorld:
    def __init__(self, grid_size=3, discount_factor=0.99, penalty=-1):
        self.GRID_SIZE = grid_size
        self.DISCOUNT_FACTOR = discount_factor
        self.PENALTY = penalty
        self.r_values = [100, 3, 0, -3]
        self.transition_probs = {
            'Up': [(0.8, -1, 0), (0.1, 0, -1), (0.1, 0, 1)],
            'Down': [(0.8, 1, 0), (0.1, 0, -1), (0.1, 0, 1)],
            'Right': [(0.8, 0, 1), (0.1, -1, 0), (0.1, 1, 0)],
            'Left': [(0.8, 0, -1), (0.1, -1, 0), (0.1, 1, 0)]
        }

    def initialize_grid(self, r):
        grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.PENALTY)
        grid[0, 0] = r
        grid[0, 2] = 10
        return grid

    def value_iteration(self, grid):
        V = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        V[0, 0] = grid[0, 0]  # terminal state
        V[0, 2] = 10  # terminal state
        policy = np.full((self.GRID_SIZE, self.GRID_SIZE), '', dtype=object)
        
        for _ in range(6):  
            for i in range(self.GRID_SIZE):
                for j in range(self.GRID_SIZE):
                    if (i, j) == (0, 0) or (i, j) == (0, 2):
                        continue
                    max_value = float('-inf')
                    best_action = None
                    for action in self.transition_probs:
                        new_value = grid[i, j] # immediate reward
                        for prob, di, dj in self.transition_probs[action]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.GRID_SIZE and 0 <= nj < self.GRID_SIZE:
                                new_value += (prob * V[ni, nj])
                        new_value = self.DISCOUNT_FACTOR * new_value
                        if new_value > max_value:
                            max_value = new_value
                            best_action = action
                    V[i, j] = max_value
                    policy[i, j] = best_action
        
        return V, policy
    
    def run_value_iter_algo(self):
        for r in self.r_values:
            grid = self.initialize_grid(r)
            V, policy = self.value_iteration(grid)
            print(f"Value function for r = {r}:")
            print(V)
            print(f"Policy for r = {r}:")
            print(policy)
    


# Testing
grid_world = GridWorld()
grid_world.run_value_iter_algo()

