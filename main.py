import numpy as np

GRID_SIZE = 3
DISCOUNT_FACTOR = 0.99
PENALTY = -1

transition_probs = {
    'Up': [(0.8, -1, 0), (0.1, 0, -1), (0.1, 0, 1)],
    'Down': [(0.8, 1, 0), (0.1, 0, -1), (0.1, 0, 1)],
    'Right': [(0.8, 0, 1), (0.1, -1, 0), (0.1, 1, 0)],
    'Left': [(0.8, 0, -1), (0.1, -1, 0), (0.1, 1, 0)]
}

def initialize_grid(r):
    grid = np.full((GRID_SIZE, GRID_SIZE), PENALTY)
    grid[0, 0] = r
    grid[0, 2] = 10
    return grid

def value_iteration(grid, discount_factor):
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    policy = np.full((GRID_SIZE, GRID_SIZE), '', dtype=object)
    for i in range(1000):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Skip updating the terminal state
                if (i, j) == (0, 2):
                    continue
                max_value = float('-inf')
                best_action = None
                for action in transition_probs:
                    new_value = 0
                    for prob, di, dj in transition_probs[action]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
                            new_value += prob * (grid[ni, nj] + discount_factor * V[ni, nj])
                        else:
                            continue
                    if new_value > max_value:
                        max_value = new_value
                        best_action = action
                V[i, j] = max_value
                policy[i, j] = best_action
    return V, policy

r_values = [100, 3, 0, -3]
for r in r_values:
    grid = initialize_grid(r)
    V, policy = value_iteration(grid, DISCOUNT_FACTOR)
    print(f"Policy for r = {r}:")
    print(policy)
    print(f"Value function for r = {r}:")
    print(V)