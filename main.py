import numpy as np


class GridWorld:
    def __init__(self, grid_size=3, discount_factor=0.99, penalty=-1):
        self.GRID_SIZE = grid_size
        self.DISCOUNT_FACTOR = discount_factor
        self.PENALTY = penalty
        self.r_values = [100, 3, 0, -3]
        self.transition_probs = {
            "Up": [(0.8, -1, 0), (0.1, 0, -1), (0.1, 0, 1)],
            "Down": [(0.8, 1, 0), (0.1, 0, -1), (0.1, 0, 1)],
            "Right": [(0.8, 0, 1), (0.1, -1, 0), (0.1, 1, 0)],
            "Left": [(0.8, 0, -1), (0.1, -1, 0), (0.1, 1, 0)],
        }

    def initialize_grid(self, r):
        grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.PENALTY)
        grid[0, 0] = r
        grid[0, 2] = 10
        return grid

    def value_iteration(self, grid, threshold=1e-6):
        # Vnew = R + y max[ for_each_direction{ SUM(prob*Vold[]) } ]
        V = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        V[0, 0] = grid[0, 0]  # terminal state
        V[0, 2] = 10  # terminal state
        policy = np.full((self.GRID_SIZE, self.GRID_SIZE), "", dtype=object)

        while True:
            delta = 0
            for i in range(self.GRID_SIZE):
                for j in range(self.GRID_SIZE):
                    if (i, j) == (0, 0) or (i, j) == (0, 2):
                        continue
                    max_value = float("-inf")
                    best_action = None

                    for action in self.transition_probs:
                        value = 0
                        for prob, di, dj in self.transition_probs[action]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.GRID_SIZE and 0 <= nj < self.GRID_SIZE:
                                value += prob * V[ni, nj]
                            else:
                                value += prob * V[i, j]
                        new_value = grid[i, j] + self.DISCOUNT_FACTOR * value
                        if new_value > max_value:
                            max_value = new_value
                            best_action = action
                    delta = max(delta, abs(max_value - V[i, j]))
                    V[i, j] = max_value
                    policy[i, j] = best_action

            if delta < threshold:
                break

        return V, policy

    def policy_evaluation(self, policy, grid):
        V = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        V[0, 0] = grid[0, 0]  # terminal state
        V[0, 2] = 10  # terminal state
        while True:
            delta = 0
            for i in range(self.GRID_SIZE):
                for j in range(self.GRID_SIZE):
                    if (i, j) == (0, 0) or (i, j) == (0, 2):
                        continue
                    action = policy[i, j]
                    # new_value = grid[i, j]
                    value = 0
                    for prob, di, dj in self.transition_probs[action]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.GRID_SIZE and 0 <= nj < self.GRID_SIZE:
                            value += prob * V[ni, nj]
                        else:
                            value += prob * V[i, j]
                    new_value = grid[i, j] + self.DISCOUNT_FACTOR * value
                    delta = max(delta, abs(new_value - V[i, j]))
                    V[i, j] = new_value
            if delta < 1e-6:
                break
        return V

    def policy_improvement(self, V, grid):
        policy = np.full((self.GRID_SIZE, self.GRID_SIZE), "", dtype=object)
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if (i, j) == (0, 0) or (i, j) == (0, 2):
                    policy[i, j] = ""  # terminal state
                    continue
                max_value = float("-inf")
                best_action = None

                for action in self.transition_probs:
                    # new_value = grid[i, j]
                    value = 0
                    for prob, di, dj in self.transition_probs[action]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.GRID_SIZE and 0 <= nj < self.GRID_SIZE:
                            value += prob * V[ni, nj]
                        else:
                            value += prob * V[i, j]
                    new_value = grid[i, j] + self.DISCOUNT_FACTOR * value
                    if new_value > max_value:
                        max_value = new_value
                        best_action = action
                policy[i, j] = best_action
        return policy

    def policy_iteration(self, grid):
        policy = np.random.choice(
            list(self.transition_probs.keys()), size=(self.GRID_SIZE, self.GRID_SIZE)
        )
        print("random policy: \n", policy, "\n\n")
        policy[0, 0] = ""
        policy[0, 2] = ""

        while True:
            V = self.policy_evaluation(policy, grid)
            new_policy = self.policy_improvement(V, grid)
            if np.array_equal(policy, new_policy):
                break
            policy = new_policy
        return V, policy

    def run_policy_iter_algo(self):
        for r in self.r_values:
            grid = self.initialize_grid(r)
            V, policy = self.policy_iteration(grid)
            print(f"Value function for r = {r}:")
            print(V)
            print(f"Policy for r = {r}:")
            print(policy, "\n\n")
            

    def run_value_iter_algo(self):
        for r in self.r_values:
            grid = self.initialize_grid(r)
            V, policy = self.value_iteration(grid)
            print(f"Value function for r = {r}:")
            print(V)
            print(f"Policy for r = {r}:")
            print(policy, "\n\n")


# Testing
grid_world = GridWorld()
# grid_world.run_value_iter_algo()
# grid_world.run_policy_iter_algo()
