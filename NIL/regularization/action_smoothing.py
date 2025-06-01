import numpy as np

class ActionSmoother:
    def __init__(self, action_dim, weight=0.05):
        self.prev_action = np.zeros(action_dim)
        self.weight = weight

    def compute(self, current_action):
        diff = np.linalg.norm(current_action - self.prev_action)
        self.prev_action = current_action.copy()
        return self.weight * diff
