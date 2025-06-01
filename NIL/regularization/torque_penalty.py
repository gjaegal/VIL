import numpy as np

class TorquePenalty:
    def __init__(self, weight=0.1):
        self.weight = weight

    def compute(self, joint_torques):
        """
        joint_torques: np.ndarray of shape (num_joints,)
        """
        return self.weight * np.sum(np.square(joint_torques))
