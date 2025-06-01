import numpy as np

class PhysicsConstraints:
    def __init__(self, vel_limit=10.0, torque_limit=200.0, weight=0.1):
        self.vel_limit = vel_limit
        self.torque_limit = torque_limit
        self.weight = weight

    def compute(self, joint_velocities, joint_torques):
        # 속도 패널티: 너무 빠른 속도는 제한
        vel_penalty = np.sum((np.abs(joint_velocities) > self.vel_limit) *
                             (np.abs(joint_velocities) - self.vel_limit) ** 2)

        # 토크 패널티: 극단적인 토크도 제약
        torque_penalty = np.sum((np.abs(joint_torques) > self.torque_limit) *
                                (np.abs(joint_torques) - self.torque_limit) ** 2)

        return self.weight * (vel_penalty + torque_penalty)
