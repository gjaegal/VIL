import numpy as np

class FootContactPenalty:
    def __init__(self, weight=0.1, foot_body_ids=None):
        """
        foot_body_ids: model에서 발로 간주할 body ID 리스트
        """
        self.weight = weight
        self.foot_body_ids = foot_body_ids or []

    def compute(self, data):
        penalty = 0.0
        reward = 0.0

        for body_id in self.foot_body_ids:
            contact_force = np.linalg.norm(data.cfrc_ext[body_id])
            velocity = np.linalg.norm(data.cvel[body_id])  # body velocity (angular+linear)

            # 발이 땅에 닿았는데 움직이면 패널티
            if contact_force > 1e-3 and velocity > 1e-3:
                penalty += velocity

            # 발이 떠 있는 경우에는 약간의 리워드
            if contact_force < 1e-4:
                reward += 0.01  # 보너스 (튜닝 가능)

        return self.weight * (penalty - reward)
