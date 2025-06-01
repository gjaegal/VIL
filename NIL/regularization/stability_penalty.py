import numpy as np

class StabilityPenalty:
    def __init__(self, torso_body_id: int, angle_threshold=np.pi/6, weight=0.1):
        """
        torso_body_id: 모델에서 torso에 해당하는 body ID
        angle_threshold: 허용 가능한 기울기 (rad), 예: 30도
        """
        self.body_id = torso_body_id
        self.threshold = angle_threshold
        self.weight = weight

    def compute(self, data):
        # orientation 행렬 (3x3)
        rot_mat = data.xmat[self.body_id].reshape(3, 3)

        # z축 단위벡터 (0, 0, 1)에 대한 현재 몸통의 z축
        torso_z = rot_mat[:, 2]

        # z축이 수직축과 이루는 각도 계산
        vertical = np.array([0, 0, 1])
        cosine = np.clip(np.dot(torso_z, vertical), -1.0, 1.0)
        angle = np.arccos(cosine)

        # 각도 초과 시만 패널티 부여
        if angle > self.threshold:
            return self.weight * (angle - self.threshold) ** 2
        else:
            return 0.0
