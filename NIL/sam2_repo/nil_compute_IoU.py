import os
import numpy as np
from PIL import Image


def load_mask(path):
    return np.array(Image.open(path).convert("L")) > 0  # 이진화된 마스크

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0.0

def compute_mean_iou_gen_sim(gen_mask_dir, frame, sim_mask_path="sim_mask/sim_mask.jpg"):
    frame = frame+1

    gen_mask_path = os.path.join(gen_mask_dir, f"{frame:05d}.jpg")

    # 마스크 불러오기
    gen_mask = load_mask(gen_mask_path)
    sim_mask = load_mask(sim_mask_path)

    # 크기 맞추기
    gen_mask_resized = Image.fromarray(gen_mask).resize((sim_mask.shape[1], sim_mask.shape[0]), Image.NEAREST)
    gen_mask_resized = np.array(gen_mask_resized)

    # IoU 계산
    iou = compute_iou(gen_mask_resized, sim_mask)
    print(f"[{frame:05d}] IoU: {iou:.4f}")  # 확인용
    return iou