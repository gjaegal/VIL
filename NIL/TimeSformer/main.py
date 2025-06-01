import os
import cv2
import torch
import numpy as np
from TimeSformer.Video_Encoder import VideoEncoder  # VideoEncoder 클래스가 정의된 파일

def load_frames_from_folder(folder, sort=True):
    frames = []
    files = sorted(os.listdir(folder)) if sort else os.listdir(folder)
    for fname in files:
        if fname.endswith(('.png', '.jpg')):
            img = cv2.imread(os.path.join(folder, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    return frames

def load_masks_from_folder(folder, sort=True):
    masks = []
    files = sorted(os.listdir(folder)) if sort else os.listdir(folder)
    for fname in files:
        if fname.endswith(('.png', '.jpg')):
            mask = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            binary = (mask > 127).astype(np.uint8)
            masks.append(binary)
    return masks

class FrameManager:
    """프레임 및 마스크 관리 클래스"""
    def __init__(self, gen_img_path, gen_mask_path):
        self.gen_frames = load_frames_from_folder(gen_img_path)
        self.gen_masks = load_masks_from_folder(gen_mask_path)
        self.sim_frames = []
        self.sim_masks = []
        
        print(f"레퍼런스 비디오 로드: {len(self.gen_frames)} 프레임, {len(self.gen_masks)} 마스크")
        
    def add_simulation_frame(self, frame, mask=None):
        """시뮬레이션 프레임 및 마스크 추가"""
        self.sim_frames.append(frame)
        
        if mask is not None:
            self.sim_masks.append(mask)
        else:
            # 기본 마스크 생성 (전체 이미지)
            default_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            self.sim_masks.append(default_mask)
    
    def compute_l2_reward_at_step(self, step):
        """특정 스텝에서의 L2 보상 계산"""
        return compute_l2_reward(
            self.gen_frames, 
            self.gen_masks, 
            self.sim_frames, 
            self.sim_masks, 
            step
        )
    
    def get_current_step(self):
        """현재 시뮬레이션 스텝 수 반환"""
        return len(self.sim_frames) - 1 if self.sim_frames else -1

# 전역 프레임 매니저 인스턴스
_global_frame_manager = None

def initialize_frame_manager(gen_img_path, gen_mask_path):
    """프레임 매니저 초기화"""
    global _global_frame_manager
    _global_frame_manager = FrameManager(gen_img_path, gen_mask_path)
    return _global_frame_manager

def get_frame_manager():
    """전역 프레임 매니저 반환"""
    return _global_frame_manager

def add_simulation_data(frame, mask=None):
    """시뮬레이션 데이터 추가 (외부 호출용)"""
    if _global_frame_manager is not None:
        _global_frame_manager.add_simulation_frame(frame, mask)

def get_l2_reward_for_current_step():
    """현재 스텝의 L2 보상 반환 (외부 호출용)"""
    if _global_frame_manager is not None:
        current_step = _global_frame_manager.get_current_step()
        return _global_frame_manager.compute_l2_reward_at_step(current_step)
    return 0.0