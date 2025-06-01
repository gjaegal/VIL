import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from TimeSformer.timesformer.models.vit import TimeSformer
import os
import cv2
import matplotlib.pyplot as plt

class VideoEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        TimeSformer의 비디오 인코더 t 초기화
        """
        self.device = device
        
        self.model = TimeSformer(
            img_size=224,
            num_classes=400,
            num_frames=8,
            attention_type='divided_space_time',
            pretrained_model='/home/jordon/seojin/ImJordon/NIL/TimeSformer/timesformer/pretrained/TimeSformer_divST_8x32_224_K400.pth'
        )
        
        self.model.eval()
        self.model.to(self.device)
        
        # 입력 이미지 전처리를 위한 변환
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),    # [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                std=[0.225, 0.225, 0.225])
        ])
    """
    def apply_mask(self, frame, mask):
        
        프레임에 마스크 적용
        Args:
            frame: numpy 배열 (H, W, 3)
            mask: numpy 배열 (H, W)
        Returns:
            마스킹된 numpy 배열
        
        masked = frame.copy()
        masked[mask == 0] = 0
        return masked
    """

    def apply_mask(self, frame, mask):
        """
        프레임에 마스크 적용 (크기 검증 추가)
        Args:
            frame: numpy 배열 (H, W, 3)
            mask: numpy 배열 (H, W)
        Returns:
            마스킹된 numpy 배열
        """
        try:
            # 프레임과 마스크 크기 확인
            if len(frame.shape) == 3:
                frame_h, frame_w = frame.shape[:2]
            else:
                print(f"경고: 예상하지 못한 프레임 모양: {frame.shape}")
                return frame
            
            if len(mask.shape) == 2:
                mask_h, mask_w = mask.shape
            elif len(mask.shape) == 3:
                mask_h, mask_w = mask.shape[:2]
                mask = mask[:, :, 0] if mask.shape[2] > 1 else mask.squeeze()
            else:
                print(f"경고: 예상하지 못한 마스크 모양: {mask.shape}")
                return frame
            
            # 크기가 다른 경우 프레임을 마스크 크기로 리사이즈
            if frame_h != mask_h or frame_w != mask_w:
                # print(f"크기 불일치 감지: frame({frame_h}, {frame_w}) vs mask({mask_h}, {mask_w})")
                frame = cv2.resize(frame, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
            
            masked = frame.copy()
            
            # 마스크 적용 - boolean indexing 대신 직접 곱셈 사용
            if mask.dtype == bool:
                mask_3d = np.stack([mask, mask, mask], axis=2)
            else:
                mask_3d = np.stack([mask > 0, mask > 0, mask > 0], axis=2)
            
            masked = masked * mask_3d.astype(masked.dtype)
            return masked
            
        except Exception as e:
            print(f"마스크 적용 중 오류: {e}")
            print(f"Frame shape: {frame.shape}, Mask shape: {mask.shape}")
            return frame
    
    def preprocess_frames(self, frames, masks=None):
        """
        프레임들을 전처리하고 텐서로 변환
        Args:
            frames: [T, H, W, 3] or PIL Image list
            masks: [T, H, W] numpy array list (선택)
        Returns:
            torch.Tensor [1, 3, T, 224, 224]
        """
        processed = []
        for i, frame in enumerate(frames):
            if isinstance(frame, np.ndarray):
                if masks is not None:
                    frame = self.apply_mask(frame, masks[i])
                frame = Image.fromarray(frame.astype(np.uint8))
            processed.append(self.transform(frame))
        tensor = torch.stack(processed).unsqueeze(0)  # [1, T, C, H, W]
        return tensor.permute(0, 2, 1, 3, 4).to(self.device)  # [1, 3, T, H, W]
    
    def create_clip(self, f_clip, m_clip):
        """
        시점 t에서 8프레임 클립 생성
        """
        return self.preprocess_frames(f_clip, m_clip)
    
    def extract_embedding(self, clip_tensor):
        """
        TimeSformer로부터 임베딩 추출
        Returns:
            torch.Tensor [D]
        """
        with torch.no_grad():
            z = self.model(clip_tensor)
        return z.squeeze(0)

    def compare_embeddings(self, z1, z2):
        """
        L2 거리 계산
        Returns:
            float
        """
        return torch.norm(z1 - z2, p=2).item()

_global_encoder = None

def get_video_encoder():
    """전역 비디오 인코더 인스턴스 반환"""
    global _global_encoder
    if _global_encoder is None:
        try:
            _global_encoder = VideoEncoder()
            print("TimeSformer 비디오 인코더 초가화")
        except Exception as e:
            print(f"TimeSformer 초기화 실패: {e}")
            _global_encoder = None
    return _global_encoder

def compute_l2_reward(gen_frame_clip, gen_mask_clip, sim_frame_clip, sim_mask_clip, step):
    """
    외부에서 호출 가능한 L2 보상 계산 함수
    
    Args:
        gen_frames: 생성된 비디오 프레임 리스트 (numpy arrays)
        gen_masks: 생성된 비디오 마스크 리스트 (numpy arrays)
        sim_frames: 시뮬레이션 프레임 리스트 (numpy arrays)
        sim_masks: 시뮬레이션 마스크 리스트 (numpy arrays)
        step: 현재 스텝
        
    Returns:
        float: L2 기반 보상 점수
    """
    encoder = get_video_encoder()
    if encoder is None:
        return 0.0
    
    try:
        clip_gen = encoder.preprocess_frames(gen_frame_clip, gen_mask_clip)
        clip_sim = encoder.preprocess_frames(sim_frame_clip, sim_mask_clip)
        z_gen = encoder.extract_embedding(clip_gen)
        z_sim = encoder.extract_embedding(clip_sim)
        
        reward = -torch.norm(z_gen - z_sim, p=2).item()
        
        return reward
        
    except Exception as e:
        print(f"L2 reward 계산 오류 발생 (step {step}): {e}")
        return 0.0