# 🎥 VIL
Video Imitation Learning

## Introduction
In robot learning, we often face tasks where little or no expert data is available for training.
This project explores how well imitation learning can perform in this situation. Inspired by the paper "NIL: No-data Imitation Learning by Leveraging Pre-trained Video Diffusion Models",
we leverage video diffusion models to generate an input video and directly use video simularity as a reward signal for imitation learning in locomotion tasks. We also experiment with an SMPL-based blender rendered video, which we created from an existing reference video.

In this project, we
1) Re-implement NIL, as the official code is not currently available
2) Explore different types of input videos to improve performance

## 👥Team Members
Hwang Soonmin (황순민) - YAI 14th \\
Lee Youngjoo (이영주) - YAI 14th \\
Kim Suran (김수란) - YAI 14th \\
Kim Seojin (김서진) - YAI 14th \\
Jaegal Gun (제갈건) - YAI 12th \\

## Download Pretrained Models
1. SAM2
download sam2.1_hiera_large.pt from this [link](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) to NIL/sam2_repo/checkpoints or run the follwing code:
```
wget -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt NIL/sam2_repo/checkpoints/sam2.1_hiera_large.pt
```

2. TimeSformer
```
mkdir -p NIL/TimeSformer/timesformer/pretrained
wget -O NIL/TimeSformer/timesformer/pretrained/TimeSformer_divST_8x32_224_K400.pyth \
   "https://www.dropbox.com/scl/fi/zcn6byf10i4r0hhojjten/TimeSformer_divST_8x32_224_K400.pyth?rlkey=azfkkmb0qalhgt9vxofhwje54&dl=1"
mv NIL/TimeSformer/timesformer/pretrained/TimeSformer_divST_8x32_224_K400.pyth \
   NIL/TimeSformer/timesformer/pretrained/TimeSformer_divST_8x32_224_K400.pth
```

## Installation

```
pip install -r requirements.txt
```

### Train
```
python train.py --env h1-walk-v0
```

## 🙏Acknowledgements

* NIL: No-data Imitation Learning by Leveraging Pre-trained Video Diffusion Models
* [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench)
* [SAM2](https://github.com/facebookresearch/sam2)
* [TimeSformer](https://github.com/facebookresearch/TimeSformer)
* [StableVideoDiffusion](https://github.com/Stability-AI/generative-models)
