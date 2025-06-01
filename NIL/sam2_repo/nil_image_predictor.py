import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # GUI 없이 이미지 저장만 가능하게 설정
import matplotlib.pyplot as plt
from PIL import Image


def run_sam2_and_save_mask(frame, sim_img_dir, sim_mask_dir):

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    from sam2_repo.sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "sam2_repo/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "sam2_repo/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            #cmap = plt.get_cmap("tab10")
            #cmap_idx = 0 if obj_id is None else obj_id
            #color = np.array([*cmap(cmap_idx)[:3], 0.6])
            color = np.array([0.0, 0.0, 0.0, 1.0])  # green with alpha
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    #video_dir = "gen_video_frames/sample_video_frames"  # replace with your video directory

    inference_state = predictor.init_state(video_path=sim_img_dir)

    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array([[168,107]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    sim_mask_path = os.path.join(sim_mask_dir, "00000.jpg")

    plt.figure(figsize=(6, 4))
    #plt.title(f"frame {out_frame_idx}")
    show_mask(out_mask_logits[0].cpu().numpy(), plt.gca(), obj_id=out_obj_ids)
    plt.axis("off")
    plt.savefig(sim_mask_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return sim_mask_path
