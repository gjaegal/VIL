import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # GUI 없이 이미지 저장만 가능하게 설정
import matplotlib.pyplot as plt
from PIL import Image
import subprocess


def run_sam2_and_save_masks(input_path, video_dir, output_dir):

    os.makedirs(video_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", input_path,
        os.path.join(video_dir, "%05d.jpg")
    ]
    subprocess.run(cmd) 
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

    """
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    """

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    #video_dir = "gen_video_frames/sample_video_frames"  # replace with your video directory

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array([[441, 323], [485, 383]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    os.makedirs(output_dir, exist_ok=True)

    # render the segmentation results every few frames
    #plt.close("all")
    for out_frame_idx in range(len(frame_names)):
        if out_frame_idx not in video_segments:
            continue  # 마스크 없으면 저장 안 함

        plt.figure(figsize=(6, 4))
        #plt.title(f"frame {out_frame_idx}")
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)  # ✅ 여기 show_mask 사용

        plt.axis("off")
        plt.savefig(f"{output_dir}/{out_frame_idx+1:05d}.jpg", bbox_inches="tight", pad_inches=0)
        plt.close()

