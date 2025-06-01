from bro.bro_learner import BRO
from replay_buffer import ParallelReplayBuffer
import logger
import utils
import envs
# from utils import mute_warning, log_to_wandb_if_time_to, evaluate_if_time_to, make_env

import argparse
import pathlib

import glob
import cv2
import gymnasium as gym
import time
import os
import numpy as np
import torch
import jax.numpy as jnp
from PIL import Image

#from sam2.nil_IoU_reward import IoU_reward_function
from sam2_repo.nil_video_predictor import run_sam2_and_save_masks
from sam2_repo.nil_compute_IoU import compute_mean_iou_gen_sim
from sam2_repo.nil_image_predictor import run_sam2_and_save_mask

from regularization.torque_penalty import TorquePenalty
from regularization.action_smoothing import ActionSmoother
from regularization.physics_constraints import PhysicsConstraints
from regularization.foot_contact_penalty import FootContactPenalty
from regularization.stability_penalty import StabilityPenalty

from TimeSformer.Video_Encoder import compute_l2_reward
from TimeSformer.main import load_frames_from_folder, load_masks_from_folder

os.environ["MUJOCO_GL"] = "egl"

import flax
import flax.serialization

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="environment test")
    parser.add_argument("--env", help="e.g. h1-walk-v0")
    parser.add_argument("--keyframe", default=None)
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--mean_path", default=None)
    parser.add_argument("--var_path", default=None)
    parser.add_argument("--policy_type", default=None)
    parser.add_argument("--blocked_hands", default="False")
    parser.add_argument("--small_obs", default="False")
    parser.add_argument("--obs_wrapper", default="False")
    parser.add_argument("--sensors", default="")
    parser.add_argument("--render_mode", default="rgb_array")  # "human" or "rgb_array".
    # NOTE: to get (nicer) 'human' rendering to work, you need to fix the compatibility issue between mujoco>3.0 and gymnasium: https://github.com/Farama-Foundation/Gymnasium/issues/749
    parser.add_argument("--log_video", default="True")
    parser.add_argument("--reference_video_path", default="gen_walk.mp4")

    parser.add_argument("--alpha", default=0.1, type=float, help="Weight for L2 reward")
    parser.add_argument("--beta", default=0.1, type=float, help="Weight for IoU reward")
    parser.add_argument("--gamma", default=-0.1, type=float, help="Weight for regularization reward")

    parser.add_argument("--gen_video_path", type=str, default="sam2_repo/gen_video_frames/sample_video.mp4")
    parser.add_argument("--gen_frame_dir", type=str, default="sam2_repo/gen_video_frames/sample_video_frames")
    parser.add_argument("--gen_mask_dir", type=str, default="sam2_repo/gen_masks")

    parser.add_argument("--sim_img_dir", type=str, default="sam2_repo/sim_img")
    parser.add_argument("--sim_mask_dir", type=str, default="sam2_repo/sim_mask")

    args = parser.parse_args()

    kwargs = vars(args).copy()
    kwargs.pop("env")
    kwargs.pop("render_mode")
    if kwargs["keyframe"] is None:
        kwargs.pop("keyframe")
    print(f"arguments: {kwargs}")

    # Set up paths
    gen_video_path = args.gen_video_path
    gen_frame_dir = args.gen_frame_dir
    gen_mask_dir = args.gen_mask_dir
    sim_img_dir = args.sim_img_dir
    sim_mask_dir = args.sim_mask_dir

    # Log directory
    data_path = "./vid_logs"
    log_dir = data_path + "/" + "walk_test_" + time.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    mylogger = logger.Logger(log_dir=log_dir)

    # Initialize wandb
    wandb.init(
    project="humanoid-nil",
    name=f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
    config=vars(args),
    dir=log_dir
    )

    # Load generated reference video
    generated_video_frames = utils.load_video_frames(args.reference_video_path)
    print(generated_video_frames[0].shape)
    
    # TODO 가비아에서 make custom env 오류 해결
    env = gym.make(args.env, render_mode=args.render_mode, **kwargs)
    # env = gym.make("Humanoid-v5", render_mode=args.render_mode)


    seed = 0
    training_steps = 1000000
    start_training_step = 5
    fps = 24
    generated_video_length = len(generated_video_frames) # 121
    # agent
    print("observation space: ", env.observation_space.sample().shape)
    print("action space: ", env.action_space.sample().shape)
    dummy_obs = jnp.zeros((1, 51))
    dummy_ac = jnp.zeros((1, 19))
    updates_per_step = 10

    agent = BRO(
        seed,
        dummy_obs, # env.observation_space.sample()[0, np.newaxis],
        dummy_ac, # env.action_space.sample()[0, np.newaxis],
        num_seeds=1,
        updates_per_step=updates_per_step,
        distributional=True,
    )
    # Replay buffer
    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], 1000000, num_seeds=1)
    
    ob, _ = env.reset()

    seg_masks = []
    generated_seg_masks= []

    torque_penalty = TorquePenalty(weight=0.00001)
    action_smoother = ActionSmoother(action_dim=env.action_space.shape[-1], weight=0.1)
    physics_constraints = PhysicsConstraints(vel_limit=8.0, torque_limit=150.0, weight=0.0001)
    foot_contact_penalty = FootContactPenalty(weight=0.1, foot_body_ids=[6, 11]) 
    stability_penalty = StabilityPenalty(torso_body_id=12, angle_threshold=0.52, weight=0.1) 

    # input_path = "sam2_repo/gen_video_frames/sample_video.mp4"  # mp4 경로
    # video_dir = "sam2_repo/gen_video_frames/sample_video_frames"  # video frames 경로
    # gen_mask_dir = "sam2_repo/gen_masks"  # gen video masks 경로

    # sim_img_path = "sam2_repo/sim_img"  # sim img 경로, 폴더 경로여야함.
    # sim_mask_dir = "sim_mask"  # sim mask 경로, 폴더 경로여야함.
    run_sam2_and_save_masks(gen_video_path, gen_frame_dir, gen_mask_dir)  # input_path는 mp4 경로, video_dir는 jpg 저장할 경로, gen_mask_dir는 mask 저장할 경로
    gen_frames = load_frames_from_folder(gen_frame_dir)
    gen_masks = load_masks_from_folder(gen_mask_dir)
    sim_frame_clip = []
    sim_mask_clip = []

    frame = 0
    # TODO for step in range(len(generated_video))
    for step in range(training_steps):
        if ob.shape[0] != 1:
            ob = ob.reshape(1, -1)

        actions = agent.sample_actions_o(ob, temperature=1.0)
        actions = actions.squeeze(0)


        next_ob, rewards, terminated, truncated, info = env.step(actions)
        print("STEP: ", step)

        # 1. Regularization rewards
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        joint_positions, joint_velocities, joint_torques = base_env.get_observations() # 19 dim
 
        model = base_env.model
        data = base_env.data

        p_j = torque_penalty.compute(joint_torques)
        p_a = action_smoother.compute(actions) # NOTE: action smoothing은 이전 스텝의 action도 필요할 수 있습니다. 현재는 최신 action만 사용.
        p_v = physics_constraints.compute(joint_velocities, joint_torques)
        p_f = foot_contact_penalty.compute(data)
        p_s = stability_penalty.compute(data)
        regularization_reward = p_j + p_a + p_v + p_f + p_s
        # print(f"Regularization rewards: torque={p_j}, action_smooth={p_a}, physics_constraints={p_v}, foot_contact={p_f}, stability={p_s}")
        
        # 2. IoU reward

        # numpy array를 PIL 이미지로 변환 후 저장
        rgb_img = base_env.render().astype(np.uint8)  # RGB 이미지로 변환
        os.makedirs(sim_img_dir, exist_ok=True)
        
        img = Image.fromarray(rgb_img)
        img.save(f"{sim_img_dir}/{frame+1:05d}.jpg")


        sim_mask_path = run_sam2_and_save_mask(frame, sim_img_dir, sim_mask_dir) #시뮬레이터 이미지를 마스크로
        iou_reward = compute_mean_iou_gen_sim(gen_mask_dir, frame, sim_mask_path)

        # 3. L2 reward
        sim_frame_clip.append(rgb_img)
        if len(sim_frame_clip) > 8:
            del sim_frame_clip[0]
        
        
        img = cv2.imread(sim_mask_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sim_mask_clip.append(img)
        if len(sim_mask_clip) > 8:
            del sim_mask_clip[0]
        
        # sim_frames = load_frames_from_folder(sim_img_dir)
        # sim_masks = load_masks_from_folder(sim_mask_dir)

        if frame > 7:
            gen_frame_clip = gen_frames[frame-7:frame+1]
            gen_mask_clip = gen_masks[frame-7:frame+1]
        else:
            gen_frame_clip = [gen_frames[0] for _ in range(7)]
            gen_frame_clip.append(gen_frames[frame])
            gen_mask_clip = [gen_masks[0] for _ in range(7)]
            gen_mask_clip.append(gen_masks[frame])
        # l2_reward = compute_l2_reward(sim_frames, sim_masks, gen_frames, gen_masks, step, frame)
        l2_reward = compute_l2_reward(gen_frame_clip, gen_mask_clip, sim_frame_clip, sim_mask_clip, step)

        weighted_l2_reward = args.alpha * l2_reward
        weighted_iou_reward = args.beta * iou_reward
        weighted_regularization_reward = args.gamma * regularization_reward
            
        nil_reward = weighted_l2_reward + weighted_iou_reward + weighted_regularization_reward
        print('weighted_l2_reward', weighted_l2_reward)
        print('weighted_iou_reward', weighted_iou_reward)
        print('weighted_regularization_reward', weighted_regularization_reward)
        print("nil_reward", nil_reward)


        # masks = env.generate_masks(terminated, truncated)
        if not truncated:
            masks = [1.0]
            replay_buffer.insert(ob, actions, nil_reward, masks, truncated, next_ob)

        # Train
        if step > start_training_step:
            batches = replay_buffer.sample_parallel_multibatch(batch_size=128, num_batches=updates_per_step)
            infos = agent.update(batches, updates_per_step, step)    
            # Log reward wandb
            wandb.log({
                "nil_reward": nil_reward,
                "l2_reward": l2_reward,
                "iou_reward": iou_reward,
                "reg_reward": regularization_reward,
                "step": step,
                "weighted_l2_reward": weighted_l2_reward,
                "weighted_iou_reward": weighted_iou_reward,
                "weighted_reg_reward": weighted_regularization_reward,
                "Q_value_mean" : infos["Q_mean"].item(),
                "Q_value_std" : infos["Q_std"].item(),
            })

        # Next frame
        frame += 1
        ob = next_ob

        if terminated or truncated or frame >= generated_video_length:
            print("Episode finished after {} frames".format(frame + 1))
            ob, _ = env.reset()
            frame = 0

        


        if (step+1) % 1000 == 0:
            if args.log_video == "True":
                # simulate with trained agent
                trajs = utils.rollout_n_trajectories(env, policy=agent, ntraj=1, max_traj_length=generated_video_length, render=True, seg_render=False)

                # tensorboard에 video 형식으로 저장
                mylogger.log_trajs_as_videos(trajs, step=0, max_videos_to_save=1, fps=fps, video_title="video.mp4")
            
                # wandb에 video 형식으로 저장
                video_file = os.path.join(log_dir, "video.mp4")
                if os.path.exists(video_file):
                    wandb.log({
                        "rollout_video": wandb.Video(video_file, caption=f"Step {step+1}", fps=fps, format="mp4"),
                        "step": step
                    })

        if (step + 1) % 10000 == 0:
            ckpt_path = os.path.join(log_dir, f"bro_model_{step+1:06d}.msgpack")
            with open(ckpt_path, "wb") as f:
                f.write(flax.serialization.to_bytes(agent.actor.params))
            print(f"Checkpoint saved to {ckpt_path}")

 
    env.close()
