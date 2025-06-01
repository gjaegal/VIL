from bro.bro_learner import BRO
from replay_buffer import ParallelReplayBuffer
# from utils import mute_warning, log_to_wandb_if_time_to, evaluate_if_time_to, make_env

# 정규화 패널티 클래스 임포트
from regularization.torque_penalty import TorquePenalty
from regularization.action_smoothing import ActionSmoother
from regularization.physics_constraints import PhysicsConstraints
from regularization.foot_contact_penalty import FootContactPenalty
from regularization.stability_penalty import StabilityPenalty

import argparse
import pathlib

import cv2
import gymnasium as gym
import time
import os
import logger
import utils
import numpy as np
import torch

from gym.vector import make as vector_make


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
    parser.add_argument("--reference_video_path", default="./NIL/ref/sample_reference_video.mp4")
    args = parser.parse_args()

    kwargs = vars(args).copy()
    kwargs.pop("env")
    kwargs.pop("render_mode")
    if kwargs["keyframe"] is None:
        kwargs.pop("keyframe")
    print(f"arguments: {kwargs}")

    # Log directory
    data_path = "./vid_logs"
    log_dir = data_path + "/" + "basketball_test" + time.strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    mylogger = logger.Logger(log_dir=log_dir)

    # TODO Generated reference video
    # reference video path
    reference_video_path = args.reference_video_path
    print(f"reference video path: {reference_video_path}")
    # generated_video = ...

    # TODO 가비아에서 make custom env 오류 해결
    num_seeds = 10
    # env = vector_make("Humanoid-v4", num_envs=num_seeds, render_mode="rgb_array")

    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    # env = gym.make("Humanoid-v5", render_mode="rgb_array", **kwargs)
    # env = GenerateMaskWrapper(env)
    # env = gym.make(args.env, render_mode=args.render_mode, **kwargs)

    seed = 0
    fps = 4

    # agent
    agent = BRO(
        seed,
        # env.observation_space.sample()[0, np.newaxis],
        # env.action_space.sample()[0, np.newaxis],
        env.observation_space.sample()[None, ...],
        env.action_space.sample()[None, ...],
        # num_seeds=10,
        num_seeds=1,
        updates_per_step=10,
        distributional=True,
    )
    # Replay buffer
    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], 1000000, num_seeds=10)
    
    ob, _ = env.reset()
    # ob_batch = np.repeat(ob[None, :], 10, axis=0)
    ob_batch = ob[None, ...]
    
    seg_masks = []
    generated_seg_masks= []

    # Regularization reward initialization (regularization folder code 활용)
    # NOTE: 필요에 따라 weight/arguments는 configs/hyperparams.yaml 에서 읽어오거나, argparse로 받을 수 있도록 수정 필요
    torque_penalty = TorquePenalty(weight=0.05)
    action_smoother = ActionSmoother(action_dim=env.action_space.shape[-1], weight=0.02)
    physics_constraints = PhysicsConstraints(vel_limit=8.0, torque_limit=150.0, weight=0.1)
    foot_contact_penalty = FootContactPenalty(weight=0.1, foot_body_ids=[7, 10]) # humanoid-v5 기준 발 body id 확인 필요
    stability_penalty = StabilityPenalty(torso_body_id=1, angle_threshold=0.52, weight=0.1) # humanoid-v5 기준 torso body id 확인 필요

# TODO for step in range(len(generated_video))
    for step in range(1000):
        # actions = agent.sample_actions_o(ob, temperature=1.0)
        # actions = agent.sample_actions_o(ob_batch, temperature=1.0)
        actions = agent.sample_actions_o(ob_batch, temperature=1.0)
        actions  = actions.squeeze(0)
        next_ob, rewards, terminated, truncated, info = env.step(actions)

        # extract joint positions / torques / velocities as numpy array
        # TODO past frame action, foot contact with ground, stability
        joint_positions = []
        model = env.unwrapped.model
        data = env.unwrapped.data
        nq = model.nq
        nv = model.nv 
        nu = model.nu
        
        # joint positions
        joint_positions = np.array(data.qpos[:nq])

        # joint velocities
        joint_velocities = np.array(data.qvel[:nv])

        # joint torques (actuator forces)
        joint_torques = np.array(data.actuator_force[:nu])

        # TODO extract segmenation masked image -> 우리 환경에서 제대로 작동하는지 확인
        # seg_mask = env.render(mode="depth", camera_name="track")
        # seg_masks.append(seg_mask)

        # TODO extract segmentation masked image from generated video
        # generated_seg_mask = SAM(generated_video[step])
        # generated_seg_masks.append(generated_seg_mask)

        # TODO clip of past 8 frames
        # CLIP(seg_masks[:], generated_seg_masks[:])

        # TODO =============== REWARD ====================
        alpha, beta, gamma = 1.0, 1.0, 1.0
        regularization_reward = 0.0
        iou_reward = 0.0
        l2_reward = 0.0
        # regularization_reward = REGULARIZATION(joint_positions, joint_velocities, joint_torques, ...)
        # iou_reward = VIDEO_SIMULARITY(seg_mask, generated_seg_mask)
        # l2_reward = IMAGE_SIMULARTIY(CLIP())

        # 정규화 패널티 계산
        p_j = torque_penalty.compute(joint_torques)
        p_a = action_smoother.compute(actions) # NOTE: action smoothing은 이전 스텝의 action도 필요할 수 있습니다. 현재는 최신 action만 사용.
        p_v = physics_constraints.compute(joint_velocities, joint_torques)
        p_f = foot_contact_penalty.compute(data)
        p_s = stability_penalty.compute(data)

        # 전체 정규화 보상 합산 (패널티는 음수 값이므로 합산 시 양수로 변환 또는 가중치에 음수 포함)
        # 여기서는 패널티 클래스가 음수 값을 반환한다고 가정하고 합산합니다.
        regularization_reward = p_j + p_a + p_v + p_f + p_s


        # 기존 reward 합산
        # NOTE: 원래의 MuJoCo 환경 reward(rewards 변수)와 사용자 정의 NIL reward를 어떻게 합칠지 결정 필요
        nil_reward = alpha * l2_reward + beta * iou_reward + gamma * regularization_reward

        masks = env.generate_masks(terminated, truncated)
        if not truncated:
            # replay_buffer.insert(ob, actions, nil_reward, masks, truncated, next_ob)
            replay_buffer.insert(ob_batch, actions, nil_reward, masks, truncated, next_ob)
        # ob = next_ob
        ob_batch = next_ob
        # TODO ob, terminated, truncated, reward_mask = env.reset_when_done(ob, terminated, truncated)
        batches = replay_buffer.sample_parallel_multibatch(batch_size=256, num_seeds=10)
        infos = agent.update(batches)

        # 각 스텝별 penalty 값을 penalty_log.csv 파일에 기록 및 터미널 출력
        print(f"[Step {step}] Penalties: torque={p_j:.4f}, action_smooth={p_a:.4f}, physics={p_v:.4f}, foot_contact={p_f:.4f}, stability={p_s:.4f}, total={regularization_reward:.4f}")
        if step == 0:
            # 첫 번째 스텝에서 파일을 새로 열고(기존 내용 삭제) 헤더 작성
            penalty_log = open("penalty_log.csv", "w")
            penalty_log.write("step,torque_penalty,action_smoothing,physics_constraints,foot_contact_penalty,stability_penalty,total_penalty\n")
        # 매 스텝 현재 penalty 값들을 CSV 형식으로 작성
        penalty_log.write(f"{step},{p_j},{p_a},{p_v},{p_f},{p_s},{regularization_reward}\n")
        # 파일에 즉시 쓰도록 flush
        penalty_log.flush()


    # Simulate with trained agent
    if args.log_video == "True":
        # simulation video의 각 프레임 별 [obs, image_obs, seg_obs, acs, rews, next_obs, terminals, seg_positions, seg_velocities, seg_torques]를 numpy array로 반환
        trajs = utils.rollout_n_trajectories(env, policy=agent, ntraj=1, max_traj_length=5000, render=True, seg_render=True)

        # tensorboard에 video 형식으로 저장
        mylogger.log_trajs_as_videos(trajs, step=0, max_videos_to_save=1, fps=10, video_title="test_basketball")

    del trajs
    env.close()