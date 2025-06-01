import numpy as np
import cv2

def rollout_trajectory(env, policy, max_traj_length, render=False, seg_render=False):

    # initialize env
    ob, _ = env.reset()

    obs, acs, rews, next_obs, terminals, image_obs = [], [], [], [], [], []
    seg_obs = []
    seg_positions = []
    seg_velocities = []
    seg_torques = []
    steps = 0
    while True:
        if render:
            if seg_render:
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env
                # seg_img = base_env.seg_render()
                # image_obs.append(seg_img)
                # print("seg_Img", seg_img)
            else:
                # render image of the simulated env
                rgb_img = env.render()
                image_obs.append(rgb_img)
        # if render and seg_render:
            # render segmentation mask image
            # seg_mask = env.render(mode="depth", camera_name="track")
            # seg_obs.append(seg_mask)
            # seg_mask = env.render(camera_name="track", depth=True) # for newer mujoco versions



        # use the most recent ob to decide what to do
        if ob.shape[0] != 1:
            ob = ob.reshape(1, -1)
        obs.append(ob)
        if policy is None:
            # if no policy is provided, take a random action
            ac = env.action_space.sample()
        else:
            ac = policy.sample_actions(ob, temperature=1.0)
            ac = ac.squeeze(0)
        acs.append(ac)
        # take that action and record results
        ob, rew, terminated, truncated, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rews.append(rew)

        # end the rollout if the rollout ended
        rollout_done = (terminated or truncated) or (steps >= max_traj_length)  
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Traj(obs, image_obs, seg_obs, acs, rews, next_obs, terminals)

def rollout_n_trajectories(env, policy, ntraj, max_traj_length, render=False, seg_render=False):
    """
    Collect ntraj rollouts.
    """
    trajs = []
    for _ in range(ntraj):
        traj = rollout_trajectory(env, policy, max_traj_length, render, seg_render)
        trajs.append(traj)

    return trajs

def Traj(obs, image_obs, seg_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    if seg_obs != []:
        seg_obs = np.stack(seg_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "seg_obs": np.array(seg_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    generated_video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        generated_video_frames.append(frame)
    cap.release()
    print("generated_video_frames", len(generated_video_frames), "frames loaded from", video_path)

    return generated_video_frames