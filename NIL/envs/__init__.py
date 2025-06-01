from gymnasium.envs import register

from envs.env import ROBOTS, TASKS

robot = "h1"
control = "pos"
for task, task_info in TASKS.items():
    task_info = task_info()
    kwargs = task_info.kwargs.copy()
    kwargs["robot"] = robot
    kwargs["control"] = control
    kwargs["task"] = task
    register(
        id=f"{robot}-{task}-v0",
        entry_point="envs.env:HumanoidEnv",
        max_episode_steps=task_info.max_episode_steps,
        kwargs=kwargs,
    )
