import torch
from torch.distributions import Categorical
import os
from envs import GoHighEnv
import atexit
import platform


from models import RecurrentActor

options = dict(
    render=True,
    player1='ai',
    player2='human',
    char1='fox',
    char2='falco',
    stage='battlefield',
)
if platform.system == 'Windows':
    options["windows"] = True


env = GoHighEnv(frame_limit=7200, options=options)
atexit.register(env.close)

net = RecurrentActor(env.observation_space.n, env.action_space.n)

obs = env.reset()
with torch.no_grad():
    while True:
        obs = torch.tensor(obs)
        logps, _ = net(obs)
        actions = Categorical(logits=logps).sample().numpy()
        obs, reward, done, infos = env.step(actions)
        if done:
            break
