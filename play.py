import torch
from torch.distributions import Categorical
import os
from envs import GoHighEnv
import atexit
import platform

checkpoint = torch.load(os.path.join("checkpoints", "agent.ckpt"))

net = checkpoint["model"]

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

obs = env.reset()
with torch.no_grad():
    while True:
        obs = torch.tensor(obs)
        logps, _ = net(obs)
        actions = Categorical(logits=logps).sample().numpy()
        obs, reward, done, infos = env.step(actions)
        if done:
            break
        # print("Player 1 action state: ", env.obs.players[0].action_state, "\t\tPlayer 2 actions_state:", env.obs.players[1].action_state)
