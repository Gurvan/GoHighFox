import torch
import torch.nn as nn
from torch.distributions import Categorical
from statistics import mean, stdev
import time
import random
from math import log10, cos, pi

def train(params, net, optimizer, env):
    print("Resetting envs")
    obs = env.reset()
    print("Envs resetted")
    total_steps = 0
    n_save = 50000
    while total_steps < params.total_steps:
        T = 10**(14.5 - log10(1 + total_steps))
        aggressivity = 0.5 * (1 + 0.8 * cos(2 * pi * total_steps / T + pi / 2))
        print("Total steps:", total_steps, "\tAggressivity:", aggressivity)
        # print("Gathering rollouts")
        steps, obs = gather_rollout(params, net, env, obs, aggressivity)
        total_steps += params.num_workers * len(steps)
        final_obs = torch.tensor(obs)
        if params.cuda: final_obs = final_obs.cuda()
        _, final_values = net(final_obs)
        steps.append((None, None, None, final_values))
        # print("Processing rollouts")
        actions, logps, values, returns, advantages = process_rollout(params, steps)

        # print("Updating network")
        update_network(params, net, optimizer, actions, logps, values, returns, advantages)

        if total_steps > n_save:
            save_model(net, optimizer, "checkpoints/" + str(total_steps) + ".ckpt")
            save_model(net, optimizer, "checkpoints/latest.ckpt")
            n_save += 250000



def gather_rollout(params, net, env, obs, aggressivity):
    steps = []
    ep_rewards = [0.] * params.num_workers
    t = time.time()
    # aggressivity = [random.random() for _ in range(params.num_workers)]
    for _ in range(params.rollout_steps):
        for i in range(len(obs)):
            obs[i][-1] = aggressivity
        obs = torch.tensor(obs)
        if params.cuda: obs = obs.cuda()
        logps, values = net(obs)
        actions = Categorical(logits=logps).sample()

        obs, rewards, dones, _ = env.step(zip(actions.cpu().numpy(), params.num_workers * [aggressivity]))

        for i, done in enumerate(dones):
            ep_rewards[i] += rewards[i]

        rewards = torch.tensor(rewards).float().unsqueeze(1)
        if params.cuda: rewards = rewards.cuda()
        steps.append((rewards, actions, logps, values))

    print(round(time.time() - t, 3), round(mean(ep_rewards), 3), round(stdev(ep_rewards), 3))
    return steps, obs


def process_rollout(params, steps):
    # bootstrap discounted returns with final value estimates
    _, _, _, last_values = steps[-1]
    returns = last_values.data

    advantages = torch.zeros(params.num_workers, 1)
    if params.cuda: advantages = advantages.cuda()

    out = [None] * (len(steps) - 1)

    # run Generalized Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, actions, logps, values = steps[t]
        _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * params.gamma

        deltas = rewards + next_values.data * params.gamma - values.data
        advantages = advantages * params.gamma * params.lambd + deltas

        out[t] = actions, logps, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))


def update_network(params, net, optimizer, actions, logps, values, returns, advantages):
    # calculate action probabilities
    log_action_probs = logps.gather(1, actions.unsqueeze(-1))
    probs = logps.exp()
    policy_loss = (-log_action_probs * advantages).sum()
    value_loss = (.5 * (values - returns) ** 2.).sum()
    entropy_loss = (logps * probs).sum()

    print("P Loss:", round(policy_loss.item() / params.num_workers, 2), "V Loss:", round(value_loss.item() / params.num_workers, 2), "E Loss:", round(entropy_loss.item() / params.num_workers, 2))
    print()
    loss = policy_loss + value_loss * params.value_coeff + entropy_loss * params.entropy_coeff
    loss.backward()
    net.h = None

    nn.utils.clip_grad_norm_(net.parameters(), params.grad_norm_limit)
    optimizer.step()
    optimizer.zero_grad()


def save_model(net, optimizer, PATH):
    torch.save({
            'model': net,
            'optimizer': optimizer,
            }, PATH)
