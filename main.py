import argparse
import torch
import torch.optim as optim

from models import Actor
from envs import GoHighEnvVec
from train import train

parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--num-workers', type=int, default=4, help='number of parallel workers')
parser.add_argument('--rollout-steps', type=int, default=600, help='steps per rollout')
parser.add_argument('--total-steps', type=int, default=int(4e7), help='total number of steps to train for')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma parameter for GAE')
parser.add_argument('--lambd', type=float, default=1.00, help='lambda parameter for GAE')
parser.add_argument('--value_coeff', type=float, default=0.5, help='value loss coeffecient')
parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy loss coeffecient')
parser.add_argument('--grad_norm_limit', type=float, default=40., help='gradient norm clipping threshold')


args = parser.parse_args()

options = dict(
    render=False,
    player1='ai',
    player2='human',
    char1='fox',
    char2='falco',
    cpu2=1,
    stage='battlefield',
)

args.cuda = torch.cuda.is_available() and not args.no_cuda

if __name__ == "__main__":
    env = GoHighEnvVec(args.num_workers, args.total_steps, options)

    net = Actor(env.observation_space.n, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    train(args, net, optimizer, env)
