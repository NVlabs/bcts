import argparse
from argparse import Namespace

from datetime import datetime
import random
import torch
from tqdm import tqdm
import wandb

import sys

sys.path.append("../Rainbow/")

from bcts_agent import BCTSAgent
from env import Env
from cule_env import CuleEnv
from memory import ReplayMemory
from test import test


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', type=str2bool, default=False, nargs='?',
                    const=True, help='Disable CUDA')
# parser.add_argument('--disable-cuda',  action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='breakout', help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', type=str2bool, default=True, nargs='?',
                    const=True, help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=150, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS',
                    help='Number of training steps between logging status')
parser.add_argument('--render', type=str2bool, default=True, nargs='?',
                    const=True, help='Display screen (testing only)')
parser.add_argument('--tree-depth', type=int, default=0, metavar='N', help='Depth of Cule tree')
parser.add_argument('--use_cule', type=str2bool, default=True, nargs='?',
                    const=True, help='Choose whether to use cule')
parser.add_argument('--use_pretrained', type=str2bool, default=True, nargs='?',
                    const=True, help='Choose whether to load the pretrained model')
parser.add_argument('--tree_top_percentile', type=str2bool, default=False, nargs='?',
                    const=True, help='Choose whether to use tree_top_percentile')
parser.add_argument('--op-scale-factor', type=float, default=1.025, metavar='w',
                    help='Off-policy scale factor for tree correction')
parser.add_argument('--override-default-op-factor', type=str2bool, default=False, nargs='?',
                    const=True, help='Whether to override the default op_factor dict per game')
parser.add_argument('--multiplicative-op-factor', type=str2bool, default=True, nargs='?',
                    const=True, help='Choose whether to use a multiplicative or additive op_factor')
parser.add_argument('--prunning-std-thresh', type=float, default=100, metavar='w',
                    help='Best-to-second-best std threshold for tree prunning')


def rename_env(snake_env):
    def to_camel_case(snake_str):
        components = snake_str.split('_')
        # We capitalize the first letter of each component except the first one
        # with the 'title' method and join them together.
        return ''.join(x.title() for x in components)

    return to_camel_case(snake_env) + 'NoFrameskip-v4'


# Setup
args = parser.parse_args()
args.evaluation_interval = 35000
args.render = False
full_env_name = rename_env(args.game)
args.model = 'pretrained/{}.pth'.format(args.game) if args.use_pretrained else ''

if args.tree_depth > 0:
    args.use_cule = True
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

wandb.init(config=args, project="rainbow")
args = Namespace(**wandb.config.as_dict())

random.seed(args.seed)
torch.manual_seed(random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(random.randint(1, 10000))
    torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
else:
    args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


# Environment
if args.use_cule:
    env = CuleEnv(args, full_env_name=full_env_name)
else:
    env = Env(args)

env.train()
action_space = env.action_space()

# Agent
dqn = BCTSAgent(args, env, full_env_name=full_env_name)
mem = ReplayMemory(args, args.memory_capacity)
priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
    if done:
        state, done = env.reset(), False

    next_state, _, done = env.step(random.randint(0, action_space - 1))
    val_mem.append(state, None, None, done)
    state = next_state
    T += 1

if args.evaluate:
    dqn.eval()  # Set DQN (online network) to evaluation mode
    results = test(args, 0, dqn, val_mem, evaluate=True, env=env, env_name=full_env_name)  # Test
    print('Avg. reward: {} +/- {}'.format(results['avg_reward'], results['std_rew']))
    print('0.25 quantile: {}, 0.75 quantile: {}'.format(results['first_quantile'], results['last_quantile']))
    wandb.log(results)
else:
    # Training loop
    dqn.train()
    done = True
    # while T < args.T_max:
    for T in tqdm(range(args.T_max)):
        if done:
            state, done = env.reset(), False

        if T % args.replay_frequency == 0:
            dqn.reset_noise()  # Draw a new set of noisy weights

        action = dqn.act(state)  # Choose an action greedily (with noisy weights)
        next_state, reward, done = env.step(action)  # Step
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        mem.append(state, action, reward, done)  # Append transition to memory
        T += 1

        if T % args.log_interval == 0:
            log('T = ' + str(T) + ' / ' + str(args.T_max))

        # Train and test
        if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase,
                                      1)  # Anneal importance sampling weight β to 1

            if T % args.replay_frequency == 0:
                dqn.learn(mem)  # Train with n-step distributional double-Q learning

            if T % args.evaluation_interval == 0:
                dqn.eval()  # Set DQN (online network) to evaluation mode
                results = test(args, T, dqn, val_mem, env=env, env_name=full_env_name)  # Test
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(
                    results['avg_reward']) + ' | Avg. Q: ' + str(results['avg_Q']))
                wandb.log({'episodic_reward': results['avg_reward'], 'average Q': results['avg_Q']}, step=T)
                dqn.train()  # Set DQN (online network) back to training mode

            # Update target network
            if T % args.target_update == 0:
                dqn.update_target_net()

        state = next_state

env.close()
