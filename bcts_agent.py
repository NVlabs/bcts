import random
import torch
from agent import Agent
from cule_bfs import CuleBFS
import os

class BCTSAgent(Agent):
    def __init__(self, args, env, full_env_name):
        super(BCTSAgent, self).__init__(args, env)
        self.args = args
        self.full_env_name = full_env_name
        if args.use_cule:
            self.cule_bfs = CuleBFS(env_name=full_env_name, tree_depth=args.tree_depth, verbose=False,
                                    ale_start_steps=1,
                                    ignore_value_function=False, perturb_reward=True, step_env=env.env, args=args)
        if args.model and os.path.isfile(args.model):
            # Loading happens in the parent class; here we only alert
            print('Loaded pretrained agent from: {}'.format(args.model))
        else:
            raise FileNotFoundError('Pretrained agent not found: {}'.format(args.model))

    # Acts based on single state (no batch)
    def act(self, state, fire_pressed=[False], max_cut_time=[0]):
        if self.args.use_cule:
            return self.cule_bfs.bfs(state, self.online_net, self.support, self.args, fire_pressed=fire_pressed,
                                     max_cut_time=max_cut_time)
        else:
            with torch.no_grad():
                return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()


    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001, discard_timing=[False], max_cut_time=[0]):
        fire_pressed = [False]
        # High ε can reduce evaluation scores drastically
        if random.random() < epsilon:
            exploring = True
            action = random.randrange(self.action_space)
        else:
            exploring = False
            action = self.act(state, fire_pressed=fire_pressed, max_cut_time=max_cut_time)
        discard_timing[0] = fire_pressed[0] or exploring
        return action
