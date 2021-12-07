import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch

from cule_env import CuleEnv
from tqdm import tqdm
import numpy as np
import time
from scipy.stats import mstats
import wandb
from copy import deepcopy


FIRE_LIST = ['Breakout', 'Beam']

# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10

def extract_stats(rew_vector):
  rew_vector_np = np.asarray(rew_vector)
  avg_reward = rew_vector_np.mean()
  std_rew, std_q = np.asarray(rew_vector_np).std(), np.asarray(rew_vector_np).std()
  max_rew, med_rew = rew_vector_np.max(), np.median(rew_vector_np)
  quantiles = mstats.mquantiles(rew_vector_np, axis=0)
  results = {}
  results['avg_reward'] = avg_reward
  results['std_rew'] = std_rew
  results['std_q'] = std_q
  results['max_rew'] = max_rew
  results['med_rew'] = med_rew
  results['first_quantile'] = quantiles[0]
  results['last_quantile'] = quantiles[-1]
  return results

# Test DQN
def test(args, T, dqn, val_mem, evaluate=False, env=None, env_name=None):
  global Ts, rewards, Qs, best_avg_reward
  if env is None:
    # env = Env(args)
    env = CuleEnv(args)
  env.eval()
  Ts.append(T)
  T_rewards, T_Qs = [], []
  fire_env = len([e for e in FIRE_LIST if e in env_name]) > 0
  # Test performance over several episodes
  done = True
  time_log = []
  mean_depth_vec = []
  for i_ep in tqdm(range(args.evaluation_episodes)):
    c = 0
    while True:
      c += 1
      if done:
        state, reward_sum, done, fire_reset = env.reset(), 0, False, fire_env
        lives = env.lives.clone()
      # if c % 5 == 0:
      #   import matplotlib.pyplot as plt
      #   plt.imshow(state[-1].cpu())
      #   plt.show()
      max_cut_time = [0]
      start_time = time.time()
      discard_timing = [fire_reset]
      action = 1 if fire_reset else dqn.act_e_greedy(state, discard_timing=discard_timing,
                                                     max_cut_time=max_cut_time)  # Choose an action ε-greedily
      end_time = time.time()
      if not discard_timing[0]:
        time_log.append(end_time - start_time - max_cut_time[0])
        # print('avg step time: {}, std step time: {}'.format(np.asarray(time_log).mean(), np.asarray(time_log).std()))

      state, reward, done = env.step(action)  # Step
      reward_sum += reward

      # Check for lost life
      new_lives = env.lives.clone()
      fire_reset = new_lives < lives and fire_env
      lives.copy_(new_lives)

      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        print('trunc count: {}/{}'.format(dqn.cule_bfs.trunc_count, c))
        print('ep rew: {}'.format(reward_sum))
        results_rolling = deepcopy(extract_stats(T_rewards))
        for key in list(results_rolling.keys()):
          results_rolling[key + '_rolling'] = results_rolling.pop(key)
        results_rolling['step_time_avg'], results_rolling['step_time_std'] = np.asarray(time_log).mean(), np.asarray(time_log).std()
        results_rolling['ep_rew'] = reward_sum
        ep_mean_depth = np.mean(dqn.cule_bfs.mean_depth_vec)
        results_rolling['mean_depth'] = ep_mean_depth; dqn.cule_bfs.reset_mean_depth_vec()
        mean_depth_vec.append(ep_mean_depth)
        wandb.log(results_rolling, step=i_ep)
        dqn.cule_bfs.trunc_count = 0
        break
  env.close()

  # Test Q-values over validation memory
  for state in val_mem:  # Iterate over valid states
    T_Qs.append(dqn.evaluate_q(state))

  results = extract_stats(T_rewards)
  results['mean_depth_final'] = np.mean(mean_depth_vec)

  if not evaluate:
    # Append to results
    rewards.append(T_rewards)
    Qs.append(T_Qs)

    # Plot
    _plot_line(Ts, rewards, 'Reward', path='results')
    _plot_line(Ts, Qs, 'Q', path='results')

    # Save model parameters if improved
    if results['avg_reward'] > best_avg_reward:
      best_avg_reward = results['avg_reward']
      dqn.save('results')

  # Return average reward and Q-value
  return results


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
