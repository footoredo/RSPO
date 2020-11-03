import copy
import glob
import os
import time
import json
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from multiprocessing import shared_memory

from argparse import Namespace

from functools import partial

from a2c_ppo_acktr.multi_agent import Agent, Environment


from a2c_ppo_acktr.multi_agent.utils import *

import logging
import multiprocessing as mp


def make_env(env_name, steps):
    if steps is None:
        steps = 32
    if env_name == "simple-tag":
        from pettingzoo.mpe import simple_tag_v1
        return simple_tag_v1.parallel_env(num_good=1, num_adversaries=3, num_obstacles=4, max_frames=steps - 3)
    elif env_name == "simple":
        from pettingzoo.mpe import simple_v1
        return simple_v1.parallel_env(num_targets=2, max_frames=steps - 3)


def train_in_turn(n_agents, i, n_iter):
    return n_iter % n_agents == i


def main(args, logger):
    num_agents = args.num_agents
    num_envs = args.num_processes

    agents = []
    main_agent_conns = []
    envs = []
    main_env_conns = []

    obs_locks = []
    act_locks = []

    from datetime import datetime
    save_dir = mkdir2(args.save_dir, datetime.now().isoformat())
    json.dump(vars(args), open(os.path.join(save_dir, "config.json"), "w"))

    for i in range(num_envs):
        _obs_locks = []
        _act_locks = []
        for j in range(num_agents):
            _obs_locks.append(mp.Event())
            # _obs_locks[-1].set()
            _act_locks.append(mp.Event())
            # _act_locks[-1].set()
        obs_locks.append(_obs_locks)
        act_locks.append(_act_locks)

    _make_env = partial(make_env, args.env_name, args.episode_steps)
    env = _make_env()
    assert num_agents == len(env.agents)
    obs_buffer_size = 0
    item_size = np.zeros(1, dtype=np.float32).nbytes
    # print(item_size)
    obs_indices = []
    for agent in env.agents:
        # print(env.observation_spaces[agent].shape)
        next_obs_buffer_size = obs_buffer_size + item_size * num_envs * (env.observation_spaces[agent].shape[0] + 2)
        # print(next_obs_buffer_size - obs_buffer_size)
        obs_indices.append((obs_buffer_size, next_obs_buffer_size))
        obs_buffer_size = next_obs_buffer_size

    obs_shm = shared_memory.SharedMemory(create=True, size=obs_buffer_size)
    act_shm = shared_memory.SharedMemory(create=True, size=num_envs * num_agents)

    input_structures = env.input_structures
    # print("input_structures:", input_structures)

    # print(len(env.agents))
    for i, agent in enumerate(env.agents):
        conn1, conn2 = mp.Pipe()
        main_agent_conns.append(conn1)
        obs_space = env.observation_spaces[agent]
        act_space = env.action_spaces[agent]
        ap = Agent(i, env.agents[i], thread_limit=8 // num_agents, logger=logger.info, args=args, obs_space=obs_space,
                   input_structure=input_structures[agent],
                   act_space=act_space, main_conn=conn2,
                   obs_shm=obs_shm, buffer_start=obs_indices[i][0], buffer_end=obs_indices[i][1],
                   obs_locks=[locks[i] for locks in obs_locks], act_shm=act_shm,
                   act_locks=[locks[i] for locks in act_locks], use_attention=args.use_attention,
                   save_dir=save_dir,
                   train=partial(train_in_turn, args.num_agents, i))
        agents.append(ap)
        ap.start()

    for i in range(num_envs):
        conn1, conn2 = mp.Pipe()
        main_env_conns.append(conn1)
        ev = Environment(i, logger.info, args, env=_make_env(), agents=env.agents, main_conn=conn2,
                         obs_shm=obs_shm, obs_locks=obs_locks[i], act_shm=act_shm, act_locks=act_locks[i])
        envs.append(ev)
        ev.start()

    for conn in main_agent_conns:
        conn.send(None)

    for conn in main_env_conns:
        conn.send(None)

    for ev in envs:
        ev.join()

    obs_shm.close()
    obs_shm.unlink()
    act_shm.close()
    act_shm.unlink()

    statistics = dict()
    for i, agent in enumerate(env.agents):
        statistics[agent] = main_agent_conns[i].recv()

    import joblib
    joblib.dump(statistics, os.path.join(save_dir, "statistics.obj"))

    if args.plot:
        plot_statistics(statistics, "reward")
        plot_statistics(statistics, "grad_norm")
        plot_statistics(statistics, "value_loss")
        plot_statistics(statistics, "action_loss")
        plot_statistics(statistics, "dist_entropy")

    env.seed(args.seed + 31234)
    obs = env.reset()
    dones = {agent: False for agent in env.agents}
    num_games = 0
    close_to = [0, 0]
    while True:
        actions = dict()
        for i, agent in enumerate(env.agents):
            main_agent_conns[i].send((obs[agent], dones[agent]))
            actions[agent] = main_agent_conns[i].recv()
            # print(agent, actions[agent])
        obs, rewards, dones, infos = env.step(actions)
        if args.render:
            env.render()
            time.sleep(0.1)

        not_done = False
        for agent in env.agents:
            not_done = not_done or not dones[agent]

        if not not_done:
            num_games += 1
            # print(infos[env.agents[0]])
            close_to[np.argmin(infos[env.agents[0]])] += 1
            if num_games >= 100:
                break
            obs = env.reset()
            dones = {agent: False for agent in env.agents}

    for i, agent in enumerate(agents):
        main_agent_conns[i].send(None)
        agent.join()

    return close_to


if __name__ == "__main__":
    args = get_args()
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)
    main(args, logger)
