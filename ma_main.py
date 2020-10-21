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

from pettingzoo.mpe import simple_tag_v1

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import logging
import multiprocessing as mp

info = mp.get_logger().info


def make_env():
    return simple_tag_v1.parallel_env(num_good=1, num_adversaries=3, num_obstacles=4, max_frames=50)


def train_in_turn(n_agents, i, n_iter):
    return n_iter % n_agents == i


def plot(statistics, keyword, agents):
    assert len(statistics) == len(agents)
    n_agents = len(agents)
    iters = []
    values = []
    names = []

    for i in range(n_agents):
        s = statistics[i][keyword]
        for it, v in s:
            iters.append(it)
            values.append(v)
            names.append(agents[i])

    df = pd.DataFrame(dict(iter=iters, value=values, agent=names))

    sns.lineplot(x="iter", y="value", hue="agent", data=df)
    plt.show()


def main():
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)
    args = get_args()
    num_agents = args.num_agents
    num_envs = args.num_processes

    agents = []
    main_agent_conns = []
    envs = []
    main_env_conns = []

    obs_locks = []
    act_locks = []

    # json.dump(vars(args), open("configs/simple_tag.json", "w"))

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

    env = make_env()
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
    print(input_structures)

    # print(len(env.agents))
    for i, agent in enumerate(env.agents):
        conn1, conn2 = mp.Pipe()
        main_agent_conns.append(conn1)
        obs_space = env.observation_spaces[agent]
        act_space = env.action_spaces[agent]
        ap = Agent(i, env.agents[i], thread_limit=12 // num_agents, logger=info, args=args, obs_space=obs_space,
                   input_structure=input_structures[agent],
                   act_space=act_space, main_conn=conn2,
                   obs_shm=obs_shm, buffer_start=obs_indices[i][0], buffer_end=obs_indices[i][1],
                   obs_locks=[locks[i] for locks in obs_locks], act_shm=act_shm,
                   act_locks=[locks[i] for locks in act_locks], use_attention=args.use_attention,
                   train=partial(train_in_turn, args.num_agents, i))
        agents.append(ap)
        ap.start()

    for i in range(num_envs):
        conn1, conn2 = mp.Pipe()
        main_env_conns.append(conn1)
        ev = Environment(i, info, args, env=make_env(), agents=env.agents, main_conn=conn2,
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

    statistics = []
    for i in range(num_agents):
        statistics.append(main_agent_conns[i].recv())

    plot(statistics, "reward", env.agents)
    plot(statistics, "grad_norm", env.agents)

    obs = env.reset()
    dones = {agent: False for agent in env.agents}
    while True:
        actions = dict()
        for i, agent in enumerate(env.agents):
            main_agent_conns[i].send((obs[agent], dones[agent]))
            actions[agent] = main_agent_conns[i].recv()
            # print(agent, actions[agent])
        obs, rewards, dones, infos = env.step(actions)
        env.render()
        time.sleep(0.1)

        not_done = False
        for agent in env.agents:
            not_done = not_done or not dones[agent]

        if not not_done:
            obs = env.reset()
            dones = {agent: False for agent in env.agents}

    for i, agent in enumerate(agents):
        main_agent_conns[i].send(None)
        agent.join()

if __name__ == "__main__":
    main()
