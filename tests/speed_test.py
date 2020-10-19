import copy
import glob
import os
import time
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
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import logging
import multiprocessing as mp
info = mp.get_logger().info


def run(agent_id):
    # torch.set_num_threads(1)
    args = get_args()

    torch.manual_seed(1234 + agent_id)
    # torch.cuda.manual_seed_all(1234)

    info(torch.random.get_rng_state())
    w = torch.empty(3, 1)
    nn.init.normal_(w)
    info(torch.rand(5))
    info(w)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    observation_space = gym.spaces.Box(0.0, 1.0, shape=(10,))
    action_space = gym.spaces.Discrete(5)

    actor_critic = Policy(
        observation_space.shape,
        action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    bs = 1024
    n_envs = 8
    rollouts = RolloutStorage(bs // n_envs, n_envs,
                              observation_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = np.random.rand(*observation_space.shape)
    obs = torch.from_numpy(obs).float().to(device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    st = time.time()
    # for n, p in actor_critic.named_parameters():
    #     info("{}: {}".format(n, p.data[:1]))
    n_runs = 100
    #
    # for _ in range(n_runs):
    #     agent.update(rollouts)

    # while True:
    # for _ in range(n_runs):
    #     actor_critic.act(rollouts.obs[:bs], rollouts.recurrent_hidden_states[:bs],  rollouts.masks[:bs])
    info("Average time: {}s".format((time.time() - st) / n_runs))


def main():
    # print(torch.get_num_threads(), torch.get_num_interop_threads())
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    ps = []
    for i in range(5):
        p = mp.Process(target=run, args=(i,))
        ps.append(p)

    for p in ps:
        p.start()

    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
