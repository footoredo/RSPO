import gym
import numpy as np

import multiprocessing as mp

from argparse import Namespace

from .utils import *


class Environment(mp.Process):
    def __init__(self, env_id, logger, args: Namespace, env, agents, main_conn, obs_shm, act_shm, obs_locks, act_locks):
        super(Environment, self).__init__()

        self.env_id = env_id
        self.logger = logger

        self.args = args
        self.seed = reseed(args.seed, "env-{}".format(self.env_id))
        self.num_steps = args.num_steps
        self.num_envs = args.num_processes
        self.num_agents = args.num_agents
        self.reward_norm = args.reward_normalization
        self.batch_size = self.num_steps * self.num_envs
        self.num_env_steps = args.num_env_steps // args.num_processes
        self.reseed_step = args.reseed_step // args.num_processes if args.reseed_step is not None else None
        self.reseed_z = args.reseed_z
        self.ref = args.use_reference
        self.dtype = np.float32

        self.env = env
        self.agents = agents

        self.main_conn = main_conn
        self.obs_shm = obs_shm
        self.act_shm = act_shm
        self.obs_locks = obs_locks
        self.act_locks = act_locks

        self.np_random = np.random.RandomState(seed=self.seed)

    def reseed(self, step, z):
        _seed = None
        for iz in range(z):
            _seed = self.np_random.tomaxint() & ((1 << 32) - 1)
            self.env.seed(_seed)
        self.log("reseed with seed {}".format(_seed))
        return _seed

    def log(self, msg):
        self.logger("Environment-{}: {}".format(self.env_id, msg))

    def write(self, place, obs, reward, done):
        obs = np.array(obs, dtype=self.dtype)
        # self.log(obs.shape[0])
        # print(obs.shape[0], len(place))
        np.copyto(place[:obs.shape[0]], obs)
        place[obs.shape[0]] = reward
        place[obs.shape[0] + 1] = done

    def run(self):
        env = self.env
        args = self.args
        # ref = self.ref
        ref = False
        reset_every = args.num_steps
        # print(env.seed)
        env.seed(self.seed)
        # acquire_all_locks(self.obs_locks)

        reward_filters = {agent: Identity() for agent in self.agents}
        if self.reward_norm:
            reward_filters = {agent: RewardFilter(reward_filters[agent], shape=(), gamma=args.gamma, clip=False)
                              for agent in self.agents}

        if self.reseed_step is not None and 0 >= self.reseed_step:
            self.reseed(0, self.reseed_z)

        last_seed = None
        if ref:
            last_seed = self.reseed(0, 1)
            self.np_random.seed(last_seed)

        init_obs = env.reset()
        # self.log(init_obs)

        obs_places = []
        obs_lens = []
        offset = 0
        item_size = np.zeros(1, dtype=self.dtype).nbytes
        actions = dict()

        for agent in self.agents:
            actions[agent] = 0

            obs_space = self.env.observation_spaces[agent]
            assert isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1
            obs_len = obs_space.shape[0]
            full_len = obs_len + 2  # reward & done
            place = np.frombuffer(self.obs_shm.buf[offset + item_size * full_len * self.env_id:
                                                   offset + item_size * full_len * (self.env_id + 1)],
                                  dtype=self.dtype)
            obs_places.append(place)
            obs_lens.append(obs_len)
            self.write(place, init_obs[agent], 0., 0.)
            # np.copyto(place[:obs_len], init_obs[agent])
            # self.log("#{} - obs for {}: {}".format(0, agent, init_obs[agent]))
            offset += item_size * full_len * self.num_envs

        # release_all_locks(self.obs_locks)

        self.main_conn.recv()
        done = False

        for step in range(self.num_env_steps):
            self.np_random.tomaxint()  # flush state for 1 step
            release_all_locks(self.obs_locks)
            # self.log(step)
            acquire_all_locks(self.act_locks)
            for i, agent in enumerate(self.agents):
                actions[agent] = self.act_shm.buf[self.env_id * self.num_agents + i]
                # print(np.isnan(actions[agent]))
                # self.log("step {} from {} - act {}".format(step, agent, actions[agent]))
            # release_all_locks(self.act_locks)
            # acquire_all_locks(self.obs_locks)

            if self.reseed_step is not None and step + 1 == self.reseed_step:
                self.reseed(step + 1, self.reseed_z)

            if ref and (step + 1) % reset_every == 0:
                c = (step + 1) // reset_every
                if c % 2 == 0:
                    last_seed = self.reseed(step + 1, 1)
                    self.np_random.seed(last_seed)
                else:
                    self.env.seed(last_seed)

            if done:
                # self.log("done {}".format(step))
                obs = env.reset()
                for agent in self.agents:
                    reward_filters[agent].reset()
                rewards = {agent: 0. for agent in self.agents}
                dones = {agent: False for agent in self.agents}
            else:
                obs, rewards, dones, _ = env.step(actions)

            not_done = False
            for i, agent in enumerate(self.agents):
                # self.log("step {} - obs for {}: {}, {}, {}".format(i + 1, agent, obs[agent], rewards[agent], dones[agent]))
                self.write(obs_places[i], obs[agent], reward_filters[agent](rewards[agent]), dones[agent])
                not_done = not_done or not dones[agent]
            done = not not_done

        release_all_locks(self.obs_locks)
