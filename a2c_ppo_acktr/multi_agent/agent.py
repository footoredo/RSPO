import os
import gym
import numpy as np
import torch

import multiprocessing as mp

from argparse import Namespace

from a2c_ppo_acktr.storage import RolloutStorage
from .utils import get_agent, release_all_locks, acquire_all_locks, mkdir2, load_actor_critic, ts, reseed


def true_func(n_iter):
    return True


class Agent(mp.Process):
    def __init__(self, agent_id, agent_name, thread_limit, logger, args: Namespace, obs_space, input_structure,
                 act_space, main_conn, obs_shm, buffer_start, buffer_end, act_shm, obs_locks, act_locks,
                 use_attention=False, save_dir=None, train=true_func, reference_agent=None):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.thread_limit = thread_limit
        self.logger = logger

        self.args = args
        self.seed = reseed(args.seed, "agent-{}".format(self.agent_id))
        self.num_steps = args.num_steps
        self.num_envs = args.num_processes
        self.num_agents = args.num_agents
        self.batch_size = self.num_steps * self.num_envs
        self.num_updates = args.num_env_steps // args.num_steps // args.num_processes
        self.save_interval = args.save_interval
        self.save_dir = mkdir2(save_dir or args.save_dir, str(agent_name))

        self.obs_space = obs_space
        self.input_structure = input_structure
        self.act_space = act_space

        self.main_conn = main_conn
        self.obs_shm = obs_shm
        self.buffer_start = buffer_start
        self.buffer_end = buffer_end
        self.act_shm = act_shm
        self.obs_locks = obs_locks
        self.act_locks = act_locks

        self.use_attention = use_attention
        self.train = train

        self.reference_agent = reference_agent

        if args.gail:
            raise ValueError("gail is not supported")

    def log(self, msg):
        self.logger("{}: {}".format(self.agent_name, msg))

    def get_obs(self):
        # self.log("acquire obs locks")
        acquire_all_locks(self.obs_locks)
        data = np.frombuffer(self.obs_shm.buf[self.buffer_start: self.buffer_end], dtype=np.float32).reshape(self.num_envs, -1)
        obs = data[:, :-2]
        reward = data[:, -2:-1]
        done = data[:, -1:]
        # if any(reward > 0.):
        #     print(data, obs, reward)
        # self.log("release obs locks")
        # release_all_locks(self.obs_locks)
        return ts(obs), ts(reward), ts(done)

    def put_act(self, i_env, act):
        self.act_shm.buf[i_env * self.num_agents + self.agent_id] = act

    @staticmethod
    def obs_distance(obs1, obs2):
        return np.linalg.norm((obs1 - obs2)[-4:])

    @staticmethod
    def obs_distance_all(obs, obs_list):
        return min([Agent.obs_distance(obs, other_obs) for other_obs in obs_list])

    def run(self):
        if self.thread_limit is not None:
            torch.set_num_threads(self.thread_limit)
        args = self.args
        use_dice = args.algo == "loaded-dice"
        # dice_lambda = args.dice_lambda if use_dice else None
        np_random = np.random.RandomState(self.seed)
        if args.reseed_step is not None and args.reseed_step < 0:
            seed = reseed(self.seed, "reseed-{}".format(args.reseed_z))
            # seed = self.seed
            # for iz in range(args.reseed_z):
            #     seed = np_random.randint(10000)
            torch.manual_seed(seed)
        else:
            torch.manual_seed(self.seed)
        # self.log(self.obs_space)
        ref = self.args.use_reference
        ref_agents = self.reference_agent
        if ref and type(ref_agents) != list:
            ref_agents = [ref_agents]
        n_ref = len(ref_agents) if ref else 0
        actor_critic, agent = get_agent(self.args, self.obs_space, self.input_structure, self.act_space,
                                        self.save_dir, n_ref=n_ref)
        agent.ref_agent = self.reference_agent
        # print(self.num_steps)
        if args.load_dir is not None and args.load:
            self.log("Loading model from {}".format(args.load_dir))
            load_actor_critic(actor_critic, args.load_dir, self.agent_name, args.load_step)
            self.log("Done.")

        rollouts = RolloutStorage(self.num_steps, self.num_envs,
                                  self.obs_space.shape, self.act_space,
                                  actor_critic.recurrent_hidden_state_size,
                                  reward_dim=n_ref + 1)

        # acquire_all_locks(self.act_locks)

        self.main_conn.recv()

        def merge_ref_strat(_obs):
            ref_strat = [ra.get_strategy(_obs, None, None) for ra in ref_agents]
            ref_strat = torch.stack(ref_strat, dim=2).max(dim=2)
            return torch.cat([_obs, ref_strat], dim=1)

        obs, _, _ = self.get_obs()

        # self.log("step {} - received {}".format(0, obs))
        # self.log(obs.shape)
        rollouts.obs[0].copy_(obs)
        statistics = dict(reward=[], grad_norm=[], value_loss=[], action_loss=[], dist_entropy=[], dist_penalty=[])
        sum_reward = 0.
        sum_dist_penalty = 0.
        # last_update_iter = -1
        update_counter = 0

        episode_steps = self.args.episode_steps
        num_envs = self.num_envs
        num_episodes = self.num_steps // episode_steps
        observations = [[[] for _ in range(num_episodes)] for _ in range(num_envs)]

        for it in range(self.num_updates):
            # cur_ref = ref and it % 2 == 0
            cur_ref = False
            cur_agent = self.reference_agent if cur_ref else actor_critic
            if cur_ref:
                for i in range(num_envs):
                    for j in range(num_episodes):
                        observations[i][j] = []
            # self.log(1)
            # self.log("iter: {}".format(it))
            for step in range(self.num_steps):
                # self.log(step)
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = cur_agent.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                    if cur_ref:
                        for i in range(self.num_envs):
                            observations[i][step // episode_steps].append(rollouts.obs[step][i].numpy())
                    # self.log("step {} - act {}".format(it * self.num_steps + step, action))

                action = action.data
                for i_env in range(self.num_envs):
                    self.put_act(i_env, action[i_env])
                # self.log("release act locks")
                release_all_locks(self.act_locks)

                obs, reward, done = self.get_obs()
                _reward = reward

                if not cur_ref and ref:
                    _dis = [ref_agent.get_probs(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                                                rollouts.masks[step], action).detach().squeeze()
                            for ref_agent in ref_agents]
                    _dis2 = [ref_agent.get_value(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                                                 rollouts.masks[step]).detach().squeeze()
                             for ref_agent in ref_agents]
                    dis = torch.stack(_dis, dim=0).transpose(1, 0)
                    dis2 = torch.stack(_dis2, dim=0).transpose(1, 0)
                    t_dis = (-torch.log(dis)).clamp(max=500.)

                    # _dis = [ref_agent.get_value(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    #                             rollouts.masks[step]).detach().squeeze()
                    #         for ref_agent in ref_agents]
                    # dis = torch.stack(_dis, dim=0).transpose(1, 0)
                    # t_dis = dis

                    # sum_dist_penalty += t_dis.sum().item()
                    _reward = torch.cat([reward, t_dis], dim=-1)
                elif ref:
                    # print(reward.size(), _reward.size())
                    _reward = torch.cat([reward] * (n_ref + 1), dim=-1)

                # print(reward)
                sum_reward += reward.sum().item()
                # self.log("step {} - received {}, {}, {}".format(it * self.num_steps + step + 1, obs, reward, done))
                # self.log("acquire act locks")
                # acquire_all_locks(self.act_locks)

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.zeros_like(masks)
                # if any(reward > 0.):
                #     print(reward)
                # print(_reward.size())
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, _reward, masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)
            sum_dist_penalty += rollouts.returns[:, 1:].mean().item()

            if self.train(it) and not cur_ref:
                # self.log("ready to update")
                value_loss, action_loss, dist_entropy, grad_norm = agent.update(rollouts)
                self.log("Update #{}, value_loss {}, action_loss {}, dist_entropy {}, grad_norm {}"
                         .format(it, value_loss, action_loss, dist_entropy, grad_norm))
                statistics["value_loss"].append((it, value_loss))
                statistics["action_loss"].append((it, action_loss))
                statistics["dist_entropy"].append((it, dist_entropy))
                statistics["grad_norm"].append((it, grad_norm))
                update_counter += 1
                if self.save_interval > 0 and update_counter % self.save_interval == 0:
                    current_save_dir = mkdir2(self.save_dir, "update-{}".format(update_counter))
                    torch.save(actor_critic.state_dict(),
                               os.path.join(current_save_dir, "model.obj"))

            if not cur_ref:
                statistics["reward"].append((it, sum_reward / self.num_steps / self.num_envs * args.episode_steps))
                statistics["dist_penalty"].append((it, sum_dist_penalty))
            # last_update_iter = it
            sum_reward = 0.
            sum_dist_penalty = 0.

            rollouts.after_update()

        torch.save(actor_critic.state_dict(), os.path.join(self.save_dir, "model.obj"))

        recurrent_hidden_states = torch.zeros((1, actor_critic.recurrent_hidden_state_size))
        self.main_conn.send(statistics)
        # print(recurrent_hidden_states)
        while True:
            command = self.main_conn.recv()
            if command is None:
                break
            obs, done = command
            _, action, _, recurrent_hidden_states = actor_critic.act(ts([obs]), recurrent_hidden_states, ts([[0.0] if done else [1.0]]))
            self.main_conn.send(action.detach()[0].item())
