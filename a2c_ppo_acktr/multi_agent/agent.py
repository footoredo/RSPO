import os
import gym
import numpy as np
import torch
import joblib

import multiprocessing as mp

from argparse import Namespace

from a2c_ppo_acktr.storage import RolloutStorage
from .utils import get_agent, release_all_locks, acquire_all_locks, mkdir2, load_actor_critic, ts, reseed


def true_func(n_iter):
    return True


class Agent(mp.Process):
    def __init__(self, agent_id, agent_name, thread_limit, logger, args: Namespace, obs_space, input_structure,
                 act_space, main_conn, obs_shm, buffer_start, buffer_end, act_shm, obs_locks, act_locks,
                 use_attention=False, save_dir=None, train=true_func, num_refs=None, reference_agent=None):
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

        self.num_refs = num_refs
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
        n_ref = self.num_refs if self.num_refs is not None else len(ref_agents) if ref else 0
        sample_n_ref = len(ref_agents) if ref else 0
        actor_critic, agent = get_agent(self.agent_name, self.args, self.obs_space, self.input_structure,
                                        self.act_space, self.save_dir, n_ref=n_ref, is_ref=False)
        # self.log("13123123")
        agent.ref_agent = self.reference_agent
        # print(self.num_steps)
        # self.log("123123")
        if args.load_dir is not None and args.load:
            self.log("Loading model from {}".format(args.load_dir))
            load_actor_critic(actor_critic, args.load_dir, self.agent_name, args.load_step)
            self.log("Done.")

        episode_steps = self.args.episode_steps
        num_envs = self.num_envs
        num_episodes = self.num_steps // episode_steps

        rollouts = RolloutStorage(episode_steps, num_envs,
                                  self.obs_space.shape, self.act_space,
                                  actor_critic.recurrent_hidden_state_size,
                                  num_refs=sample_n_ref, num_value_refs=n_ref)

        valid_rollouts = RolloutStorage(self.num_steps * num_envs, 1,
                                        self.obs_space.shape, self.act_space,
                                        actor_critic.recurrent_hidden_state_size,
                                        num_refs=sample_n_ref, num_value_refs=n_ref)

        self.main_conn.recv()

        def merge_ref_strat(_obs):
            ref_strat = [ra.get_strategy(_obs, None, None) for ra in ref_agents]
            ref_strat = torch.stack(ref_strat, dim=2).max(dim=2)
            return torch.cat([_obs, ref_strat], dim=1)

        observations = [[[] for _ in range(num_episodes)] for _ in range(num_envs)]

        statistics = dict(reward=[], grad_norm=[], value_loss=[], action_loss=[], dist_entropy=[], dist_penalty=[],
                          likelihood=[], total_episodes=[], accepted_episodes=[], efficiency=[])
        sum_reward = 0.
        sum_dist_penalty = 0.
        if ref_agents is not None:
            sum_likelihood = np.zeros(len(ref_agents))
            sum_likelihood_capped = np.zeros((num_envs, len(ref_agents)))
            likelihood_threshold = args.likelihood_threshold
            if type(likelihood_threshold) != list:
                likelihood_threshold = [likelihood_threshold] * len(ref_agents)
            likelihood_threshold = np.array(likelihood_threshold)
        else:
            sum_likelihood = np.zeros(1)
            sum_likelihood_capped = 0.
            likelihood_threshold = 0.
        # print(sum_likelihood)
        # last_update_iter = -1
        update_counter = 0

        for it in range(self.num_updates):
            total_samples = 0
            total_episodes = 0
            accepted_episodes = 0
            cur_agent = actor_critic
            # self.log(it)
            # self.log("iter: {}".format(it))
            remain_episodes = num_episodes * num_envs
            tmp_likelihood = []
            while True:
                total_episodes += num_envs
                decay = 1.
                obs, _, _ = self.get_obs()

                # self.log("step {} - received {}".format(0, obs))
                # self.log(obs.shape)
                rollouts.obs[0].copy_(obs)
                for step in range(episode_steps):
                    # self.log(step)
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = cur_agent.act(
                            rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                            rollouts.masks[step])
                        # self.log("step {} - act {}".format(it * self.num_steps + step, action))

                    action = action.data
                    for i_env in range(self.num_envs):
                        self.put_act(i_env, action[i_env])
                    # self.log("release act locks")
                    release_all_locks(self.act_locks)

                    obs, reward, done = self.get_obs()
                    _reward = reward

                    if ref:
                        _dis = [ref_agent.get_probs(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                                                    rollouts.masks[step], action).detach().squeeze(dim=-1)
                                for ref_agent in ref_agents]
                        # _dis2 = [ref_agent.get_value(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        #                              rollouts.masks[step]).detach().squeeze()
                        #          for ref_agent in ref_agents]
                        # print(len(_dis), _dis[1])
                        dis = torch.stack(_dis, dim=0).transpose(1, 0)
                        # dis2 = torch.stack(_dis2, dim=0).transpose(1, 0)
                        t_dis = (-torch.log(dis)).clamp(max=5000.)
                        sum_likelihood += t_dis.sum(dim=0).numpy()
                        old_sum_likelihood_capped = sum_likelihood_capped.copy()
                        sum_likelihood_capped += t_dis.numpy() * decay
                        decay *= args.likelihood_gamma
                        if args.use_likelihood_reward_cap:
                            sum_likelihood_capped = sum_likelihood_capped.clip(max=likelihood_threshold)
                        # print(t_dis)

                        # _dis = [ref_agent.get_value(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        #                             rollouts.masks[step]).detach().squeeze()
                        #         for ref_agent in ref_agents]
                        # dis = torch.stack(_dis, dim=0).transpose(1, 0)
                        # t_dis = dis

                        # sum_dist_penalty += t_dis.sum().item()
                        # print(sum_likelihood_capped)
                        _reward = torch.cat([reward, t_dis * 0.01, t_dis], dim=1)
                        _reward[:, 0] += args.likelihood_alpha * (sum_likelihood_capped - old_sum_likelihood_capped).mean(axis=1)

                    # print(reward)
                    total_samples += 1
                    sum_reward += reward.sum().item()
                    # self.log("step {} - received {}, {}, {}".format(it * self.num_steps + step + 1, obs, reward, done))
                    # self.log("acquire act locks")
                    # acquire_all_locks(self.act_locks)

                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    # if step == episode_steps - 1:
                    #     self.log("{}, {}".format(step, done[0]))
                    # sum_likelihood_capped *= masks.numpy()
                    bad_masks = torch.zeros_like(masks)
                    # if any(reward > 0.):
                    #     print(reward)
                    # print(_reward.size())
                    rollouts.insert(obs, recurrent_hidden_states, action,
                                    action_log_prob, value, _reward, masks, bad_masks)

                # available = np.zeros(num_envs, dtype=int)
                # self.log(sum_likelihood_capped)
                for i_env in range(num_envs):
                    # self.log(args.reject_sampling)
                    if not args.reject_sampling or all(sum_likelihood_capped[i_env] > likelihood_threshold):
                        # self.log(sum_likelihood_capped[i_env])
                        if ref:
                            tmp_likelihood.append(np.copy(sum_likelihood_capped[i_env]))
                        remain_episodes -= 1
                        accepted_episodes += int(all(sum_likelihood_capped[i_env] > likelihood_threshold))
                        if remain_episodes >= 0 and self.train(it):
                            keywords0 = ["actions", "action_log_probs", "value_preds", "rewards"]
                            keywords1 = ["obs", "recurrent_hidden_states", "masks", "bad_masks"]
                            step = valid_rollouts.step
                            for k in keywords0:
                                valid_rollouts.__getattribute__(k)[step: step + episode_steps, 0] = \
                                    rollouts.__getattribute__(k)[:episode_steps, i_env]
                            for k in keywords1:
                                valid_rollouts.__getattribute__(k)[step + 1: step + episode_steps + 1, 0] = \
                                    rollouts.__getattribute__(k)[1: episode_steps + 1, i_env]
                            valid_rollouts.step += episode_steps
                            assert valid_rollouts.step <= valid_rollouts.num_steps
                if ref:
                    sum_likelihood_capped.fill(0.)

                # self.log("{}, {}".format(remain_episodes, remain_episodes <= 0))
                self.main_conn.send(remain_episodes <= 0 or not self.train(it))
                # self.log(remain_episodes)
                if self.main_conn.recv():
                    break

            efficiency = accepted_episodes / total_episodes
            self.log("Total explored episodes: {}. Accepted episodes: {}. Efficiency: {:.2%}".
                     format(total_episodes, accepted_episodes, efficiency))

            # print(valid_rollouts.returns[0, 0, 1:])
            # self.log("{}, {}".format(tmp_likelihood[0], valid_rollouts.returns[0, 0, 1:]))
            # diff = []
            # for i in range(num_episodes * num_envs):
            #     diff.append(np.absolute(tmp_likelihood[i] - valid_rollouts.returns[episode_steps * i, 0, 1:].numpy()).sum())
            # self.log(np.max(diff))
            # sum_dist_penalty += valid_rollouts.returns[:, :, 1:].mean().item()

            if self.train(it):
                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        valid_rollouts.obs[-1], valid_rollouts.recurrent_hidden_states[-1],
                        valid_rollouts.masks[-1]).detach()

                valid_rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                               args.gae_lambda, args.likelihood_gamma, args.use_proper_time_limits)
                # self.log("ready to update")
                # valid_rollouts.action_loss_coef = 0.0 if efficiency < 0.1 else 1.0
                # self.log("before update")
                value_loss, action_loss, dist_entropy, grad_norm = agent.update(valid_rollouts)
                valid_rollouts.step = 0
                self.log("Update #{}, reward {}, likelihood {}, value_loss {}, action_loss {}, dist_entropy {}, grad_norm {}"
                         .format(it, sum_reward / total_episodes,
                                 sum_likelihood / total_episodes,
                                 value_loss, action_loss, dist_entropy, grad_norm))
                statistics["value_loss"].append((it, value_loss))
                statistics["action_loss"].append((it, action_loss))
                statistics["dist_entropy"].append((it, dist_entropy))
                statistics["grad_norm"].append((it, grad_norm))
                update_counter += 1
                if self.save_interval > 0 and update_counter % self.save_interval == 0:
                    current_save_dir = mkdir2(self.save_dir, "update-{}".format(update_counter))
                    torch.save(actor_critic.state_dict(),
                               os.path.join(current_save_dir, "model.obj"))

            statistics["reward"].append((it, sum_reward / total_episodes))
            if ref:
                statistics["dist_penalty"].append((it, sum_dist_penalty))
                statistics["likelihood"].append((it, sum_likelihood / total_episodes))
                statistics["total_episodes"].append((it, total_episodes))
                statistics["accepted_episodes"].append((it, accepted_episodes))
                statistics["efficiency"].append((it, accepted_episodes / total_episodes))
            # last_update_iter = it
            sum_reward = 0.
            sum_dist_penalty = 0.
            sum_likelihood.fill(0.)

            valid_rollouts.after_update()

            joblib.dump(statistics, os.path.join(self.save_dir, "statistics.obj"))

        torch.save(actor_critic.state_dict(), os.path.join(self.save_dir, "model.obj"))

        recurrent_hidden_states = torch.zeros((1, actor_critic.recurrent_hidden_state_size))
        self.main_conn.send(statistics)
        # print(recurrent_hidden_states)
        while True:
            command = self.main_conn.recv()
            if command is None:
                break
            obs, done = command
            # _, action, action_log_prob, recurrent_hidden_states = actor_critic.act(ts([obs]), recurrent_hidden_states, ts([[0.0] if done else [1.0]]))
            if obs[2] != 0 or obs[3] != 0:
                # if obs[0] > 0:
                #     action = 2
                # elif obs[1] > 0:
                #     action = 0
                # else:
                #     action = 4
                if obs[0] < 2:
                    action = 3
                elif obs[0] > 2:
                    action = 2
                elif obs[1] < 2:
                    action = 1
                elif obs[1] > 2:
                    action = 0
                else:
                    action = 4
                # if abs(obs[2]) > 0:
                #     if obs[2] < -1:
                #         action = 2
                #     elif obs[2] == -1:
                #         action = 4
                #     else:
                #         action = 3
                # elif abs(obs[3]) > 0:
                #     if obs[3] < -1:
                #         action = 0
                #     elif obs[3] == -1:
                #         action = 4
                #     else:
                #         action = 1
            else:
                # action = 4
                if abs(obs[4]) + abs(obs[5]) > 1:
                    # if obs[4] <= -1 and obs[0] >= 2:
                    #     action = 2
                    # elif obs[4] >= 1 and obs[0] <= 2:
                    #     action = 3
                    # elif obs[5] <= -1 and obs[1] >= 2:
                    #     action = 0
                    # elif obs[5] >= 1 and obs[1] <= 2:
                    #     action = 1
                    # else:
                    #     action = 4

                    if abs(obs[4]) > abs(obs[5]) or obs[4] == obs[5] and np_random.rand() < 0.5:
                        if obs[4] <= -1:
                            action = 2
                        else:
                            action = 3
                    else:
                        if obs[5] <= -1:
                            action = 0
                        else:
                            action = 1
                else:
                    action = 4
            strategy = actor_critic.get_strategy(ts([obs]), recurrent_hidden_states, ts([[0.0] if done else [1.0]])).detach().squeeze().numpy()
            action = np_random.choice(strategy.shape[0], p=strategy)
            # action = np.argmax(strategy)
            np.set_printoptions(precision=3, suppress=True)
            # print(self.name, strategy, strategy[action])
            self.main_conn.send(action)
