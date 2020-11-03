import gym
import numpy as np
import torch

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
import multiprocessing as mp

from argparse import Namespace

from ..model import AttentionBase

from .utils import *


def get_agent(args, obs_space, input_structure, act_space, use_attention, save_dir):
    if use_attention:
        actor_critic = Policy(
            obs_space.shape,
            act_space,
            AttentionBase,
            base_kwargs={'recurrent': args.recurrent_policy, 'input_structure': input_structure})
    else:
        actor_critic = Policy(
            obs_space.shape,
            act_space,
            base_kwargs={'recurrent': args.recurrent_policy})

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            clip_grad_norm=not args.no_grad_norm_clip)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
    elif args.algo == 'loaded-dice':
        agent = algo.LoadedDiCE(
            actor_critic,
            args.dice_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            args.dice_lambda,
            args.episode_steps,
            args.dice_task,
            lr=args.lr,
            eps=args.eps,
            save_dir=save_dir
        )
    else:
        raise ValueError("algo {} not supported".format(args.algo))
    return actor_critic, agent


def true_func(n_iter):
    return True


class Agent(mp.Process):
    def __init__(self, agent_id, agent_name, thread_limit, logger, args: Namespace, obs_space, input_structure,
                 act_space, main_conn, obs_shm, buffer_start, buffer_end, act_shm, obs_locks, act_locks,
                 use_attention=False, save_dir=None, train=true_func):
        super(Agent, self).__init__()
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.thread_limit = thread_limit
        self.logger = logger

        self.args = args
        self.seed = args.seed + self.agent_id if args.seed is not None else None
        self.num_steps = args.num_steps
        self.num_envs = args.num_processes
        self.num_agents = args.num_agents
        self.batch_size = self.num_steps * self.num_envs
        self.num_updates = args.num_env_steps // args.num_steps // args.num_processes
        self.save_interval = args.save_interval
        self.save_dir = mkdir2(save_dir or args.save_dir, agent_name)

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
        # self.log("release obs locks")
        # release_all_locks(self.obs_locks)
        return ts(obs), ts(reward), ts(done)

    def put_act(self, i_env, act):
        self.act_shm.buf[i_env * self.num_agents + self.agent_id] = act

    def run(self):
        if self.thread_limit is not None:
            torch.set_num_threads(self.thread_limit)
        args = self.args
        use_dice = args.algo == "loaded-dice"
        # dice_lambda = args.dice_lambda if use_dice else None
        torch.manual_seed(self.seed)
        # self.log(self.obs_space)
        actor_critic, agent = get_agent(self.args, self.obs_space, self.input_structure, self.act_space,
                                        self.use_attention, self.save_dir)
        # print(self.num_steps)
        if args.load_dir is not None and args.load_step is not None and args.load:
            self.log("Loading model from {}".format(args.load_dir))
            actor_critic.load_state_dict(torch.load(os.path.join(args.load_dir, self.agent_name,
                                                                 "update-{}".format(args.load_step), "model.obj")))
            self.log("Done.")
            # while True:

        rollouts = RolloutStorage(self.num_steps, self.num_envs,
                                  self.obs_space.shape, self.act_space,
                                  actor_critic.recurrent_hidden_state_size)

        # acquire_all_locks(self.act_locks)

        self.main_conn.recv()

        obs, _, _ = self.get_obs()
        # self.log("step {} - received {}".format(0, obs))
        # self.log(obs.shape)
        rollouts.obs[0].copy_(obs)
        statistics = dict(reward=[], grad_norm=[], value_loss=[], action_loss=[], dist_entropy=[])
        sum_reward = 0.
        # last_update_iter = -1
        update_counter = 0

        for it in range(self.num_updates):
            # self.log("iter: {}".format(it))
            for step in range(self.num_steps):
                # self.log(step)
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                    # self.log("step {} - act {}".format(it * self.num_steps + step, action))

                action = action.data
                for i_env in range(self.num_envs):
                    self.put_act(i_env, action[i_env])
                # self.log("release act locks")
                release_all_locks(self.act_locks)

                obs, reward, done = self.get_obs()
                # print(reward)
                sum_reward += reward.detach().numpy().sum()
                # self.log("step {} - received {}, {}, {}".format(it * self.num_steps + step + 1, obs, reward, done))
                # self.log("acquire act locks")
                # acquire_all_locks(self.act_locks)

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.zeros_like(masks)
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            if self.train(it):
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

            statistics["reward"].append((it, sum_reward / self.num_steps / self.num_envs * args.episode_steps))
            # last_update_iter = it
            sum_reward = 0.

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
