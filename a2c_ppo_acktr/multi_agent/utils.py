import numpy as np
import torch
import os
import pathlib
import json
import gym

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering


# CONFIDENTIAL = json.load(open("./confidential.json"))
CONFIDENTIAL = None
SSH = None


# plt.rcParams["font.family"] = "monospace"
# plt.rcParams["lines.linewidth"] = 2.0
# plt.rcParams["axes.edgecolor"] = "#6b6b76"
# # plt.rcParams["axes.labelcolor"] = "#6b6b76"
# plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.spines.right"] = False
# plt.rcParams["grid.alpha"] = 0.5
# plt.rcParams["grid.alpha"] = 0.5


def show_play_statistics(env_name, statistics, episode_steps=None):
    try:
        np.set_printoptions(precision=3, suppress=True)
        keys = []
        if "episode_steps" in statistics:
            episode_steps = statistics["episode_steps"]
        if env_name == "stag-hunt-gw":
            keys = ["gather_count"]
        elif env_name == "escalation-gw":
            max_step = 0
            for i in reversed(range(episode_steps)):
                if statistics["cnt_after_meeting"][i] > 0:
                    max_step = i + 1
                    break
            for i in range(max_step):
                print(i + 1, "\t", statistics["cnt_after_meeting"][i], "\t",
                      statistics["actions_after_meeting"][i][0],
                      statistics["actions_after_meeting"][i][1])
        elif env_name == "simple-key":
            keys = ["reach_1", "reach_key", "reach_2"]
        elif env_name == "prisoners-dilemma":
            for i in range(2):
                print("player {}:".format(i))
                print("initial_strategy:", statistics[f"initial_strategy_{i}"])
                for j in range(2):
                    for k in range(2):
                        print("({}, {}) strat:".format(j, k), statistics[f"strategy_matrix_{i}"][j, k],
                              "cnt:", statistics[f"cnt_matrix_{i}"][j, k])
        elif env_name == "half-cheetah":
            keys = ["normal", "reversed", "front_upright", "back_upright"]
        else:
            raise NotImplementedError

        for key in keys:
            print(key, statistics[key])

    except KeyError:
        pass


def get_action_size(action_space, in_buffer=False):
    if action_space.__class__.__name__ == "Discrete":
        return 1 if not in_buffer else action_space.n
    elif action_space.__class__.__name__ == "Box":
        return action_space.shape[0]
    elif isinstance(action_space, gym.spaces.Tuple):
        size = 0
        for subspace in action_space:
            size += get_action_size(subspace, in_buffer)
        return size
    else:
        raise NotImplementedError


def get_action_recover_fn(action_space):
    if action_space.__class__.__name__ == "Discrete":
        return lambda a: int(a[0])
    elif action_space.__class__.__name__ == "Box":
        return lambda a: a
    elif isinstance(action_space, gym.spaces.Tuple):
        return lambda a: a
    else:
        raise NotImplementedError


def smooth_data(data, gamma):
    last = data[0]
    smoothed_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        smoothed_data[i] = last * gamma + data[i] * (1 - gamma)
        # if i == 0:
        #     print(smoothed_data[i], data[i])
        last = smoothed_data[i]
    # from scipy import signal
    # smoothed_data = signal.savgol_filter(data, **args)
    return smoothed_data


def get_ssh():
    from paramiko import SSHClient

    global SSH
    if SSH is not None:
        return SSH

    ssh_config = CONFIDENTIAL["ssh"]

    which = "local"

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(ssh_config["hostname"][which], username=ssh_config["username"],
                key_filename=ssh_config["key_filename"], port=ssh_config["port"][which])

    SSH = ssh
    return ssh


def get_remote_file(filename):
    from scp import SCPClient, SCPException

    ssh = get_ssh()

    ssh_config = CONFIDENTIAL["ssh"]
    home_dir = ssh_config["remote_home_dir"]
    tmp_dir = "/tmp/get_remote"

    local_path = os.path.join(tmp_dir, get_timestamp())
    mkdir(tmp_dir)

    with SCPClient(ssh.get_transport()) as scp:
        try:
            scp.get(os.path.join(home_dir, filename), local_path)
        except SCPException:
            return None

    return local_path


def remote_listdir(path):
    import time

    ssh = get_ssh()

    ssh_config = CONFIDENTIAL["ssh"]
    home_dir = ssh_config["remote_home_dir"]
    remote_path = os.path.join(home_dir, path)

    stdin, stdout, stderr = ssh.exec_command(f"ls {remote_path}")
    time.sleep(1)  # fixed a bug in paramiko

    folders = []
    for line in stdout:
        folders.append(line.strip("\n"))
    return folders


def get_timestamp():
    from datetime import datetime
    import binascii
    return "{}#{}".format(datetime.now().isoformat(), binascii.hexlify(os.urandom(2)).decode())


def make_env(env_name, steps, env_config=None):
    if steps is None:
        steps = 32
    env_config_file = env_config or f"./env-configs/{env_name}-default.json"
    env_config = json.load(open(env_config_file))
    if env_name == "simple-tag":
        from pettingzoo.mpe import simple_tag_v1
        return simple_tag_v1.parallel_env(max_frames=steps, **env_config)
    elif env_name == "simple":
        from pettingzoo.mpe import simple_v1
        return simple_v1.parallel_env(max_frames=steps, **env_config)
    elif env_name == "simple-key":
        from pettingzoo.mpe import simple_key_v1
        return simple_key_v1.parallel_env(max_frames=steps, **env_config)
    elif env_name == "simple-more":
        from pettingzoo.mpe import simple_more_v1
        return simple_more_v1.parallel_env(max_frames=steps, **env_config)
    elif env_name == "simple-more-3":
        from pettingzoo.mpe import simple_more_v1
        return simple_more_v1.parallel_env(max_frames=steps, **env_config)
    elif env_name == 'stag-hunt-gw':
        from pettingzoo.mappo_ssd import stag_hunt_gw_v1
        return stag_hunt_gw_v1.parallel_env(max_frames=steps, **env_config)
    elif env_name == 'escalation-gw':
        from pettingzoo.mappo_ssd import escalation_gw_v1
        return escalation_gw_v1.parallel_env(max_frames=steps, **env_config)
    elif env_name == 'prisoners-dilemma':
        from pettingzoo.matrix_game import prisoners_dilemma_v1
        return prisoners_dilemma_v1.parallel_env(max_frames=steps, **env_config)
    elif env_name == 'half-cheetah':
        from pettingzoo.mujoco import half_cheetah_v3
        return half_cheetah_v3.parallel_env(max_frames=steps, **env_config)
    elif env_name == 'agar':
        from pettingzoo.agar.Agar_Env import AgarEnv
        return AgarEnv(max_frames=steps, **env_config)
    else:
        raise NotImplementedError(env_name)


def reseed(seed, phrase):
    import hashlib
    m = hashlib.sha1()
    m.update(seed.to_bytes(2, "big"))
    m.update(phrase.encode())
    return int.from_bytes(m.digest()[:2], "big")


def save_gif(images, save_path, fps):
    import imageio
    imageio.mimwrite(save_path, images, fps=fps)


def to_numpy(data):
    if type(data) == list:
        data = np.array(data)
    elif type(data) == torch.Tensor:
        data = data.numpy()
    elif type(data) == np.ndarray:
        return data
    else:
        raise NotImplementedError("Unrecognizable data type {}".format(type(data)))


def jointplot(data1, data2, save_path=None, title="likelihood-return"):
    data1 = to_numpy(data1).reshape(-1)
    data2 = to_numpy(data2).reshape(-1)
    df = pd.DataFrame(dict(x=data1, y=data2))
    sns.jointplot(data=df, x="x", y="y", kind="kde", xlim=(0, 500), ylim=(0.0, 10.))
    # sns.jointplot(data=df, x="x", y="y", kind="kde")
    plt.title(title)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def displot(data):
    data = to_numpy(data)
    df = pd.DataFrame(dict(x=data.reshape(-1)))
    sns.displot(df, x="x")
    plt.show()


def load_actor_critic(actor_critic, load_dir, agent_name, load_step=None):
    if load_step is not None:
        load_path = os.path.join(load_dir, str(agent_name), "update-{}".format(load_step), "model.obj")
    else:
        load_path = os.path.join(load_dir, str(agent_name), "model.obj")
    # print(load_path)
    loaded = torch.load(load_path)
    if type(loaded) is tuple and len(loaded) == 2:
        state_dict, obs_rms = loaded
    else:
        state_dict = loaded
        obs_rms = None
    actor_critic.load_state_dict(state_dict)
    return obs_rms


def get_agent(agent_name, args, obs_space, input_structure, act_space, save_dir, n_ref=0, is_ref=False):
    from a2c_ppo_acktr import algo
    from a2c_ppo_acktr.model import Policy, AttentionBase, LinearBase

    if args.use_attention:
        actor_critic = Policy(
            obs_space.shape,
            act_space,
            AttentionBase,
            base_kwargs={'recurrent': args.recurrent_policy, 'input_structure': input_structure})
    elif args.use_linear:
        actor_critic = Policy(
            obs_space.shape,
            act_space,
            LinearBase)
    else:
        # if not is_ref:
        #     print("A")
        actor_critic = Policy(
            obs_space.shape,
            act_space,
            base_kwargs={'recurrent': args.recurrent_policy,
                         'critic_dim': n_ref * 2 + 1,
                         'is_ref': is_ref,
                         'predict_reward': args.use_reward_predictor})
        # if not is_ref:
        #     print("B")

    # if not is_ref:
    #     print("!!@!@")

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
            agent_name,
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            clip_grad_norm=not args.no_grad_norm_clip,
            task=args.task,
            direction=args.direction,
            save_dir=save_dir,
            args=args,
            is_ref=is_ref
        )
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
    elif args.algo == 'hessian':
        agent = algo.Hessian(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            clip_grad_norm=not args.no_grad_norm_clip,
            task=args.task,
            direction=args.direction,
            args=args
        )
    else:
        raise ValueError("algo {} not supported".format(args.algo))
    return actor_critic, agent


def calc_sim(direc1, direc2):
    sim = direc1 @ direc2 / np.linalg.norm(direc1) / np.linalg.norm(direc2)
    return sim


def extract_trajectory(rollout):
    length = rollout.actions.size()[0]
    width = rollout.actions.size()[1]

    trajectories = []
    cur_traj = []
    for i in range(width):
        for j in range(length):
            obs = rollout.obs[j, i]
            act = rollout.actions[j, i]
            rew = rollout.rewards[j, i]
            cur_traj.append((obs, act, rew))
            if rollout.masks[j, i] < 0.5:
                trajectories.append(cur_traj)
                cur_traj = []
    return trajectories


def barred_argmax(vec, threshold=1e-2):
    ret = []
    for i in range(vec.size()[0]):
        top2 = torch.topk(vec[i], 2)
        if top2.values[0] > top2.values[1] + threshold:
            ret.append(top2.indices[0])
        else:
            ret.append(-1)
    return torch.tensor(ret)


def cluster(data, k):
    X = np.array(data)
    clusters = SpectralClustering(n_clusters=k).fit(X)
    return clusters


def tsne(v, h=None, s=None, pdf=False, **kwargs):
    print(kwargs)
    v_embedded = TSNE(n_components=2, **kwargs).fit_transform(v)
    xs = v_embedded[:, 0]
    ys = v_embedded[:, 1]
    hs = h[:] if h is not None else [np.linalg.norm(_) for _ in v]
    ss = s[:] if s is not None else [1. for _ in v]
    df = pd.DataFrame(dict(x=xs, y=ys, h=hs, s=ss))
    df = df.sort_values(by="h")
    # rel = sns.relplot(x="x", y="y", hue="h", size="s", sizes=(15, 200), palette="ch:r=-1.,l=1.", data=df)
    rel = sns.relplot(x="x", y="y", hue="h", size="s", sizes=(15, 200), data=df)
#     rel.fig.axes[0].scatter([v_embedded[-1, 0]], [v_embedded[-1, 0]], facecolors='none', edgecolors='r', s=80)
    if pdf:
        plt.savefig("tsne.pdf")
    else:
        plt.show()


def get_hessian(net, obj):
    g = flat_view(ggrad(net, obj), net)
    h = [flat_view(ggrad(net, _g), net) for _g in g]
    return torch.stack(h, dim=0)


def ggrad(net, obj):
    return torch.autograd.grad(obj, net.parameters(), create_graph=True, allow_unused=True)


def net_add(net, vec):
    s = 0
    for p in net.parameters():
        # print(p)
        l = p.view(-1).size()[0]
        p.data += vec[s:s + l].view(p.size())
        s += l
        # print(vec[s:s + l].view(p.size()))
        # print(p)


def flat_view(grads, net=None):
    flat_gs = []
    sizes = []
    if net is not None:
        for p in net.parameters():
            sizes.append(p.view(-1).size())
    # print(grads, net)
    for i, g in enumerate(grads):
        if g is not None:
            flat_gs.append(g.view(-1))
        elif net is not None:
            flat_gs.append(torch.zeros(sizes[i]))
    return torch.cat(flat_gs, dim=0)


def plot_agent_statistics(statistics, keyword, ref_indices=None):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    if keyword != "likelihood":
        iters = []
        values = []

        for it, v in statistics[keyword]:
            iters.append(it)
            values.append(v)

        df = pd.DataFrame(dict(iteration=iters, value=values))
        fig, ax = plt.subplots()
        sns.lineplot(x="iteration", y="value", data=df, ax=ax)
        ax.set_title(keyword)
        plt.tight_layout()
        plt.show()
    else:
        iters = []
        values = []
        refs = []
        for it, v in statistics[keyword]:
            # print(v.shape)
            if ref_indices is None:
                ref_indices = range(v.shape[0])
            for i in ref_indices:
                iters.append(it)
                values.append(v[i])
                refs.append("ref-{}".format(i))

        df = pd.DataFrame(dict(iteration=iters, value=values, ref=refs))
        fig, ax = plt.subplots()
        sns.lineplot(x="iteration", y="value", hue="ref", data=df, ax=ax)
        ax.set_title(keyword)
        # plt.tight_layout()
        plt.show()


def _plot_statistics(statistics, keyword, max_iter=None, smooth=None):
    iters = []
    values = []
    names = []

    agents = list(statistics.keys())

    for agent in agents:
        s = statistics[agent][keyword]
        _v = []
        for it, v in s:
            if max_iter is None or it <= max_iter:
                iters.append(it)
                _v.append(v)
                names.append(agent)

        if smooth is not None:
            _v = smooth_data(np.array(_v), smooth).tolist()
        values += _v

    return iters, values, names


def plot_statistics(statistics, keyword, max_iter=None, smooth=None, xscale="linear", yscale="linear"):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    if type(statistics) == list:
        iters = []
        values = []
        names = []
        for s in statistics:
            _iters, _values, _names = _plot_statistics(s, keyword, max_iter, smooth)
            iters += _iters
            values += _values
            names += _names
    else:
        iters, values, names = _plot_statistics(statistics, keyword, max_iter, smooth)

    df = pd.DataFrame(dict(iteration=iters, value=values, agent=names))
    fig, ax = plt.subplots()
    # print(df["value"])
    sns.lineplot(x="iteration", y="value", hue="agent", data=df, ax=ax)
    ax.set(xscale=xscale, yscale=yscale)
    ax.set_title(keyword)
    # plt.tight_layout()
    plt.show()


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def mkdir2(path1, path2):
    return mkdir(os.path.join(path1, path2))


def ts(ar):
    return torch.from_numpy(np.array(ar, dtype=np.float32)).float()


def acquire_all_locks(locks):
    for lock in locks:
        lock.wait()
        lock.clear()


def release_all_locks(locks):
    for lock in locks:
        lock.set()


class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass


class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape, (x.shape, self._M.shape)
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class RewardFilter:
    """
    "Incorrect" reward normalization [copied from OAI code]
    Incorrect in the sense that we
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean
    """

    def __init__(self, prev_filter, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()