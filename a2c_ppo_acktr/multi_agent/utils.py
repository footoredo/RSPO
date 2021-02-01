import numpy as np
import torch
import os
import pathlib

from a2c_ppo_acktr.storage import RolloutStorage

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering


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


def jointplot(data1, data2, save_path=None):
    data1 = to_numpy(data1).reshape(-1)
    data2 = to_numpy(data2).reshape(-1)
    df = pd.DataFrame(dict(x=data1, y=data2))
    sns.jointplot(data=df, x="x", y="y", kind="kde", xlim=(0, 350), ylim=(0.0, 6.0))
    # sns.jointplot(data=df, x="x", y="y", kind="kde")
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
        load_path = os.path.join(load_dir, agent_name, "update-{}".format(load_step), "model.obj")
    else:
        load_path = os.path.join(load_dir, agent_name, "model.obj")
    # print(load_path)
    actor_critic.load_state_dict(torch.load(load_path))


def get_agent(args, obs_space, input_structure, act_space, save_dir, n_ref=0):
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
        actor_critic = Policy(
            obs_space.shape,
            act_space,
            base_kwargs={'recurrent': args.recurrent_policy,
                         'critic_dim': n_ref + 1})

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
            clip_grad_norm=not args.no_grad_norm_clip,
            task=args.task,
            direction=args.direction,
            save_dir=save_dir,
            args=args
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


def extract_trajectory(rollout: RolloutStorage):
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


def plot_statistics(statistics, keyword):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    iters = []
    values = []
    names = []

    agents = list(statistics.keys())

    for agent in agents:
        s = statistics[agent][keyword]
        for it, v in s:
            iters.append(it)
            values.append(v)
            names.append(agent)

    df = pd.DataFrame(dict(iteration=iters, value=values, agent=names))
    fig, ax = plt.subplots()
    sns.lineplot(x="iteration", y="value", hue="agent", data=df, ax=ax)
    ax.set_title(keyword)
    plt.tight_layout()
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
        assert x.shape == self._M.shape
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