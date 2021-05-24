import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams


def generate(alg_name, alg_data, num_iterations, mode, plot_data, delta=0.0):
    for _data in alg_data:
        for i, s in enumerate(_data[:num_iterations]):
            plot_data["iter"].append(i + 1)
            plot_data["num"].append(s + delta)
            plot_data["type"].append(alg_name)
            plot_data["mode"].append(mode)
        for i in range(len(_data), num_iterations):
            plot_data["iter"].append(i + 1)
            plot_data["num"].append(_data[-1] + delta)
            plot_data["type"].append(alg_name)
            plot_data["mode"].append(mode)


def simulate(alg_name, num_targets, p, num_iterations, num_tests, mode, data, delta=0.0):
    for i in range(num_tests):
        past = set()
        for j in range(num_iterations):
            current = np.random.choice(num_targets, p=p)
            past.add(current)
            data["iter"].append(j + 1)
            data["num"].append(len(past) + delta)
            data["type"].append(alg_name)
            data["mode"].append(mode)

    # generate("DIPG", normal_digp, data, num_iterations)

    # df = pd.DataFrame(data)
    # # plt.figure(figsize=(20, 20))
    # sns.catplot(x="iter", y="num", data=df, kind="point", hue="type", height=3, aspect=1.0)
    # plt.show()


def generate_rspo(num_targets, num_iterations, num_tests, mode, data, delta=0.0):
    for i in range(num_tests):
        for j in range(num_iterations):
            data["iter"].append(j + 1)
            data["num"].append(min(num_targets, j + 1) + delta)
            data["type"].append("RSPO")
            data["mode"].append(mode)


def plot_simple_more_4_balls_all():
    easy_dipg = [[1, 2, 3, 3, 4],
                 [1, 2, 3, 4],
                 [1, 2, 3, 4],
                 [1, 2, 3, 4],
                 [1, 2, 3, 4]]

    normal_digp = [[1, 2, 2, 2, 2, 2, 2],
                   [1, 2, 3, 3, 3, 3, 3],
                   [1, 2, 2, 3, 3, 4, 4],
                   [1, 1, 1, 2, 3, 3, 3],
                   [1, 2, 3, 3, 3, 4, 4]]

    normal_clip_only = [[1, 2, 3, 4]]

    data = {
        "iter": [],
        "num": [],
        "type": [],
        "mode": []
    }

    simulate("VPG", 4, np.ones(4) / 4, 7, 5, "easy", data)
    generate("DIPG", easy_dipg, 7, "easy", data)
    generate_rspo(4, 7, 5, "easy", data)

    simulate("VPG", 4, np.ones(4) / 4, 7, 5, "medium", data)
    generate("DIPG", normal_digp, 7, "medium", data)
    generate_rspo(4, 7, 5, "medium", data)

    simulate("VPG", 4, [1., 0., 0., 0.], 7, 5, "hard", data, delta=-0.05)
    simulate("DIPG", 4, [1., 0., 0., 0.], 7, 5, "hard", data, delta=0.05)
    generate_rspo(4, 7, 5, "hard", data)

    df = pd.DataFrame(data)
    # print(df)
    # g = sns.FacetGrid(df, col="mode", hue="type")
    # g.map(sns.catplot, "iter", "num")
    # g.add_legend()
    g = sns.catplot(data=df, x="iter", y="num", hue="type", col="mode", kind="point", height=3, legend=False)
    g.set_axis_labels("# iterations", "# distinct strategies found")
    g.set(yticks=[1, 2, 3, 4])
    # g.despine(left=True)
    # g.set_xlabels(["", "# iterations", ""])
    g.add_legend(title="")
    g.tight_layout()

    # plt.show()
    plt.savefig("./plots/4-balls.pdf")


HARD_REWARDS = [1., 1.1, 1.2, 1.3]


def calc_reward(cnt, rewards, total=100):
    return (np.array(cnt) * np.array(rewards) / total).sum()


def generate_from_cnt(alg_name, cnt_data, data):
    for _data in cnt_data:
        data["type"].append(alg_name)
        data["reward"].append(calc_reward(_data, HARD_REWARDS))


def generate_from_reward(alg_name, reward_data, num_iterations, data):
    np.random.shuffle(reward_data)
    n_data = len(reward_data)
    n_test = n_data // num_iterations
    for i in range(n_test):
        max_reward = 0.
        for reward in reward_data[num_iterations * i: num_iterations * (i + 1)]:
            max_reward = max(reward, max_reward)

        data["type"].append(alg_name)
        data["reward"].append(max_reward)


def plot_simple_more_4_balls_hard():
    rspo_data = [[3, 0, 1, 96],
                 [4, 3, 2, 90],
                 [3, 6, 1, 91],
                 [0, 3, 0, 96],
                 [4, 3, 3, 90]]
    vpg_data = [1.0070000000000001, 1.003, 1.008, 0.99, 1.005, 1.006, 1.011, 1.005, 1.009, 1.003, 1.005, 0.995, 1.0070000000000001, 0.9989999999999999, 1.006, 1.004, 1.005, 1.009, 1.008, 1.004]
    data = {
        "type": [],
        "reward": []
    }

    generate_from_reward("VPG", vpg_data, 4, data)
    generate_from_reward("DIPG", vpg_data, 4, data)
    generate_from_cnt("RSPO", rspo_data, data)

    df = pd.DataFrame(data)

    g = sns.catplot(data=df, x="type", y="reward", kind="bar", height=3)
    g.set(ylim=(0.8, 1.4))
    g.set_axis_labels("algorithm", "max reward")
    g.tight_layout()
    # plt.show()
    plt.savefig("./plots/4-balls-hard.pdf")


if __name__ == "__main__":
    # simulate(4, np.ones(4) / 4, 7, 5)
    # simulate(4, [1.0, 0.0, 0.0, 0.0], 7, 5)
    plot_simple_more_4_balls_hard()
