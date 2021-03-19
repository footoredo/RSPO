import os
import joblib
from a2c_ppo_acktr.multi_agent.utils import plot_statistics, plot_agent_statistics, get_remote


def main(save_path, max_iter=None):
    statistics = joblib.load(save_path)
    plot_statistics(statistics, "reward", max_iter=max_iter)
    plot_statistics(statistics, "efficiency", max_iter=max_iter)
    # plot_statistics(statistics, "grad_norm")
    # plot_statistics(statistics, "value_loss")
    # plot_statistics(statistics, "action_loss")
    # plot_statistics(statistics, "dist_entropy")
    # plot_statistics(statistics, "dist_penalty")


def agent(save_path):
    statistics = joblib.load(save_path)
    plot_agent_statistics(statistics, "reward")
    # plot_agent_statistics(statistics, "dist_entropy")
    plot_agent_statistics(statistics, "efficiency")
    # plot_agent_statistics(statistics, "grad_norm")
    # plot_agent_statistics(statistics, "value_loss")
    # plot_agent_statistics(statistics, "likelihood")
    # plot_agent_statistics(statistics, "action_loss")


if __name__ == "__main__":
    remote = True
    save_dir = "./results/stag-hunt-gw/ultimate/2021-03-19T12:11:45.123159#f992/0"
    # remote = False
    # save_dir = "./sync-results/stag-hunt-gw/ultimate/2021-03-18T20:19:23.060669#125b/0"
    path = os.path.join(save_dir, "statistics.obj")
    if remote:
        path = get_remote(path)

    # main(path)
    agent(path)
