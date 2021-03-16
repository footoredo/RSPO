import os
import joblib
from a2c_ppo_acktr.multi_agent.utils import plot_statistics, plot_agent_statistics


def main(save_dir, max_iter=None):
    statistics = joblib.load(os.path.join(save_dir, "statistics.obj"))
    plot_statistics(statistics, "reward", max_iter=max_iter)
    plot_statistics(statistics, "dist_penalty")
    plot_statistics(statistics, "efficiency")
    plot_statistics(statistics, "grad_norm")
    plot_statistics(statistics, "value_loss")
    plot_statistics(statistics, "action_loss")
    plot_statistics(statistics, "dist_entropy")


def agent(save_dir):
    statistics = joblib.load(os.path.join(save_dir, "statistics.obj"))
    plot_agent_statistics(statistics, "reward")
    plot_agent_statistics(statistics, "dist_entropy")
    plot_agent_statistics(statistics, "efficiency")
    # plot_agent_statistics(statistics, "grad_norm")
    # plot_agent_statistics(statistics, "value_loss")
    # plot_agent_statistics(statistics, "likelihood")
    # plot_agent_statistics(statistics, "action_loss")


if __name__ == "__main__":
    # main("./sync-results/stag-hunt-gw/ultimate/2021-03-15T17:16:37.080772#4e29")
    agent("./sync-results/stag-hunt-gw/ultimate/2021-03-16T12:27:49.871832#3afa/0")
