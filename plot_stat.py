import os
import joblib
from a2c_ppo_acktr.multi_agent.utils import plot_statistics


def main(save_dir):
    statistics = joblib.load(os.path.join(save_dir, "statistics.obj"))
    plot_statistics(statistics, "reward")
    # plot_statistics(statistics, "dist_penalty")
    # plot_statistics(statistics, "grad_norm")
    # plot_statistics(statistics, "value_loss")
    # plot_statistics(statistics, "action_loss")
    # plot_statistics(statistics, "dist_entropy")


if __name__ == "__main__":
    main("./sync-results/ppo-simple-key-overlap/tests/2021-02-01T14:49:19.162587")
