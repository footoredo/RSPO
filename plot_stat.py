import os
import joblib
from a2c_ppo_acktr.multi_agent.utils import plot_statistics, plot_agent_statistics, get_remote_file, remote_listdir


# SMOOTH_ARGS = {
#     "window_length": 53,
#     "polyorder": 3,
# }


def plot_full(statistics, max_iter=None):
    plot_statistics(statistics, "reward", max_iter=max_iter)
    plot_statistics(statistics, "reward_prediction_loss", max_iter=max_iter)
    # plot_statistics(statistics, "efficiency", max_iter=max_iter)
    plot_statistics(statistics, "grad_norm", max_iter=max_iter)
    plot_statistics(statistics, "value_loss", max_iter=max_iter)
    # plot_statistics(statistics, "action_loss", max_iter=max_iter)
    plot_statistics(statistics, "dist_entropy", max_iter=max_iter)
    # plot_statistics(statistics, "dist_penalty", max_iter=max_iter)


def plot_agent(statistics):
    plot_agent_statistics(statistics, "reward")
    # plot_agent_statistics(statistics, "dist_entropy")
    plot_agent_statistics(statistics, "efficiency")
    # plot_agent_statistics(statistics, "grad_norm")
    # plot_agent_statistics(statistics, "value_loss")
    plot_agent_statistics(statistics, "likelihood")
    # plot_agent_statistics(statistics, "action_loss")


def get_statistics(path, remote):
    path = os.path.join(path, "statistics.obj")
    if remote:
        path = get_remote_file(path)
    if path is not None:
        return joblib.load(path)
    else:
        return None
    

def plot_single(path, agent, remote, max_iter=None):
    if agent is not None:
        path = os.path.join(path, str(agent))
    statistics = get_statistics(path, remote)
    if agent is not None:
        plot_agent(statistics)
    else:
        plot_full(statistics, max_iter=max_iter)
        

def plot_all(path, agent, remote):
    assert agent is None
    statistics = []
    if not remote:
        folders = os.listdir(path)
    else:
        folders = remote_listdir(path)
    # print(folders)
    # return
    for folder in folders:
        _path = os.path.join(path, folder)
        if agent is not None:
            _path = os.path.join(_path, str(agent))
        _statistics = get_statistics(_path, remote)
        if _statistics is not None:
            statistics.append(_statistics)
    if agent is not None:
        plot_agent(statistics)
    else:
        plot_full(statistics)


if __name__ == "__main__":
    plot_single("./sync-results/half-cheetah/0-step-action-mask-256-1000/2021-05-13T14:08:03.131780#7eb1",
                None, True, max_iter=None)
    # plot_all("./sync-results/escalation-gw/1-prediction-new-2", None, True)
