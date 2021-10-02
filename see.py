import os
from pathlib import Path
import joblib
import json
import numpy as np
import sys
from a2c_ppo_acktr.multi_agent.utils import show_play_statistics

# DIR = "./sync-results/stag-hunt-gw/no-fruit-new-9"
# DIR = "./sync-results/stag-hunt-gw/ultimate-new-6/"
env = sys.argv[1]
DIR = f"./sync-results/{env}/{sys.argv[2]}/"
# DIR = "./sync-results/escalation-gw/1-d63b/"


total_gather_count = np.zeros((5, 5), dtype=int)

simple_more_hard_rewards = []

escalation_coop_cnt = np.zeros(50, dtype=int)
escalation_coop_lengths = []


def show(root_dir, i_copy=None):
    path = Path(os.path.abspath(root_dir))
    if i_copy is not None:
        path = path / f'copy-{i_copy}'
    full_path = path / "play_statistics.obj"
    statistics = joblib.load(full_path)
    # gather_count = statistics["gather_count"]
    config_path = path / "config.json"
    config = json.load(open(config_path))
    if i_copy is not None:
        if i_copy == 0:
            print(f'num_copies: {config["num_copies"]}, coef: {config["dvd_coef"]}')
        print(f'\n------ copy-{i_copy} ------')
    print(f'seed: {config["seed"]}, num_steps: {config["num_steps"]}, train_in_turn: {config["train_in_turn"]}')
    print(f'env_config: {config["env_config"]}')
    print(f'rewards: {statistics["rewards"]}')
    returns = show_play_statistics(config["env_name"], statistics, config["episode_steps"])
    if config["env_name"] == "escalation-gw":
        escalation_coop_cnt[returns["max_coop"]] += 1
        escalation_coop_lengths.append(returns["max_coop"])

    if config["env_name"] == "simple-more" and config["env_config"].endswith("-hard.json"):
        cnt = statistics["reach_cnt"]
        hard_rewards = [1.0, 1.1, 1.2, 1.3]
        reward = 0.0
        for i in range(4):
            reward += cnt[i] * hard_rewards[i]
        reward /= config["num_games_after_training"]
        simple_more_hard_rewards.append(reward)


for folder in sorted(Path(os.path.abspath(DIR)).iterdir(), key=os.path.getmtime):
    try:
        print(f'\n\n{folder}')
        n_copies = 0
        while True:
            copy_folder = os.path.join(folder, "copy-{}".format(n_copies))
            if not os.path.isdir(os.path.join(DIR, copy_folder)):
                break
            n_copies += 1
        if n_copies == 0:
            show(folder)
        else:
            for i in range(n_copies):
                show(folder, i)
    except FileNotFoundError:
        pass
    except NotADirectoryError:
        pass

if env == "stag-hunt-gw":
    print("Total")
    print(total_gather_count)

if env == "escalation-gw":
    print(escalation_coop_cnt)
    print(escalation_coop_lengths)

if len(simple_more_hard_rewards) > 0:
    print(simple_more_hard_rewards)

# json.dump(ref_config, open("ref_config.json", "w"), indent=4)

