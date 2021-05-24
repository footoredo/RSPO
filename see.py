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

# ref_config = dict()
# ref_config["0"] = []
# ref_config["1"] = []
#
# def new_entry(load_dir, num_refs, load_agent):
#     entry = {
#             "load_dir": load_dir,
#             "load_step": None,
#             "num_refs": num_refs,
#             "load_agent": load_agent
#         }
#     return entry
#
# def add_entry(load_dir, num_refs):
#     entry_0 = new_entry(load_dir, num_refs, 0)
#     entry_1 = new_entry(load_dir, num_refs, 1)
#     ref_config["0"].append(entry_0)
#     ref_config["0"].append(entry_1)
#     ref_config["1"].append(entry_0)
#     ref_config["1"].append(entry_1)
#
#
# add_entry("./sync-results/stag-hunt-gw/first/2021-02-26T16:24:13.694554", 0)
# add_entry("./sync-results/stag-hunt-gw/first/2021-02-26T17:27:38.373291", 0)
# add_entry("./sync-results/stag-hunt-gw/first/2021-02-26T18:31:03.140068", 0)

total_gather_count = np.zeros((5, 5), dtype=int)

simple_more_hard_rewards = []


for folder in sorted(Path(os.path.abspath(DIR)).iterdir(), key=os.path.getmtime):
    try:
        # full_path = os.path.join(DIR, folder, "gather_count.obj")
        # gather_count = joblib.load(full_path)
        full_path = os.path.join(DIR, folder, "play_statistics.obj")
        statistics = joblib.load(full_path)
        # gather_count = statistics["gather_count"]
        config_path = os.path.join(DIR, folder, "config.json")
        config = json.load(open(config_path))
        print(folder)
        print(config["seed"], config["num_steps"], config["train_in_turn"])
        print(f'rewards: {statistics["rewards"]}')
        show_play_statistics(config["env_name"], statistics, config["episode_steps"])

        if config["env_name"] == "simple-more" and config["env_config"].endswith("-hard.json"):
            cnt = statistics["reach_cnt"]
            hard_rewards = [1.0, 1.1, 1.2, 1.3]
            reward = 0.0
            for i in range(4):
                reward += cnt[i] * hard_rewards[i]
            reward /= config["num_games_after_training"]
            simple_more_hard_rewards.append(reward)

        # if env == "escalation-gw":
        #     # print(statistics["cnt_after_meeting"][:20])
        #     max_step = 0
        #     for i in reversed(range(config["episode_steps"])):
        #         if statistics["cnt_after_meeting"][i] > 0:
        #             max_step = i + 1
        #             break
        #     np.set_printoptions(precision=3, suppress=True)
        #     for i in range(max_step):
        #         print(i + 1, "\t", statistics["cnt_after_meeting"][i], "\t",
        #               statistics["actions_after_meeting"][i][0], statistics["actions_after_meeting"][i][1])
        # elif env == "stag-hunt-gw":
        #     gather_count = statistics["gather_count"]
        #     print(gather_count)
        #     total_gather_count += np.array(gather_count)
        # elif env == "simple-key":
        #     print("reach_1:", statistics["reach_1"])
        #     print("reach_key:", statistics["reach_key"])
        #     print("reach_2:", statistics["reach_2"])
        # elif env == "prisoners-dilemma":
        #     np.set_printoptions(precision=3, suppress=True)
        #     try:
        #         for i in range(2):
        #             print("player {}:".format(i))
        #             print("initial_strategy:", statistics[f"initial_strategy_{i}"])
        #             for j in range(2):
        #                 for k in range(2):
        #                     print("({}, {}) strat:".format(j, k), statistics[f"strategy_matrix_{i}"][j, k],
        #                           "cnt:", statistics[f"cnt_matrix_{i}"][j, k])
        #     except KeyError:
        #         pass
        # add_entry(os.path.join(DIR, folder), 6)
    except FileNotFoundError:
        pass
    except NotADirectoryError:
        pass

if env == "stag-hunt-gw":
    print("Total")
    print(total_gather_count)

if len(simple_more_hard_rewards) > 0:
    print(simple_more_hard_rewards)

# json.dump(ref_config, open("ref_config.json", "w"), indent=4)

