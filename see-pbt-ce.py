import os
from pathlib import Path
import joblib
import json
import numpy as np
import sys
from pprint import pprint

root_dir = Path(os.path.abspath(sys.argv[1]))
keyword = sys.argv[2] if len(sys.argv) > 2 else None

for folder in sorted(root_dir.iterdir(), key=os.path.getmtime):
    try:
        config = json.load(open(root_dir / "copy-0" / "config.json"))
        env_name = config["env_name"]
        rewards = []
        print(folder)
        print(f'coef: {config["dvd_coef"]}, num_copies: {config["num_copies"]}')

        for i in range(config["num_copies"]):
            copy_folder = root_dir / f"copy-{i}"
            print("---- copy {} ----".format(i))

        for i, _findings in enumerate(findings):
            try:
                if env_name == "stag-hunt-gw":
                    try:
                        rewards.append(_findings[0]["rewards"].sum())
                    except TypeError:
                        pass
                if keyword is None:
                    print("Stage {}".format(i))
                    pprint(_findings, indent=1, sort_dicts=False)
                else:
                    print("Stage {}: {}".format(i, [_finding["statistics"][keyword] for _finding in _findings]))
            except KeyError:
                pass
            except IndexError:
                pass
        config_keywords = ["exploration_reward_alpha", "prediction_reward_alpha", "use_dynamic_prediction_alpha", "auto_threshold"]
        for k in config_keywords:
            if k in config.keys():
                print(k, config[k])
        if env_name == "stag-hunt-gw":
            print(rewards)
        # if len(findings) == 2:
        #     for i, _findings in enumerate(findings[0]):
        #         print("stage {}:".format(i), _findings)
        #     print(findings[1])
    except FileNotFoundError:
        pass
    except NotADirectoryError:
        pass
# json.dump(ref_config, open("ref_config.json", "w"), indent=4)

