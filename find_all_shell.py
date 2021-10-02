import os
import json
import copy
import argparse
from pathlib import Path
from a2c_ppo_acktr.multi_agent.utils import mkdir, get_timestamp
import subprocess
import joblib
import random
from pprint import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-config")
    parser.add_argument("--default-likelihood", type=float)
    parser.add_argument("--project-dir")
    parser.add_argument("--current-stage", type=int)
    parser.add_argument("--agents", type=str, nargs="+")
    parser.add_argument("--keywords", type=str, nargs="+")

    args = parser.parse_args()

    current_stage = args.current_stage
    project_dir = args.project_dir

    if current_stage == 0:
        initial_config = json.load(open(args.initial_config))
    else:
        initial_config = json.load(open(os.path.join(project_dir, "stage-{}".format(current_stage - 1), "config.json")))

    env_name = initial_config["env_name"]
    if env_name == "half-cheetah":
        agents = ["0"]
        keywords = ["normal", "reversed", "front_upright", "back_upright"]
    elif env_name == "stag-hunt-gw":
        agents = ["0", "1"]
        keywords = ["gather_count"]
    elif env_name == "escalation-gw":
        agents = ["0", "1"]
        keywords = ["cnt_after_meeting"]
    elif env_name == "simple-more":
        agents = ["agent_0"]
        keywords = ["reach_cnt", "reach_steps", "total_reach"]
    elif env_name == "hopper" or env_name == "humanoid" or env_name == "walker2d" or \
            env_name == "ant" or env_name == "point":
        agents = ["0"]
        keywords = []
    else:
        agents = args.agents
        keywords = args.keywords

    initial_config["config"] = None
    # initial_config["log_level"] = "error"
    if not initial_config["use_reference"]:
        ref_config = {agent: [] for agent in agents}
        n_ref = 0
        likelihoods = []
    else:
        ref_config = json.load(open(initial_config["ref_config"]))
        n_ref = len(ref_config[agents[0]])
        likelihoods = initial_config["likelihood_threshold"]

    if current_stage > 0:
        last_save_dir = os.path.join(project_dir, "stage-{}".format(current_stage - 1))
        findings_file = os.path.join(project_dir, "findings.obj")

        if current_stage > 1:
            findings = joblib.load(findings_file)
            # print(findings)
        else:
            findings = []

        _findings = []
        for folder in sorted(Path(os.path.abspath(last_save_dir)).iterdir(), key=os.path.getmtime):
            _save_dir = os.path.join(last_save_dir, folder)

            try:
                play_statistics = joblib.load(os.path.join(_save_dir, "play_statistics.obj"))
                rewards = play_statistics["rewards"]
                _finding = {
                    "save_dir": _save_dir,
                    "rewards": rewards,
                    "statistics": dict()
                }

                for keyword in keywords:
                    _finding["statistics"][keyword] = play_statistics[keyword]

                _findings.append(_finding)
            except NotADirectoryError:
                pass

        for _finding in _findings:
            for i_agent in agents:
                likelihoods.append(args.default_likelihood)
                for j_agent in agents:
                    ref_config[i_agent].append({
                        "load_dir": _finding["save_dir"],
                        "load_step": None,
                        "num_refs": n_ref,
                        "load_agent": j_agent
                    })

        n_ref += len(_findings) * len(agents)

        findings.append(_findings)
        # print("Findings:", _findings)
        # for _finding in _findings:
        #     # print("save_dir\t{}".format(_finding["save_dir"]))
        #     # print("rewards\t{}".format(_finding["rewards"]))
        #     # print(_finding["statistics"])
        #     # print("")
        #     # print(json.dumps(_finding, indent=2))
        #     pprint(_finding)
        pprint(_findings, width=1, sort_dicts=False)

        joblib.dump(findings, findings_file)

    save_dir = os.path.join(project_dir, f"stage-{current_stage}")
    mkdir(save_dir)
    config = copy.deepcopy(initial_config)
    ref_config_path = os.path.join(save_dir, "ref_config.json")
    json.dump(ref_config, open(ref_config_path, "w"))
    if n_ref > 0:
        config["likelihood_threshold"] = likelihoods
        config["ref_config"] = ref_config_path
        config["use_reference"] = True
    else:
        config["use_reference"] = False

    config["save_dir"] = save_dir
    config_path = os.path.join(save_dir, "config.json")
    json.dump(config, open(config_path, "w"))
    # print(project_dir)


if __name__ == "__main__":
    main()
