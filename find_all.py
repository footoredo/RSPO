import os
import json
import copy
import argparse
from a2c_ppo_acktr.multi_agent.utils import mkdir, get_timestamp
import subprocess
import joblib
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-config")
    parser.add_argument("--default-likelihood", type=float)
    parser.add_argument("--project-dir")
    parser.add_argument("--runs-per-stage", default=1, type=int)
    parser.add_argument("--max-stages", type=int)
    parser.add_argument("--agents", type=str, nargs="+")
    parser.add_argument("--keywords", type=str, nargs="+")

    args = parser.parse_args()

    project_dir = os.path.join(args.project_dir, get_timestamp())

    initial_config = json.load(open(args.initial_config))
    initial_config["config"] = None
    # initial_config["log_level"] = "error"
    agents = args.agents
    if not initial_config["use_reference"]:
        ref_config = {agent: [] for agent in agents}
        n_ref = 0
        likelihoods = []
    else:
        ref_config = json.load(open(initial_config["ref_config"]))
        n_ref = len(ref_config["0"])
        likelihoods = initial_config["likelihood_threshold"]

    n_stage = 0

    findings = []
    findings_file = os.path.join(project_dir, "findings.obj")
    while True:
        print("Stage:", n_stage)
        save_dir = os.path.join(project_dir, f"stage-{n_stage}")
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
        # run_args = argparse.Namespace(**config)

        _findings = []

        # print(ref_config, config)

        for i in range(args.runs_per_stage):
            print(f"run #{i}", end=" ... ", flush=True)
            subprocess.run(["python", "ma_main.py", f"--config={config_path}",
                            f"--seed={random.getrandbits(16)}"], capture_output=True)
            print("done")

        for folder in os.listdir(save_dir):
            _save_dir = os.path.join(save_dir, folder)

            try:
                play_statistics = joblib.load(os.path.join(_save_dir, "play_statistics.obj"))
                rewards = play_statistics["rewards"]
                _finding = {
                    "save_dir": _save_dir,
                    "rewards": rewards,
                    "statistics": dict()
                }

                for keyword in args.keywords:
                    _finding["statistics"][keyword] = play_statistics[keyword]

                _findings.append(_finding)
            except NotADirectoryError:
                pass

        for _finding in _findings:
            likelihoods.append(args.default_likelihood)
            likelihoods.append(args.default_likelihood)
            for i_agent in agents:
                for j_agent in agents:
                    ref_config[i_agent].append({
                        "load_dir": _finding["save_dir"],
                        "load_step": None,
                        "num_refs": n_ref,
                        "load_agent": j_agent
                    })

        n_ref += len(_findings) * len(agents)

        findings.append(_findings)
        print("Findings:", _findings)

        joblib.dump(findings, findings_file)

        n_stage += 1

        if args.max_stages is not None and n_stage >= args.max_stages:
            break


if __name__ == "__main__":
    main()
