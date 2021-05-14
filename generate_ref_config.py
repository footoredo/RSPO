import os
import json


def make_entry(load_dir, num_refs, load_agent):
    return {
        "load_dir": load_dir,
        "load_step": None,
        "num_refs": num_refs,
        "load_agent": load_agent
    }


def generate(path, num_agents, num_refs):
    ref_configs = [[] for _ in range(num_agents)]
    for folder in os.listdir(path):
        _path = os.path.join(path, folder)
        for i in range(num_agents):
            for j in range(num_agents):
                ref_configs[i].append(make_entry(_path, num_refs, j))
    ref_config = dict()
    for i in range(num_agents):
        ref_config[str(i)] = ref_configs[i]
    json.dump(ref_config, open("ref_config.json", "w"), indent=2)


if __name__ == "__main__":
    generate("./sync-results/half-cheetah/0-step", 1, 0)
