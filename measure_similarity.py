import os
import itertools
import json
import numpy as np

from pathlib import Path

from ma_main import run
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.multi_agent.utils import get_timestamp, make_env, mkdir


def add_extra_args(parser):
    parser.add_argument('--load-dirs', nargs="*")
    parser.add_argument('--all-load-dir')
    parser.add_argument('--dvd', action='store_true', default=False)
    parser.add_argument('--num-stages', type=int)
    parser.add_argument('--max-policies', type=int)
    parser.add_argument('--list-num-refs', nargs="*", type=int)
    parser.add_argument('--load-steps', nargs="*")
    parser.add_argument('--test-steps', type=int)
    parser.add_argument('--symmetry')  # format: "0,1|3,4,5" (agent_id)
    return parser


def search_permutations(mapping, sym, perms):
    if len(sym) == 0:
        perms.append(mapping)
    else:
        n = len(sym[0])
        cur_perms = list(itertools.permutations(sym[0]))
        op = cur_perms[0]
        for cp in cur_perms:
            for i in range(n):
                mapping[op[i]] = cp[i]
            search_permutations(mapping, sym[1:], perms)


def get_permutations(sym_groups):
    permutations = []
    search_permutations(list(range(n)), sym_groups, permutations)
    return permutations


def get_symmetry(sym_str):
    if sym_str is None:
        return []
    sym_groups = list(map(lambda group: list(map(int, group.split(","))), sym_str.split("|")))
    return sym_groups


def load_all(load_dir, n_agents, max_policies):
    load_dirs = []
    load_steps = []
    list_num_refs = []
    for folder in os.listdir(load_dir):
        try:
            path = os.path.join(load_dir, folder)
            _config = json.load(open(os.path.join(path, "config.json")))
            # print(path)
            load_dirs.append(path)
            load_steps.append(None)
            num_refs = 0 if not _config["use_reference"] else len(_config["likelihood_threshold"])
            for j in range(n_agents):
                list_num_refs.append(num_refs)
            if len(load_dirs) == max_policies:
                break
        except NotADirectoryError:
            pass
    return load_dirs, load_steps, list_num_refs


def main():
    args = get_args(add_extra_args)

    args.train_in_turn = False
    args.play = False
    tmp_dir = "/tmp/measure_similarity/{}".format(get_timestamp())
    mkdir(tmp_dir)
    args.save_dir = os.path.join(tmp_dir, "save_dir")
    args.num_steps = args.test_steps // args.num_processes
    args.num_env_steps = args.test_steps
    args.ref_config = os.path.join(tmp_dir, "ref-config.json")
    args.load = True
    args.use_reference = True
    args.save_interval = 0
    args.train = False
    args.reject_sampling = False
    args.load_dvd_weights_dir = None
    n_agents = args.num_agents

    if args.env_name in ["hopper", 'half-cheetah', 'walker2d']:
        args.deterministic = True

    load_dirs = []
    load_steps = []
    list_num_refs = []

    if args.load_dirs is not None:
        load_dirs += args.load_dirs
        if args.load_steps is None:
            load_steps += [None] * len(args.load_dirs)
        else:
            load_steps += args.load_steps
        list_num_refs += args.list_num_refs

    if args.all_load_dir is not None:
        all_load_dir = args.all_load_dir
        num_stages = args.num_stages
        if num_stages is not None:
            for i in range(num_stages):
                stage_dir = os.path.join(all_load_dir, "stage-{}".format(i))
                ld, ls, lnr = load_all(stage_dir, n_agents, None)
                load_dirs += ld
                load_steps += ls
                list_num_refs += lnr
        else:
            ld, ls, lnr = load_all(all_load_dir, n_agents, args.max_policies)
            load_dirs += ld
            load_steps += ls
            list_num_refs += lnr

    n_policies = len(load_dirs)
    list_num_refs = np.array(list_num_refs).reshape(n_policies, n_agents).tolist()

    # permutations = get_permutations(args.num_agents, args.symmetry)
    # n_perm = len(permutations)
    # print("# permutations:", n_perm)

    sym_groups = get_symmetry(args.symmetry)
    which_group = []
    for i in range(n_agents):
        which_group.append([i])
    for g in sym_groups:
        for i in g:
            which_group[i] = g

    env = make_env(args.env_name, args.episode_steps)

    ref_config = dict()
    ref_indices = []
    for i in range(n_agents):
        refs = []
        _ref_indices = []
        for j in which_group[i]:
            for k in range(n_policies):
                ref = {
                    "load_dir": load_dirs[k],
                    "load_step": load_steps[k],
                    "num_refs": list_num_refs[k][j],
                    "load_agent": env.agents[j]
                }
                refs.append(ref)
                _ref_indices.append(k)
        ref_config[env.agents[i]] = refs
        ref_indices.append(_ref_indices)

    json.dump(ref_config, open(args.ref_config, "w"))

    dir_matrix = np.zeros((n_policies, n_policies))

    for i in range(n_policies):
        args.load_dir = load_dirs[i]
        args.load_step = load_steps[i]
        args.num_refs = list_num_refs[i]
        result = run(args)
        likelihood = np.zeros(n_policies)
        for j in range(n_agents):
            _likelihood = result["statistics"][env.agents[j]]["likelihood"][0][1]
            # print(len(_likelihood), len(ref_indices[j]))
            assert len(_likelihood) == len(ref_indices[j])
            for k in range(len(_likelihood)):
                likelihood[ref_indices[j][k]] += _likelihood[k] / len(which_group[j])

        dir_matrix[i] = likelihood / n_agents

    print(dir_matrix)

    sim_matrix = np.zeros((n_policies, n_policies))

    for i in range(n_policies):
        for j in range(n_policies):
            sim_matrix[i, j] = (dir_matrix[i, j] + dir_matrix[j, i] - dir_matrix[i, i] - dir_matrix[j, j]) / 2

    print(sim_matrix)

    dir_matrix.dump("dir_matrix.obj")
    sim_matrix.dump("sim_matrix.obj")

    if args.all_load_dir is not None:
        dir_matrix.dump(os.path.join(args.all_load_dir, "dir_matrix.obj"))
        sim_matrix.dump(os.path.join(args.all_load_dir, "sim_matrix.obj"))

    # print(np.linalg.det(sim_matrix))


if __name__ == "__main__":
    main()
