import os
import itertools
import json
import numpy as np

from ma_main import run
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.multi_agent.utils import get_timestamp, make_env, mkdir


def add_extra_args(parser):
    parser.add_argument('--load-dirs', nargs="*")
    parser.add_argument('--list-num-refs', nargs="*", type=int)
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
    n_agents = args.num_agents
    load_dirs = args.load_dirs
    n_policies = len(load_dirs)
    list_num_refs = np.array(args.list_num_refs).reshape(n_policies, n_agents).tolist()

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
                    "load_step": None,
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


if __name__ == "__main__":
    main()
