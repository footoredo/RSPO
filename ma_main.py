import time
import json

from a2c_ppo_acktr.arguments import get_args
from multiprocessing import shared_memory

from functools import partial

from a2c_ppo_acktr.multi_agent import Agent, RefAgent, Environment


from a2c_ppo_acktr.multi_agent.utils import *

import logging
import multiprocessing as mp


def train_in_turn(n_agents, i, n_iter):
    return n_iter % n_agents == i


def train_simultaneously(n_agents, i, n_iter):
    return True


def no_train(n_agents, i, n_iter):
    return False


def _run(args, logger):
    result = dict()

    num_agents = args.num_agents
    num_envs = args.num_processes

    # if args.use_reference:
    #     args.num_env_steps *= 2

    agents = []
    main_agent_conns = []
    envs = []
    main_env_conns = []

    _make_env = partial(make_env, args.env_name, args.episode_steps)
    env = _make_env()

    input_structures = env.input_structures
    if args.use_reference and args.ref_config is not None:
        ref_config = json.load(open(args.ref_config))
    else:
        ref_config = None

    ref_agents = []
    num_refs_all = []
    if args.use_reference:
        for i, agent in enumerate(env.agents):
            obs_space = env.observation_spaces[agent]
            act_space = env.action_spaces[agent]

            if ref_config is not None:
                ref_load_dirs = []
                ref_load_steps = []
                ref_num_refs = []
                ref_load_agents = []
                for ref in ref_config[str(env.agents[i])]:
                    ref_load_dirs.append(ref["load_dir"])
                    ref_load_steps.append(ref["load_step"])
                    ref_num_refs.append(ref["num_refs"])
                    ref_load_agents.append(ref["load_agent"])
            else:
                ref_load_dirs = args.ref_load_dir
                ref_load_steps = args.ref_load_step
                ref_num_refs = args.ref_num_ref
                if type(ref_load_dirs) != list:
                    ref_load_dirs = [ref_load_dirs]
                if type(ref_load_steps) != list:
                    ref_load_steps = [ref_load_steps] * len(ref_load_dirs)
                if type(ref_num_refs) != list:
                    ref_num_refs = [ref_num_refs] * len(ref_load_dirs)
                ref_load_agents = [env.agents[i]] * len(ref_load_dirs)
            ref_agents_i = []
            # print(ref_load_steps, args.ref_use_ref)
            for ld, ls, nr, la in zip(ref_load_dirs, ref_load_steps, ref_num_refs, ref_load_agents):
                ref_agent, _ = get_agent(la, args, obs_space, input_structures[agent], act_space, None, n_ref=nr,
                                         is_ref=True)
                load_actor_critic(ref_agent, ld, la if type(la) == str else env.agents[la], ls)
                ref_agents_i.append(ref_agent)
            ref_agents.append(ref_agents_i)
            num_refs_all.append(len(ref_agents_i))
            # print(num_refs)

    save_dir = mkdir2(args.save_dir, get_timestamp())
    json.dump(vars(args), open(os.path.join(save_dir, "config.json"), "w"), indent=2)

    obs_locks = []
    act_locks = []
    ref_locks = []

    for i in range(num_envs):
        _obs_locks = []
        _act_locks = []
        for j in range(num_agents):
            # __obs_locks = []
            # __act_locks = []
            # for k in range(1 + num_refs_all[j]):
            #     __obs_locks.append(mp.Event())
            #     # _obs_locks[-1].set()
            #     __act_locks.append(mp.Event())
            # # _act_locks[-1].set()
            # _obs_locks.append(__obs_locks)
            # _act_locks.append(__act_locks)
            _obs_locks.append(mp.Event())
            _act_locks.append(mp.Event())
        obs_locks.append(_obs_locks)
        act_locks.append(_act_locks)

    # for i in range(num_agents):
    #     _ref_locks = []
    #     for k in range(num_refs_all[i]):
    #         _ref_locks.append(mp.Event())
    #     ref_locks.append(_ref_locks)

    assert num_agents == len(env.agents)
    obs_buffer_size = 0
    item_size = np.zeros(1, dtype=np.float32).nbytes
    # print(item_size)
    obs_indices = []

    for i, agent in enumerate(env.agents):
        # print(env.observation_spaces[agent].shape)
        next_obs_buffer_size = obs_buffer_size + item_size * num_envs * (env.observation_spaces[agent].shape[0] + 2)
        # print(next_obs_buffer_size - obs_buffer_size)
        obs_indices.append((obs_buffer_size, next_obs_buffer_size))
        obs_buffer_size = next_obs_buffer_size

    obs_shm = shared_memory.SharedMemory(create=True, size=obs_buffer_size)
    act_shm = shared_memory.SharedMemory(create=True, size=num_envs * num_agents)
    ref_shms = []

    # print("input_structures:", input_structures)

    # print(len(env.agents))
    # print(env.agents)

    train_fn = no_train
    if args.train:
        if args.train_in_turn:
            train_fn = train_in_turn
        else:
            train_fn = train_simultaneously

    for i, agent in enumerate(env.agents):
        conn1, conn2 = mp.Pipe()
        main_agent_conns.append(conn1)

        obs_space = env.observation_spaces[agent]
        act_space = env.action_spaces[agent]

        ref_shm = None

        if args.use_reference:
            num_actions = act_space.n

            # print(num_refs_all[i], args.num_processes, num_actions, item_size)
            # ref_shm = shared_memory.SharedMemory(create=True, size=num_refs_all[i] * args.num_processes * num_actions * item_size)
            # ref_processes_i = []
            # for j, ref_agent in enumerate(ref_agents[i]):
            #     ref_process = RefAgent(agent=ref_agent, agent_id=i, ref_id=j, num_refs=num_refs_all[i],
            #                            num_actions=num_actions, args=args, obs_shm=obs_shm,
            #                            buffer_start=obs_indices[i][0], buffer_end=obs_indices[i][1],
            #                            ref_shm=ref_shm, obs_locks=[locks[i][1 + j] for locks in obs_locks],
            #                            act_locks=[locks[i][1 + j] for locks in act_locks],
            #                            ref_locks=[locks[j] for locks in ref_locks])
            #     ref_processes_i.append(ref_process)
            #     # ref_process.start()
            #
            #     # while True:
            #     #     obs = list(map(float, input().split()))
            #     #     obs = torch.tensor(obs, dtype=torch.float)
            #     #     print(ref_agent.get_strategy(obs, None, None))
            # ref_shms.append(ref_shm)

        # print("123123132")
        num_refs = None if args.num_refs is None else args.num_refs[i]
        thread_limit = args.parallel_limit // num_agents
        # thread_limit = None
        ap = Agent(i, env.agents[i], thread_limit=thread_limit, logger=logger.info, args=args,
                   obs_space=obs_space,
                   input_structure=input_structures[agent],
                   act_space=act_space, main_conn=conn2,
                   obs_shm=obs_shm, buffer_start=obs_indices[i][0], buffer_end=obs_indices[i][1],
                   obs_locks=[locks[i] for locks in obs_locks], act_shm=act_shm, ref_shm=None,
                   act_locks=[locks[i] for locks in act_locks], ref_locks=None,
                   use_attention=args.use_attention,
                   save_dir=save_dir,
                   train=partial(train_fn, args.num_agents, i),
                   num_refs=num_refs,
                   reference_agent=ref_agents[i])
        # print("123123123", i)
        agents.append(ap)
        ap.start()

    for i in range(num_envs):
        conn1, conn2 = mp.Pipe()
        main_env_conns.append(conn1)
        ev = Environment(i, logger.info, args, env=_make_env(), agents=env.agents, main_conn=conn2,
                         obs_shm=obs_shm,
                         obs_locks=obs_locks[i],
                         act_shm=act_shm,
                         act_locks=act_locks[i])
        envs.append(ev)
        ev.start()

    for conn in main_agent_conns:
        conn.send(None)

    for conn in main_env_conns:
        conn.send(None)

    num_updates = args.num_env_steps // args.num_steps // args.num_processes

    if args.reject_sampling:
        for i in range(num_updates):
            while True:
                # This happens every batch (times num_envs)
                finish = True
                for j in range(num_agents):
                    cf = main_agent_conns[j].recv()
                    # print(j, cf)
                    finish = finish and cf
                    # print(j, finish)
                for j in range(num_agents):
                    main_agent_conns[j].send(finish)
                for j in range(num_envs):
                    main_env_conns[j].send(finish and i == num_updates - 1)
                if finish:
                    break

    for ev in envs:
        ev.join()

    obs_shm.close()
    obs_shm.unlink()
    act_shm.close()
    act_shm.unlink()

    statistics = dict()
    for i, agent in enumerate(env.agents):
        statistics[agent] = main_agent_conns[i].recv()

    result["statistics"] = statistics

    import joblib
    joblib.dump(statistics, os.path.join(save_dir, "statistics.obj"))

    if args.plot:
        plot_statistics(statistics, "reward")
        if args.use_reference:
            plot_statistics(statistics, "dist_penalty")
        # plot_statistics(statistics, "grad_norm")
        # plot_statistics(statistics, "value_loss")
        # plot_statistics(statistics, "action_loss")
        # plot_statistics(statistics, "dist_entropy")

    close_to = [0, 0, 0, 0, 0]
    gather_count = np.zeros((5, 5), dtype=int)
    sum_reward = np.zeros(num_agents)
    if args.play:
        env.seed(np.random.randint(10000))
        obs = env.reset()
        if args.env_name == "stag-hunt-gw":
            if args.render:
                for agent in env.env.agents:
                    print(agent.agent_id, agent.pos, agent.collective_return)
                print("monster", env.env.stag_pos)
                print("plant1", env.env.hare1_pos)
                print("plant2", env.env.hare2_pos)
                print('---------------')
        dones = {agent: False for agent in env.agents}
        num_games = 0
        images = []
        while True:
            actions = dict()
            for i, agent in enumerate(env.agents):
                main_agent_conns[i].send((obs[agent], dones[agent]))
                actions[agent] = main_agent_conns[i].recv()
                # print(agent, actions[agent])
            obs, rewards, dones, infos = env.step(actions)
            for i, agent in enumerate(env.agents):
                sum_reward[i] += rewards[agent]
            if args.env_name == "stag-hunt-gw":
                if env.env.agents[0].pos[0] == env.env.agents[1].pos[0] and \
                        env.env.agents[0].pos[1] == env.env.agents[1].pos[1]:
                    gather_count[env.env.agents[0].pos[0]][env.env.agents[0].pos[1]] += 1
                if args.render:
                    print(actions)
                    for agent in env.env.agents:
                        print(agent.agent_id, agent.pos, agent.collective_return)
                    print("monster", env.env.stag_pos)
                    print("plant1", env.env.hare1_pos)
                    print("plant2", env.env.hare2_pos)
                    print(env.steps, env.max_frames, dones)
                    env.env.render()
                    print('---------------')
            if args.render:
                if args.env_name != "stag-hunt-gw":
                    env.render()
                    # input()
                    time.sleep(0.1)
            elif args.gif:
                image = env.render(mode='rgb_array')
                # print(image)
                images.append(image)

            not_done = False
            for agent in env.agents:
                not_done = not_done or not dones[agent]

            if not not_done:
                num_games += 1
                # print(infos[env.agents[0]])
                if args.env_name[:6] == "simple":
                    close_to[np.argmin(infos[env.agents[0]])] += 1
                if num_games >= args.num_games_after_training or args.num_games_after_training == -1:
                    break
                obs = env.reset()
                dones = {agent: False for agent in env.agents}

        if args.gif:
            from a2c_ppo_acktr.multi_agent.utils import save_gif
            save_gif(images, os.path.join(save_dir, "plays.gif"), 15)

    for i, agent in enumerate(agents):
        main_agent_conns[i].send(None)
        agent.join()

    # print(close_to)
    result["close_to"] = close_to
    print(gather_count)
    # import seaborn as sns
    # sns.heatmap(gather_count)
    # plt.show()
    print(sum_reward / args.num_games_after_training)

    # print(result["statistics"][env.agents[0]]["likelihood"])
    # print(result["statistics"][env.agents[1]]["likelihood"])

    return result


def run(args):
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)
    return _run(args, logger)


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
