import torch.nn

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.multi_agent.utils import *

from functools import partial
import torch.nn.functional as F

import seaborn as sns


def action_one_hot(n, i):
    one_hot = np.zeros(n, dtype=np.float32)
    one_hot[i] = 1
    return one_hot


def ts(v):
    return torch.tensor(v)


def sample_games(env, agents, num_games_total, args):
    experiences = dict()
    for agent in env.agents:
        experiences[agent] = dict(obs=[], strategy=[], action=[], action_onehot=[], next_obs=[])

    obs = env.reset()
    dones = {agent: False for agent in env.agents}
    num_games = 0
    recurrent_hidden_states = torch.zeros((1, agents[0].recurrent_hidden_state_size))
    while True:
        actions = dict()
        strategies = dict()
        reward_predictions = []
        random_net_values = []
        random_net_predictions = []
        for i, agent in enumerate(env.agents):
            _obs = np.array(obs[agent], dtype=np.float32)
            ts_obs = ts([_obs])
            masks = ts([[0.0] if dones[agent] else [1.0]])
            strategy = agents[i].get_strategy(ts_obs, recurrent_hidden_states, masks).detach().squeeze().numpy()
            action = np.random.choice(strategy.shape[0], p=strategy)
            prediction = agents[i].get_reward_prediction(ts_obs, recurrent_hidden_states, masks, ts([[action]]))
            reward_prediction, random_net_value, random_net_prediction = prediction
            reward_predictions.append(reward_prediction.item())
            # random_net_values.append(random_net_value.item())
            # random_net_predictions.append(random_net_prediction.item())
            experiences[agent]["obs"].append(_obs)
            experiences[agent]["strategy"].append(strategy)
            experiences[agent]["action"].append(action)
            experiences[agent]["action_onehot"].append(action_one_hot(strategy.shape[0], action))
            actions[agent] = action
            strategies[agent] = strategy
        old_obs = obs
        obs, rewards, dones, infos = env.step(actions)

        if type(rewards) == dict:
            rewards = [rewards[agent] for agent in env.agents]

        print("obs:", old_obs)
        print("actions:", actions)
        print("strategies:", strategies)
        print("rewards:", rewards)
        print("predictions:", np.array(reward_predictions) / args.reward_prediction_multiplier)
        print("prediction errors:", np.square(np.array(reward_predictions) / args.reward_prediction_multiplier - np.array(rewards)))
        print("")
        # print("random_net_values:", random_net_values)
        # print("random_net_predictions:", random_net_predictions)
        # print("random_net_prediction_errors:", np.array(random_net_predictions) - np.array(random_net_values))

        for agent in env.agents:
            _obs = np.array(obs[agent], dtype=np.float32)
            experiences[agent]["next_obs"].append(_obs)

        not_done = False
        for agent in env.agents:
            not_done = not_done or not dones[agent]

        if not not_done:
            num_games += 1
            # print(infos[env.agents[0]])
            if num_games >= num_games_total:
                break
            obs = env.reset()
            dones = {agent: False for agent in env.agents}

    for agent in env.agents:
        n_exp = len(experiences[agent]["obs"])
        shuffle_indices = list(range(n_exp))
        np.random.shuffle(shuffle_indices)
        keywords = ["obs", "strategy", "action", "action_onehot", "next_obs"]
        for keyword in keywords:
            experiences[agent][keyword] = ts(experiences[agent][keyword])[shuffle_indices]

    return experiences


def main():
    torch.set_printoptions(precision=5, sci_mode=False)

    args = get_args()

    _make_env = partial(make_env, args.env_name, args.episode_steps, args.env_config)
    env = _make_env()

    agents = []

    for i, agent in enumerate(env.agents):
        obs_space = env.observation_spaces[agent]
        act_space = env.action_spaces[agent]
        input_structure = env.input_structures[agent]

        actor_critic, _ = get_agent(agent, args, obs_space, input_structure,
                                    act_space, None, n_ref=args.num_refs[i], is_ref=False)
        load_actor_critic(actor_critic, args.load_dir, agent, args.load_step)
        agents.append(actor_critic)

    env.seed(np.random.randint(10000))

    experiences = sample_games(env, agents, 1, args)

    # score = np.zeros(9)
    # weights = np.zeros(9)

    # for agent_id in range(env.num_agents):
    #     exp = experiences[agent_id]
    #     n_exp = len(exp["obs"])
    #     print(n_exp)
    #     # for i_exp in range(n_exp):
    #     #     for j_exp in range(n_exp):
    #     #         for i in range(9):
    #     #             w = (exp["obs"][i_exp][i] - exp["obs"][j_exp][i]) ** 2
    #     #             d = (exp["strategy"][i_exp] - exp["strategy"][j_exp]).square().sum()
    #     #             score[i] += d * w
    #     #             weights[i] += w
    #
    #     for i in range(n_exp):


    # print(score / weights)

    # obss = [[0.1, 0., 1., 4., 2., 0., 0., 0., -1.],
    #         [0.1, 0., 1., 4., 2., 0., 0., 0., 1.],
    #         [0.1, 0., 1., 4., 2., 0., 0., -1., 0.],
    #         [0.1, 0., 1., 4., 2., 0., 0., 1., 0.]]
    # obss = [[0.1, 0., 0., 4., 2., 0., 0., 1., -1.],   # [0, 3]
    #         [0.1, 0., 0., 4., 2., 0., 0., 1., 1.],    # [1, 3]
    #         [0.1, 0., 0., 4., 2., 0., 0., -1., 1.],   # [1, 2]
    #         [0.1, 0., 0., 4., 2., 0., 0., -1., -1.]]  # [0, 2]
    obss = [[0.1, 0., 0., 4., 2., 0., 0., 0., -1.],
            [0.1, 0., 0., 4., 2., 0., 0., 0., 1.],
            [0.1, 0., 0., 4., 2., 0., 0., -1., 0.],
            [0.1, 0., 0., 4., 2., 0., 0., 1., 0.]]

    while True:
        obs = input("obs: ")
        agent = 0
        obs = torch.tensor(list(map(float, obs.split())))
        strategy = agents[agent].get_strategy(obs, None, None).detach()
        print(strategy)
        print(-torch.log(strategy))
        for i in range(env.action_spaces[agent].n):
            action = torch.LongTensor([i])
            prediction = agents[agent].get_reward_prediction(obs, None, None, action)
            reward_prediction, random_net_value, random_net_prediction = prediction
            # print(reward_prediction.item(), random_net_value.item(), random_net_prediction.item(), (random_net_value - random_net_prediction).item())
            print(reward_prediction.item() / args.reward_prediction_multiplier)

    # obss = experiences[0]["obs"]

    n_obs = len(obss)

    obs = torch.tensor(obss, requires_grad=True)
    strategy = agents[0].get_strategy(obs, None, None)
    grads = np.zeros((9,))
    for i in range(n_obs):
        _grads = np.zeros((9,))
        for j in range(5):
            grad = torch.autograd.grad(strategy[i, j], obs, retain_graph=True)[0][i].detach().abs().numpy()
            # print(grad.shape)
            _grads += grad
        grads += _grads
    print(-torch.log(strategy.detach()))
    # sns.heatmap(grads.reshape(1, 9), square=True, vmax=1000, vmin=0)
    # sns.heatmap(grads.reshape(1, 9)[:, 3:], square=True)
    # plt.show()


if __name__ == "__main__":
    main()
