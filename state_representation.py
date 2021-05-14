import wandb

import torch.nn

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.multi_agent.utils import *

from functools import partial
import torch.nn.functional as F


def action_one_hot(n, i):
    one_hot = np.zeros(n, dtype=np.float32)
    one_hot[i] = 1
    return one_hot


def ts(v):
    return torch.tensor(v)


def sample_games(env, agents, num_games_total):
    experiences = dict()
    for agent in env.agents:
        experiences[agent] = dict(obs=[], strategy=[], action=[], action_onehot=[], next_obs=[])

    obs = env.reset()
    dones = {agent: False for agent in env.agents}
    num_games = 0
    recurrent_hidden_states = torch.zeros((1, agents[0].recurrent_hidden_state_size))
    while True:
        actions = dict()
        for i, agent in enumerate(env.agents):
            _obs = np.array(obs[agent], dtype=np.float32)
            strategy = agents[i].get_strategy(ts([_obs]), recurrent_hidden_states,
                                                 ts([[0.0] if dones[agent] else [1.0]])).detach().squeeze().numpy()
            action = np.random.choice(strategy.shape[0], p=strategy)
            experiences[agent]["obs"].append(_obs)
            experiences[agent]["strategy"].append(strategy)
            experiences[agent]["action"].append(action)
            experiences[agent]["action_onehot"].append(action_one_hot(strategy.shape[0], action))
            actions[agent] = action
        obs, rewards, dones, infos = env.step(actions)

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


def sim_games(env, agents, state_rep_nets, inverse_nets, num_games_total):
    obs = env.reset()
    dones = {agent: False for agent in env.agents}
    num_games = 0
    recurrent_hidden_states = torch.zeros((1, agents[0].recurrent_hidden_state_size))
    sum_rewards = [0. for _ in env.agents]
    step = 0
    while True:
        actions = dict()
        print(f"========step {step}==========")
        for i, agent in enumerate(env.agents):
            _obs = np.array(obs[agent], dtype=np.float32)
            state_rep = state_rep_nets[i](ts([_obs])).detach()
            strategy = inverse_nets[i](state_rep).detach().squeeze(0).numpy()
            action = np.random.choice(strategy.shape[0], p=strategy)
            actions[agent] = action
            print("------------------------------")
            print(f"player #{i}")
            print(f"obs:\t{_obs}")
            print(f"state rep:\t{state_rep.squeeze(0)}")
            print(f"strategy:\t{strategy}")
            print(f"action:\t{action}")

        print("------------------------------")

        obs, rewards, dones, infos = env.step(actions)

        for i, agent in enumerate(env.agents):
            print(f"player #{i} reward:\t{rewards[i]}")
            sum_rewards[i] += rewards[i]

        not_done = False
        for agent in env.agents:
            not_done = not_done or not dones[agent]

        step += 1

        if not not_done:
            step = 0
            print(f"sum_rewards:\t{sum_rewards}")
            sum_rewards = [0. for _ in env.agents]
            num_games += 1
            # print(infos[env.agents[0]])
            if num_games >= num_games_total:
                break
            obs = env.reset()
            dones = {agent: False for agent in env.agents}


def cross_entropy(a, b):
    return -torch.mul(a, torch.log(b)).sum(dim=1) + torch.mul(a, torch.log(a)).sum(dim=1)


def main():
    torch.set_printoptions(precision=5, sci_mode=False)

    args = get_args()

    wandb.init(project='state-representation-learning', entity='footoredo', group="test", config=args)

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
    experiences = sample_games(env, agents, 1)

    state_rep_nets = []
    inverse_nets = []

    for agent in env.agents:
        num_hidden = 32
        num_rep = 16
        num_inputs = env.observation_spaces[agent].shape[0]
        num_actions = env.action_spaces[agent].n
        state_rep_net = torch.nn.Sequential(torch.nn.Linear(num_inputs, num_hidden),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(num_hidden, num_rep, bias=False))
        forward_net = torch.nn.Sequential(torch.nn.Linear(num_rep + num_actions, num_hidden),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(num_hidden, num_rep))
        inverse_net = torch.nn.Sequential(torch.nn.Linear(num_rep, num_hidden),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(num_hidden, num_actions),
                                          torch.nn.Softmax())

        wandb.watch(state_rep_net, log="all", log_freq=100)
        wandb.watch(forward_net, log="all", log_freq=100)
        wandb.watch(inverse_net, log="all", log_freq=100)

        obs = experiences[agent]["obs"]
        strategy = experiences[agent]["strategy"]
        action = experiences[agent]["action"]
        action_onehot = experiences[agent]["action_onehot"]
        next_obs = experiences[agent]["next_obs"]

        state_rep_optimizer = torch.optim.RMSprop(state_rep_net.parameters(), lr=1e-4, weight_decay=0.0)
        forward_optimizer = torch.optim.RMSprop(forward_net.parameters(), lr=1e-4)
        inverse_optimizer = torch.optim.RMSprop(inverse_net.parameters(), lr=1e-4)

        for i in range(10000):
            if i % 10 == 0:
                experiences = sample_games(env, agents, 10)
                obs = experiences[agent]["obs"]
                strategy = experiences[agent]["strategy"]
                action = experiences[agent]["action"]
                action_onehot = experiences[agent]["action_onehot"]
                next_obs = experiences[agent]["next_obs"]

            rep = state_rep_net(obs)
            next_rep = state_rep_net(next_obs)
            # action_pred = inverse_net(torch.cat([rep, next_rep], dim=1))
            action_pred = inverse_net(rep)
            loss_forward = (forward_net(torch.cat([rep, action_onehot], dim=1)) - next_rep).square().mean()
            loss_inverse = (cross_entropy(strategy, action_pred)).mean()
            reg_loss = torch.sum(torch.square(state_rep_net[0].weight)) + \
                       torch.sum(torch.square(state_rep_net[2].weight))
            loss = 1.0 * loss_forward + 1 * loss_inverse + 0.01 * reg_loss
            state_rep_optimizer.zero_grad()
            forward_optimizer.zero_grad()
            inverse_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state_rep_net.parameters(), 40.)
            torch.nn.utils.clip_grad_norm_(forward_net.parameters(), 40.)
            torch.nn.utils.clip_grad_norm_(inverse_net.parameters(), 40.)
            state_rep_optimizer.step()
            forward_optimizer.step()
            inverse_optimizer.step()

            wandb.log({f"agent-{agent}/loss_forward": loss_forward.item(),
                       f"agent-{agent}/loss_inverse": loss_inverse.item(),
                       f"agent-{agent}/reg_loss": reg_loss.item(),
                       f"agent-{agent}/loss": loss.item()})

            if i % 200 == 0:
                print(f"\n#{i}")
                print(i, loss_forward.item(), loss_inverse.item(), reg_loss.item(), loss.item())
                print("obs", obs[0])
                print("rep", rep[0])
                print("action_pred", action_pred[0])
                print("action", action[0])
                print("strategy", strategy[0])
                # print(obs[0], rep[0], action_pred[0], action[0], strategy[0])

        state_rep_nets.append(state_rep_net)
        inverse_nets.append(inverse_net)

        torch.save(state_rep_net.state_dict(), f"agent-{agent}_state_rep_net.obj")
        torch.save(inverse_net.state_dict(), f"agent-{agent}_inverse_net.obj")
        wandb.save(f"agent-{agent}_state_rep_net.obj")
        wandb.save(f"agent-{agent}_inverse_net.obj")

    sim_games(env, agents, state_rep_nets, inverse_nets, 5)


if __name__ == "__main__":
    main()
