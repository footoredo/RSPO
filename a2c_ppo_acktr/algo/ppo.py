import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from a2c_ppo_acktr.multi_agent.utils import flat_view, net_add, ggrad, tsne


D_MAP = "SLRDU"


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 clip_grad_norm=True,
                 use_clipped_value_loss=True,
                 task=None,
                 direction=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.clip_grad_norm = clip_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.task = task
        self.direction = direction

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # advantages = rollouts.returns[:-1] # - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        grad_norm_epoch = 0

        cnt = [0, 0]

        # the_generator = rollouts.feed_forward_generator(advantages, mini_batch_size=1)
        # the_sample = next(the_generator)
        # the_obs = the_sample[0]
        the_obs = torch.tensor([0., 0., 0., -1., 0., 1., 0.])
        the_strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()
        print(the_obs.size())
        print(the_obs)
        print(the_strategy)
        fgs = []
        advs = []
        nears = []
        fgmax = []

        episode_steps = 32

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch, episode_steps=episode_steps)

            mini_batch_cnt = 0
            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                batch_size = value_preds_batch.view(-1).size()[0]

                z_mask = []
                for z in range(batch_size // 32):
                    near = 0
                    # for i in range(32):
                    #     if obs_batch[z][i][3:5].norm(2) < 0.11:
                    #         cnt[0] += 1
                    #         near = 1
                    #         break
                    #         # print(0, obs_batch[0], D_MAP[actions_batch.item()], adv_targ.item())
                    #     if obs_batch[z][i][5:7].norm(2) < 0.11:
                    #         cnt[1] += 1
                    #         near = 2
                    #         break
                    # z_mask.append(near != 2)
                    z_mask.append(1.)

                adv_targ = torch.mul(adv_targ, torch.tensor(z_mask, dtype=torch.float).unsqueeze(-1))

                obs_batch = obs_batch.view(batch_size, -1)
                recurrent_hidden_states_batch = recurrent_hidden_states_batch.view(batch_size, -1)
                actions_batch = actions_batch.view(batch_size, -1)
                value_preds_batch = value_preds_batch.view(batch_size, -1)
                return_batch = return_batch.view(batch_size, -1)
                masks_batch = masks_batch.view(batch_size, -1)
                old_action_log_probs_batch = old_action_log_probs_batch.view(batch_size, -1)
                adv_targ = adv_targ.view(batch_size, -1)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, dists = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                # print(obs_batch[0], dists.probs[0])

                # if adv_targ.item() > 0. and int(obs_batch[0][0] + 0.5) == 31:

                # if near != 1:
                #     mini_batch_cnt += 1
                #     # print(mini_batch_cnt)
                #     if mini_batch_cnt >= self.num_mini_batch:
                #         break
                #     continue
                    # print(1, obs_batch[0], D_MAP[actions_batch.item()], adv_targ.item())

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # with torch.no_grad():
                #     fg = torch.autograd.grad(action_loss, self.actor_critic.parameters(), create_graph=True, allow_unused=True)
                #     fgs.append(fg)

                if self.task is None:
                    action_loss_mask = 1.
                    g = flat_view(ggrad(self.actor_critic, action_loss)).detach()
                    # print(g.norm(2))
                    net_add(self.actor_critic, -g)
                    strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()
                    # print(the_strategy.size())
                    fg = strategy - the_strategy
                    # print(fg)
                    # print(self.direction)
                    if fg.argmax() == self.direction:
                        action_loss_mask = 0.
                    # if fg.argmax() == 1:
                    #     cnt[0] += 1
                    # elif fg.argmax() == 2:
                    #     cnt[1] += 1
                    net_add(self.actor_critic, g)

                    self.optimizer.zero_grad()
                    (value_loss * self.value_loss_coef + action_loss_mask * action_loss -
                     dist_entropy * self.entropy_coef).backward()

                    total_norm = 0.
                    for p in self.actor_critic.parameters():
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    grad_norm_epoch += total_norm

                    if self.clip_grad_norm:
                        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                                 self.max_grad_norm)
                    self.optimizer.step()
                elif self.task[:4] == "grad":
                    if return_batch[0].item() > 0.5:
                        g = flat_view(ggrad(self.actor_critic, action_loss)).detach()
                        # print(g.norm(2))
                        net_add(self.actor_critic, -g)
                        strategy = self.actor_critic.get_strategy(the_obs, None, None).detach()
                        # print(the_strategy.size())
                        fg = strategy - the_strategy
                        fgmax.append(fg.argmax())
                        fgs.append(fg.reshape(-1).numpy())
                        advs.append(adv_targ.mean().item())
                        nears.append(["no", "near-1", "near-2"][near])
                        net_add(self.actor_critic, g)

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                mini_batch_cnt += 1
                # print(mini_batch_cnt)
                if mini_batch_cnt >= self.num_mini_batch:
                    break

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # import pandas as pd
        #
        # df = pd.DataFrame(dict(x=fgmax, y=nears))
        # sns.displot(data=df, x="x", y="y")
        # plt.show()
        #
        # print(cnt)
        if type(self.task) == str:
            if self.task[:3] == "cnt":
                import joblib
                joblib.dump(cnt, "{}.obj".format(self.task))
            elif self.task[:4] == "grad":
                pass
                # n = len(fgs)
                # dim = fgs[0].shape[0]
                # c = 2
                # ids = [[] for _ in range(c)]
                # xs = [np.zeros(dim) for _ in range(c)]
                #
                # def norm(vec):
                #     return np.linalg.norm(vec)
                #
                # for i in range(n):
                #     j = np.random.choice(c)
                #     ids[j].append(i)
                #     xs[j] += fgs[i] / norm(fgs[i])
                #
                # for j in range(c):
                #     xs[j] /= len(ids[j])
                #
                # for n_iter in range(20):
                #     ids = [[] for _ in range(c)]
                #     new_xs = [np.zeros(dim) for _ in range(c)]
                #     # print(xs)
                #     for i in range(n):
                #         nearest_dis = -5.
                #         nearest_j = 0
                #         for j in range(c):
                #             dis = fgs[i] @ xs[j] / (norm(fgs[i]) * norm(xs[j]))
                #             # print(dis, norm(fgs[i]), norm(xs[j]))
                #             if dis > nearest_dis:
                #                 nearest_dis = dis
                #                 nearest_j = j
                #         ids[nearest_j].append(i)
                #         new_xs[nearest_j] += fgs[i] / norm(fgs[i])
                #     for j in range(c):
                #         xs[j] = new_xs[j] / len(ids[j])
                #
                #     avg = [0. for _ in range(c)]
                #     for j in range(c):
                #         print(len(ids[j]))
                #         for i in ids[j]:
                #             avg[j] += int(nears[i] == "near-2") / len(ids[j])
                #     print(avg)
                # # print(len(fgs), fgs[0])
                # tsne(fgs, nears)
                # import joblib
                # joblib.dump((fgs, advs), "{}.obj".format(self.task))

        # mean_fg = []
        # for i, g in enumerate(fgs[0]):
        #     if g is None:
        #         mean_fg.append(None)
        #     else:
        #         mean_g = torch.zeros_like(g)
        #         for fg in fgs:
        #             mean_g += fg[i]
        #         mean_fg.append(mean_g / len(fgs))
        # fgs.append(tuple(mean_fg))
        #
        # print("generating fingerprints")
        # data_generator = rollouts.feed_forward_generator(advantages, 1)
        # fingerprints = []
        # for i, fg in enumerate(fgs):
        #     for p, g in zip(self.actor_critic.parameters(), fg):
        #         if g is not None:
        #             p.data += 1e-4 * g
        #
        #     fingerprint = []
        #
        #     for sample in data_generator:
        #         obs_batch, recurrent_hidden_states_batch, actions_batch, \
        #         value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
        #         adv_targ = sample
        #
        #         if i == 0:
        #             for j, obs in enumerate(list(obs_batch)):
        #                 if obs[0] < 0.01 and (obs[3] * obs[5] < 0 or obs[4] * obs[6] < 0):
        #                     print(j, obs)
        #
        #         # Reshape to do in a single forward pass for all steps
        #         values, action_log_probs, dist_entropy, _, dist = self.actor_critic.evaluate_actions(
        #             obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
        #
        #         fingerprint.append(dist.probs)
        #         break
        #
        #     # print(len(fingerprint), fingerprint[0].size())
        #     fingerprint = torch.cat(fingerprint, dim=0).view(-1)
        #     fingerprints.append(fingerprint.detach().numpy())
        #
        #     for p, g in zip(self.actor_critic.parameters(), fg):
        #         if g is not None:
        #             p.data -= 1e-4 * g
        #
        # np.array(fingerprints).dump("grads.15.fingerprint.data")

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        grad_norm_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, grad_norm_epoch
