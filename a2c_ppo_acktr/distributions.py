import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def likelihoods(self, actions):
        return self.log_probs(actions)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)
        # _log_probs = (super().log_prob(actions) + torch.log(self.scale) + math.log(math.sqrt(2 * math.pi))) / 100.
        # return _log_probs.sum(-1, keepdim=True)

    def likelihoods(self, actions):
        log_probs = super().log_prob(actions) + torch.log(self.scale) + math.log(math.sqrt(2 * math.pi))
        return log_probs.sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def likelihoods(self, actions):
        return self.log_probs(actions)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class FixedNormalCategoricalMixture:
    # NOTE: this implementation is just for Agar! it is not general!
    def __init__(self, logits, mean, std, real_action_dims):
        self.real_action_dims = real_action_dims
        self.c = FixedCategorical(logits=logits)
        self.n = FixedNormal(mean, std)
    
    def sample(self):
        return torch.cat([self.n.sample(), self.c.sample()], -1)

    def log_probs(self, actions):
        continuous_a, discrete_a = actions.split(self.real_action_dims, -1)
        return torch.cat([self.n.log_probs(continuous_a), self.c.log_probs(discrete_a)], -1)

    def likelihoods(self, actions):
        continuous_a, discrete_a = actions.split(self.real_action_dims, -1)
        return self.n.likelihoods(continuous_a) + self.c.likelihoods(discrete_a)

    def entropy(self):
        return self.c.entropy() + self.n.entropy()

    def mode(self):
        return torch.cat((self.n.mean, self.c.mode()), -1)


class NormalCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, real_action_dims, is_ref=False):
        super().__init__()

        if is_ref:
            init_ = lambda m: m
        else:
            init_ = lambda m: init(
                m,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
                gain=0.01)
        
        self.num_outputs = num_outputs
        self.real_action_dims = real_action_dims
        self.c_linear = init_(nn.Linear(num_inputs, num_outputs[1]))  # 2
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs[0]))  # 4
        # TODO: now we simply fix standard deviation
        self.logstd = nn.Parameter(torch.zeros(num_outputs[0]), requires_grad=True)
    
    def forward(self, x):
        mean = self.fc_mean(x)
        logits = self.c_linear(x)
        return FixedNormalCategoricalMixture(logits=logits, mean=mean, std=self.logstd.exp(), real_action_dims=self.real_action_dims)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, is_ref=False):
        super(Categorical, self).__init__()

        if is_ref:
            init_ = lambda m: m
        else:
            init_ = lambda m: init(
                m,
                nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0),
                gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        # self.linear = nn.Linear(num_inputs, num_outputs, bias=False)

    def forward(self, x):
        x = self.linear(x)
        # print(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, is_ref):
        super(DiagGaussian, self).__init__()

        if is_ref:
            init_ = lambda m: m
        else:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.num_outputs = num_outputs
        # in original implementation, std of gaussian is fixed
        # however, logstd of gaussian can be a part of outputs of the linear layer
        self.fc_mean_std = init_(nn.Linear(num_inputs, num_outputs * 2))

    def forward(self, x):
        action_mean, action_logstd = self.fc_mean_std(x).split(self.num_outputs, -1)

        # print(action_mean)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
