import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointLikeMixer(nn.Module):
    def __init__(self, args):
        super(PointLikeMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_groups = args.mixing_group_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.group_dim = args.mixing_group_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_groups)
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.group_dim, self.n_agents)
        group_qs = agent_qs.sum(dim=2).view(-1, 1, self.group_dim)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_groups, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(group_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
