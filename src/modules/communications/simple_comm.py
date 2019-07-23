import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SimpleComm(nn.Module):
    def __init__(self, input_shape, args):
        super(SimpleComm, self).__init__()

        self.args = args

        self.g_fc1 = nn.Linear(input_shape, 2 * input_shape)
        self.g_fc2 = nn.Linear(2 * input_shape, 2*input_shape)
        self.g_fc3 = nn.Linear(2 * input_shape, 2*input_shape)
        self.g_fc4 = nn.Linear(2 * input_shape, 1)

    def forward(self, input):
        g = F.relu(self.g_fc1(input))
        g = F.relu(self.g_fc2(g))
        g = F.relu(self.g_fc3(g))
        g = th.sigmoid(self.g_fc4(g))

        return input * g, g
