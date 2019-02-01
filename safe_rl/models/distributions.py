import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent, Normal, Categorical as FixedCategorical


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_input, num_output):
        super(DiagGaussian, self).__init__()

        self.mean_predict = nn.Linear(num_input, num_output)
        self.std = nn.Parameter(torch.zeros(num_output))

    def forward(self, input):
        mu = self.mean_predict(input)
        sig = F.softplus(self.std)
        return Normal(mu, sig)
