import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.nfe = 0

        self.layers = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.dim)
        )
    
    def forward(self, t, z):
        self.nfe += 1
        return self.layers(z)


class ODEBlock(nn.Module):
    def __init__(self, in_channel, hidden_dim, horizon, atol=1e-9, rtol=1e-7, solver='rk4'):
        super(ODEBlock, self).__init__()
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim
        self.atol = atol
        self.rtol = rtol
        self.horizon = horizon
        self.solver = solver

        self.t = torch.linspace(0, 1, self.horizon).float()
        self.func = ODEFunc(self.in_channel, self.hidden_dim)
    
    def forward(self, z0):
        t = self.t.to(z0.device)
        zt = odeint(self.func, z0, t, atol=self.atol, rtol=self.rtol, method=self.solver).transpose(0, 1)
        return zt
