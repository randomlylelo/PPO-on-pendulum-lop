import numpy as np
import torch
import torch.nn as nn
import collections
from cbp_linear import CBPLinear

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
        def __init__(self, input_size, output_size, hidden_size=64, activation=nn.functional.relu):
                super(Net, self).__init__()
                self.layer1 = nn.Linear(input_size, hidden_size)
                self.layer2 = nn.Linear(hidden_size, hidden_size)
                self.layer3 = nn.Linear(hidden_size, output_size)
                self.act = activation

                REPLACEMENT_RATE = 0
                MATURITY_THRESHOLD = 100
                INIT = 'default'
                self.cbp1 = CBPLinear(in_layer=self.layer1, out_layer=self.layer2, replacement_rate=REPLACEMENT_RATE, maturity_threshold=MATURITY_THRESHOLD, init=INIT)
                self.cbp2 = CBPLinear(in_layer=self.layer2, out_layer=self.layer3, replacement_rate=REPLACEMENT_RATE, maturity_threshold=MATURITY_THRESHOLD, init=INIT)

        def forward(self, x):
                x = self.cbp1(self.act(self.layer1(x)))
                x = self.cbp2(self.act(self.layer2(x)))
                # x = self.act(self.layer1(x))
                # x = self.act(self.layer2(x))
                out = self.layer3(x)

                return out

class PolicyNet(nn.Module):
        def __init__(self, input_size, output_size, hidden_size=64, activation=nn.functional.relu):
                super(PolicyNet, self).__init__()
                self.layer1 = nn.utils.spectral_norm(nn.Linear(input_size, hidden_size))
                self.layer2 = nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size))
                self.layer3 = nn.utils.spectral_norm(nn.Linear(hidden_size, output_size))
                self.act = activation

        def forward(self, x):
                x = self.act(self.layer1(x))
                x = self.act(self.layer2(x))
                out = self.layer3(x)

                return out

class ReplayMemory():
    def __init__(self, batch_size=10000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []
        self.batch_size = batch_size

    def push(self, state, action, reward, reward_togo, advantage, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.rewards_togo.append(reward_togo)
        self.advantages.append(advantage)
        self.values.append(value)  
        self.log_probs.append(log_prob)

    def sample(self):
        num_states = len(self.states)
        batch_start = torch.arange(0, num_states, self.batch_size)
        indices = torch.randperm(num_states)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (torch.tensor(self.states).to(device),
                torch.tensor(self.actions).to(device),
                torch.tensor(self.rewards).to(device),
                torch.tensor(self.rewards_togo).to(device),
                torch.tensor(self.advantages).to(device),
                torch.tensor(self.values).to(device),
                torch.tensor(self.log_probs).to(device),
                batches)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []


        



