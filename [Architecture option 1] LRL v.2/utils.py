import math
import random
import numpy as np
import pickle
import time

from collections import defaultdict
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal

import gym
import gym.spaces        # to avoid warnings
gym.logger.set_level(40) # to avoid warnings

USE_CUDA = torch.cuda.is_available()
Tensor = lambda *args, **kwargs: torch.FloatTensor(*args, **kwargs).cuda() if USE_CUDA else torch.FloatTensor(*args, **kwargs)
LongTensor = lambda *args, **kwargs: torch.LongTensor(*args, **kwargs).cuda() if USE_CUDA else torch.LongTensor(*args, **kwargs)
device = "cuda" if torch.cuda.is_available() else "cpu"

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# allegedly takes less memory than tuples 
class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.done

class TransitionBatch(Transition):
    def __iter__(self):
        for s, a, r, ns, done in zip(self.state, self.action, self.reward, self.next_state, self.done):
            yield Transition(s, a, r, ns, done)

    def __len__(self):
        return self.state.shape[0]

# TODO: make part of agent class?
class Batch():
    def __init__(self, state, action, reward, next_state, done, n_steps, ActionTensor):
        self.state = Tensor(np.concatenate(state))
        self.action = ActionTensor(action)
        self.reward = Tensor(reward)
        self.next_state = Tensor(np.concatenate(next_state)) 
        self.done = Tensor(done)
        self.n_steps = n_steps
        self.losses = {}
        
    def __len__(self):
        return self.state.shape[0]

# TODO: WTF?
# def align(tensor, i):
#     """
#     Adds i singleton dimensions to the end of tensor 
#     """
#     for _ in range(i):
#         tensor = tensor[:, None]
#     return tensor

