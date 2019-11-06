from .DQN import *

class toDuelingQ(Hat):
    def __init__(self, agent):
        super().__init__()
        self.num_actions = agent.num_actions

    def required_shape(self):
        return self.num_actions + 1
        
    def forward(self, x):
        assert x.shape[-1] == self.num_actions + 1
        v, a = torch.split(x, [1, self.num_actions], dim=-1)
        return Q(v + a - a.mean(dim=-1, keepdim=True))

    def extra_repr(self):    
        return 'output interpreted as V + A - A.mean() for {} actions'.format(self.num_actions)