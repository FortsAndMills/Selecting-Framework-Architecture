from .utils import *
from copy import deepcopy

def Target(parclass, target_update=100):
    '''
    Target network heuristic implementation.
    
    Args:
        target_update - frequency in frames of updating target network
    '''
    
    class Target(parclass):    
        def __init__(self):
            super().__init__()

            self.target_update = target_update
            self.target_net = deepcopy(self.q_net)
            self.unfreeze()

        def unfreeze(self):
            '''copy policy net weights to target net'''
            self.target_net.load_state_dict(self.q_net.state_dict())

        def see(self, transitionBatch):
            super().see(transitionBatch)

            if self.frames_done % self.target_update < len(transitionBatch):
                self.unfreeze()
        
        def estimate_next_state(self, next_state_b):
            return self.target_net(next_state_b).value()

        def load(self, name, *args, **kwargs):
            super().load(name, *args, **kwargs)
            self.unfreeze()
    return Target
