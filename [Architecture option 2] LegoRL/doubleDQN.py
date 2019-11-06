from .utils import *

def Double(parclass):
    """
    Double DQN implementation.
    Based on: https://arxiv.org/abs/1509.06461
    """
    
    class Double(parclass):                    
        def estimate_next_state(self, next_state_b):
            chosen_actions = self.q_net(next_state_b).greedy()
            return self.target_net(next_state_b).gather(chosen_actions)
    return Double
