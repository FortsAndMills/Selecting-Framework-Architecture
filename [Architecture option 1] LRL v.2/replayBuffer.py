from .utils import *

def ReplayBuffer(parclass, replay_buffer_capacity=100000):
    """
    Replay Memory storing all transitions seen with basic uniform batch sampling.
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        replay_buffer_capacity - size of buffer, int
    """

    class ReplayBuffer(parclass):
        def __init__(self):
            super().__init__()

            self.replay_buffer_nsteps = 1
            self.replay_buffer_capacity = replay_buffer_capacity
            self.buffer = []
            self.buffer_pos = 0
        
        def store_transition(self, transition):
            """
            Remember given transition:
            input: Transition
            """        
            # preparing for concatenation into batch in future
            transition.state      = transition.state[None]
            transition.next_state = transition.next_state[None]
            
            # this seems to be the quickest way of working with experience memory
            if len(self.buffer) < self.replay_buffer_capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.buffer_pos] = transition
            
            self.buffer_pos = (self.buffer_pos + 1) % self.replay_buffer_capacity
        
        def see(self, transitionBatch):
            super().see(transitionBatch)
            for transition in transitionBatch:
                self.store_transition(transition)

        # TODO: this is not true for n-step!
        def sample(self, batch_size):
            """
            Generate batch of given size.
            input: batch_size - int
            output: Batch
            """
            return Batch(*zip(*random.sample(self.buffer, batch_size)), self.replay_buffer_nsteps, self.ActionTensor)
        
        # TODO: trigger may allow to delete this
        def process_batch(self, batch):
            pass
    return ReplayBuffer
