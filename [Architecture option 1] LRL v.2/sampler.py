from .utils import *

def Sampler(parclass, batch_size=32, replay_buffer_init=100, samples_per_transition=1.):
    """
    Samples from replay buffer memory each t-th transition.

    Args:
        # DO NOT NEED FOR NOW: triggers - name of event to trigger
        batch_size - size of batch for optimization on each frame, int
        replay_buffer_init - size of buffer launching q-network optimization, int
        samples_per_transition - number of samples per transition, can be fractional, float
    """

    class Sampler(parclass):
        def __init__(self):
            super().__init__()
            self.batch_size = batch_size
            self.replay_buffer_init = replay_buffer_init
            self.samples_per_transition = samples_per_transition
            
            assert self.replay_buffer_init >= self.batch_size, "Batch size must be smaller than replay_buffer_init!"
            
            self.sampler_iteration_charges = 0

        def see(self, transitionBatch):
            super().see(transitionBatch)

            self.sampler_iteration_charges += self.samples_per_transition * len(transitionBatch)
            while self.sampler_iteration_charges >= 1:
                self.sampler_iteration_charges -= 1
                
                if len(self.buffer) >= self.replay_buffer_init:
                    batch = self.sample(self.batch_size)
                    self.process_batch(batch)
    return Sampler