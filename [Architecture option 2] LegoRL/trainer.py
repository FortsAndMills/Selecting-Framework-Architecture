from .system import *

class Trainer(RLmodule):
    """
    Trains network using data from replay buffer memory each t-th transition.

    Args:
        replay - RLmodule, providing method "sample" and __len__
        network - RLmodule, providing method "optimize"
        batch_size - size of batch for optimization on each frame, int
        replay_buffer_init - size of buffer launching q-network optimization, int
        samples_per_transition - number of samples per transition, can be fractional, float
    """

    def __init__(self, system, replay, network, batch_size=32, replay_buffer_init=100, samples_per_transition=1.):
        super().__init__(system)

        self.replay = replay
        self.network = network
        self.batch_size = batch_size
        self.replay_buffer_init = replay_buffer_init
        self.samples_per_transition = samples_per_transition
        
        assert self.replay_buffer_init >= self.batch_size, "Batch size must be smaller than replay_buffer_init!"
        
        self.sampler_iteration_charges = 0

    def see(self, transitionBatch):
        self.replay.see(transitionBatch)

        self.sampler_iteration_charges += self.samples_per_transition * len(transitionBatch)
        while self.sampler_iteration_charges >= 1:
            self.sampler_iteration_charges -= 1
            
            if len(self.replay) >= self.replay_buffer_init:
                batch = self.replay.sample(self.batch_size)
                self.network.optimize(batch)