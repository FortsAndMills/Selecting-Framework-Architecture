from .system import *

class RandomActor(RLmodule):
    def act(self, state):
        return [self.system.env.action_space.sample() for _ in range(state.shape[0])]