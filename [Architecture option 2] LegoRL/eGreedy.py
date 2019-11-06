from .system import *
   
class eGreedy(RLmodule):
    """
    Basic e-Greedy exploration strategy.

    Args:
        greedy_policy - RLmodule with "act" method
        epsilon_start - value of epsilon at the beginning, float, from 0 to 1
        epsilon_final - minimal value of epsilon, float, from 0 to 1
        epsilon_decay - degree of exponential damping of epsilon, int

    Provides: act
    """
    def __init__(self, system, greedy_policy, epsilon_start=1, epsilon_final=0.01, epsilon_decay=500):
        super().__init__(system)
        
        self.greedy_policy = greedy_policy
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay

    def epsilon_by_frame(self):
        '''
        Returns current value of eps
        output: float, from 0 to 1 
        '''
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
            math.exp(-1. * self.system.frames_done / self.epsilon_decay)

    def act(self, state):
        if self.system.is_learning:
            eps = self.epsilon_by_frame()
            self.system.log("eps", eps, "training iteration", "annealing hyperparameter")

            explore = np.random.uniform(0, 1, size=state.shape[0]) <= eps
            
            actions = np.zeros((state.shape[0], *self.system.action_shape), dtype=self.system.env.action_space.dtype)
            if explore.any():
                actions[explore] = np.array([self.system.env.action_space.sample() for _ in range(explore.sum())])
            if (~explore).any():
                actions[~explore] = self.greedy_policy.act(state[~explore])
            return actions
        else:
            return self.greedy_policy.act(state)
