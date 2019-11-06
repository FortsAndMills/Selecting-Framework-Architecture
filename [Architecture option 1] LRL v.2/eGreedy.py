from .utils import *

def eGreedy(parclass, epsilon_start=1, epsilon_final=0.01, epsilon_decay=500):
    """
    Basic e-Greedy exploration strategy.
    
    Args:
        epsilon_start - value of epsilon at the beginning, float, from 0 to 1
        epsilon_final - minimal value of epsilon, float, from 0 to 1
        epsilon_decay - degree of exponential damping of epsilon, int
    """
    
    class eGreedy(parclass):    
        def epsilon_by_frame(self):
            '''
            Returns current value of eps
            output: float, from 0 to 1 
            '''
            return epsilon_final + (epsilon_start - epsilon_final) * \
                math.exp(-1. * self.frames_done / epsilon_decay)

        def act(self, state):
            if self.is_learning:
                eps = self.epsilon_by_frame()
                self.log("eps", eps, "training iteration", "annealing hyperparameter")

                explore = np.random.uniform(0, 1, size=state.shape[0]) <= eps
                
                actions = np.zeros((state.shape[0], *self.action_shape), dtype=self.env.action_space.dtype)
                if explore.any():
                    actions[explore] = np.array([self.env.action_space.sample() for _ in range(explore.sum())])
                if (~explore).any():
                    actions[~explore] = super().act(state[~explore])
                return actions
            else:
                return super().act(state)
    return eGreedy
