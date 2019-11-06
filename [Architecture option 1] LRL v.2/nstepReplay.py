from .utils import *

def NstepReplay(parclass, replay_buffer_nsteps=3):
    """
    Stores transitions more than on one step.
    
    Args:
        replay_buffer_nsteps - N steps, int
    """

    class NstepReplay(parclass):        
        def __init__(self, replay_buffer_nsteps=3):
            super().__init__()
            
            self.replay_buffer_nsteps = replay_buffer_nsteps
            self.nstep_buffer = []
            
        def reset(self):
            super().reset()
            self.nstep_buffer = []
        
        def see(self, transitionBatch):        
            self.nstep_buffer.append(transitionBatch)
            
            if len(self.nstep_buffer) == self.replay_buffer_nsteps:      
                nstep_reward = sum([self.nstep_buffer[i].reward * (self.gamma**i) for i in range(self.replay_buffer_nsteps)])
                actual_done = max([self.nstep_buffer[i].done for i in range(self.replay_buffer_nsteps)])
                
                oldestTransitions = self.nstep_buffer.pop(0)
                state, action = oldestTransitions.state, oldestTransitions.action
                
                T = TransitionBatch(state, action, nstep_reward, transitionBatch.next_state, actual_done)

                super().see(transitionBatch)

            if len(self.nstep_buffer) == self.replay_buffer_nsteps:
                raise Exception("Error! Nstep buffer is >N")
    return NstepReplay
  
# TODO AND WHO WILL STORE N IN REPLAYBUFFER!?!?!?!?
# def CollectiveNstepReplayBufferAgent(parclass):
#   """
#   Requires parclass inherited from ReplayBufferAgent.
#   Already inherits from NstepReplay
#   """
  
#   class CollectiveNstepReplayBufferAgent(NstepReplay(parclass)):
#     """
#     Experimental. Stores all transitions from transitions on one step to transitions on n steps.
#     """
#     __doc__ += NstepReplay(parclass).__doc__
        
#     def memorize(self, state, action, reward, next_state, done):
#         self.nstep_buffer.append((state, action, reward, next_state, done))
        
#         R = np.zeros((state.shape[0]))
#         actual = np.zeros((state.shape[0]))
#         for i in reversed(range(len(self.nstep_buffer))):
#             R *= self.gamma
#             R += self.nstep_buffer[i][2] * self.gamma
#             actual = max(self.nstep_buffer[i][4], actual)
#             ReplayBufferAgent.memorize(self, self.nstep_buffer[i][0], self.nstep_buffer[i][1], R, next_state, actual)           
        
#         if len(self.nstep_buffer) >= self.config.replay_buffer_nsteps:      
#             self.nstep_buffer.pop(0)
#   return CollectiveNstepReplayBufferAgent

