from .system import *

class DQN_loss(RLmodule):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        dqn - RLmodule for DQN with ...

    Provides: loss
    """
    def __init__(self, system, dqn):
        super().__init__(system)
        self.dqn = dqn

    def estimate_next_state(self, next_state_b):
        '''
        Calculates estimation of next state.
        input: next_state_b - FloatTensor, (batch_size x state_dim)
        output: FloatTensor, batch_size
        '''
        return self.dqn.q_net(next_state_b).value()

    def loss(self, batch):
        '''
        Loss calculation based on TD-error from DQN algorithm.
        input: Batch
        output: Tensor, (batch_size)
        '''

        # getting q values for state and next state
        self.q_net.train()
        q = self.dqn.q_net(batch.state).gather(batch.action)
        with torch.no_grad():
            next_q_values = self.estimate_next_state(batch.next_state)
            target = self.dqn.batch_target(batch, next_q_values)
            
        # getting loss
        loss_b = self.dqn.regression_loss(q, target)        
        assert len(loss_b.shape) == 1, loss_b
        
        return loss_b