from .network import *
    
class Q():
    '''
    FloatTensor representing Q-function, (batch_size x num_actions)
    '''
    def __init__(self, tensor):
        self.tensor = tensor

    def greedy(self):
        '''
        Returns greedy action based on the output of net
        output: LongTensor, (batch_size)
        '''
        return self.tensor.max(-1)[1]
        
    def gather(self, action_b):
        '''
        Returns output of net for given batch of actions
        input: action_b - LongTensor, (batch_size)
        output: FloatTensor, (batch_size)
        '''
        return self.tensor.gather(-1, action_b.unsqueeze(-1)).squeeze(-1)
    
    def value(self):
        '''
        Returns value of action, chosen greedy
        output: FloatTensor, (batch_size)
        '''
        return self.tensor.max(-1)[0]

class toQ(Hat):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def required_shape(self):
        return self.num_actions

    def forward(self, x):
        assert x.shape[-1] == self.num_actions
        return Q(x)

    def extra_repr(self):    
        return 'output interpreted as Q-function for {} actions'.format(self.num_actions)

def DQN(parclass, HeadNetwork=nn.Linear, hat=toQ, backbone="the only network"):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        HeadNetwork - nn.Module class for head
        backbone - name of network module, providing backbone for DQN network, str
    """
    class DQN(parclass):
        def __init__(self):
            super().__init__()

            assert backbone in self.personal_data, "Backbone network does not exist"

            self.q_net = self.mount_head(backbone, self.observation_shape, HeadNetwork, hat(self.num_actions))
            self.add_loss(backbone, "dqn", self.dqn_loss)

        def act(self, state):
            self.q_net.train()
            
            # TODO: now what?
            #if self.is_learning:
            #    self.q_net.train()
            #else:
            #    self.q_net.eval()
            
            with torch.no_grad():
                q = self.q_net(Tensor(state))
                
                # TODO: now what?
                #if self.is_recording:
                #    self.record["q"].append(q[0:1].cpu().numpy())
                
                return q.greedy().cpu().numpy()

        def estimate_next_state(self, next_state_b):
            '''
            Calculates estimation of next state.
            input: next_state_b - FloatTensor, (batch_size x state_dim)
            output: FloatTensor, batch_size
            '''
            return self.q_net(next_state_b).value()
        
        def batch_target(self, batch):
            '''
            Calculates target for batch to learn
            input: Batch
            output: FloatTensor, (batch_size)
            '''
            next_q_values = self.estimate_next_state(batch.next_state)
            return batch.reward + (self.gamma**batch.n_steps) * next_q_values * (1 - batch.done)

        def regression_loss(self, q, guess):
            '''
            Calculates batch loss
            input: q - current model output, FloatTensor, (batch_size)
            input: guess - target, FloatTensor, (batch_size)
            output: FloatTensor, (batch_size)
            '''
            return (guess - q).pow(2)
        
        def get_priorities(self, batch):
            '''
            Calculates importance of transitions in batch by loss
            input: batch - FloatTensor, (batch_size)
            output: FloatTensor, (batch_size)
            '''
            # TODO: wtf?
            return batch.dqn_loss**0.5

        def dqn_loss(self, batch):
            '''
            Loss calculation based on TD-error from DQN algorithm.
            input: Batch
            output: Tensor, (batch_size)
            '''

            # getting q values for state and next state
            self.q_net.train()
            q = self.q_net(batch.state).gather(batch.action)
            with torch.no_grad():
                target = self.batch_target(batch)
                
            # getting loss
            loss_b = self.regression_loss(q, target)        
            assert len(loss_b.shape) == 1, loss_b

            # TODO: shouldn't it happen in network module?
            batch.dqn_loss = loss_b
            
            return loss_b

        # TODO: what to do now?   
        #def show_record(self):
        #    show_frames_and_distribution(self.record["frames"], np.array(self.record["qualities"]), "Qualities", np.arange(self.config["num_actions"]))
    return DQN