from .network import *
from .system import *
    
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
    
class DQN(RLmodule):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        backbone - RLmodule for backbone with "mount_head"
        HeadNetwork - nn.Module class for head
        hat - ?

    Provides: act
    """
    def __init__(self, system, backbone, headNetwork=nn.Linear, hat=toQ):
        super().__init__(system)

        self.q_net = backbone.mount_head(self.system.observation_shape, headNetwork, hat(self.system.num_actions))
        
    def act(self, state):
        self.q_net.train()
        
        if self.system.is_learning:
           self.q_net.train()
        else:
           self.q_net.eval()
        
        with torch.no_grad():
            q = self.q_net(Tensor(state))
            
            # TODO: now what?
            #if self.is_recording:
            #    self.record["q"].append(q[0:1].cpu().numpy())
            
            return q.greedy().cpu().numpy()    
    
    def batch_target(self, batch, next_q_values):
        '''
        Calculates target for batch to learn
        input: Batch
        input: next_q_values, TODO
        output: FloatTensor, (batch_size)
        '''
        return batch.reward + (self.system.gamma**batch.n_steps) * next_q_values * (1 - batch.done)

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

    # TODO: what to do now?   
    #def show_record(self):
    #    show_frames_and_distribution(self.record["frames"], np.array(self.record["qualities"]), "Qualities", np.arange(self.config["num_actions"]))