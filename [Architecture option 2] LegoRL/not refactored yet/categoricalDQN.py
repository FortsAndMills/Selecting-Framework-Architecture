from .DQN import *

class CategoricalQ(Q):
    '''
    FloatTensor representing Categorical Q-function, (batch_size x num_actions x num_atoms)
    '''
    def __init__(self, tensor, support):
        super().__init__(tensor)
        self.support = support

    def greedy(self):
        return (self.tensor * self.support).sum(-1).max(-1)[1]
    
    def gather(self, action_b):
        return self.tensor.gather(-2, action_b[..., None, None].expand(*self.tensor.shape[:-2], 1, self.tensor.shape[-1])).squeeze(-2)
    
    def value(self):
        return self.gather(self.greedy())

class toCategoricalQ(Hat):
    def __init__(self, agent):
        super().__init__()
        self.support = agent.support
        self.num_actions = agent.num_actions
        self.num_atoms = agent.num_atoms

    def required_shape(self):
        return self.num_actions * self.num_atoms

    def forward(self, x):
        x = x.view(*x.shape[:-1], self.num_actions, self.num_atoms)
        x = F.softmax(x, dim=-1)
        return CategoricalQ(x, self.support)

    def extra_repr(self):    
        return 'output interpreted as categorical {} atoms of categorical Q-function for {} actions'.format(self.num_atoms, self.num_actions)

def CategoricalDQN(parclass, Vmin=-10, Vmax=10, num_atoms=51):
    """
    Categorical DQN.
    Based on: https://arxiv.org/pdf/1707.06887.pdf
    
    Args:
        Vmin - minimum value of approximation distribution, int
        Vmax - maximum value of approximation distribution, int
        num_atoms - number of atoms in approximation distribution, int
    """

    class CategoricalDQN(parclass):
        def __init__(self):
            assert Vmin < Vmax, "Vmin must be less than Vmax!"
            self.Vmin = Vmin
            self.Vmax = Vmax
            self.num_atoms = num_atoms   
            self.support = torch.linspace(Vmin, Vmax, num_atoms).to(device)
            
            super().__init__()            
                
        def batch_target(self, batch):
            offset = torch.linspace(0, (len(batch) - 1) * self.num_atoms, len(batch)).long().unsqueeze(1).expand(len(batch), self.num_atoms).to(device)
            delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
            
            next_dist = self.estimate_next_state(batch.next_state)

            reward_b = batch.reward.unsqueeze(1).expand_as(next_dist)
            done_b   = batch.done.unsqueeze(1).expand_as(next_dist)
            support = self.support.unsqueeze(0).expand_as(next_dist)

            Tz = reward_b + (1 - done_b) * (self.system.gamma**batch.n_steps) * support
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            b  = (Tz - self.Vmin) / delta_z
            l  = b.floor().long()
            u  = b.ceil().long()        
            
            proj_dist = Tensor(next_dist.size()).zero_()              
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float()+ (b.ceil() == b).float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
            proj_dist /= proj_dist.sum(1).unsqueeze(1)
            return proj_dist

        def regression_loss(self, q, guess):
            '''
            Calculates batch loss
            input: guess - target, FloatTensor, (batch_size, num_atoms)
            input: q - current model output, FloatTensor, (batch_size, num_atoms)
            output: FloatTensor, (batch_size)
            '''
            q.data.clamp_(1e-8, 1 - 1e-8)   # TODO doesn't torch have cross entropy? Taken from source code.
            return -(guess * q.log()).sum(1)
            
        def get_priorities(self, batch):
            # TODO: wtf?
            return batch.dqn_loss
        
        # TODO: now what?
        # def show_record(self):
        #     show_frames_and_distribution(self.record["frames"], np.array(self.record["qualities"])[:, 0], "Future reward distribution", self.support.cpu().numpy())
    return CategoricalDQN
